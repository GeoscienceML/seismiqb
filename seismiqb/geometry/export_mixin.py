""" Mixin to hold methods for exporting array-like data as SEG-Y files. """
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import segyio

from batchflow.notifier import Notifier

from .memmap_loader import MemmapLoader


class ExportMixin:
    """ Methods for exporting arrays (or array-likes) to a SEG-Y files with given spec. """
    #pylint: disable=redefined-builtin, protected-access

    # Specs for export
    def make_export_spec(self, array_like, origin=(0, 0, 0)):
        """ Create a description of the current geometry.
        Includes file-wide attributes: `sample_interval`, `delay`, `format` and `sorting`,
        coordinate descriptions: `ilines`, `xlines` and `samples`, and matrices for other headers.
        Can be used directly to create SEG-Y file by `segyio`.

        Parameters
        ----------
        array_like : array like
            An object with numpy-like getitem semantics.
        origin : tuple of three integers
            Coordinates of the upper leftmost point of the `array_like`.
        """
        # Parse parameters
        file_handler = self.loader.file_handler
        slices = tuple(slice(o, o+s) for o, s in zip(origin, array_like.shape))

        spec = segyio.spec()

        # File-wide values
        spec.sample_interval = self.sample_interval
        spec.delay = int(self.delay) + int(self.sample_interval * origin[-1])
        spec.format = 5 if file_handler.format is None else int(file_handler.format)
        spec.sorting = 2 if file_handler.sorting is None else int(file_handler.sorting)

        # Structure
        spec.ilines = self.index_sorted_uniques[0][slices[0]]
        spec.xlines = self.index_sorted_uniques[1][slices[1]]
        spec.samples = np.arange(array_like.shape[2], dtype=np.int32)

        # Additional matrices
        spec.cdp_x_matrix = self.compute_header_values_matrix('CDP_X')[slices[:2]]
        spec.cdp_y_matrix = self.compute_header_values_matrix('CDP_Y')[slices[:2]]
        return spec

    @staticmethod
    def default_export_spec(array_like, origin=(0, 0, 0), sample_interval=2.0, delay=0, sorting=2, format=5,
                            iline_shift=1, iline_step=1, xline_shift=1, xline_step=1,
                            cdp_x_shift=100_000, cdp_x_step=25, cdp_y_shift=300_000, cdp_y_step=25):
        """ Create default description of SEG-Y file.
        Includes file-wide attributes: `sample_interval`, `delay`, `format` and `sorting`,
        coordinate descriptions: `ilines`, `xlines` and `samples`, and matrices for other headers.
        Can be used directly to create SEG-Y file by `segyio`.

        Parameters
        ----------
        array_like : array like
            An object with numpy-like getitem semantics.
        origin : tuple of three integers
            Coordinates of the upper leftmost point of the `array_like`.
        sample_interval, delay, sorting, format : numbers
            Directly used in SEG-Y creation, according to SEG-Y standard.
        *_shift : int
            Starting (minimum) value of corresponding header.
        *_step : int
            Increment between consecutive values of corresponding header.
        """
        spec = segyio.spec()

        # File-wide values
        spec.sample_interval = sample_interval
        spec.delay = int(delay) + int(sample_interval * origin[-1])
        spec.sorting = sorting
        spec.format = format

        # Structure
        i_start, x_start = iline_shift + origin[0], xline_shift + origin[1]
        spec.ilines = np.arange(i_start, i_start + iline_step * array_like.shape[0], dtype=np.int32)
        spec.xlines = np.arange(x_start, x_start + xline_step * array_like.shape[1], dtype=np.int32)
        spec.samples = np.arange(array_like.shape[2], dtype=np.int32)

        # Additional matrices
        spec.cdp_x_matrix = np.tile(cdp_x_shift + cdp_x_step * spec.ilines.reshape(-1, 1), array_like.shape[1])
        spec.cdp_y_matrix = np.tile(cdp_y_shift + cdp_y_step * spec.xlines.reshape(-1, 1), array_like.shape[0]).T
        return spec

    # Public APIs
    @staticmethod
    def array_to_segy(array_like, path, spec=None, origin=(0, 0, 0), pbar='t', zip_segy=False, remove_segy=False,
                      engine='memmap', format=5, transform=None, endian_symbol='>', chunk_size=20, max_workers=4,
                      **kwargs):
        """ Convert an `array_like` object to a SEG-Y file.
        In order to determine values of bin/trace headers, one should provide `spec`:
            - if no spec provided, we use the default one. It fills coordinate values with shifted ranges.
            - if a path to SEG-Y file or the instance of :class:`GeometrySEGY` is provided, we use its
            headers to create a matching spec. The resulting file will have the same headers, as that file,
            and it is easy to compare two such SEG-Y files in geological software.

        Parameter `origin`, coupled with the fact that this function works with arbitrary shaped `array_like`,
        allows one to save `array_like` which is a subset of the original (used for `spec`) SEG-Y file.
        For example, if the spec SEG-Y has the (1000, 2000, 3000) shape,
        and `array_like` has the (500, 1000, 1500) shape, we can use origin (250, 50, 750) meaning that `array_like`
        should be placed in the middle 1/8 of the volume.

        This method has two underlying implementations for actually writing the data:
            - engine `segyio` uses segyio library to write data on a trace-by-trace basis.
            It is slow, but well-tested and easy to understand/maintain.
            - engine `memmap` uses numpy memory mapping mechanism to write data in chunks in multiple threads.
            It is a lot faster and also allows to write SEG-Ys with arbitrary data formats: for example, integer values.

        # TODO: write the textual header, as well as the extended ones.
        # TODO: full-copy of trace headers or additional headers to write.

        Parameters
        ----------
        array_like : array like
            An object with numpy-like getitem semantics.
        path : str
            Path to save the SEG-Y to.
        spec : object
            Object with the following mandatory attributes:
            - file-wide parameters `sample_interval`, `delay`, `format` and `sorting`
            - coordinate grid along each axis `ilines`, `xlines` and `samples`
            - mapping from ordinal spatial coordinates to header values: `cdp_x_matrix` and `cdp_y_matrix`.
            Refer to :meth:`make_export_spec` and :meth:`default_export_spec` for details.
        origin : tuple of three integers
            Coordinates of the upper leftmost point of the `array_like`.
        pbar : bool, str
            If bool, then whether to display progress bar.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        zip_segy : bool
            Whether to compress the created SEG-Y into a zip archive. May take a lot of time with no progress bar.
        remove_segy : bool
            Whether to remove the SEG-Y. Useful when combined with `zip_segy` to keep only the zipped version.
        engine : {'segyio', 'memmap'}
            Which engine for file writing to use.
        format : int
            Target SEG-Y format. Refer to SEG-Y standard for detailed description.
        transform : callable, optional
            Function to transform array values before writing. Useful to change the dtype.
            Must return the same dtype, as specified by `format`.
        endian_symbol, chunk_size, max_workers
            Directly passed to :meth:`array_to_segy_memmap`.
        """
        #pylint: disable=import-outside-toplevel
        from .segy import GeometrySEGY

        # Select the spec
        if spec is None:
            spec = ExportMixin.default_export_spec(array_like=array_like, origin=origin, format=format, **kwargs)
        if isinstance(spec, str):
            spec = GeometrySEGY(spec)
        if isinstance(spec, GeometrySEGY):
            spec = spec.make_export_spec(array_like=array_like, origin=origin)

        # Export the data
        if engine == 'segyio':
            ExportMixin.array_to_segy_segyio(array_like=array_like, path=path, spec=spec, pbar=pbar)
        else:
            ExportMixin.array_to_segy_memmap(array_like=array_like, path=path, spec=spec, pbar=pbar,
                                             format=format, endian_symbol=endian_symbol, transform=transform,
                                             chunk_size=chunk_size, max_workers=max_workers)

        # Finalize: optionally, compress to `zip` and remove `SEG-Y`
        if zip_segy:
            dir_name = os.path.dirname(os.path.abspath(path))
            file_name = os.path.basename(path)
            shutil.make_archive(os.path.splitext(path)[0], 'zip', dir_name, file_name)
        if remove_segy:
            os.remove(path)

    def export_array(self, array_like, path, origin=(0, 0, 0), pbar='t', zip_segy=False, remove_segy=False,
                     engine='memmap', format=5, transform=None, endian_symbol='>', chunk_size=20, max_workers=4):
        """ An alias to :meth:`array_to_segy` which uses `self` as spec. """
        spec = self.make_export_spec(array_like=array_like, origin=origin)
        ExportMixin.array_to_segy(array_like=array_like, path=path, spec=spec, pbar=pbar,
                                  zip_segy=zip_segy, remove_segy=remove_segy,
                                  engine=engine, format=format, transform=transform, endian_symbol=endian_symbol,
                                  chunk_size=chunk_size, max_workers=max_workers)


    # Export engines
    @staticmethod
    def array_to_segy_segyio(array_like, path, spec, pbar='t'):
        """ Write `array_like` as a SEG-Y file to `path` according to `spec`.
        Does so on a trace-by-trace basis.
        """
        with segyio.create(path, spec) as dst_file:
            # Write binary header
            dst_file.bin.update({
                segyio.BinField.Samples: len(spec.samples),
                segyio.BinField.Interval: int(spec.sample_interval * 1000),
            })

            # Iterate over traces, writing headers/data to the dst
            basename = os.path.basename(path)
            notifier = Notifier(pbar, total=len(spec.ilines), desc=f'Writing `{basename}`')

            c = 0
            for i, il in notifier(enumerate(spec.ilines)):
                # Load full slice: speeds-up the case where `array_like` is HDF5 dataset
                slide = array_like[i]

                for x, xl in enumerate(spec.xlines):
                    # Write trace header values
                    dst_file.header[c].update({
                        segyio.TraceField.INLINE_3D: il,
                        segyio.TraceField.CROSSLINE_3D: xl,
                        segyio.TraceField.CDP_X: spec.cdp_x_matrix[i, x],
                        segyio.TraceField.CDP_Y: spec.cdp_y_matrix[i, x],

                        segyio.TraceField.TRACE_SAMPLE_COUNT: len(spec.samples),
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(spec.sample_interval * 1000),
                        segyio.TraceField.DelayRecordingTime: spec.delay
                    })

                    # Write trace data values
                    dst_file.trace[c] = slide[x]
                    c += 1


    @staticmethod
    def array_to_segy_memmap(array_like, path, spec, format=5, endian_symbol='>', transform=None,
                             chunk_size=20, max_workers=4, pbar='t'):
        """ Write `array_like` as a SEG-Y file to `path` according to `spec`.
        Does so by chunks along the inline direction in multiple threads.
        Threads are used instead of processes to avoid the need to pass `array_like` between processes.

        Parameters
        ----------
        format : int
            Target SEG-Y format. Refer to SEG-Y standard for detailed description.
        transform : callable, optional
            Function to transform array values before writing. Useful to change the dtype.
            Must return the same dtype, as specified by `format`.
        endian_symbol : {'>', '<'}
            Symbol of big/little endianness.
        chunk_size : int
            Maximum number of full inlines to include in one chunk.
        max_workers : int
            Maximum number of threads for parallelization.
        pbar : bool, str
            If bool, then whether to display progress bar.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        """
        # Parse parameters
        n_traces = len(spec.ilines) * len(spec.xlines)
        n_samples = len(spec.samples)
        spec.format = format

        # Compute target dtype, itemsize, size of the dst file
        dst_dtype = endian_symbol + MemmapLoader.SEGY_FORMAT_TO_TRACE_DATA_DTYPE[spec.format]
        dst_itemsize = np.dtype(dst_dtype).itemsize
        dst_size = 3600 + n_traces * (MemmapLoader.TRACE_HEADER_SIZE + n_samples * dst_itemsize)

        # Create new file
        dst_mmap = np.memmap(path, mode='w+', shape=(dst_size, ), dtype=np.uint8)
        dst_mmap[:3600] = 0

        # Write file-wide 'Interval', 'Samples' and 'Format' headers
        # TODO: can be changed to a custom 400-bytes long np.dtype
        dst_mmap[3217-1:3217-1+2] = np.array([int(spec.sample_interval * 1000)], dtype=endian_symbol + 'u2').view('u1')
        dst_mmap[3221-1:3221-1+2] = np.array([n_samples], dtype=endian_symbol + 'u2').view('u1')
        dst_mmap[3225-1:3225-1+2] = np.array([spec.format], dtype=endian_symbol + 'u2').view('u1')

        # Zero-fill all of headers, if on Windows.
        # On POSIX complaint systems, memmap is initialized with zeros by default.
        if os.name == 'nt':
            mmap_trace_dtype = np.dtype([('headers', np.uint8, MemmapLoader.TRACE_HEADER_SIZE),
                                         ('data', dst_dtype, n_samples)])
            dst_mmap = np.memmap(path, mode='r+', offset=3600, shape=(n_traces, ), dtype=mmap_trace_dtype)
            dst_mmap['headers'] = 0

        # Prepare the export dtype
        mmap_trace_headers_dtype = MemmapLoader._make_mmap_headers_dtype(('INLINE_3D', 'CROSSLINE_3D',
                                                                          'CDP_X', 'CDP_Y',
                                                                          'TRACE_SAMPLE_COUNT',
                                                                          'TRACE_SAMPLE_INTERVAL',
                                                                          'DelayRecordingTime'))
        mmap_trace_dtype = np.dtype([*mmap_trace_headers_dtype,
                                     ('data', dst_dtype, n_samples)])

        # Split the whole file along ilines into chunks no larger than `chunk_size`
        n_chunks, last_chunk_size = divmod(len(spec.ilines), chunk_size)
        chunk_sizes = [chunk_size] * n_chunks
        if last_chunk_size:
            chunk_sizes += [last_chunk_size]
        chunk_starts = np.cumsum([0] + chunk_sizes[:-1])

        # Write trace headers and values in chunks in multiple threads
        basename = os.path.basename(path)
        with Notifier(pbar, total=len(spec.ilines), desc=f'Writing `{basename}`') as progress_bar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                def callback(future):
                    chunk_size = future.result()
                    progress_bar.update(chunk_size)

                for start, chunk_size_ in zip(chunk_starts, chunk_sizes):
                    future = executor.submit(write_chunk, path=path,
                                             shape=n_traces, offset=3600, dtype=mmap_trace_dtype,
                                             spec=spec, array_like=array_like, transform=transform,
                                             start=start, chunk_size=chunk_size_)
                    future.add_done_callback(callback)


def write_chunk(path, shape, offset, dtype, spec, array_like, transform, start, chunk_size):
    """ Write one chunk on disk: headers values and actual data.
    We create memory mapping anew in each worker, as it is easier and creates no significant overhead.
    """
    # Create memory mapping and compute correct trace indices (TRACE_SEQUENCE_FILE)
    dst_mmap = np.memmap(path, mode='r+', shape=shape, offset=offset, dtype=dtype)
    tsf_start = start * len(spec.xlines)
    tsf_end = tsf_start + chunk_size * len(spec.xlines)
    dst_traces = dst_mmap[tsf_start : tsf_end]

    # Write trace headers
    dst_traces['INLINE_3D'] = np.repeat(spec.ilines[start:start+chunk_size], len(spec.xlines))
    dst_traces['CROSSLINE_3D'] = np.tile(spec.xlines, chunk_size)
    dst_traces['CDP_X'] = spec.cdp_x_matrix[start:start+chunk_size].ravel()
    dst_traces['CDP_Y'] = spec.cdp_y_matrix[start:start+chunk_size].ravel()

    dst_traces['TRACE_SAMPLE_COUNT'] = len(spec.samples)
    dst_traces['TRACE_SAMPLE_INTERVAL'] = int(spec.sample_interval * 1000)
    dst_traces['DelayRecordingTime'] = spec.delay

    # Write trace data
    data = array_like[start:start+chunk_size]
    if transform is not None:
        data = transform(data)
    dst_traces['data'] = data.reshape(-1, len(spec.samples))
    return chunk_size

# Convenient aliases for staticmethod
array_to_segy = array_to_sgy = ExportMixin.array_to_segy
