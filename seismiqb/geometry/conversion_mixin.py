""" Mixin for geometry conversions. """
import os

import cv2
import numpy as np
import h5pickle as h5py
import hdf5plugin

from batchflow import Notifier, Quantizer


def resize_3D(array, factor):
    """ Resize 3D array along the last axis. """
    resampled_depth = int(array.shape[2] * factor)
    buffer = np.empty(shape=(*array.shape[:2], resampled_depth), dtype=array.dtype)

    for i, item in enumerate(array):
        cv2.resize(item, dsize=(resampled_depth, array.shape[1]), dst=buffer[i])
    return buffer



class ConversionMixin:
    """ Methods for converting data to other formats. """
    #pylint: disable=redefined-builtin, import-outside-toplevel
    PROJECTION_NAMES = {0: 'projection_i', 1: 'projection_x', 2: 'projection_d'}    # names of projections
    TO_PROJECTION_TRANSPOSITION = {0: [0, 1, 2], 1: [1, 0, 2], 2: [2, 0, 1]}        # re-order axis to given projection
    FROM_PROJECTION_TRANSPOSITION = {0: [0, 1, 2], 1: [1, 0, 2], 2: [1, 2, 0]}      # revert the previous re-ordering

    @staticmethod
    def compute_axis_transpositions(axis):
        """ Compute transpositions of original (inline, crossline, depth) axes to a given projection.
        Returns a transposition to that projection and from it.
        """
        return ConversionMixin.TO_PROJECTION_TRANSPOSITION[axis], ConversionMixin.FROM_PROJECTION_TRANSPOSITION[axis]


    # Quantization
    def compute_quantization_parameters(self, ranges=0.99, clip=True, center=False, dtype=np.int8,
                                        n_quantile_traces=100_000, seed=42):
        """ Compute parameters, needed for quantizing data to required range.
        Also evaluates quantization error by comparing subset of data with its dequantized quantized version.
        On the same subset, stats like mean, std and quantile values are computed.

        Parameters
        ----------
        ranges : float or sequence of two numbers
            Ranges to quantize data to.
            If float, then used as quantile to clip data to. If two numbers, then this exact range is used.
        clip : bool
            Whether to clip data to selected ranges.
        center : bool
            Whether to make data have 0-mean before quantization.
        n_quantile_traces : int
            Size of the subset to compute quantiles.
        seed : int
            Seed for quantile traces subset selection.

        Returns
        -------
        quantization_parameters : dict
            Dictionary with keys for stats and methods of data transformation.
            `'quantizer'` key is the instance, which can be `called` to quantize arbitrary array.
        """
        if isinstance(ranges, float):
            qleft, qright = self.get_quantile([1 - ranges, ranges])
            value = min(abs(qleft), abs(qright))
            ranges = (-value, +value)

        if center:
            ranges = tuple(item - self.v_mean for item in ranges)

        quantizer = Quantizer(ranges=ranges, clip=clip, center=center, mean=self.mean, dtype=dtype)

        # Load subset of data to compute quantiles
        alive_traces_indices = self.index_matrix[~self.dead_traces_matrix].ravel()
        indices = np.random.default_rng(seed=seed).choice(alive_traces_indices, size=n_quantile_traces)
        data = self.load_by_indices(indices)
        quantized_data = quantizer.quantize(data)

        mean, std = quantized_data.mean(), quantized_data.std()
        quantile_values = np.quantile(quantized_data, q=self.quantile_support)
        quantile_values[0], quantile_values[-1] = -127, +128

        # Estimate quantization error
        dequantized_data = quantizer.dequantize(quantized_data)
        quantization_error = np.mean(np.abs(dequantized_data - data)) / self.std

        return {
            'ranges': quantizer.ranges, 'center': quantizer.center, 'clip': clip,

            'quantizer': quantizer,
            'transform': quantizer.quantize,
            'dequantize': quantizer.dequantize,
            'quantization_error': quantization_error,

            'min': -127, 'max': +127,
            'mean': mean, 'std': std,
            'quantile_values': quantile_values,
        }

    # Convert SEG-Y
    def convert_to_hdf5(self, path=None, overwrite=True, postfix=False, projections='ixd',
                        quantize=False, quantization_parameters=None, dataset_kwargs=None, chunk_size_divisor=1,
                        pbar='t', store_meta=True, **kwargs):
        """ Convert SEG-Y file to a more effective storage.

        Parameters
        ----------
        path : str
            If provided, then path to save file to.
            Otherwise, file is saved under the same name with different extension.
        postfix : bool or str
            Whether to add before extension. Used only if the `path` is not provided. If True, it will be
            created automatically depending on conversion parameters.
        projections : str
            Which projections of data to store: `i` for the inline one, `x` for the crossline, `d` for depth.
        quantize : bool
            Whether to quantize data to `int8` dtype. If True, then `q` is appended to extension.
        quantization_parameters : dict, optional
            If provided, then used as parameters for quantization.
            Otherwise, parameters from the call to :meth:`compute_quantization_parameters` are used.
        pbar : bool, str
            If bool, then whether to display progress bar.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        store_meta : bool
            Whether to store meta in the same file.
        dataset_kwargs : dict, optional
            Parameters, passed directly to the dataset constructor.
            If not provided, we use the blosc compression with `lz4hc` compressor, clevel 6 and no bit shuffle.
        kwargs : dict
            Other parameters, passed directly to the file constructor.
        """
        # Quantization
        if quantize:
            if quantization_parameters is None:
                quantization_parameters = self.compute_quantization_parameters()
            dtype, transform = np.int8, quantization_parameters['transform']
        else:
            dtype, transform = np.float32, lambda array: array

        # Default path: right next to the original file with new extension
        if path is None:
            path = self.make_output_path(format='hdf5', quantize=quantize, postfix=postfix, projections=projections,
                                         chunk_size_divisor=chunk_size_divisor)

        # Dataset creation parameters
        if dataset_kwargs is None:
            dataset_kwargs = dict(hdf5plugin.Blosc(cname='lz4hc', clevel=6, shuffle=0))

        # Remove file, if exists
        if os.path.exists(path) and overwrite:
            os.remove(path)

        # Create file and datasets inside
        with h5py.File(path, mode='w-', **kwargs) as file:
            total = sum((letter in projections) * self.shape[idx]
                        for idx, letter in enumerate('ixd'))
            progress_bar = Notifier(pbar, total=total, ncols=110)
            name = os.path.basename(path)

            for p in projections:
                # Projection parameters
                axis = self.parse_axis(p)
                projection_name = self.PROJECTION_NAMES[axis]
                projection_transposition = self.TO_PROJECTION_TRANSPOSITION[axis]
                projection_shape = self.shape[projection_transposition]

                # Create dataset
                dataset_kwargs_ = {'chunks': (1, *projection_shape[1:] // chunk_size_divisor),
                                   **dataset_kwargs}
                projection = file.create_dataset(projection_name, shape=projection_shape, dtype=dtype,
                                                 **dataset_kwargs_)

                # Write data on disk
                progress_bar.set_description(f'Converting to {name}:{p}')
                for idx in range(self.shape[axis]):
                    slide = self.load_slide(idx, axis=axis)
                    slide = transform(slide)
                    projection[idx, :, :] = slide

                    progress_bar.update()
            progress_bar.close()

        # Save meta to the same file. If quantized, replace stats with the correct ones
        from .base import Geometry
        geometry = Geometry.new(path)

        if store_meta:
            self.dump_meta(path=path)

            if quantize:
                quantization_parameters['quantization_ranges'] = quantization_parameters['ranges']
                for key in ['quantization_ranges', 'center', 'clip', 'quantization_error',
                            'min', 'max', 'mean', 'std', 'quantile_values']:
                    geometry.meta_storage.store_item(key=key, value=quantization_parameters[key], overwrite=True)
        return geometry

    def repack_segy(self, path=None, format=8, transform=None, quantization_parameters=None,
                    chunk_size=25_000, max_workers=4, pbar='t', store_meta=True, overwrite=True):
        """ Repack SEG-Y file with a different `format`: dtype of data values.
        Keeps the same binary header (except for the 3225 byte, which stores the format).
        Keeps the same header values for each trace: essentially, only the values of each trace are changed.

        The most common scenario of this function usage is to convert float32 SEG-Y into int8 one:
        the latter is a lot faster and takes ~4x less disk space at the cost of some information loss.

        Parameters
        ----------
        path : str, optional
            Path to save file to. If not provided, we use the path of the current cube with an added postfix.
        format : int
            Target SEG-Y format.
            Refer to :attr:`~.MemmapLoader.SEGY_FORMAT_TO_TRACE_DATA_DTYPE` for
            list of available formats and their data value dtype.
        transform : callable, optional
            Callable to transform data from the current file to the ones, saved in `path`.
            Must return the same dtype, as specified by `format`.
        quantization_parameters : dict, optional
            If provided, then used as parameters for quantization.
            Otherwise, parameters from the call to :meth:`compute_quantization_parameters` are used.
        chunk_size : int
            Maximum amount of traces in each chunk.
        max_workers : int or None
            Maximum number of parallel processes to spawn. If None, then the number of CPU cores is used.
        pbar : bool, str
            If bool, then whether to display progress bar.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        overwrite : bool
            Whether to overwrite existing `path` or raise an exception. Also removes `meta` files.
        """
        if format == 8 and transform is None:
            quantization_parameters = quantization_parameters or self.compute_quantization_parameters()
            transform = quantization_parameters['transform']

        path = self.loader.convert(path=path, format=format, transform=transform,
                                   chunk_size=chunk_size, max_workers=max_workers, pbar=pbar, overwrite=overwrite)

        meta_path = path + '_meta'
        if overwrite and os.path.exists(meta_path):
            os.remove(meta_path)

        # Re-open geometry, store values that were used for quantization
        from .base import Geometry
        geometry = Geometry.new(path, collect_stats=True)

        quantization_parameters['quantization_ranges'] = quantization_parameters['ranges']
        for key in ['quantization_ranges', 'center', 'clip', 'quantization_error']:
            geometry.meta_storage.store_item(key=key, value=quantization_parameters[key], overwrite=True)
        return geometry


    def make_output_path(self, format='hdf5', quantize=False, postfix=False, projections='ixd',
                         chunk_size_divisor=1, sgy_format=8):
        """ Compute output path for converted file, based on conversion parameters. """
        format = format.lower()

        if format.startswith('q'):
            quantize = True
            format = format[1:]

        fmt_prefix = 'q' if quantize else ''

        if not isinstance(postfix, str):
            if not postfix:
                postfix = ''
            else:
                if format == 'hdf5':
                    if len(projections) < 3:
                        postfix = '_' + projections
                    if chunk_size_divisor != 1:
                        postfix = '_' + f'c{chunk_size_divisor}'

                if format == 'sgy':
                    if quantize:
                        postfix = '_' + f'f{sgy_format}'

        dirname = os.path.dirname(self.path)
        basename = os.path.basename(self.path)
        shortname = os.path.splitext(basename)[0]
        path = os.path.join(dirname, shortname + postfix + '.' + fmt_prefix + format)
        return path


    def convert(self, format='qsgy', path=None, postfix=False, projections='ixd', overwrite=True,
                quantize=False, quantization_parameters=None, dataset_kwargs=None, chunk_size_divisor=1,
                pbar='t', store_meta=True, sgy_format=8, transform=None, chunk_size=25_000, max_workers=4, **kwargs):
        """ Convert SEG-Y file to a more effective storage.
        Automatically select the conversion format, based on `format` parameter.
        Available formats are {'hdf5', 'qhdf5', 'qsgy}.

        Parameters are passed to either :meth:`.convert_to_hdf5` or :meth:`.repack_sgy`:
        refer to their documentation for parameters description.
        """
        format = format.lower()

        if format.startswith('q'):
            quantize = True
            format = format[1:]

        if path is None:
            path = self.make_output_path(format=format, postfix=postfix, quantize=quantize, projections=projections,
                                         chunk_size_divisor=chunk_size_divisor, sgy_format=sgy_format)

        # Actual conversion
        if 'hdf5' in format:
            geometry = self.convert_to_hdf5(path=path, overwrite=overwrite, projections=projections,
                                            quantize=quantize, quantization_parameters=quantization_parameters,
                                            dataset_kwargs=dataset_kwargs, chunk_size_divisor=chunk_size_divisor,
                                            pbar=pbar, store_meta=store_meta)
        elif 'sgy' in format and quantize:
            geometry = self.repack_segy(path=path, overwrite=overwrite, format=sgy_format,
                                        transform=transform, quantization_parameters=quantization_parameters,
                                        chunk_size=chunk_size, max_workers=max_workers, pbar=pbar)
        else:
            raise ValueError(f'Unknown/unsupported combination of format={format} and quantize={quantize}!')

        return geometry


    # Resample SEG-Y
    def resample(self, path=None, factor=2, quantize=True, quantization_parameters=None, pbar='t',
                 overwrite=True, **kwargs):
        """ Resample SEG-Y file along the depth dimension with optional quantization.

        Parameters
        ----------
        path : str, optional
            Path to save file to. If not provided, we use the path of the current cube with an added postfix.
        factor : number
            Scale factor along the depth axis.
        quantize : bool
            Whether to quantize SEG-Y data.
            If the geometry is already using quantized values, no quantization is applied.
        quantization_parameters : dict, optional
            If provided, then used as parameters for quantization.
            Otherwise, parameters from the call to :meth:`compute_quantization_parameters` are used.
        pbar : bool, str
            If bool, then whether to display progress bar.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        overwrite : bool
            Whether to overwrite existing `path` or raise an exception. Also removes `meta` files.
        """
        # Path
        path = path or self.make_output_path('sgy', quantize=quantize, postfix=f'_r{factor}')

        # Quantization parameters
        if quantize and not self.quantized:
            quantization_parameters = quantization_parameters or self.compute_quantization_parameters()
            quantization_transform = quantization_parameters['transform']
        else:
            quantization_transform = lambda array: array

        # Spec: use `self` as `array_like` to infer shapes
        spec = self.make_export_spec(self)
        spec.sample_interval /= factor
        spec.samples = np.arange(self.depth * factor, dtype=np.int32)
        spec.format = 8 if quantize else 5

        # Final data transform: resample and optional quantization
        transform = lambda array: quantization_transform(resize_3D(array, factor=factor))

        self.array_to_segy(self, path=path, spec=spec, transform=transform, format=spec.format,
                        pbar=pbar, zip_segy=False, **kwargs)

        # Re-open geometry, store values that were used for quantization
        meta_path = path + '_meta'
        if overwrite and os.path.exists(meta_path):
            os.remove(meta_path)

        from .base import Geometry
        geometry = Geometry.new(path, collect_stats=True)

        if quantize and not self.quantized:
            quantization_parameters['quantization_ranges'] = quantization_parameters['ranges']
            for key in ['quantization_ranges', 'center', 'clip', 'quantization_error']:
                geometry.meta_storage.store_item(key=key, value=quantization_parameters[key], overwrite=True)
        return geometry
