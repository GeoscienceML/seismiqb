""" Class to work with seismic data in SEG-Y format. """
#pylint: disable=not-an-iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit, prange
import cv2

from batchflow import Notifier

from .base import Geometry
from .segyio_loader import SegyioLoader
from .memmap_loader import MemmapLoader



class GeometrySEGY(Geometry):
    """ Class to infer information about SEG-Y cubes and provide convenient methods for working with them.

    In order to initialize instance, one must supply `path`, `index_headers` and `additional_headers`:
        - `path` is a location of SEG-Y file
        - `index_headers` are used as the gather/trace unique identifier:
          for example, `INLINE_3D` and `CROSSLINE_3D` has a one-to-one correspondence with trace numbers.
          Another example is `FieldRecord` and `TraceNumber`.
        - `additional_headers` are also loaded.
    Default value of `index_headers` is ['INLINE_3D', 'CROSSLINE_3D'] with additional ['CDP_X', 'CDP_Y'],
    so that post-stack cube can be loaded by providing path only.

    For brevity, we use the 'inline/crossline' words to refer to the first/second indexing header in documentation
    and developer comments, as that is the most common scenario.

    To simplify indexing, we use ordinals of unique values of each indexing header pretty much everywhere after init.
    In the simplest case of regular structure, we can convert ordinals into unique values by using
    `value = value_min + ordinal * value_step`, where `value_min` and `value_step` are inferred from trace headers.

    For faster indexing of the traces we use indexing matrix, that maps
    `(ordinal_for_indexing_header_0, ordinal_for_indexing_header_1)` into the actual trace number to be loaded.

    At initialization or by manually calling method :meth:`collect_stats` we make a full pass through
    the cube in order to analyze distribution of amplitudes, storing global, spatial and depth-wise stats.
    They are available as attributes, e.g. `mean`, `mean_matrix` and `mean_vector`.

    Refer to the documentation of the base class :class:`Geometry` for more information about attributes and parameters.
    """
    # Headers to use as a unique id of a trace
    INDEX_HEADERS_PRESTACK = ('FieldRecord', 'TraceNumber')
    INDEX_HEADERS_POSTSTACK = ('INLINE_3D', 'CROSSLINE_3D')
    INDEX_HEADERS_CDP = ('CDP_Y', 'CDP_X')

    # Headers to load from SEG-Y cube
    ADDITIONAL_HEADERS_PRESTACK_FULL = ('FieldRecord', 'TraceNumber', 'TRACE_SEQUENCE_FILE',
                                        'CDP', 'CDP_TRACE', 'offset')
    ADDITIONAL_HEADERS_POSTSTACK_FULL = ('INLINE_3D', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y')

    def init(self, path, index_headers=INDEX_HEADERS_POSTSTACK, additional_headers=ADDITIONAL_HEADERS_POSTSTACK_FULL,
             loader_class=MemmapLoader, reload_headers=False, dump_headers=True, load_headers_params=None,
             collect_stats=True, recollect_stats=False, collect_stats_params=None, dump_meta=True,
             **kwargs):
        """ Init for SEG-Y geometry. The sequence of actions:
            - initialize loader instance
            - load headers by reading SEG-Y or reading from meta
            - compute additional attributes from indexing headers
            - validate structure of the coordinate system, created by the indexing headers
            - collect stats by full SEG-Y sweep or reading from meta
            - dump meta for future inits.
        """
        # Store attributes
        self.index_headers = list(index_headers)
        self.additional_headers = list(additional_headers)
        self.index_length = len(index_headers)
        self.converted = False

        # Initialize loader
        self.loader = self._infer_loader_class(loader_class)(path)

        # Retrieve some of the attributes directly from the `loader`
        self.n_traces = self.loader.n_traces
        self.depth = self.loader.n_samples
        self.delay = self.loader.delay
        self.sample_interval = self.loader.sample_interval
        self.sample_rate = self.loader.sample_rate

        self.dtype = self.loader.dtype
        self.quantized = (self.dtype == np.int8)

        self.segy_path = self.loader.path
        try:
            self.segy_text = [item.decode('ascii') for item in self.loader.text]
        except: #pylint: disable=bare-except
            self.segy_text = ['*'*3200]

        # If all stats are already available in meta, use them
        required_attributes = self.PRESERVED + self.PRESERVED_LAZY + self.PRESERVED_LAZY_CACHED
        meta_exists_and_has_attributes = self.meta_storage.exists and self.meta_storage.has_items(required_attributes)

        if meta_exists_and_has_attributes and not (reload_headers or recollect_stats):
            self.load_meta(keys=self.PRESERVED)
            self.has_stats = True
            return

        # Load all of the requested headers, either from SEG-Y directly or previously stored dump
        headers_to_load = list(set(index_headers) | set(additional_headers))

        if self.meta_storage.has_item(key='headers') and not reload_headers:
            headers = self.meta_storage.read_item(key='headers')
        else:
            load_headers_params = load_headers_params or {}
            headers = self.load_headers(headers_to_load, **load_headers_params)
            if dump_headers:
                self.meta_storage.store_item(key='headers', value=headers)
        self.headers = headers

        # Infer attributes based on indexing headers: values and coordinates
        self.add_index_attributes()

        if 'INLINE_3D' in self.index_headers and 'CROSSLINE_3D' in self.index_headers:
            self.rotation_matrix = self.compute_rotation_matrix()

        # Collect amplitude stats, either by passing through SEG-Y or from previously stored dump
        required_attributes = self.PRESERVED + self.PRESERVED_LAZY
        meta_exists_and_has_attributes = self.meta_storage.exists and self.meta_storage.has_items(required_attributes)

        if meta_exists_and_has_attributes and not recollect_stats:
            self.load_meta(keys=self.PRESERVED)
            self.has_stats = True
        elif collect_stats:
            collect_stats_params = collect_stats_params or {}
            self.collect_stats(**collect_stats_params)
            self.has_stats = True
        else:
            self.compute_dead_traces()
            self.has_stats = False

        if hasattr(self, 'n_alive_traces') and self.n_alive_traces is not None:
            try:
                self.area = self.compute_area()
            except IndexError:
                self.area = -1.

        # Dump inferred attributes to a separate file for later loads
        if dump_meta and not meta_exists_and_has_attributes:
            self.dump_meta()

    def _infer_loader_class(self, loader_class):
        """ Select appropriate loader class. """
        if isinstance(loader_class, type):
            return loader_class
        if 'seg' in loader_class:
            return SegyioLoader
        return MemmapLoader

    def load_headers(self, headers_to_load, reconstruct_tsf=True, chunk_size=25_000, max_workers=4, pbar=False):
        """ Load all of the requested headers into dataframe. """
        return self.loader.load_headers(headers_to_load, reconstruct_tsf=reconstruct_tsf,
                                        chunk_size=chunk_size, max_workers=max_workers, pbar=pbar)

    def add_index_attributes(self):
        """ Add attributes, based on the values of indexing headers. """
        # For each indexing headers compute set of its values, its sorted version,
        # and the mapping from each unique value to its ordinal in sorted list
        self.index_unsorted_uniques = [np.unique(self.headers[index_header])
                                       for index_header in self.index_headers]
        self.index_sorted_uniques = [np.sort(item) for item in self.index_unsorted_uniques]
        self.index_value_to_ordinal = [{value: i for i, value in enumerate(item)}
                                       for item in self.index_sorted_uniques]

        # Infer coordinates for indexing headers
        self.shifts = [np.min(item) for item in self.index_sorted_uniques]
        self.lengths = [len(item) for item in self.index_sorted_uniques]
        self.ranges = [(np.min(item), np.max(item)) for item in self.index_sorted_uniques]
        self.shape = np.array([*self.lengths, self.depth])

        # Check if indexing headers provide regular structure
        self.increments = []
        regular_structure = True
        for i, index_header in enumerate(self.index_headers):
            increments = np.diff(self.index_sorted_uniques[i])
            unique_increments = set(increments) or set([1])

            if len(unique_increments) > 1:
                print(f'`{index_header}` has irregular spacing! {unique_increments}')
                regular_structure = False
            else:
                self.increments.append(unique_increments.pop())
        self.regular_structure = regular_structure

        # Create indexing matrix
        if self.index_length == 2:
            index_matrix = self.compute_header_values_matrix('TRACE_SEQUENCE_FILE')
            index_matrix[index_matrix != -1] -= 1
            self.index_matrix = index_matrix

            self.absent_traces_matrix = (self.index_matrix == -1).astype(np.bool_)

    def compute_dead_traces(self, frequency=100):
        """ Fallback for dead traces matrix computation, if no full stats are collected. """
        slices = self.loader.load_depth_slices(list(range(0, self.depth, frequency)))

        if slices.shape[-1] == np.prod(self.lengths):
            slices = slices.reshape(slices.shape[0], *self.lengths)
            std_matrix = np.std(slices, axis=0)

            self.dead_traces_matrix = (std_matrix == 0).astype(np.bool_)
            self.n_dead_traces = np.sum(self.dead_traces_matrix)
            self.n_alive_traces = np.prod(self.lengths) - self.n_dead_traces


    def compute_header_values_matrix(self, header):
        """ Mapping from ordinal inline/crossline coordinate to the value of header. """
        index_values = self.headers[self.index_headers].values
        index_ordinals = self.lines_to_ordinals(index_values)
        idx_0, idx_1 = index_ordinals[:, 0], index_ordinals[:, 1]

        dtype = self.headers[header].dtype
        matrix = np.full(self.lengths, -1, dtype=dtype)
        matrix[idx_0, idx_1] = self.headers[header]
        return matrix


    # Compute additional stats from CDP/LINES correspondence
    def compute_rotation_matrix(self, n_points=10):
        """ Compute transform from INLINE_3D/CROSSLINE_3D coordinates to CDP_X/CDP_Y system. """
        ix_points = []
        cdp_points = []

        for _ in range(n_points):
            idx = np.random.randint(self.n_traces)
            row = self.headers.iloc[idx]

            # INLINE_3D -> CDP_X, CROSSLINE_3D -> CDP_Y
            ix_point = (row['INLINE_3D'], row['CROSSLINE_3D'])
            cdp_point = (row['CDP_X'], row['CDP_Y'])

            ix_points.append(ix_point)
            cdp_points.append(cdp_point)
        rotation_matrix, inliers = cv2.estimateAffine2D(np.float32(ix_points), np.float32(cdp_points))

        if 0 in inliers:
            return None
        return rotation_matrix

    def compute_area(self, shift=50):
        """ Compute approximate area of the cube in square kilometers. """
        central_i = self.shape[0] // 2
        central_x = self.shape[1] // 2

        tsf = self.index_matrix[central_i, central_x]
        tsf_di = self.index_matrix[central_i, central_x + shift]
        tsf_dx = self.index_matrix[central_i + shift, central_x]

        row = self.headers.iloc[tsf]
        row_di = self.headers.iloc[tsf_di]
        row_dx = self.headers.iloc[tsf_dx]

        # CDP_X/CDP_Y coordinate system is rotated on 90 degrees with respect to INLINE_3D/CROSSLINE_3D
        if row_di['CDP_X'] - row['CDP_X'] == 0 and row_dx['CDP_Y'] - row['CDP_Y'] == 0:
            row_di, row_dx = row_dx, row_di

        # Size of one "trace bin"
        cdp_x_delta_km = abs(row_di['CDP_X'] - row['CDP_X']) / shift / 1000
        cdp_y_delta_km = abs(row_dx['CDP_Y'] - row['CDP_Y']) / shift / 1000
        area = cdp_x_delta_km * cdp_y_delta_km * self.n_alive_traces
        return round(area, 2)


    # Collect stats
    def collect_stats(self, chunk_size=20, max_workers=16,
                      n_quantile_traces=100_000, quantile_precision=3, seed=42, pbar='t'):
        """ One sweep through the entire SEG-Y data to collects stats, which are available as instance attributes:
            - global: one number for the entire cube, e.g. `mean`
            - spatial: a matrix of values for each trace, e.g. `mean_matrix`
            - depth-wise: one value for each depth slice, e.g. `mean_vector`.
        Other than `mean`, we also collect `min`, `max` and `std`.
        Moreover, we compute a certain amount of quantiles: they are computed from a random subset of the traces.
        TODO: add `limits`?

        The traces are iterated over in chunks: chunking is performed along the first indexing header, e.g. `INLINE_3D`.
        Computation of stats is performed in multiple threads to speed up the process.

        Implementation detail: we store buffers for stats, e.g. `mean_matrix` in the instance itself.
        Each thread has the access to buffers and modifies them in-place.
        Moreover, even the underlying numba functions are using the same buffers in-place:
        this way we avoid unnecessary copies and data conversions.

        Parameters
        ----------
        chunk_size : int
            Number of full inlines to include in one chunk.
        max_workers : int
            Maximum number of threads for parallelization.
        n_quantile_traces : int
            Size of the subset to compute quantiles.
        quantile_precision : int
            Compute an approximate quantile for each value with that number of decimal places.
        seed : int
            Seed for quantile traces subset selection.
        pbar : bool, str
            If bool, then whether to display progress bar.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        """
        # pylint: disable=too-many-statements
        # Prepare chunks
        n = self.lengths[0]
        n_chunks, last_chunk_size = divmod(n, chunk_size)
        chunk_sizes = [chunk_size] * n_chunks
        if last_chunk_size:
            chunk_sizes += [last_chunk_size]
            n_chunks += 1

        chunk_starts = np.cumsum([0] + chunk_sizes[:-1])
        chunk_ends = np.cumsum(chunk_sizes)
        chunk_weights = np.array(chunk_sizes, dtype=np.float64) / n

        # Define buffers: chunked vectors
        self.min_vector_chunked = np.full((n_chunks, self.depth), np.inf, dtype=np.float32)
        self.max_vector_chunked = np.full((n_chunks, self.depth), -np.inf, dtype=np.float32)
        self.mean_vector_chunked = np.zeros((n_chunks, self.depth), dtype=np.float64)
        self.var_vector_chunked = np.zeros((n_chunks, self.depth), dtype=np.float64)

        # Define buffers: matrices
        self.min_matrix = np.full(self.lengths, np.inf, dtype=np.float32)
        self.max_matrix = np.full(self.lengths, -np.inf, dtype=np.float32)
        self.mean_matrix = np.zeros(self.lengths, dtype=np.float64)
        self.var_matrix = np.zeros(self.lengths, dtype=np.float64)

        # Read data in chunks, compute stats for each of them, store into buffer
        description = f'Collecting stats for `{self.name}`'
        with Notifier(pbar, total=n, desc=description, ncols=110) as progress_bar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                def callback(future):
                    chunk_size = future.result()
                    progress_bar.update(chunk_size)

                for chunk_i, (start, end) in enumerate(zip(chunk_starts, chunk_ends)):
                    future = executor.submit(self.collect_stats_chunk,
                                             start=start, end=end, chunk_i=chunk_i)
                    future.add_done_callback(callback)

        # Finalize vectors
        self.min_vector = np.average(self.min_vector_chunked, axis=0, weights=chunk_weights)
        self.max_vector = np.average(self.max_vector_chunked, axis=0, weights=chunk_weights)
        mean_vector = np.average(self.mean_vector_chunked, axis=0, weights=chunk_weights)
        var_vector = np.average(self.var_vector_chunked + (self.mean_vector_chunked - mean_vector)**2,
                                axis=0, weights=chunk_weights)

        self.mean_vector = mean_vector.astype(np.float32)
        self.std_vector = np.sqrt(var_vector).astype(np.float32)

        # Finalize matrices
        self.mean_matrix = self.mean_matrix.astype(np.float32)
        self.std_matrix = np.sqrt(self.var_matrix).astype(np.float32)
        self.dead_traces_matrix = (self.min_matrix == self.max_matrix).astype(np.bool_)

        # Clean-up redundant buffers
        del (self.min_vector_chunked, self.max_vector_chunked,
             self.mean_vector_chunked, self.var_vector_chunked,
             self.var_matrix)

        # Add scalar values
        self.min = self.min_matrix[~self.dead_traces_matrix].min()
        self.max = self.max_matrix[~self.dead_traces_matrix].max()
        self.mean = self.mean_matrix[~self.dead_traces_matrix].mean()

        n_dead_traces = np.sum(self.dead_traces_matrix)
        n_alive_traces = np.prod(self.lengths) - n_dead_traces
        self.std = np.sqrt((self.std_matrix[~self.dead_traces_matrix] ** 2).sum() / n_alive_traces)

        # Load subset of data to compute quantiles
        alive_traces_indices = self.index_matrix[~self.dead_traces_matrix].reshape(-1)
        indices = np.random.default_rng(seed=seed).choice(alive_traces_indices,
                                                          size=min(self.n_traces, n_quantile_traces))
        data = self.load_by_indices(indices)

        quantile_support = np.round(np.linspace(0, 1, num=10**quantile_precision+1),
                                    decimals=quantile_precision)
        quantile_values = np.quantile(data, q=quantile_support)
        quantile_values[0], quantile_values[-1] = self.min, self.max

        # Store stats of the subset to compare against fair ones
        self.subset_min = data.min()
        self.subset_max = data.max()
        self.subset_mean = data.mean()
        self.subset_std = data.std()
        self.n_value_uniques = len(np.unique(data))
        self.quantile_precision = quantile_precision
        self.quantile_support, self.quantile_values = quantile_support, quantile_values

        # Store the number of alive/dead traces
        self.n_alive_traces = n_alive_traces
        self.n_dead_traces = n_dead_traces

    def collect_stats_chunk(self, start, end, chunk_i):
        """ Read requested chunk, compute stats for it. """
        # Retrieve chunk data
        indices = self.index_matrix[start:end].reshape(-1)

        data = self.load_by_indices(indices)
        data = data.reshape(end - start, self.lengths[1], self.depth)

        # Actually compute all of the stats. Modifies buffers in-place
        _collect_stats_chunk(data,
                             min_vector=self.min_vector_chunked[chunk_i],
                             max_vector=self.max_vector_chunked[chunk_i],
                             mean_vector=self.mean_vector_chunked[chunk_i],
                             var_vector=self.var_vector_chunked[chunk_i],
                             min_matrix=self.min_matrix[start:end],
                             max_matrix=self.max_matrix[start:end],
                             mean_matrix=self.mean_matrix[start:end],
                             var_matrix=self.var_matrix[start:end])
        return end - start


    # Data loading: arbitrary trace indices
    def load_by_indices(self, indices, limits=None, buffer=None):
        """ Read requested traces from SEG-Y file.
        Value `-1` is interpreted as missing trace, and corresponding traces are filled with zeros.

        Parameters
        ----------
        indices : sequence
            Indices (TRACE_SEQUENCE_FILE) of the traces to read.
        limits : sequence of ints, slice, optional
            Slice of the data along the depth (last) axis.
        buffer : np.ndarray, optional
            Buffer to read the data into. If possible, avoids copies.
        """
        if buffer is None:
            limits = self.process_limits(limits)
            buffer = np.empty((len(indices), self.depth), dtype=self.dtype)[:, limits]
        else:
            buffer = buffer.reshape((len(indices), -1))

        if -1 in indices:
            # Create new buffer to avoid copy on advanced indexing
            mask = indices >= 0
            buffer_ = np.empty_like(buffer)[:mask.sum()]
            self.loader.load_traces(indices=indices[mask], limits=limits, buffer=buffer_)

            buffer[mask] = buffer_
            buffer[~mask] = self.FILL_VALUE
        else:
            self.loader.load_traces(indices=indices, limits=limits, buffer=buffer)
        return buffer

    def load_depth_slices(self, indices, buffer=None):
        """ Read requested depth slices from SEG-Y file. """
        if buffer is None:
            buffer = np.empty((len(indices), self.n_traces), dtype=self.dtype)
        else:
            buffer = buffer.reshape((len(indices), self.n_traces))

        buffer = self.loader.load_depth_slices(indices, buffer=buffer)
        if buffer.shape[-1] == np.prod(self.lengths):
            buffer = buffer.reshape(len(indices), *self.lengths)
        else:
            idx = np.nonzero(self.index_matrix >= 0)
            matrix = np.zeros(shape=(*self.spatial_shape, len(indices)), dtype=self.dtype)
            matrix[idx] = buffer.T
            buffer = matrix.transpose(2, 0, 1)
        return buffer

    @property
    def mmap(self):
        """ 3D memory map, that views the entire SEG-Y as one 3D array. """
        return self.loader.data_mmap.reshape(self.shape)

    # Data loading: 2D
    def load_slide_native(self, index, axis=0, limits=None, buffer=None, safe=False):
        """ Load one slide of data along specified axis.

        Parameters
        ----------
        index : int, str
            If int, then interpreted as the ordinal along the specified axis.
            If `'random'`, then we generate random index along the axis.
            If string of the `'#XXX'` format, then we interpret it as the exact indexing header value.
        axis : int
            Axis of the slide.
        limits : sequence of ints, slice, optional
            Slice of the data along the depth (last) axis.
        buffer : np.ndarray, optional
            Buffer to read the data into. If possible, avoids copies.
        """
        _ = safe
        if axis in {0, 1}:
            index = self.get_slide_index(index=index, axis=axis)
            indices = np.take(self.index_matrix, indices=index, axis=axis)
            slide = self.load_by_indices(indices=indices, limits=limits, buffer=buffer)
        else:
            slide = self.load_depth_slices([index], buffer=buffer).squeeze(0)
        return slide

    # Data loading: 3D
    def load_crop_native(self, locations, buffer=None, safe=False):
        """ Load crop (3D subvolume) from the cube.

        Parameters
        ----------
        locations : sequence
            A triplet of slices to specify the location of a subvolume.
        buffer : np.ndarray, optional
            Buffer to read the data into. If possible, avoids copies.
        """
        _ = safe
        shape = self.locations_to_shape(locations)
        axis = np.argmin(shape)

        if axis in {0, 1} or shape[-1] > 50: #TODO: explain this constant
            indices = self.index_matrix[locations[0], locations[1]].reshape(-1)
            buffer = self.load_by_indices(indices=indices, limits=locations[-1], buffer=buffer)

            shape = [((slc.stop or stop) - (slc.start or 0)) for slc, stop in zip(locations, self.shape)]
            buffer = buffer.reshape(shape)
        else:
            indices = np.arange(locations[-1].start, locations[-1].stop)
            data = self.load_depth_slices(indices).transpose(1, 2, 0)[locations[0], locations[1]]

            if buffer is None:
                buffer = data
            else:
                buffer[:] = data
        return buffer

    def get_optimal_axis(self, locations=None, shape=None):
        """ Choose the fastest axis for loading given locations. """
        shape = shape or self.locations_to_shape(locations)
        return np.argsort(shape)[0]

    def load_section(self, locations, dtype=None):
        """ Load section through `locations`.

        Parameters
        ----------
        locations : iterable
            Locations of traces to construct section.

        dtype : None or numpy.dtype, optional
            Type of the resulting image, by default None (transforms to self.dtype)

        Returns
        -------
        section, indices, nodes: tuple with 3 elements
            section : numpy.ndarray
                2D array with loaded and interpolated traces of section.
            indices : numpy.ndarray
                Float coordinates of section traces.
            nodes : numpy.ndarray
                Positions of node traces (from `locations`) in `traces` array.
        """
        locations = np.array(locations)
        dtype = dtype or self.dtype

        indices = []
        for start, stop in zip(locations[:-1], locations[1:]):
            indices.append(get_line_coordinates(start, stop)[:-1])
        indices.append(np.array([locations[-1]], dtype='float32'))

        support, weights = get_line_support(np.concatenate(indices))

        all_support_traces = np.concatenate(support)
        unique_support, traces_indices = np.unique(all_support_traces, axis=0, return_inverse=True)
        traces_indices = traces_indices.reshape(-1, 4)

        traces = self.load_by_indices(self.index_matrix[unique_support[:, 0], unique_support[:, 1]])
        section = interpolate(traces, traces_indices, weights)
        if np.issubdtype(dtype, np.integer):
            section = section.astype(dtype)
        nodes = np.cumsum([0] + [len(item) for item in indices[:-1]])
        indices = np.concatenate(indices)
        return section, indices, nodes

@njit(nogil=True)
def _collect_stats_chunk(data,
                         min_vector, max_vector, mean_vector, var_vector,
                         min_matrix, max_matrix, mean_matrix, var_matrix):
    """ Compute stats of a 3D array: min, max, mean, variance.

    We use provided buffers to avoid unnecessary copies.
    We use buffers for mean and var to track the running sum of values / squared values.
    """
    shape = data.shape

    for i in range(shape[0]):
        for x in range(shape[1]):
            for d in range(shape[2]):
                # Read traces values
                trace_value = data[i, x, d]
                trace_value64 = np.float64(trace_value)

                # Update vectors
                min_vector[d] = min(min_vector[d], trace_value)
                max_vector[d] = max(max_vector[d], trace_value)
                mean_vector[d] += trace_value64
                var_vector[d] += trace_value64 ** 2

                # Update matrices
                min_matrix[i, x] = min(min_matrix[i, x], trace_value)
                max_matrix[i, x] = max(max_matrix[i, x], trace_value)
                mean_matrix[i, x] += trace_value64
                var_matrix[i, x] += trace_value64 ** 2

    # Finalize vectors
    area = shape[0] * shape[1]
    mean_vector /= area
    var_vector /= area
    var_vector -= mean_vector ** 2

    # Finalize matrices
    mean_matrix /= shape[2]
    var_matrix /= shape[2]
    var_matrix -= mean_matrix ** 2

    return (min_vector, max_vector, mean_vector, var_vector,
            min_matrix, max_matrix, mean_matrix, var_matrix)

@njit
def get_line_coordinates(start, stop):
    """ Get float coordinates of traces for line from `start` to `stop`.

    Parameters
    ----------
    start : numpy.ndarray

    stop : numpy.ndarray

    Returns
    -------
    locations : numpy.ndarray
        array of shape (N, 2) and dtype float32 with coordinates for section traces.
    """
    direction = stop - start
    distance = np.power(direction, 2).sum() ** 0.5
    locations = np.empty((int(np.ceil(distance)) + 1, 2), dtype=np.float32)
    for i in [0, 1]:
        locations[:, i] = np.linspace(start[i], stop[i], int(np.ceil(distance)) + 1)
    return locations

@njit
def get_line_support(locations):
    """ Get support for non-integer locations.

    Parameters
    ----------
    locations : numpy.ndarray
        array of shape (N, 2) and dtype float32

    Returns
    -------
    support, weights : tuple with two elements

        support : numpy.ndarray
            array of shape (N, 4, 2) and dtype int32 with coordinates of support traces for each location.
        weights : numpy.ndarray
            array of shape (N, 4) with weights for support traces for interpolation. If some location has integer
            coordinates, support will have duplicated traces and nan weights.
    """
    ceil, floor = np.ceil(locations), np.floor(locations)
    support = np.empty((len(locations), 4, 2), dtype='int32')
    support[:, 0] = floor
    support[:, 1, 0] = floor[:, 0]
    support[:, 1, 1] = ceil[:, 1]
    support[:, 2, 0] = ceil[:, 0]
    support[:, 2, 1] = floor[:, 1]
    support[:, 3] = ceil

    distances = ((support - np.expand_dims(locations, 1)) ** 2).sum(axis=-1) ** 0.5
    weights = 1 / distances
    weights = weights / np.expand_dims(weights.sum(axis=-1), 1)

    return support, weights

@njit(parallel=True)
def interpolate(traces, traces_indices, weights, dtype='float32'):
    """ Interpolate traces with float coordinates by traces from support.

    Parameters
    ----------
    traces : numpy.ndarray
        array of shape (M, geometry.shape[2]) with loaded traces from support.
    traces_indices : numpy.ndarray
        array of shape (N, 4) with indices of corresponging traces in `traces` for each support trace.
    weights : numpy.ndarray
        array of shape (N, 4) with weights for support traces for interpolation. If some location has integer
        coordinates, support will have duplicated traces and nan weights.
    dtype : str, optional
        resulting dtype, by default 'float32'

    Returns
    -------
    numpy.ndarray
        array of shape (N, geometry.shape[2])
    """
    image = np.empty((len(traces_indices), traces.shape[1]), dtype=dtype)
    for i in prange(len(traces_indices)):
        trace_weights = weights[i]
        image[i] = traces[traces_indices[i][0]]
        if not np.isnan(trace_weights[0]):
            image[i] *= trace_weights[0]
            for j in range(1, 4):
                image[i] += traces[traces_indices[i][j]] * trace_weights[j]
    return image
