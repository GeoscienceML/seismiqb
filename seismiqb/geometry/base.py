""" Base class for working with seismic data. """
import os
import sys
from textwrap import dedent
from contextlib import contextmanager

import numpy as np

import cv2
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import binary_erosion

from batchflow import Normalizer

from .benchmark_mixin import BenchmarkMixin
from .conversion_mixin import ConversionMixin
from .export_mixin import ExportMixin
from .metric_mixin import MetricMixin

from ..utils import SQBStorage, lru_cache, CacheMixin, TransformsMixin, select_printer, transformable, take_along_axis
from ..plotters import plot



class Geometry(BenchmarkMixin, CacheMixin, ConversionMixin, ExportMixin, MetricMixin, TransformsMixin):
    """ Class to infer information about seismic cube in various formats and provide format agnostic interface to them.

    During the SEG-Y processing, a number of statistics are computed. They are saved next to the cube under the
    `.segy_meta` extension, so that subsequent loads (in, possibly, other formats) don't have to recompute them.
    Most of them are loaded at initialization, but the most memory-intensive ones are loaded on demand.

    Based on the extension of the path, a different subclass is used to implement key methods for data indexing.
    Currently supported extensions are SEG-Y and TODO:
    The last two are created by converting the original SEG-Y cube.
    During the conversion, an extra step of `int8` quantization can be performed to reduce the disk usage.

    Independent of the exact format, `Geometry` provides the following:
        - attributes to describe shape and structure of the cube like `shape` and `lengths`,
        as well as exact values of file-wide headers, for example, `depth`, `delay`,
        `sample_rate` and `sample_interval`.

        - method :meth:`collect_stats` to infer information about the amplitudes distribution:
        under the hood, we make a full pass through the cube data to collect global, spatial and depth-wise stats.

        - :meth:`load_slide` (2D entity) or :meth:`load_crop` (3D entity) methods to load data from the cube:
            - :meth:`load_slide` takes an ordinal index of the slide and its axis;
            - :meth:`load_crop` works off of complete location specification (triplet of slices).

        - textual representation of cube geometry: method `print` shows the summary of an instance with
        information about its location and values; `print_textual` allows to see textual header from a SEG-Y.

        - visual representation of cube geometry:
            - :meth:`show` to display top view on cube with computed statistics;
            - :meth:`show_slide` to display front view on various slices of data.

    Parameters
    ----------
    path : str
        Path to seismic cube. Supported formats are `segy`, TODO.
    meta_path : str, optional
        Path to pre-computed statistics. If not provided, use the same as `path` with `_meta` postfix.

    SEG-Y parameters
    ----------------
    TODO

    HDF5 parameters
    ---------------
    TODO
    """
    # Headers to use as a unique id of a trace
    INDEX_HEADERS_PRESTACK = ('FieldRecord', 'TraceNumber')
    INDEX_HEADERS_POSTSTACK = ('INLINE_3D', 'CROSSLINE_3D')
    INDEX_HEADERS_CDP = ('CDP_Y', 'CDP_X')

    # Headers to load from SEG-Y cube
    ADDITIONAL_HEADERS_PRESTACK_FULL = ('FieldRecord', 'TraceNumber', 'TRACE_SEQUENCE_FILE',
                                        'CDP', 'CDP_TRACE', 'offset')
    ADDITIONAL_HEADERS_POSTSTACK_FULL = ('INLINE_3D', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y')

    # Value to use in dead traces
    FILL_VALUE = 0.0

    # Attributes to store in a separate file with meta
    PRESERVED = [ # loaded at instance initialization
        # Crucial geometry properties
        'n_traces', 'depth', 'delay', 'sample_interval', 'sample_rate', 'shape',
        'shifts', 'lengths', 'ranges', 'increments', 'regular_structure',
        'index_matrix', 'absent_traces_matrix', 'dead_traces_matrix',
        'n_alive_traces', 'n_dead_traces',

        # Additional info from SEG-Y
        'segy_path', 'segy_text',

        # Scalar stats for cube values: computed for the entire SEG-Y / its subset
        'min', 'max', 'mean', 'std', 'n_value_uniques',
        'subset_min', 'subset_max', 'subset_mean', 'subset_std',
        'quantile_precision', 'quantile_support', 'quantile_values',
    ]

    PRESERVED_LAZY = [ # loaded at the time of the first access
        'index_unsorted_uniques', 'index_sorted_uniques', 'index_value_to_ordinal',
        'min_vector', 'max_vector', 'mean_vector', 'std_vector',
        'min_matrix', 'max_matrix', 'mean_matrix', 'std_matrix',
    ]

    PRESERVED_LAZY_CACHED = [ # loaded at the time of the first access, stored in the instance
        'headers',
    ]

    PRESERVED_LAZY_MISC = [ # additional stats that may be absent. loaded at the time of the first access
        'quantization_ranges', 'quantization_error', 'rotation_matrix', 'area',
    ]

    PRESERVED_LAZY_ALL = PRESERVED_LAZY + PRESERVED_LAZY_CACHED + PRESERVED_LAZY_MISC

    @staticmethod
    def new(path, *args, **kwargs):
        """ A convenient selector of appropriate (SEG-Y or HDF5) geometry. """
        #pylint: disable=import-outside-toplevel
        extension = os.path.splitext(path)[1][1:]

        if extension in {'sgy', 'segy', 'seg', 'qsgy'}:
            from .segy import GeometrySEGY
            cls = GeometrySEGY
        elif extension in {'hdf5', 'qhdf5'}:
            from .converted import GeometryHDF5
            cls = GeometryHDF5
        else:
            raise TypeError(f'Unknown format of the cube: {extension}')
        return cls(path, *args, **kwargs)

    def __init__(self, path, meta_path=None, safe=False, use_cache=False, init=True, **kwargs):
        # Path to the file
        self.path = path

        # Names
        self.name = os.path.basename(self.path)
        self.short_name, self.format = os.path.splitext(self.name)

        # Meta
        self._meta_path = meta_path
        self.meta_storage = SQBStorage(self.meta_path)

        # Instance flags
        self.safe = safe
        self.use_cache = use_cache

        # Lazy properties
        self._quantile_interpolator = None
        self._normalization_stats = None
        self._quantization_stats = None
        self._quantizer = None
        self._normalizer = None

        # Init from subclasses
        if init:
            self._init_kwargs = kwargs
            self.init(path, **kwargs)

    # Meta: store/load pre-computed statistics and attributes from disk
    def __getattr__(self, key):
        """ Load item from stored meta. """
        if key not in self.__dict__ and (key in self.PRESERVED_LAZY_ALL) \
            and self.meta_storage.exists and self.meta_storage.has_item(key):
            value =  self.meta_storage.read_item(key)
            if key in self.PRESERVED_LAZY_CACHED:
                setattr(self, key, value)
            return value
        return object.__getattribute__(self, key)

    @property
    def meta_path(self):
        """ Paths to the file with stored meta. """
        if self._meta_path is not None:
            return self._meta_path

        if hasattr(self, 'path'):
            if 'hdf5' in self.path:
                return self.path
            return self.path + '_meta'
        raise ValueError('No `meta_path` exists!')

    def load_meta(self, keys):
        """ Load `keys` from meta storage and setattr them to `self`. """
        items = self.meta_storage.read(keys)
        for key, value in items.items():
            setattr(self, key, value)

    def dump_meta(self, path=None):
        """ Dump all attributes, referenced in  `PRESERVED_*` lists, to a storage.
        If no `path` is provided, uses `meta_storage` of the `self`. """
        storage = self.meta_storage if path is None else SQBStorage(path)
        items = {key : getattr(self, key) for key in self.PRESERVED + self.PRESERVED_LAZY + self.PRESERVED_LAZY_MISC
                 if getattr(self, key, None) is not None}
        items['type'] = 'geometry-meta'
        storage.store(items)


    # Redefined protocols
    def __getnewargs__(self):
        return (self.path, )

    def __getstate__(self):
        self.reset_cache()
        state = self.__dict__.copy()
        for name in ['loader', 'axis_to_projection'] + self.PRESERVED_LAZY_CACHED:
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)

        if self.converted:
            self.init(self.path, **self._init_kwargs)
        else:
            self.loader = self._infer_loader_class(self._init_kwargs.get('loader_class', 'memmap'))(self.path)


    # Data loading
    def __getitem__(self, key):
        """ Slice the cube using the usual `NumPy`-like semantics. """
        key, axis_to_squeeze = self.process_key(key)

        crop = self.load_crop(key)
        if axis_to_squeeze:
            crop = np.squeeze(crop, axis=tuple(axis_to_squeeze))
        return crop

    def process_key(self, key):
        """ Convert tuple of slices/ints into locations. """
        # Convert to list
        if isinstance(key, (int, slice)):
            key = [key]
        elif isinstance(key, tuple):
            key = list(key)

        # Pad not specified dimensions
        if len(key) != len(self.shape):
            key += [slice(None)] * (len(self.shape) - len(key))

        # Parse each subkey. Remember location of integers for later squeeze
        key_, axis_to_squeeze = [], []
        for i, (subkey, limit) in enumerate(zip(key, self.shape)):
            if isinstance(subkey, slice):
                slc = slice(max(subkey.start or 0, 0),
                            min(subkey.stop or limit, limit), subkey.step)

            elif isinstance(subkey, (int, np.integer)):
                subkey = subkey if subkey >= 0 else limit - subkey
                slc = slice(subkey, subkey + 1)
                axis_to_squeeze.append(i)

            if slc.start < 0 or slc.stop > limit:
                raise ValueError(f'Slice `{slc}` is outside geometry boundaries!')
            key_.append(slc)

        return key_, axis_to_squeeze


    # Data loading: cache
    def load_slide(self, index, axis=0, limits=None, buffer=None, safe=None, use_cache=None):
        """ Load one slide of data along specified axis.
        Under the hood, relies on :meth`:`load_slide_native`, implemented in subclasses.
        Also allows to use slide cache to speed up the loading process.

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
        safe : bool or None
            Whether to force usage of public (safe) or private API of data loading.
            If None, then uses instance-wide value (default False).
        use_cache : bool or None
            Whether to use cache for lines.
            If None, then uses instance-wide value (default False).
            If bool, forces that behavior.
        """
        # Parse parameters
        index = self.get_slide_index(index=index, axis=axis)
        axis = self.parse_axis(axis)

        if limits is not None and axis==2:
            raise ValueError('Providing `limits` with `axis=2` is meaningless!')

        safe = safe if safe is not None else self.safe
        use_cache = use_cache if use_cache is not None else self.use_cache

        # Actual data loading
        if use_cache is False:
            return self.load_slide_native(index=index, axis=axis, limits=limits, buffer=buffer, safe=safe)

        slide = self.load_slide_cached(index=index, axis=axis, limits=limits)
        if buffer is not None:
            buffer[:] = slide
        else:
            buffer = slide
        return slide

    take = load_slide # for compatibility with numpy API

    @lru_cache(128)
    def load_slide_cached(self, index, axis=0, limits=None):
        """ Cached version of :meth:`load_slide_native`. """
        return self.load_slide_native(index=index, axis=axis, limits=limits, buffer=None, safe=True)

    def load_crop(self, locations, buffer=None, safe=None, use_cache=None):
        """ Load crop (3D subvolume) from the cube.
        Uses either public or private API of `h5py`: the latter reads data directly into preallocated buffer.
        Also allows to use slide cache to speed up the loading process.

        Parameters
        ----------
        locations : sequence
            A triplet of slices to specify the location of a subvolume.
        buffer : np.ndarray, optional
            Buffer to read the data into. If possible, avoids copies.
        safe : bool
            Whether to force usage of public (safe) or private API of data loading.
        use_cache : bool or None
            Whether to use cache for lines.
            If None, then uses instance-wide value (default False).
            If bool, forces that behavior.
        """
        safe = safe if safe is not None else self.safe
        use_cache = use_cache if use_cache is not None else self.use_cache

        if use_cache is False:
            return self.load_crop_native(locations=locations, buffer=buffer, safe=safe)
        return self.load_crop_cached(locations=locations, buffer=buffer)

    def load_crop_cached(self, locations, axis=None, buffer=None):
        """ Cached version of :meth:`load_crop`. """
        # Parse parameters
        shape = self.locations_to_shape(locations)
        axis = axis or self.get_optimal_axis(shape=shape)
        to_projection_transposition, from_projection_transposition = self.compute_axis_transpositions(axis)

        locations = [locations[idx] for idx in to_projection_transposition]
        locations = tuple(locations)

        # Prepare buffer
        if buffer is None:
            buffer = np.empty(shape, dtype=np.float32)
        buffer = buffer.transpose(to_projection_transposition)

        # Load data
        for i, idx in enumerate(range(locations[0].start, locations[0].stop)):
            buffer[i] = self.load_slide_cached(index=idx, axis=axis)[locations[1], locations[2]]

        # View buffer in original ordering
        buffer = buffer.transpose(from_projection_transposition)
        return buffer

    def add_to_mask(self, mask, locations=None, **kwargs):
        """ Load data from `locations` and put into `mask`. Is used for labels which are geometries. """
        mask[:] = self.load_crop(locations)
        return mask

    def enable_cache(self):
        """ Enable cache for loaded slides. """
        self.use_cache = True

    def disable_cache(self):
        """ Disable cache for loaded slides, and clear existing cache. """
        self.use_cache = False
        self.reset_cache()

    @contextmanager
    def enabled_cache(self, enable=True):
        """ Context manager for enabling cache. """
        try:
            if enable:
                self.enable_cache()
            yield self
        finally:
            self.disable_cache()

    def load_subset(self, n_traces=100_000, seed=42):
        """ Load a subset of data. Returns an array of (n_traces, depth) shape. """
        rng = np.random.default_rng(seed=seed)
        if self.converted is False:
            alive_traces_indices = self.index_matrix[~self.dead_traces_matrix].ravel()
            indices = rng.choice(alive_traces_indices, size=n_traces)
            data = self.load_by_indices(indices)
        else:
            indices = rng.choice(self.shape[0], size=n_traces // self.shape[1], replace=False)
            data = []
            for index in indices:
                slide = self.load_slide(index=index, axis=0)
                slide_bounds = self.compute_auto_zoom(index=index, axis=0)[0]
                data.append(slide[slide_bounds].ravel())
            data = np.concatenate(data)
        return data


    # Coordinate system conversions
    def lines_to_ordinals(self, array):
        """ Convert values from inline-crossline coordinate system to their ordinals.
        In the simplest case of regular grid `ordinal = (value - value_min) // value_step`.
        In the case of irregular spacings between values, we have to manually map values to ordinals. TODO.
        """
        # Indexing headers
        if self.regular_structure:
            for i in range(self.index_length):
                array[:, i] -= self.shifts[i]
                if self.increments[i] != 1:
                    array[:, i] //= self.increments[i]
        else:
            raise NotImplementedError

        # Depth to units
        if array.shape[1] == self.index_length + 1:
            array = array.astype(np.float32)
            array[:, self.index_length] -= self.delay
            array[:, self.index_length] /= self.sample_interval
        return array

    def ordinals_to_lines(self, array):
        """ Convert ordinals to values in inline-crossline coordinate system.
        In the simplest case of regular grid `value = value_min + ordinal * value_step`.
        In the case of irregular spacings between values, we have to manually map ordinals to values. TODO.
        """
        array = array.astype(np.float32)

        # Indexing headers
        if self.regular_structure:
            for i in range(self.index_length):
                if self.increments[i] != 1:
                    array[:, i] *= self.increments[i]
                array[:, i] += self.shifts[i]
        else:
            raise NotImplementedError

        # Units to depth
        if array.shape[1] == self.index_length + 1:
            array[:, self.index_length] *= self.sample_interval
            array[:, self.index_length] += self.delay
        return array

    def lines_to_cdp(self, points):
        """ Convert lines to CDP. """
        return (self.rotation_matrix[:, :2] @ points.T + self.rotation_matrix[:, 2].reshape(2, -1)).T

    def cdp_to_lines(self, points):
        """ Convert CDP to lines. """
        inverse_matrix = np.linalg.inv(self.rotation_matrix[:, :2])
        lines = (inverse_matrix @ points.T - inverse_matrix @ self.rotation_matrix[:, 2].reshape(2, -1)).T
        return np.rint(lines)


    # Stats and normalization
    @property
    def quantile_interpolator(self):
        """ Quantile interpolator for arbitrary values. """
        if self._quantile_interpolator is None:
            self._quantile_interpolator = interp1d(self.quantile_support, self.quantile_values)
        return self._quantile_interpolator

    def get_quantile(self, q):
        """ Get q-th quantile of the cube data. Works with any `q` in [0, 1] range. """
        #pylint: disable=not-callable
        return self.quantile_interpolator(q).astype(np.float32)

    def make_normalization_stats(self):
        """ Values for performing normalization of data from the cube. """
        q_01, q_05, q_95, q_99 = self.get_quantile(q=[0.01, 0.05, 0.95, 0.99])
        self._normalization_stats = {
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'q_01': q_01,
            'q_05': q_05,
            'q_95': q_95,
            'q_99': q_99,
        }
        return self._normalization_stats

    @property
    def normalization_stats(self):
        """ Property with default normalization stats for synthetic images. """
        if self._normalization_stats is None:
            self.make_normalization_stats()
        return self._normalization_stats

    def make_normalizer(self, mode='meanstd', clip_to_quantiles=False, q=(0.01, 0.99), normalization_stats=None):
        """ Create normalizer. """
        if normalization_stats == 'field':
            normalization_stats = self.normalization_stats
        self._normalizer = Normalizer(mode=mode, clip_to_quantiles=clip_to_quantiles, q=q,
                                      normalization_stats=normalization_stats)
        return self._normalizer

    @property
    def normalizer(self):
        """ Normalizer instance. If it doesn't already exist, it will be created with default parameters. """
        if self._normalizer is not None:
            return self._normalizer
        return self.make_normalizer()

    def make_quantization_stats(self, ranges=0.99, clip=True, center=False, dtype=np.int8,
                                n_quantile_traces=100_000, seed=42):
        """ Compute quantization statistics. """
        self._quantization_stats = self.compute_quantization_parameters(ranges=ranges, clip=clip, center=center,
                                                                        dtype=dtype,
                                                                        n_quantile_traces=n_quantile_traces, seed=seed)
        return self._quantization_stats

    @property
    def quantization_stats(self):
        """ Property with default normalization stats for synthetic images. """
        if self._quantization_stats is None:
            self.make_quantization_stats()
        return self._quantization_stats

    def make_quantizer(self, ranges=0.99, clip=True, center=False, dtype=np.int8,
                       n_quantile_traces=100_000, seed=42):
        """ Compute quantization statistics and create quantizer. """
        self.make_quantization_stats(ranges=ranges, clip=clip, center=center, dtype=dtype,
                                     n_quantile_traces=n_quantile_traces, seed=seed)
        self._quantizer = self.quantization_stats['quantizer']
        return self._quantizer

    @property
    def quantizer(self):
        """ Quantizer instance. If it doesn't already exist, it will be created with default parameters. """
        if self._quantizer is not None:
            return self._quantizer
        return self.make_quantizer()

    def estimate_impulse(self, wavelet_length=40, n_traces=10_000, seed=42):
        """ Estimate impulse on a random subset of data.
        The idea is to average traces in the frequency domain to get the frequencies of an impulse, produced them.
        """
        data = self.load_subset(n_traces=n_traces, seed=seed)

        # FFT domain
        data_fft = np.fft.rfft(data, axis=-1)
        wavelet_fft = np.mean(np.abs(data_fft), axis=0)

        # Depth domain
        wavelet_estimation = np.real(np.fft.irfft(wavelet_fft)[:wavelet_length//2])
        wavelet_estimation = np.concatenate((wavelet_estimation[1:][::-1], wavelet_estimation), axis=0)
        return wavelet_estimation


    # Spatial matrices
    @property
    def snr(self):
        """ Signal-to-noise ratio. """
        eps = 1
        snr = np.log((self.mean_matrix**2 + eps) / (self.std_matrix**2 + eps))
        snr[self.std_matrix == 0] = np.nan
        return snr

    @transformable
    def get_dead_traces_matrix(self):
        """ Dead traces matrix.
        Due to decorator, allows for additional transforms at loading time.

        Parameters
        ----------
        dilation_iterations : int, optional
            Number of dilation iterations to apply.
        """
        return self.dead_traces_matrix.copy()

    @transformable
    def get_alive_traces_matrix(self):
        """ Alive traces matrix.
        Due to decorator, allows for additional transforms at loading time.

        Parameters
        ----------
        dilation_iterations : int, optional
            Number of dilation iterations to apply.
        """
        return 1 - self.dead_traces_matrix

    def get_grid(self, frequency=100, iline=True, xline=True, margin=20):
        """ Compute the grid over alive traces. """
        #pylint: disable=unexpected-keyword-arg
        # Parse parameters
        frequency = frequency if isinstance(frequency, (tuple, list)) else (frequency, frequency)

        # Prepare dilated `dead_traces_matrix`
        dead_traces_matrix = self.get_dead_traces_matrix(dilation_iterations=margin)

        if margin:
            dead_traces_matrix[:+margin, :] = 1
            dead_traces_matrix[-margin:, :] = 1
            dead_traces_matrix[:, :+margin] = 1
            dead_traces_matrix[:, -margin:] = 1

        # Select points to keep
        idx_i, idx_x = np.nonzero(~dead_traces_matrix)
        grid = np.zeros_like(dead_traces_matrix)
        if iline:
            mask = (idx_i % frequency[0] == 0)
            grid[idx_i[mask], idx_x[mask]] = 1
        if xline:
            mask = (idx_x % frequency[1] == 0)
            grid[idx_i[mask], idx_x[mask]] = 1
        return grid


    # Properties
    def __len__(self):
        """ Number of meaningful traces in a Geometry. """
        if hasattr(self, 'n_alive_traces'):
            return self.n_alive_traces
        return self.n_traces

    @property
    def axis_names(self):
        """ Names of the axes: indexing headers and `DEPTH` as the last one. """
        return list(self.index_headers) + ['DEPTH']

    @property
    def bbox(self):
        """ Bounding box with geometry limits. """
        return np.array([[0, s] for s in self.shape])

    @property
    def spatial_shape(self):
        """ Shape of the cube along indexing headers. """
        return tuple(self.shape[:2])

    @property
    def textual(self):
        """ Wrapped textual header of SEG-Y file. """
        text = self.segy_text[0]
        lines = [text[start:start + 80] for start in range(0, len(text), 80)]
        return '\n'.join(lines)

    @property
    def file_size(self):
        """ Storage size in GB. """
        return round(os.path.getsize(self.path) / (1024**3), 3)

    @property
    def nbytes(self):
        """ Size of the instance in bytes. """
        attributes = set(['headers'])
        attributes.update({attribute for attribute in self.__dict__
                           if 'matrix' in attribute or '_quality' in attribute})

        return self.cache_size + sum(sys.getsizeof(getattr(self, attribute))
                                     for attribute in attributes if hasattr(self, attribute))

    @property
    def ngbytes(self):
        """ Size of instance in gigabytes. """
        return self.nbytes / (1024 ** 3)


    # Attribute retrieval. Used by `Field` instances
    def load_attribute(self, src, **kwargs):
        """ Load instance attribute from a string, e.g. `snr` or `std_matrix`.
        Used from a field to re-direct calls.
        """
        return self.get_property(src=src, **kwargs)

    @transformable
    def get_property(self, src, **_):
        """ Load a desired instance attribute. Decorated to allow additional postprocessing steps. """
        return getattr(self, src)


    # Textual representation
    def __repr__(self):
        msg = f'geometry `{self.short_name}`'
        if not hasattr(self, 'shape'):
            return f'<Unprocessed {msg}>'
        return f'<Processed {msg}: {tuple(self.shape)} at {hex(id(self))}>'

    def __str__(self):
        if not hasattr(self, 'shape'):
            return f'<Unprocessed geometry `{self.short_path}`>'

        msg = f"""
        Processed geometry for cube        {self.path}
        Index headers:                     {self.index_headers}
        Traces:                            {self.n_traces:,}
        Shape:                             {tuple(self.shape)}
        Time delay:                        {self.delay} ms
        Sample interval:                   {self.sample_interval} ms
        Sample rate:                       {self.sample_rate} Hz
        Area:                              {self.area:4.1f} km²

        File size:                         {self.file_size:4.3f} GB
        Instance (in-memory) size:         {self.ngbytes:4.3f} GB
        """

        if self.converted and os.path.exists(self.segy_path):
            segy_size = os.path.getsize(self.segy_path) / (1024 ** 3)
            msg += f'SEG-Y original size:               {segy_size:4.3f} GB\n'

        if hasattr(self, 'dead_traces_matrix'):
            msg += f"""
        Number of dead  traces:            {self.n_dead_traces:,}
        Number of alive traces:            {self.n_alive_traces:,}
        Fullness:                          {self.n_alive_traces / self.n_traces:2.2f}
        """

        if self.has_stats:
            msg += f"""
        Value statistics:
        mean | std:                        {self.mean:>10.2f} | {self.std:<10.2f}
        min | max:                         {self.min:>10.2f} | {self.max:<10.2f}
        q01 | q99:                         {self.get_quantile(0.01):>10.2f} | {self.get_quantile(0.99):<10.2f}
        Number of unique values:           {self.n_value_uniques:>10}
        """

        if self.quantized:
            try:
                msg += f"""
        Quantization ranges:               {self.quantization_ranges[0]:>10.2f} | {self.quantization_ranges[1]:<10.2f}
        Quantization error:                {self.quantization_error:>10.3f}
            """
            except AttributeError:
                pass
        return dedent(msg).strip()

    def print(self, printer=print):
        """ Show textual representation. """
        select_printer(printer)(self)

    def print_textual(self, printer=print):
        """ Show textual header from original SEG-Y. """
        select_printer(printer)(self.textual)

    def print_location(self, printer=print):
        """ Show ranges for each of the headers. """
        msg = '\n'.join(f'{header+":":<35} [{uniques[0]}, {uniques[-1]}]'
                        for header, uniques in zip(self.index_headers, self.index_sorted_uniques))
        select_printer(printer)(msg)

    def log(self):
        """ Log info about geometry to a file next to the cube. """
        self.print(printer=os.path.dirname(self.path) + '/CUBE_INFO.log')


    # Visual representation
    def show(self, matrix='snr', plotter=plot, **kwargs):
        """ Show geometry related top-view map. """
        matrix_name = matrix if isinstance(matrix, str) else kwargs.get('matrix_name', 'custom matrix')
        kwargs = {
            'cmap': 'magma',
            'title': f'`{matrix_name}` map of cube `{self.short_name}`',
            'xlabel': self.index_headers[0],
            'ylabel': self.index_headers[1],
            'colorbar': True,
            **kwargs
            }
        matrix = getattr(self, matrix) if isinstance(matrix, str) else matrix
        return plotter(matrix, **kwargs)

    def show_histogram(self, n_traces=100_000, seed=42, bins=50, plotter=plot, **kwargs):
        """ Show distribution of amplitudes in a random subset of the cube. """
        data = self.load_subset(n_traces=n_traces, seed=seed)

        kwargs = {
            'title': (f'Amplitude distribution for {self.short_name}' +
                      f'\n Mean/std: {np.mean(data):3.3f}/{np.std(data):3.3f}'),
            'label': 'Amplitudes histogram',
            'xlabel': 'amplitude',
            'ylabel': 'density',
            **kwargs
        }
        return plotter(data, bins=bins, mode='histogram', **kwargs)

    def show_slide(self, index, axis=0, zoom=None, plotter=plot, **kwargs):
        """ Show seismic slide in desired index.
        Under the hood relies on :meth:`load_slide`, so works with geometries in any formats.
        Parameters
        ----------
        index : int, str
            Index of the slide to show.
            If int, then interpreted as the ordinal along the specified axis.
            If `'random'`, then we generate random index along the axis.
            If string of the `'#XXX'` format, then we interpret it as the exact indexing header value.
        axis : int
            Axis of the slide.
        zoom : tuple, None or 'auto'
            Tuple of slices to apply directly to 2d images. If None, slicing is not applied.
            If 'auto', zero traces on bounds will be dropped.
        plotter : instance of `plot`
            Plotter instance to use.
            Combined with `positions` parameter allows using subplots of already existing plotter.
        """
        axis = self.parse_axis(axis)
        slide = self.load_slide(index=index, axis=axis)
        xmin, xmax, ymin, ymax = 0, slide.shape[0], slide.shape[1], 0

        if zoom == 'auto':
            zoom = self.compute_auto_zoom(index, axis)
        if zoom:
            slide = slide[zoom]
            xmin = zoom[0].start or xmin
            xmax = zoom[0].stop or xmax
            ymin = zoom[1].stop or ymin
            ymax = zoom[1].start or ymax

        # Plot params
        if len(self.index_headers) > 1:
            title = f'{self.axis_names[axis]} {index} out of {self.shape[axis]}'

            if axis in [0, 1]:
                xlabel = self.index_headers[1 - axis]
                ylabel = 'DEPTH'
            else:
                xlabel = self.index_headers[0]
                ylabel = self.index_headers[1]
        else:
            title = '2D seismic slide'
            xlabel = self.index_headers[0]
            ylabel = 'DEPTH'

        kwargs = {
            'title': title,
            'suptitle':  f'Field `{self.short_name}`',
            'xlabel': xlabel,
            'ylabel': ylabel,
            'cmap': 'Greys_r',
            'colorbar': True,
            'extent': (xmin, xmax, ymin, ymax),
            'labeltop': False,
            'labelright': False,
            **kwargs
        }
        return plotter(slide, **kwargs)

    def show_spectrum(self, n_traces=1000, seed=42, frequency_threshold=150,
                      filter_length=31, filter_order=3, **kwargs):
        """ Show power and phase spectrums of a random subset of a cube data. """
        data = self.load_subset(n_traces=n_traces, seed=seed)
        spectrum = np.fft.rfft(data, axis=-1)

        power_spectrum = np.abs(spectrum).mean(axis=0)
        phase_spectrum = np.angle(spectrum).mean(axis=0)
        frequencies = np.fft.rfftfreq(n=self.depth, d=self.sample_interval * 1e-3)

        if frequency_threshold is not None:
            mask = frequencies <= frequency_threshold
            frequencies = frequencies[mask]
            power_spectrum = power_spectrum[mask]
            phase_spectrum = phase_spectrum[mask]

        power_spectrum_smoothed = savgol_filter(power_spectrum, filter_length, filter_order)
        f1, f2 = frequencies[[np.argmax(power_spectrum), np.argmax(power_spectrum_smoothed)]]

        kwargs = {
            'combine': 'separate',
            'ncols': 2,
            'suptitle': f'Spectrum on `{self.short_name}`',
            'title': ['power spectrum', 'phase spectrum'],
            'label': [[f'power spectrum: max at {f1:3.1f} Hz', f'power spectrum smoothed: max at {f2:3.1f} Hz'], ''],
            **kwargs
        }
        plotter = plot([[(frequencies, power_spectrum), (frequencies, power_spectrum_smoothed)],
                        (frequencies, phase_spectrum)], mode='curve', **kwargs)
        plotter.subplots[0].ax.axvline(f1, color='cornflowerblue', linewidth=1)
        plotter.subplots[0].ax.axvline(f2, color='goldenrod', linewidth=1)
        return plotter

    def show_section(self, locations, zoom=None, plotter=plot, linecolor='gray', linewidth=3, show=True,
                     savepath=None, **kwargs):
        """ Show seismic section via desired traces.
        Under the hood relies on :meth:`load_section`, so works with geometries in any formats.

        Parameters
        ----------
        locations : iterable
            Locations of traces to construct section.
        zoom : tuple, None or 'auto'
            Tuple of slices to apply directly to 2d images. If None, slicing is not applied.
            If 'auto', zero traces on bounds will be dropped.
        plotter : instance of `plot`
            Plotter instance to use.
            Combined with `positions` parameter allows using subplots of already existing plotter.
        linecolor : str or None
            Color of line to mark node traces. If None, lines will not be drawn.
        linewidth : int
            With of the line.
        show : bool
            Whether to show created plot or not.
        savepath : str
            Path to save the plot to.
        kwargs : dict
            kwargs for plotter
        """
        section, indices, nodes = self.load_section(locations)
        xmin, xmax, ymin, ymax = 0, section.shape[0], section.shape[1], 0

        if zoom == 'auto':
            nonzero = np.nonzero((section != 0).any(axis=1))[0]
            if len(nonzero) > 0:
                start, stop = nonzero[[0, -1]]
                zoom = (slice(start, stop + 1), slice(None))
            else:
                zoom = None
        if zoom:
            section = section[zoom]
            xmin = zoom[0].start or xmin
            xmax = zoom[0].stop or xmax
            ymin = zoom[1].stop or ymin
            ymax = zoom[1].start or ymax

        # Plot params
        title = f'Section via {str(locations)[1:-1]}'
        xlabel = f'{self.index_headers[0]}/{self.index_headers[1]}'
        ylabel = 'DEPTH'

        kwargs = {
            'title': title,
            'suptitle':  f'Field `{self.short_name}`',
            'xlabel': xlabel,
            'ylabel': ylabel,
            'cmap': 'Greys_r',
            'colorbar': True,
            'extent': (xmin, xmax, ymin, ymax),
            'labeltop': False,
            'labelright': False,
            **kwargs
        }

        plt = plotter(section, show=show, **kwargs)

        xticks = plt[0].ax.get_xticks().astype('int32')
        nearest_ticks = np.argmin(np.abs(xticks.reshape(-1, 1) - nodes.reshape(1, -1)), axis=0)
        xticks[nearest_ticks] = nodes
        labels = np.array(list(map('\n'.join, indices.astype('int32').astype(str))))[xticks % section.shape[0]]

        plt[0].ax.set_xticks(xticks[:-1])
        plt[0].ax.set_xticklabels(labels[:-1])

        if linecolor:
            for pos in nodes:
                plt[0].ax.plot([pos, pos], [0, section.shape[1]], color=linecolor, linewidth=linewidth)

        if savepath is not None:
            plt.save(savepath=savepath)

        return plt

    def show_section_map(self, locations, linecolor='green', linewidth=3, pointcolor='blue',
                         pointsize=100, marker='*', show=True, savepath=None, **kwargs):
        """ Show section line on 2D geometry map.

        Parameters
        ----------
        locations : iterable
            Locations of traces to construct section.
        linecolor : str, optional
            Color of section line, by default 'green'
        linewidth : int, optional
            Width of section line, by default 3
        pointcolor : str, optional
            Color of points at locations, by default 'blue'
        pointsize : int, optional
            Size of points at locations, by default 100
        marker : str, optional
            Points marker, by default '*'
        show : bool
            Whether to show created plot or not.
        savepath : str
            Path to save the plot to.
        kwargs : dict
            kwargs for `show` method to plot geometry map (e.g., 'matrix')

        Returns
        -------
        plotter
            Plot instance
        """
        title = f'Section via {str(locations)[1:-1]}'
        locations = np.array(locations)

        kwargs = {
            'title': title,
            'labeltop': False,
            'labelright': False,
            'matrix': 'snr',
            **kwargs
        }

        plotter = self.show(show=show, **kwargs)
        plotter[0].ax.scatter(locations[:, 0], locations[:, 1], c=pointcolor, s=pointsize, marker=marker)
        plotter[0].ax.plot(locations[:, 0], locations[:, 1], color=linecolor, linewidth=linewidth)

        if savepath is not None:
            plotter.save(savepath=savepath)

        return plotter

    # Utilities for 2D slides
    def get_slide_index(self, index, axis=0):
        """ Get the slide index along specified axis.
        Integer `12` means 12-th (ordinal) inline.
        String `#244` means inline 244.

        Parameters
        ----------
        index : int, str
            If int, then interpreted as the ordinal along the specified axis.
            If `'random'`, then we generate random index along the axis.
            If string of the `'#XXX'` format, then we interpret it as the exact indexing header value.
        axis : int
            Axis of the slide.
        """
        if isinstance(index, (int, np.integer)):
            if index >= self.shape[axis]:
                raise KeyError(f'Index={index} is out of geometry bounds={self.shape[axis]}!')
            return index
        if index == 'random':
            return np.random.randint(0, self.lengths[axis])
        if isinstance(index, str) and index.startswith('#'):
            index = int(index[1:])
            return self.index_value_to_ordinal[axis][index]
        raise ValueError(f'Unknown type of index={index}')

    def get_slide_bounds(self, index, axis=0):
        """ Compute bounds of the slide: indices of the first/last alive traces of it.

        Parameters
        ----------
        index : int
            Ordinal index of the slide.
        axis : int
            Axis of the slide.
        """
        dead_traces = take_along_axis(self.dead_traces_matrix, index=index, axis=axis)
        left_bound = np.argmin(dead_traces)
        right_bound = len(dead_traces) - np.argmin(dead_traces[::-1]) # the first dead trace
        return left_bound, right_bound

    def compute_auto_zoom(self, index, axis=0):
        """ Compute zoom for a given slide. """
        return slice(*self.get_slide_bounds(index=index, axis=axis)), slice(None)

    # General utility methods
    STRING_TO_AXIS = {
        'i': 0, 'il': 0, 'iline': 0, 'inline': 0,
        'x': 1, 'xl': 1, 'xline': 1, 'xnline': 1,
        'd': 2, 'depth': 2,
    }

    def parse_axis(self, axis):
        """ Convert string representation of an axis into integer, if needed. """
        if isinstance(axis, str):
            if axis in self.index_headers:
                axis = self.index_headers.index(axis)
            elif axis in self.STRING_TO_AXIS:
                axis = self.STRING_TO_AXIS[axis]
        return axis

    def make_slide_locations(self, index, axis=0):
        """ Create locations (sequence of slices for each axis) for desired slide along given axis. """
        locations = [slice(0, item) for item in self.shape]

        axis = self.parse_axis(axis)
        locations[axis] = slice(index, index + 1)
        return locations

    def process_limits(self, limits):
        """ Convert given `limits` to a `slice`. """
        if limits is None:
            return slice(0, self.depth, 1)
        if isinstance(limits, (tuple, list)):
            limits = slice(*limits)
        return limits

    @staticmethod
    def locations_to_shape(locations):
        """ Compute shape of a location. """
        return tuple(slc.stop - slc.start for slc in locations)

    def get_slide_mask(self, index, axis=0, kernel_size=9, threshold=None, erosion=11, dilation=60):
        """ Get mask with dead pixels on a given slide.
        Under the hood, we compute ptp value for each pixel, and deem everything lower than `threshold` be a dead pixel.

        Parameters
        ----------
        index : int, str
            Index of the slide to load.
            If int, then interpreted as the ordinal along the specified axis.
            If `'random'`, then we generate random index along the axis.
            If string of the `'#XXX'` format, then we interpret it as the exact indexing header value.
        axis : int
            Axis of the slide.
        kernel_size : int
            Window size for computations.
        threshold : number
            Minimum ptp value to consider a pixel to be a dead one.
        erosion : int
            Amount of binary erosion for postprocessing.
        dilation : int
            Amount of binary dilataion for postprocessing.

        Returns
        -------
        mask : np.ndarray
            Boolean mask with 1`s at dead pixels and 0`s at alive ones.
        """
        locations = [slice(None)]
        locations[axis] = slice(index, index+1)
        return self.get_crop_mask(tuple(locations), axis, kernel_size, threshold, erosion, dilation)

    def get_crop_mask(self, locations, axis=0, kernel_size=9, threshold=None, erosion=11, dilation=60):
        """ Get mask with dead pixels on a given crop.
        Under the hood, we compute ptp value for each pixel, and deem everything lower than `threshold` be a dead pixel.

        Parameters
        ----------
        locations : tuple of slices
            Slices of the crop to load.
        axis : int
            Direction to split crop into slides to process.
        kernel_size : int
            Window size for computations.
        threshold : number
            Minimum ptp value to consider a pixel to be a dead one.
        erosion : int
            Amount of binary erosion for postprocessing.
        dilation : int
            Amount of binary dilataion for postprocessing.

        Returns
        -------
        mask : np.ndarray
            Boolean mask with 1`s at dead pixels and 0`s at alive ones.
        """
        threshold = threshold or self.std / 10
        array = self.load_crop(locations)
        array = array if array.dtype == np.float32 else array.astype(np.uint8)
        mask = np.zeros_like(array, dtype=np.bool_)

        ptp_kernel = np.ones((kernel_size, kernel_size), dtype=array.dtype)
        erosion_kernel = np.ones((erosion, erosion), dtype=array.dtype) if erosion else None
        dilation_kernel = np.ones((dilation, dilation), dtype=array.dtype)

        for i in range(array.shape[axis]):
            slide = take_along_axis(array, i, axis)
            ptps = cv2.dilate(slide, ptp_kernel) - cv2.erode(slide, ptp_kernel)
            mask_ = ptps <= threshold
            if erosion:
                mask_ = binary_erosion(mask_, structure=erosion_kernel, border_value=True)
            if dilation:
                mask_ = cv2.dilate(mask_.astype('uint8'), dilation_kernel).astype('bool')
            slc = [slice(None)] * 3
            slc[axis] = i
            mask[tuple(slc)] = mask_

        return mask
