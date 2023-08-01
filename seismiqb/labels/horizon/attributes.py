""" Mixin with computed along horizon geological attributes. """
# pylint: disable=too-many-statements
from copy import copy
from ast import literal_eval

from math import isnan
import numpy as np
from numba import njit, prange

from cv2 import dilate
from scipy.signal import ricker
from scipy.ndimage import convolve
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion
from skimage.measure import label
from sklearn.decomposition import PCA

from ...functional import compute_spectral_decomposition, compute_instantaneous_amplitude, compute_instantaneous_phase
from ...utils import transformable, lru_cache



class AttributesMixin:
    """ Geological attributes along horizon:
    - scalars computed from its depth map only: number of holes, perimeter, coverage
    - matrices computed from its depth map only: presence mask, gradients along directions, etc
    - properties of a carcass
    - methods to cut data from the cube along horizon
    - matrices derived from amplitudes along horizon: instant amplitudes/phases, decompositions, etc.

    Method for getting desired attributes is `load_attribute`. It works with nested keys, i.e. one can get attributes
    of horizon subsets. Address method documentation for further details.
    """
    # Modify computed matrices
    def _dtype_to_fill_value(self, dtype):
        if dtype == np.int32:
            fill_value = self.FILL_VALUE
        elif dtype == np.float32:
            fill_value = np.nan
        elif np.issubdtype(dtype, np.bool_):
            fill_value = False
        else:
            raise TypeError(f'Incorrect dtype: `{dtype}`')
        return fill_value

    def matrix_set_dtype(self, matrix, dtype):
        """ Change the dtype and fill_value to match it. """
        matrix = matrix.astype(dtype)
        mask = self._matrix_absence_mask(matrix)
        matrix[mask] = self._dtype_to_fill_value(dtype)
        return matrix

    def matrix_put_on_full(self, matrix):
        """ Convert matrix from being horizon-shaped to cube-shaped. """
        if matrix.shape[:2] != self.field.spatial_shape:
            background = np.full(shape=self.field.spatial_shape,
                                 fill_value=self._dtype_to_fill_value(matrix.dtype),
                                 dtype=matrix.dtype)
            background[self.i_min:self.i_max + 1, self.x_min:self.x_max + 1] = matrix
        else:
            background = matrix.copy()
        return background

    def matrix_fill_to_num(self, matrix, value):
        """ Change the matrix values at points where horizon is absent to a supplied one. """
        mask = self._matrix_absence_mask(matrix)
        matrix[mask] = value
        return matrix

    def _matrix_absence_mask(self, matrix):
        """ Provide bool mask of horizon absence points consistent with shape of given matrix. """
        if matrix.shape[:2] == self.shape:
            return ~self.binary_matrix
        if matrix.shape[:2] == self.field.spatial_shape:
            return ~self.full_binary_matrix
        msg = f"Can't define horizon absence mask with respect to provided matrix since its shape {matrix.shape} "\
                f"doesn't coincide with either horizon shape {self.shape} or field shape {self.field.spatial_shape}."
        raise ValueError(msg)

    def matrix_num_to_fill(self, matrix, value):
        """ Mark points equal to value as absent ones. """
        if value is np.nan:
            mask = np.isnan(matrix)
        else:
            mask = (matrix == value)

        matrix[mask] = self._dtype_to_fill_value(matrix.dtype)
        return matrix

    def matrix_normalize(self, matrix, mode):
        """ Normalize matrix values.

        Parameters
        ----------
        mode : bool, str, optional
            If `min-max` or True, then use min-max scaling.
            If `mean-std`, then use mean-std scaling.
        """
        values = matrix[self.full_binary_matrix]

        if mode in ['min-max', True]:
            min_, max_ = np.nanmin(values), np.nanmax(values)
            matrix = (matrix - min_) / (max_ - min_)
        elif mode == 'mean-std':
            mean, std = np.nanmean(values), np.nanstd(values)
            matrix = (matrix - mean) / std
        else:
            raise ValueError(f'Unknown normalization mode `{mode}`.')
        return matrix

    def matrix_dilate(self, matrix, dilation_iterations=1):
        """ Dilate matrix with (3, 3) kernel with special care for filling values. """
        fill_value = np.nan if issubclass(matrix.dtype.type, np.floating) else self.FILL_VALUE

        matrix = np.nan_to_num(matrix)
        matrix[matrix == self.FILL_VALUE] = 0

        matrix = dilate(matrix, kernel=np.ones((3, 3), np.uint8), iterations=dilation_iterations)

        matrix[self.field.dead_traces_matrix == 1] = fill_value
        return matrix

    def matrix_enlarge(self, matrix, width=3):
        """ Increase visibility of a sparse carcass metric. Should be used only for visualization purposes. """
        if matrix.ndim == 3 and matrix.shape[-1] != 1:
            return matrix

        # Convert all the nans to a number, so that `dilate` can work with it
        matrix = matrix.copy().astype(np.float32).squeeze()
        matrix[np.isnan(matrix)] = self.FILL_VALUE

        # Apply dilations along both axis
        structure = np.ones((1, 3), dtype=np.uint8)
        dilated1 = dilate(matrix, structure, iterations=width)
        dilated2 = dilate(matrix, structure.T, iterations=width)

        # Mix matrices
        matrix = np.full_like(matrix, np.nan)
        matrix[dilated1 != self.FILL_VALUE] = dilated1[dilated1 != self.FILL_VALUE]
        matrix[dilated2 != self.FILL_VALUE] = dilated2[dilated2 != self.FILL_VALUE]

        mask = (dilated1 != self.FILL_VALUE) & (dilated2 != self.FILL_VALUE)
        matrix[mask] = (dilated1[mask] + dilated2[mask] + 1) // 2

        # Fix zero traces
        matrix[np.isnan(self.field.std_matrix)] = np.nan
        return matrix

    @staticmethod
    def pca_transform(data, n_components=3, **kwargs):
        """ Reduce number of channels along the depth axis. """
        flattened = data.reshape(-1, data.shape[-1])
        mask = np.isnan(flattened).any(axis=-1)

        pca = PCA(n_components, **kwargs)
        transformed = pca.fit_transform(flattened[~mask])
        n_components = transformed.shape[-1]

        result = np.full((*data.shape[:2], n_components), np.nan).reshape(-1, n_components)
        result[~mask] = transformed
        result = result.reshape(*data.shape[:2], n_components)
        return result


    # Technical matrices
    @property
    def full_matrix(self):
        """ A method for getting matrix in cubic coordinates. Allows for introspectable cache. """
        return self.matrix_put_on_full(self.matrix)

    @property
    @lru_cache(maxsize=1, apply_by_default=True, copy_on_return=True)
    def binary_matrix(self):
        """ Boolean matrix with `True` values at places where horizon is present and `False` everywhere else. """
        return (self.matrix != self.FILL_VALUE).astype(np.bool_)

    @property
    @lru_cache(maxsize=1, apply_by_default=True, copy_on_return=True)
    def full_binary_matrix(self):
        """ A method for getting binary matrix in cubic (ordinal) coordinates. Allows for introspectable cache. """
        return self.matrix_put_on_full(self.binary_matrix)

    @property
    def mask(self):
        """ An alias. """
        return self.full_binary_matrix

    @property
    @lru_cache(maxsize=1, apply_by_default=True, copy_on_return=True)
    def float_matrix(self):
        """ Matrix with smoothed float values in cubic (ordinal) coordinates. """
        if self.dtype == np.float32:
            return self.full_matrix
        smoothed = self.smooth_out(mode='convolve', kernel_size=5, sigma_spatial=3, max_depth_difference=5,
                                   inplace=False, dtype=np.float32)
        return smoothed.full_matrix

    # Scalars computed from depth map
    @property
    def coverage(self):
        """ Ratio between number of present values and number of good traces in cube. """
        coverage = len(self) / self.field.n_alive_traces
        return round(coverage, 5)

    @property
    def filled_coverage(self):
        """ Ratio between number of points inside horizon filled contour and number of good traces in cube. """
        coverage = np.count_nonzero(self.filled_matrix) / self.field.n_alive_traces
        return round(coverage, 5)

    @property
    def number_of_holes(self):
        """ Number of holes inside horizon borders. """
        holes_array = self.filled_matrix != self.binary_matrix
        _, num = label(holes_array, connectivity=2, return_num=True, background=0)
        return num

    @property
    def perimeter(self):
        """ Number of points in the borders. """
        return np.sum((self.borders_matrix == 1).astype(np.int32))

    @property
    def solidity(self):
        """ Ratio of area covered by horizon to total area inside borders. """
        return len(self) / np.sum(self.filled_matrix)

    @property
    def d_ptp(self):
        """ Horizon spread across the depth. """
        return self.d_max - self.d_min

    # Matrices computed from depth map
    @property
    def borders_matrix(self):
        """ Borders of horizons (borders of holes inside are not included). """
        filled_matrix = self.filled_matrix
        structure = np.ones((3, 3))
        eroded = binary_erosion(filled_matrix, structure, border_value=0)
        return filled_matrix ^ eroded # binary difference operation

    @property
    def boundaries_matrix(self):
        """ Borders of horizons (borders of holes inside included). """
        binary_matrix = self.binary_matrix
        structure = np.ones((3, 3))
        eroded = binary_erosion(binary_matrix, structure, border_value=0)
        return binary_matrix ^ eroded # binary difference operation

    @property
    def filled_matrix(self):
        """ Binary matrix with filled holes (except dead traces). """
        return self.filled_full_matrix[self.bbox[0, 0]:self.bbox[0, 1]+1, self.bbox[1, 0]:self.bbox[1, 1]+1]

    @property
    def filled_full_matrix(self):
        """ Full binary matrix with filled holes (except dead traces). """
        structure = np.ones((3, 3))
        filled_matrix = binary_fill_holes(self.full_binary_matrix, structure)

        filled_matrix[self.field.dead_traces_matrix] = 0
        return filled_matrix

    def grad_along_axis(self, axis=0):
        """ Change of depths along specified direction. """
        grad = np.diff(self.matrix, axis=axis, prepend=self.FILL_VALUE)
        grad[np.abs(grad) > self.d_min] = self.FILL_VALUE
        grad[self.matrix == self.FILL_VALUE] = self.FILL_VALUE
        return grad

    @property
    def grad_i(self):
        """ Change of depths along iline direction. """
        return self.grad_along_axis(1)

    @property
    def grad_x(self):
        """ Change of depths along xline direction. """
        return self.grad_along_axis(0)


    # Carcass properties: should be used only if the horizon is a carcass
    @property
    def is_carcass(self):
        """ Check if the horizon is a sparse carcass. """
        return len(self) / self.filled_matrix.sum() < 0.5

    @property
    def carcass_ilines(self):
        """ Labeled inlines in a carcass. """
        uniques, counts = np.unique(self.points[:, 0], return_counts=True)
        return uniques[counts > 256]

    @property
    def carcass_xlines(self):
        """ Labeled xlines in a carcass. """
        uniques, counts = np.unique(self.points[:, 1], return_counts=True)
        return uniques[counts > 256]

    @property
    def probabilities(self):
        """ Map of the horizon presence probabilities. """
        if hasattr(self, 'proba_points'):
            _map = np.zeros(self.full_matrix.shape, dtype=np.float32)
            _map[self.proba_points[:, 0].astype(np.int32),
                 self.proba_points[:, 1].astype(np.int32)] = self.proba_points[:, 2]

            _map[~self.full_binary_matrix] = np.nan
            return _map

        raise AttributeError(f'Horizon `{self.displayed_name}` hasn\'t `proba_points` attribute. Check, whether'
                             ' the horizon was initialized `from_mask` with `save_probabilities=True` option.')

    def compute_sparse_coverage(self, frequency, margin=1):
        """ Ratio between number of present values on grid and number of grid traces.

        Helpful for evaluating carcass-like horizon coverage relationally to carcass.
        """
        grid = self.field.get_grid(margin=margin, frequency=frequency)
        horizon_on_grid = self.full_binary_matrix & grid
        return np.count_nonzero(horizon_on_grid) / np.count_nonzero(grid)


    # Retrieve data from seismic along horizon
    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_cube_values(self, window=1, offset=0, chunk_size=256, src_geometry=None, apply_float_correction=False, **_):
        """ Get values from the cube along the horizon.

        Parameters
        ----------
        window : int
            Width of data slice along the horizon.
        offset : int
            Offset of data slice with respect to horizon depths matrix.
        chunk_size : int
            Size of data along depth axis processed at a time.
        apply_float_correction : bool
            Whether to apply float correction to fix step-like artifacts.
        """
        geometry = self.field.geometry if src_geometry is None else getattr(self.field, src_geometry)

        if apply_float_correction:
            window += 2

        low = window // 2 - offset
        high = max(window - low, 0)
        chunk_size = min(chunk_size, self.d_max - self.d_min + window)
        result = np.full((*self.field.spatial_shape, window), fill_value=np.nan, dtype=np.float32)

        for d_start in range(max(low, self.d_min), self.d_max + 1, chunk_size):
            d_end = min(d_start + chunk_size, self.d_max + 1)

            # Get chunk from the cube (depth-wise)
            location = (slice(None), slice(None),
                        slice(d_start - low, min(d_end + high, self.field.depth)))
            location, _ = geometry.process_key(location)
            data_chunk = geometry.load_crop(location, use_cache=False)

            # Check which points of the horizon are in the current chunk (and present)
            idx_i, idx_x = np.asarray((self.matrix != self.FILL_VALUE) &
                                      (self.matrix >= d_start) &
                                      (self.matrix < d_end)).nonzero()
            depths = self.matrix[idx_i, idx_x]

            # Convert spatial coordinates to cubic, convert depth to current chunk local system
            idx_i += self.i_min
            idx_x += self.x_min
            depths -= d_start

            # Subsequently add values from the cube to result, then shift horizon 1 unit lower
            for j in range(window):
                result[idx_i, idx_x, j] = data_chunk[idx_i, idx_x, depths]
                depths += 1
                mask = depths < data_chunk.shape[2]
                idx_i = idx_i[mask]
                idx_x = idx_x[mask]
                depths = depths[mask]

        result[~self.full_binary_matrix] = np.nan

        if apply_float_correction:
            result = self._apply_float_correction(result)
        return result

    def get_layer_values(self, other, window=0, offset=0, src_geometry=None):
        """ Get values from the cube between `self` and `other` horizons.

        Parameters
        ----------
        window : int or sequence of two ints
            Additional samples to get along depth axis.
            If two ints, then number of samples above `self` and below `other` horizon.
        offset : int
            Offset of data slice with respect to extraction window.

        TODO: implement chunking
        """
        # Parse parameters
        if self.d_mean > other.d_mean:
            self, other = other, self
        geometry = self.field.geometry if src_geometry is None else getattr(self.field, src_geometry)

        window = window if isinstance(window, (tuple, list)) else (window, window)
        d_start = max( self.d_min - window[0] - offset, 0)
        d_stop  = min(other.d_max + window[1] - offset, geometry.depth)

        # Prepare depth bounds between `self` and `other`
        presence_matrix = self.full_binary_matrix & other.full_binary_matrix
        idx_i, idx_x = np.nonzero(presence_matrix)
        self_depths = self.full_matrix[idx_i, idx_x]
        other_depths = other.full_matrix[idx_i, idx_x]

        self_depths  -= d_start + window[1]
        other_depths -= d_start - window[0]

        # Load data chunk, prepare output array
        data = geometry[:, :, d_start:d_stop]
        n = d_stop - d_start
        result = np.full((*presence_matrix.shape, n), fill_value=np.nan, dtype=np.float32)

        # Subsequently add values from the cube to result, then shift indices 1 unit lower
        for i in range(n):
            mask = self_depths < other_depths
            idx_i = idx_i[mask]
            idx_x = idx_x[mask]
            self_depths = self_depths[mask]
            other_depths = other_depths[mask]

            result[idx_i, idx_x, i] = data[idx_i, idx_x, self_depths]
            self_depths += 1

        result[~presence_matrix] = np.nan
        return result

    def _apply_float_correction(self, array):
        """ Compute a three-point correction from int to float.

        As most of the loaded horizons are stored as int32 depths and only integer samples can be retrieved from
        the cube, extracted values (amplitudes) and their derivatives may have step-like artifacts.
        To eliminate them, we interpolate between depth-wise channels, using the size of difference between
        int32 and float32 horizons as the weight for interpolation.

        Parameters
        ----------
        array : array-like
            Array of (I, X, D) shape.

        Returns
        -------
        Array of (I, X, D-2) shape.
        """
        shifts_matrix = self.float_matrix - self.full_matrix
        middle = array[:, :, 1:-1]
        output = middle.copy()

        mask = shifts_matrix > 0
        output[mask] = (1 - shifts_matrix[mask]).reshape(-1, 1) * middle[mask] + \
                          +(shifts_matrix[mask]).reshape(-1, 1) * array[:, :, 2:][mask]

        mask = shifts_matrix < 0
        output[mask] = (1 + shifts_matrix[mask]).reshape(-1, 1) * middle[mask] + \
                         +(-shifts_matrix[mask]).reshape(-1, 1) * array[:, :, :-2][mask]

        output[np.isnan(shifts_matrix)] = np.nan
        return output


    # Generic attributes loading
    ATTRIBUTE_TO_ALIAS = {
        # Properties
        'full_matrix': ['full_matrix', 'depths'],
        'full_binary_matrix': ['full_binary_matrix', 'mask'],
        'probabilities': ['proba', 'probabilities'],

        # Created by `get_*` methods
        'amplitudes': ['amplitudes', 'cube_values'],
        'metric': ['metric', 'metrics'],
        'instantaneous_phases': ['instant_phases', 'iphases'],
        'instantaneous_amplitudes': ['instant_amplitudes', 'iamplitudes'],

        'fourier_decomposition': ['fourier', 'fourier_decomposition'],
        'wavelet_decomposition': ['wavelet', 'wavelet_decomposition'],
        'spectral_decomposition': ['spectral', ],

        'median_diff': ['median_diff', 'mdiff', 'median_faults'],
        'grad': ['grad', 'gradient', 'gradient_diff', 'gradient_faults'],
        'max_grad': ['max_grad', 'max_gradient', 'maximum_gradient'],
        'max_abs_grad': ['max_abs_grad', 'max_abs_gradient', 'maximum_abs_gradient'],
    }
    ALIAS_TO_ATTRIBUTE = {alias: name for name, aliases in ATTRIBUTE_TO_ALIAS.items() for alias in aliases}

    ATTRIBUTE_TO_METHOD = {
        'amplitudes' : 'get_cube_values',
        'metric' : 'get_metric',
        'instantaneous_phases' : 'get_instantaneous_phases',
        'instantaneous_amplitudes' : 'get_instantaneous_amplitudes',
        'fourier_decomposition' : 'get_fourier_decomposition',
        'wavelet_decomposition' : 'get_wavelet_decomposition',
        'spectral_decomposition' : 'get_spectral_decomposition',

        'median_diff': 'get_median_diff_map',
        'grad': 'get_gradient_map',
        'max_grad': 'get_max_gradient_map',
        'max_abs_grad': 'get_max_abs_gradient_map',
        'spikes': 'get_spikes_mask'
    }

    def load_attribute(self, src, location=None, use_cache=True, enlarge=False, **kwargs):
        """ Load horizon attribute values at requested location.
        This is the intended interface of loading matrices along the horizon, and should be preferred in all scenarios.

        To retrieve the attribute, we either use `:meth:~.get_property` or `:meth:~.get_*` methods: as all of them are
        wrapped with `:func:~.transformable` decorator, you can use its arguments to modify the behavior.

        Parameters
        ----------
        src : str
            Key of the desired attribute. Valid attributes are either properties or aliases, defined
            by `ALIAS_TO_ATTRIBUTE` mapping, for example:

            - 'cube_values' or 'amplitudes': cube values at horizon points;
            - 'metrics' or 'metric': horizon random support metrics.
            - 'instantaneous_phases', 'instant_phases' or 'iphases': instantaneous phase;
            - 'instantaneous_amplitudes', 'instant_amplitudes' or 'iamplitudes': instantaneous amplitude;
            - 'fourier_decomposition' or 'fourier': fourier transform with optional PCA;
            - 'wavelet decomposition' or 'wavelet': wavelet transform with optional PCA;
            - 'full_matrix' or 'depths': horizon depth map in cubic coordinates;
            - 'full_binary_matrix' or 'mask': mask of horizon presence;
        location : sequence of 3 slices
            First two slices are used as `iline` and `xline` ranges to cut crop from.
            Last 'depth' slice is not used, since points are sampled exactly on horizon.
            If None, `src` is returned uncropped.
        enlarge : bool, optional
            Whether to enlarge carcass maps. Defaults to True, if the horizon is a carcass, False otherwise.
            Should be used only for visualization purposes.
        kwargs :
            Passed directly to attribute-evaluating methods from :attr:`.ALIAS_TO_ATTRIBUTE` depending on `src`.

        Examples
        --------
        Load 'depths' attribute for whole horizon:
        >>> horizon.load_attribute('depths')

        Load 'cube_values' attribute for requested slice of fixed width:
        >>> horizon.load_attribute('cube_values', (x_slice, i_slice, 1), window=10)

        Load 'metrics' attribute with specific evaluation parameter and following normalization.
        >>> horizon.load_attribute('metrics', metric='local_corrs', normalize='min-max')
        """
        src = copy(src)
        if isinstance(src, str):
            src_name = src
        if isinstance(src, dict):
            src_name = src.pop('src')
            kwargs.update(src)

        if '[' in src_name:
            pos_, _pos = src_name.find('['), src_name.find(']')
            channels = literal_eval(src_name[1+pos_:_pos])
            src_name = src_name[:pos_]
        else:
            channels = None

        src_name = self.ALIAS_TO_ATTRIBUTE.get(src_name, src_name)
        enlarge = enlarge and self.is_carcass

        if src_name in self.ATTRIBUTE_TO_METHOD:
            method = self.ATTRIBUTE_TO_METHOD[src_name]
            data = getattr(self, method)(use_cache=use_cache, enlarge=enlarge, **kwargs)
        else:
            data = self.get_property(src_name, use_cache=use_cache, enlarge=enlarge, **kwargs)

        # TODO: Someday, we would need to re-write attribute loading methods
        # so they use locations not to crop the loaded result, but to load attribute only at location.
        if location is not None:
            i_slice, x_slice, _ = location
            data = data[i_slice, x_slice]

        if channels == 'middle':
            channels = data.shape[-1] // 2
        if channels is not None:
            data = data[..., channels]
        return data


    # Specific attributes loading
    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=False)
    @transformable
    def get_property(self, src, **_):
        """ Load a desired instance attribute. Decorated to allow additional postprocessing steps. """
        data = getattr(self, src, None)
        if data is None:
            aliases = list(self.ALIAS_TO_ATTRIBUTE.keys())
            raise ValueError(f'Unknown `src` {src}. Expected a matrix-property or one of {aliases}.')
        return data

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_instantaneous_amplitudes(self, window=11, offset=0, **kwargs):
        """ Calculate instantaneous amplitude along the horizon.

        Parameters
        ----------
        window : int
            Width of cube values cutout along horizon to use for attribute calculation.
        offset : int
            Constant shift of cube values cutout up or down from the horizon surface.
        kwargs :
            Passed directly to :meth:`.get_cube_values`.

        Notes
        -----
        Since Hilbert transform produces artifacts at signal start and end, if one's intenston is to use `n` channels
        of the resulting array, the `window` parameter value should better be somewhat bigger than the value of `n`.
        """
        amplitudes = self.get_cube_values(window=window, offset=offset, use_cache=False, **kwargs)
        return compute_instantaneous_amplitude(amplitudes)

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_instantaneous_phases(self, window=11, offset=0, **kwargs):
        """ Calculate instantaneous phase along the horizon.

        Parameters
        ----------
        window : int
            Width of cube values cutout along horizon to use for attribute calculation.
        offset : int
            Constant shift of cube values cutout up or down from the horizon surface.
        kwargs : dict
            Passed directly to :meth:`.get_cube_values`.
        """
        amplitudes = self.get_cube_values(window=window, offset=offset, use_cache=False, **kwargs)
        return compute_instantaneous_phase(amplitudes)

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_metric(self, metric='support_corrs', supports=50, agg='nanmean', **kwargs):
        """ Cached metrics calculation with disabled plotting option.

        Parameters
        ----------
        metric, supports, agg :
            Passed directly to :meth:`.HorizonMetrics.evaluate`.
        kwargs :
            Passed directly to :meth:`.HorizonMetrics.evaluate`.
        """
        metrics = self.metrics.evaluate(metric=metric, supports=supports, agg=agg,
                                        enlarge=False, visualize=False, savepath=None, **kwargs)
        return metrics


    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_fourier_decomposition(self, window=50, **_):
        """ Cached fourier transform calculation follower by dimensionality reduction via PCA.

        Parameters
        ----------
        window : int
            Width of amplitudes slice to calculate fourier transform on.
        """
        amplitudes = self.load_attribute('amplitudes', window=window)
        result = np.abs(np.fft.rfft(amplitudes))
        return result

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_wavelet_decomposition(self, widths=range(1, 14, 3), window=50, **_):
        """ Cached wavelet transform calculation followed by dimensionality reduction via PCA.

        Parameters
        ----------
        widths : list of numbers
            Widths of wavelets to calculate decomposition for.
        window : int
            Width of amplitudes slice to calculate wavelet transform on.
        """
        amplitudes = self.load_attribute('amplitudes', window=window)
        result_shape = *amplitudes.shape[:2], len(widths)
        result = np.empty(result_shape, dtype=np.float32)
        for idx, width in enumerate(widths):
            wavelet = ricker(window, width).reshape(1, 1, -1)
            result[:, :, idx] = convolve(amplitudes, wavelet, mode='constant')[:, :, window // 2]

        return result

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_spectral_decomposition(self, frequencies=(15, 25, 35), wavelet='mexh', method='fft',
                                   normalize=True, window=33, offset=0, **kwargs):
        """ Compute spectral decomposition by convolving data with wavelets at different scales.
        Scales are computed from frequencies to roughly match them.

        Parameters
        ----------
        frequencies : sequence
            Frequencies to compute wavelets scales.
        normalize : bool
            Whether to normalize each frequency-channel by removing outliers and scaling values to [0, 1] range.
        window : int
            Width of amplitudes slice to calculate wavelet transform on.
        """
        amplitudes = self.get_cube_values(window=window, offset=offset, use_cache=False, **kwargs)

        sample_rate = self.field.sample_rate
        spectral =  compute_spectral_decomposition(amplitudes, frequencies=frequencies, wavelet=wavelet,
                                                   sample_rate=sample_rate, method=method)
        spectral = spectral[..., window // 2].transpose(1, 2, 0)

        if normalize:
            q = np.nanquantile(spectral, (0.01, 0.99), axis=(0, 1)).reshape(2, 1, 1, -1)
            spectral = np.clip(spectral, a_min=q[0], a_max=q[1])
            spectral -= q[0]
            spectral /= (q[1] - q[0])
        return spectral


    def get_zerocrossings(self, side, window=15):
        """ Get matrix of depths shifted to nearest point of sign change in cube values.

        Parameters
        ----------
        side : -1 or 1
            Whether to look for sign change above the horizon (-1) or below (1).
        window : positive int
            Width of data slice above/below the horizon made along its surface.
        """
        values = self.get_cube_values(window=window, offset=window // 2 * side, fill_value=0)
        # reverse array along depth axis for invariance
        values = values[:, :, ::side]

        sign = np.sign(values)
        # value 2 in the array below mark cube values sign change along depth axis
        cross = np.abs(np.diff(sign, axis=-1))

        # put 2 at points, where cube values are precisely equal to zero
        zeros = sign[:, :, :-1] == 0
        cross[zeros] = 2

        # obtain indices of first sign change occurrences for every trace
        # if trace doesn't change sign, corresponding index of sign change is 0
        cross_indices = np.argmax(cross == 2, axis=-1)

        # get cube values before sign change
        start_points = self.matrix_to_points(cross_indices).T
        start_values = values[tuple(start_points)]

        # get cube values after sign change
        stop_points = start_points + np.array([[0], [0], [1]])
        stop_values = values[tuple(stop_points)]

        # calculate additional float shifts towards true zero-crossing point
        float_shift = start_values - stop_values
        # do not perform division at points, where both 'start' and 'stop' values are 0
        np.divide(start_values, float_shift, out=float_shift, where=float_shift != 0)

        # treat obtained indices as shifts for label depths matrix
        shift = cross_indices.astype(np.float32)
        # apply additional float shifts to shift matrix
        shift += float_shift.reshape(shift.shape)
        # account for shift matrix sign change
        shift *= side

        result = self.full_matrix + shift
        return result


    # Maps with faults and spikes
    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_median_diff_map(self, iters=2, window_size=11, max_depth_difference=0,
                            threshold=2, dilation_iterations=0, **_):
        """ Compute difference between depth map and its median filtered counterpart.

        Parameters
        ----------
        iters : int
            Number of median filter iterations to perform.
        window_size : int
            A window size to compute the median in.
        max_depth_difference : number
            If the distance between anchor point and the point inside filter is bigger than the threshold,
            then the point is ignored in filter.
        threshold : number
            Threshold to consider a difference between matrix and median value is insignificant.
        dilation_iterations : int
            Number of iterations for binary dilation algorithm to increase areas with significant
            differences between matrix and median filter.
        """
        _ = dilation_iterations # transformable decorator argument

        medfilt = self.full_matrix.astype(np.float32)
        medfilt[self.full_matrix == self.FILL_VALUE] = np.nan

        # Apply `_medfilt` multiple times. Note that there is no dtype conversion in between
        # Also the method returns a new object
        for _ in range(iters):
            medfilt = _medfilt(src=medfilt, window_size=window_size, preserve_missings=True,
                               max_depth_difference=max_depth_difference)

        median_diff = self.full_matrix - medfilt

        if threshold is not None:
            median_diff[np.abs(median_diff) < threshold] = 0

        median_diff[self.full_matrix == self.FILL_VALUE] = np.nan
        return median_diff

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_gradient_map(self, threshold=1, dilation_iterations=2, **_):
        """ Compute combined gradient map along both directions.

        Parameters
        ----------
        threshold : number
            Threshold to consider a gradient value is insignificant.
        dilation_iterations : int
            Number of iterations for binary dilation algorithm to increase areas with significant gradients values.
        """
        _ = dilation_iterations # transformable decorator argument

        grad_i = self.load_attribute('grad_i', on_full=True, dtype=np.float32, use_cache=False)
        grad_x = self.load_attribute('grad_x', on_full=True, dtype=np.float32, use_cache=False)

        if threshold is not None:
            grad_i[np.abs(grad_i) <= threshold] = 0
            grad_x[np.abs(grad_x) <= threshold] = 0

        grad_i[grad_i == self.FILL_VALUE] = np.nan
        grad_x[grad_x == self.FILL_VALUE] = np.nan

        grad = grad_i + grad_x
        grad[np.abs(grad) > self.d_min] = np.nan

        grad[self.field.dead_traces_matrix == 1] = np.nan
        return grad

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_max_gradient_map(self, **_):
        """ Compute maximum of gradients along both directions. """
        grad_i = self.load_attribute('grad_i', on_full=True, dtype=np.float32, use_cache=False)
        grad_x = self.load_attribute('grad_x', on_full=True, dtype=np.float32, use_cache=False)

        matrix = np.nanmax([grad_i, grad_x], axis=0)
        matrix[matrix == self.FILL_VALUE] = np.nan
        matrix = np.abs(matrix)
        return matrix

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_max_abs_gradient_map(self, **_):
        """ Compute maximum of abs gradients along both directions. """
        grad_i = self.load_attribute('grad_i', on_full=True, dtype=np.float32, use_cache=False)
        grad_x = self.load_attribute('grad_x', on_full=True, dtype=np.float32, use_cache=False)
        grad_i[grad_i == self.FILL_VALUE] = np.nan
        grad_x[grad_x == self.FILL_VALUE] = np.nan

        matrix = np.nanmax([np.abs(grad_i), np.abs(grad_x)], axis=0)
        matrix[matrix == self.FILL_VALUE] = np.nan
        matrix[matrix == -self.FILL_VALUE] = np.nan
        return matrix

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_spikes_mask(self, max_spike_width=7, min_spike_size=5, max_depths_distance=2,
                        dilation_iterations=0, nan_amount_to_ignore=1, points_distance=3):
        """ Get spikes mask for the horizon.

        Parameters
        ----------
        max_spike_width : int
            Maximum possible spike size along the iline or xline axes.
        min_spike_size : int
            Minimum possible spike size along the depth axis.
        max_depths_distance : int
            Threshold to consider that depths are close.
            If points has difference in depth not more than this threshold, then we
            assume that depths are almost the same.
        dilation_iterations : int
            Number of iterations for binary dilation algorithm to increase the spikes.
        nan_amount_to_ignore : int
            Number of nan values to skip while finding spike start point.
        points_distance : int
            Distance between points to compare depths to find spikes. Must be more than 0.
        """
        _ = dilation_iterations # transformable decorator argument

        matrix = self.full_matrix.astype(np.float32)
        matrix[matrix == self.FILL_VALUE] = np.nan

        spikes = np.zeros_like(matrix)

        # We try to find spikes on four directions:
        # from left to right, from up to down, from right to left, from down to up
        for rotation_num in range(1, 5):
            matrix = np.rot90(matrix)
            rotated_spikes = _get_spikes_along_line(matrix=matrix,
                                                    max_spike_width=max_spike_width, min_spike_size=min_spike_size,
                                                    max_depths_distance=max_depths_distance,
                                                    nan_amount_to_ignore=nan_amount_to_ignore,
                                                    points_distance=points_distance)
            spikes += np.rot90(rotated_spikes, k=4-rotation_num)

        spikes[spikes > 0] = 1
        spikes[self.field.dead_traces_matrix == 1] = np.nan
        return spikes

# Helper functions
@njit
def _get_spikes_along_line(matrix, max_spike_width=5, min_spike_size=4, max_depths_distance=2,
                           nan_amount_to_ignore=1, points_distance=3):
    """ Find spikes on a matrix for the fixed search direction: from up to down, from left to right.

    Under the hood, the function iterates over matrix lines and find too huge depth differences on neighboring points.
    These points might be spike's starting points.

    If start points were found, we check points on the right of them to find spike's end point.
    We suppose that a depth on the point next to the spikes' end point is close to a depth
    on the point before the spike's start point.
    """
    spikes_mask = np.zeros_like(matrix)
    line_length = matrix.shape[1]

    for line_idx in range(matrix.shape[0]):
        line = matrix[line_idx]

        previous_not_nan_idx = -100
        previous_idx = 0

        while previous_idx < line_length - points_distance:
            # Check that point can be a spike's start point: find too huge depth difference
            previous_depth = line[previous_idx]

            current_idx = previous_idx + points_distance
            current_depth = line[current_idx]

            previous_idx += 1 # for next iter

            if isnan(previous_depth) and isnan(current_depth):
                continue

            if (not isnan(previous_depth)) and isnan(current_depth):
                previous_not_nan_idx = current_idx - points_distance
                continue

            if isnan(previous_depth) and (not isnan(current_depth)):
                if current_idx - previous_not_nan_idx < nan_amount_to_ignore + points_distance:
                    previous_depth = line[previous_not_nan_idx]
                else:
                    continue

            depths_diff = current_depth - previous_depth

            if np.abs(depths_diff) < min_spike_size:
                continue

            # Check a range of points indices where the spike can be and
            # find a point with a depth close to a depth before the spike
            spike_start_idx = current_idx
            spike_potential_end_idx = spike_start_idx + max_spike_width

            if spike_potential_end_idx >= line_length:
                spike_potential_end_idx = line_length - 1

            standard_depth = previous_depth # depth before the spike

            for spike_potential_point_idx in range(spike_start_idx+1, spike_potential_end_idx+1):
                depth = line[spike_potential_point_idx]

                if not isnan(depth):
                    depths_diff = np.abs(standard_depth - depth)

                    if depths_diff <= max_depths_distance:
                        spikes_mask[line_idx, spike_start_idx:spike_potential_point_idx] = 1
                        previous_idx = spike_potential_point_idx # we can skip checked and founded spikes area
                        break

    return spikes_mask

@njit(parallel=True)
def _medfilt(src, window_size, preserve_missings, max_depth_difference):
    """ Jit-accelerated function to apply 2d median filter with special care for `np.nan` values. """
    # max_depth_difference = 0: median across all non-equal-to-self elements in window
    # max_depth_difference = -1: median across all elements in window
    #pylint: disable=too-many-nested-blocks, consider-using-enumerate, not-an-iterable
    k = window_size // 2

    i_range, x_range = src.shape
    dst = src.copy()

    for iline in prange(0, i_range):
        for xline in range(0, x_range):
            central = src[iline, xline]

            if preserve_missings and isnan(central):
                continue

            element = src[max(0, iline-k):min(iline+k+1, i_range),
                          max(0, xline-k):min(xline+k+1, x_range)].ravel()

            # Find elements which are close or distant for the `central`
            # 0 for close, 1 for distant, 2 for nan
            indicator = np.zeros_like(element)

            for i, item in enumerate(element):
                if not isnan(item):
                    if (abs(item - central) > max_depth_difference) or isnan(central):
                        indicator[i] = np.float32(1)
                else:
                    indicator[i] = np.float32(2)

            # If there are more distant points than close in the window, then find median of distant points
            n_close = (indicator == np.float32(0)).sum()
            mask_distant = indicator == np.float32(1)
            n_distant = mask_distant.sum()
            if n_distant > n_close:
                dst[iline, xline] = np.median(element[mask_distant])
    return dst
