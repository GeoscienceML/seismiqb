""" Metrics for seismic objects: cubes and horizons. """
from warnings import warn
from textwrap import dedent
from itertools import zip_longest

import numpy as np
import pandas as pd

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False
try:
    import bottleneck
    import numexpr
    BOTTLENECK_NUMEXPR_AVAILABLE = True
except ImportError:
    BOTTLENECK_NUMEXPR_AVAILABLE = False

from batchflow.notifier import Notifier

from .labels import Horizon
from .utils import Accumulator, to_list
from .plotters import plot



# Device management
def to_device(array, device='cpu'):
    """ Transfer array to chosen GPU, if possible.
    If `cupy` is not installed, does nothing.

    Parameters
    ----------
    device : str or int
        Device specificator. Can be either string (`cpu`, `gpu:4`) or integer (`4`).
    """
    if isinstance(device, str) and ':' in device:
        device = int(device.split(':')[1])
    if device in ['cuda', 'gpu']:
        device = 0

    if isinstance(device, int):
        if CUPY_AVAILABLE:
            with cp.cuda.Device(device):
                array = cp.asarray(array)
        else:
            warn('Performance Warning: computing metrics on CPU as `cupy` is not available', RuntimeWarning)
    return array

def from_device(array):
    """ Move the data from GPU, if needed.
    If `cupy` is not installed or supplied array already resides on CPU, does nothing.
    """
    if CUPY_AVAILABLE and hasattr(array, 'device'):
        array = cp.asnumpy(array)
    return array



# Functions to compute various distances between two atleast 2d arrays
def correlation(array1, array2, std1, std2, **kwargs):
    """ Compute correlation. """
    _ = kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np
    if xp is np and BOTTLENECK_NUMEXPR_AVAILABLE:
        covariation = bottleneck.nanmean(numexpr.evaluate('array1 * array2'), axis=-1)
        result = numexpr.evaluate('covariation / (std1 * std2)')
    else:
        covariation = (array1 * array2).mean(axis=-1)
        result = covariation / (std1 * std2)
    return result


def crosscorrelation(array1, array2, std1, std2, **kwargs):
    """ Compute crosscorrelation. """
    _ = std1, std2, kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np
    window = array1.shape[-1]
    pad_width = [(0, 0)] * (array2.ndim - 1) + [(window//2, window - window//2)]
    padded = xp.pad(array2, pad_width=tuple(pad_width))

    accumulator = Accumulator('argmax')
    for i in range(window):
        corrs = (array1 * padded[..., i:i+window]).sum(axis=-1)
        accumulator.update(corrs)
    return accumulator.get(final=True).astype(float) - window//2

class BaseMetrics:
    """ Base class for seismic metrics.
    Child classes have to implement access to `data` and `bad_traces` attributes.
    """
    # pylint: disable=attribute-defined-outside-init, blacklisted-name
    PLOT_DEFAULTS = {
        'cmap': 'Metric',
        'mask_color': 'black'
    }

    LOCAL_DEFAULTS = {
        'kernel_size': 3,
        'agg': 'nanmean',
        'device': 'gpu',
        'amortize': True,
    }

    SUPPORT_DEFAULTS = {
        'supports': 100,
        'safe_strip': 50,
        'agg': 'nanmean',
        'device': 'gpu',
        'amortize': True,
    }

    SMOOTHING_DEFAULTS = {
        'kernel_size': 21,
        'sigma': 10.0,
    }

    EPS = 0.00001


    def evaluate(self, metric, plot_supports=False, enlarge=True,
                 width=5, visualize=True, savepath=None, plotter=plot, **kwargs):
        """ Calculate desired metric, apply aggregation, then plot resulting metric-map.
        To plot the results, set `plot` argument to True.

        Parameters
        ----------
        metric : str
            Name of metric to evaluate.
        plot_supports : bool
            Whether to show support traces on resulting image. Works only if `plot` set to True.
        enlarge : bool
            Whether to apply `:meth:.Horizon.matrix_enlarge` to the result.
        width : int
            Widening for the metric. Works only if `enlarge` set to True.
        visualize : bool
            Whether to use `:func:.plot` to show the result.
        savepath : None or str
            Where to save visualization.
        plotter : instance of `plot`
            Plotter instance to use.
            Combined with `positions` parameter allows using subplots of already existing plotter.
        kwargs : dict
            Arguments to be passed in metric-calculation methods
            (see `:meth:.compute_local` and `:meth:.compute_support`),
            as well as plotting arguments (see `:func:.plot`).
        """
        if 'support' in metric:
            kwargs = {**self.SUPPORT_DEFAULTS, **kwargs}
        elif 'local' in metric:
            kwargs = {**self.LOCAL_DEFAULTS, **kwargs}

        self._last_evaluation = {**kwargs}
        metric_fn = getattr(self, metric)
        metric_map, plot_config = metric_fn(**kwargs)

        if cp is not np and cp.cuda.is_available():
            # pylint: disable=protected-access
            cp._default_memory_pool.free_all_blocks()

        if hasattr(self, 'horizon') and self.horizon.is_carcass and enlarge:
            metric_map = self.horizon.matrix_enlarge(metric_map, width)

        if visualize:
            plot_config = {**self.PLOT_DEFAULTS, **plot_config}
            if savepath is not None:
                plot_config['savepath'] = self.horizon.field.make_path(savepath, name=self.name)
            plotter = plotter(metric_map, **plot_config)

            if 'support' in metric and plot_supports:
                support_coords = self._last_evaluation['support_coords']
                plotter[0].ax.scatter(support_coords[:, 0], support_coords[:, 1], s=33, marker='.', c='blue')

            # Store for debug / introspection purposes
            self._last_evaluation['plotter'] = plotter
        return metric_map

    def compute_local(self, function, data, bad_traces, kernel_size=3,
                      normalize=True, agg='mean', amortize=False, axis=0, device='cpu', pbar=None):
        """ Compute metric in a local fashion, using `function` to compare nearest traces.
        Under the hood, each trace is compared against its nearest neighbours in a square window
        of `kernel_size` size. Results of comparisons are aggregated via `agg` function.

        Works on both `cpu` (via standard `NumPy`) and GPU (with the help of `cupy` library).
        The returned array is always on CPU.

        Parameters
        ----------
        function : callable
            Function to compare two arrays. Must have the following signature:
            `(array1, array2, std1, std2)`, where `std1` and `std2` are pre-computed standard deviations.
            In order to work properly on GPU, must be device-agnostic.
        data : ndarray
            3D array of data to evaluate on.
        bad_traces : ndarray
            2D matrix of traces where the metric should not be computed.
        kernel_size : int
            Window size for comparison traces.
        normalize : bool
            Whether the data should be zero-meaned before computing metric.
        agg : str
            Function to aggregate values for each trace. See :class:`.Accumulator` for details.
        amortize : bool
            Whether the aggregation should be sequential or by stacking all the matrices.
            See :class:`.Accumulator` for details.
        axis : int
            Axis to stack arrays on. See :class:`.Accumulator` for details.
        device : str
            Device specificator. Can be either string (`cpu`, `gpu:4`) or integer (`4`).
        pbar : type or None
            Progress bar to use.
        """
        i_range, x_range = data.shape[:2]
        k = kernel_size // 2 + 1

        # Transfer to GPU, if needed
        data = to_device(data, device)
        bad_traces = to_device(bad_traces, device)
        xp = cp.get_array_module(data) if CUPY_AVAILABLE else np

        # Compute data statistics
        data_stds = data.std(axis=-1)
        bad_traces[data_stds == 0.0] = 1
        if normalize:
            data_n = data - data.mean(axis=-1, keepdims=True)
        else:
            data_n = data

        # Pad everything
        padded_data = xp.pad(data_n, ((0, k), (k, k), (0, 0)), constant_values=xp.nan)
        padded_stds = xp.pad(data_stds, ((0, k), (k, k)), constant_values=0.0)
        padded_bad_traces = xp.pad(bad_traces, k, constant_values=1)

        # Compute metric by shifting arrays
        total = kernel_size * kernel_size - 1
        pbar = Notifier(pbar, total=total)

        accumulator = Accumulator(agg=agg, amortize=amortize, axis=axis, total=total)
        for i in range(k):
            for j in range(-k+1, k):
                # Comparison between (x, y) and (x+i, y+j) vectors is the same as comparison between (x+i, y+j)
                # and (x, y). So, we can compare (x, y) with (x+i, y+j) and save computed result twice:
                # matrix associated with vector (x, y) and matrix associated with (x+i, y+j) vector.
                if (i == 0) and (j <= 0):
                    continue
                shifted_data = padded_data[i:i+i_range, k+j:k+j+x_range]
                shifted_stds = padded_stds[i:i+i_range, k+j:k+j+x_range]
                shifted_bad_traces = padded_bad_traces[k+i:k+i+i_range, k+j:k+j+x_range]

                computed = function(data, shifted_data, data_stds, shifted_stds)
                # Using symmetry property:
                symmetric_bad_traces = padded_bad_traces[k-i:k-i+i_range, k-j:k-j+x_range]
                symmetric_computed = computed[:i_range-i, max(0, -j):min(x_range, x_range-j)]
                symmetric_computed = xp.pad(symmetric_computed,
                                            ((i, 0), (max(0, j), -min(0, j))),
                                            constant_values=xp.nan)

                computed[shifted_bad_traces == 1] = xp.nan
                symmetric_computed[symmetric_bad_traces == 1] = xp.nan
                accumulator.update(computed)
                accumulator.update(symmetric_computed)
                pbar.update(2)
        pbar.close()

        result = accumulator.get(final=True)
        return from_device(result)

    @staticmethod
    def find_supports(supports, bad_traces, safe_strip, carcass_mode=False, horizon=None,
                      device='cpu', seed=None):
        """ Find valid supports coordinates.

        Parameters
        ----------
        supports : int or ndarray
            If int, then number of supports to generate randomly from non-bad traces.
            If ndarray, then should be of (N, 2) shape and contain coordinates of reference traces.
        bad_traces : ndarray
            2D matrix of traces where the metric should not be computed.
        safe_strip : int
            Margin for computing metrics safely.
        carcass_mode : bool
            Whether to use carcass intersection nodes as supports traces.
            Notice that it works only for a carcass.
            Note, if `carcass_mode` is True, then the `horizon` argument must be provided.
        horizon : :class:`.Horizon`, optional
            Instance of a carcass horizon for which to create supports in intersection points.
        device : str
            Device specificator. Can be either string (`cpu`, `gpu:4`) or integer (`4`).
        seed : int, optional
            Seed the random numbers generator.
        """
        xp = cp if (CUPY_AVAILABLE and device != 'cpu') else np

        if isinstance(supports, int):
            if safe_strip:
                bad_traces = bad_traces.copy()
                bad_traces[:, :safe_strip], bad_traces[:, -safe_strip:] = 1, 1
                bad_traces[:safe_strip, :], bad_traces[-safe_strip:, :] = 1, 1

            valid_traces = xp.where(bad_traces == 0)

            if carcass_mode and (horizon is not None) and horizon.is_carcass:
                carcass_ilines = horizon.carcass_ilines
                carcass_xlines = horizon.carcass_xlines

                carcass_ilines = to_device(carcass_ilines, device)
                carcass_xlines = to_device(carcass_xlines, device)

                mask_i = xp.in1d(valid_traces[0], carcass_ilines)
                mask_x = xp.in1d(valid_traces[1], carcass_xlines)
                mask = mask_i & mask_x

                valid_traces = (valid_traces[0][mask], valid_traces[1][mask])

            rng = xp.random.default_rng(seed=seed)
            indices = rng.integers(low=0, high=len(valid_traces[0]), size=supports)

            support_coords = xp.asarray([valid_traces[0][indices], valid_traces[1][indices]]).T

        elif isinstance(supports, (tuple, list, np.ndarray)):
            support_coords = xp.asarray(supports)

        else:
            raise TypeError('Unknown type for the `supports` argument. '
                            'It must be one of: int, tuple, list or np.ndarray.')

        if len(support_coords) == 0:
            raise ValueError('No valid support coordinates was found. '
                             'Check input surfaces for available common points.')

        return support_coords

    def compute_support(self, function, data, bad_traces, supports, safe_strip=0, carcass_mode=False,
                        normalize=True, agg='mean', amortize=False, axis=0, device='cpu', pbar=None, seed=None):
        """ Compute metric in a support fashion, using `function` to compare all the traces
        against a set of (randomly chosen or supplied) reference ones.
        Results of comparisons are aggregated via `agg` function.

        Works on both `cpu` (via standard `NumPy`) and GPU (with the help of `cupy` library).
        The returned array is always on CPU.

        Parameters
        ----------
        function : callable
            Function to compare two arrays. Must have the following signature:
            `(array1, array2, std1, std2)`, where `std1` and `std2` are pre-computed standard deviations.
            In order to work properly on GPU, must be device-agnostic.
        data : ndarray
            3D array of data to evaluate on.
        bad_traces : ndarray
            2D matrix of traces where the metric should not be computed.
        supports : int or ndarray
            If int, then number of supports to generate randomly from non-bad traces.
            If ndarray, then should be of (N, 2) shape and contain coordinates of reference traces.
        safe_strip : int
            Margin for computing metrics safely.
        carcass_mode : bool
            Whether to use carcass intersection nodes as supports traces.
            Notice that it works only for a carcass.
        normalize : bool
            Whether the data should be zero-meaned before computing metric.
        agg : str
            Function to aggregate values for each trace. See :class:`.Accumulator` for details.
        amortize : bool
            Whether the aggregation should be sequential or by stacking all the matrices.
            See :class:`.Accumulator` for details.
        axis : int
            Axis to stack arrays on. See :class:`.Accumulator` for details.
        device : str
            Device specificator. Can be either string (`cpu`, `gpu:4`) or integer (`4`).
        pbar : type or None
            Progress bar to use.
        seed : int, optional
            Seed the random numbers generator for supports coordinates.
        """
        # Transfer to GPU, if needed
        data = to_device(data, device)
        bad_traces = to_device(bad_traces, device)
        xp = cp.get_array_module(data) if CUPY_AVAILABLE else np

        # Compute data statistics
        data_stds = data.std(axis=-1)
        bad_traces[data_stds == 0.0] = 1
        if normalize:
            data_n = data - data.mean(axis=-1, keepdims=True)
        else:
            data_n = data

        horizon = getattr(self, 'horizon', None)
        support_coords = BaseMetrics.find_supports(supports=supports, bad_traces=bad_traces,
                                                   safe_strip=safe_strip, carcass_mode=carcass_mode,
                                                   horizon=horizon, device=device, seed=seed)

        # Save for plot and introspection
        if not hasattr(self, '_last_evaluation'):
            self._last_evaluation = {}

        self._last_evaluation['support_coords'] = from_device(support_coords)

        # Generate support traces
        support_traces = data_n[support_coords[:, 0], support_coords[:, 1]]
        support_stds = data_stds[support_coords[:, 0], support_coords[:, 1]]

        # Compute metric
        pbar = Notifier(pbar, total=len(support_traces))
        accumulator = Accumulator(agg=agg, amortize=amortize, axis=axis, total=len(support_traces))

        valid_data = data_n[bad_traces != 1]
        valid_stds = data_stds[bad_traces != 1]

        for i, _ in enumerate(support_traces):
            computed = function(valid_data, support_traces[i], valid_stds, support_stds[i])
            accumulator.update(computed)
            pbar.update()
        pbar.close()

        result = xp.full(shape=(data_n.shape[0], data_n.shape[1]), fill_value=xp.nan, dtype=data_n.dtype)
        result[bad_traces != 1] = accumulator.get(final=True)

        return from_device(result)


    def local_corrs(self, kernel_size=3, normalize=True, agg='mean', amortize=False,
                    device='cpu', pbar=None, **kwargs):
        """ Compute correlation in a local fashion. """
        metric = self.compute_local(function=correlation, data=self.data, bad_traces=self.bad_traces,
                                    kernel_size=kernel_size, normalize=normalize, agg=agg, amortize=amortize,
                                    device=device, pbar=pbar)

        title, plot_defaults = self.get_plot_defaults()
        title = f'Local correlation, k={kernel_size}, with `{agg}` aggregation\nfor {title}'
        plot_config = {
            **plot_defaults,
            'title': title,
            'vmin': -1.0, 'vmax': 1.0,
            **kwargs
        }
        return metric, plot_config

    def support_corrs(self, supports=100, safe_strip=0, carcass_mode=False, normalize=True, agg='mean', amortize=False,
                      device='cpu', pbar=None, **kwargs):
        """ Compute correlation against reference traces. """
        metric = self.compute_support(function=correlation, data=self.data, bad_traces=self.bad_traces,
                                      supports=supports, safe_strip=safe_strip, carcass_mode=carcass_mode,
                                      normalize=normalize, agg=agg, device=device, amortize=amortize,
                                      pbar=pbar)

        title, plot_defaults = self.get_plot_defaults()
        n_supports = supports if isinstance(supports, int) else len(supports)
        title = f'Support correlation with {n_supports} supports\nwith `{agg}` aggregation\nfor {title}'
        plot_config = {
            **plot_defaults,
            'title': title,
            'vmin': -1.0, 'vmax': 1.0,
            'colorbar': True,
            'bad_color': 'k',
            **kwargs
        }
        return metric, plot_config


    def local_crosscorrs(self, kernel_size=3, normalize=False, agg='mean', amortize=False,
                         device='cpu', pbar=None, **kwargs):
        """ Compute cross-correlation in a local fashion. """
        metric = self.compute_local(function=crosscorrelation, data=self.data, bad_traces=self.bad_traces,
                                    kernel_size=kernel_size, normalize=normalize, agg=agg, amortize=amortize,
                                    device=device, pbar=pbar)
        zvalue = np.nanquantile(np.abs(metric), 0.98).astype(np.int32)

        title, plot_defaults = self.get_plot_defaults()
        title = f'Local cross-correlation, k={kernel_size}, with `{agg}` aggregation\nfor {title}'
        plot_config = {
            **plot_defaults,
            'title': title,
            'cmap': 'seismic_r',
            'vmin': -zvalue, 'vmax': zvalue,
            **kwargs
        }
        return metric, plot_config

    def support_crosscorrs(self, supports=100, safe_strip=0, carcass_mode=False, normalize=False,
                           agg='mean', amortize=False, device='cpu', pbar=None, **kwargs):
        """ Compute cross-correlation against reference traces. """
        metric = self.compute_support(function=crosscorrelation, data=self.data, bad_traces=self.bad_traces,
                                      supports=supports, safe_strip=safe_strip, carcass_mode=carcass_mode,
                                      normalize=normalize, agg=agg, amortize=amortize, device=device, pbar=pbar)
        zvalue = np.nanquantile(np.abs(metric), 0.98).astype(np.int32)

        title, plot_defaults = self.get_plot_defaults()
        n_supports = supports if isinstance(supports, int) else len(supports)
        title = f'Support cross-correlation with {n_supports} supports with `{agg}` aggregation\nfor {title}'
        plot_config = {
            **plot_defaults,
            'title': title,
            'cmap': 'seismic_r',
            'vmin': -zvalue, 'vmax': zvalue,
            **kwargs
        }
        return metric, plot_config



class HorizonMetrics(BaseMetrics):
    """ Evaluate metric(s) on horizon(s).
    During initialization, data along the horizon is cut with the desired parameters.
    To get the value of a particular metric, use :meth:`.evaluate`::
        HorizonMetrics(horizon).evaluate('support_corrs', supports=20, agg='mean')

    To plot the results, set `plot` argument of :meth:`.evaluate` to True.

    Parameters
    horizons : :class:`.Horizon` or sequence of :class:`.Horizon`
        Horizon(s) to evaluate.
        Can be either one horizon, then this horizon is evaluated on its own,
        or sequence of two horizons, then they are compared against each other,
        or nested sequence of horizon and list of horizons, then the first horizon is compared against the
        best match from the list.
    other parameters
        Passed direcly to :meth:`.Horizon.get_cube_values` or :meth:`.Horizon.get_cube_values_line`.
    """
    AVAILABLE_METRICS = [
        'local_corrs', 'support_corrs',
        'local_btch', 'support_btch',
        'local_kl', 'support_kl',
        'local_js', 'support_js',
        'local_hellinger', 'support_hellinger',
        'local_tv', 'support_tv',
        'instantaneous_phase',
    ]

    def __init__(self, horizons, window=23, offset=0, normalize=False, chunk_size=256):
        super().__init__()
        horizons = list(horizons) if isinstance(horizons, tuple) else horizons
        horizons = horizons if isinstance(horizons, list) else [horizons]
        self.horizons = horizons

        # Save parameters for later evaluation
        self.window, self.offset, self.normalize, self.chunk_size = window, offset, normalize, chunk_size

        # The first horizon is used to evaluate metrics
        self.horizon = horizons[0]
        self.name = self.horizon.short_name

        # Properties
        self._data = None
        self._probs = None
        self._bad_traces = None


    @classmethod
    def evaluate_support(cls, horizons, metric='support_corrs', supports=100, bad_traces=None, safe_strip=0,
                         device='cpu', seed=None, **kwargs):
        """ Evaluate support metric for given horizons using same support coordinates.

        Parameters
        ----------
        horizons : list of :class:`.Horizon`
            List of horizon instances for which evaluate the metric.
        metric : str
            Name of metric to evaluate.
        supports : int or ndarray
            If int, then number of supports to generate randomly from non-bad traces.
            If ndarray, then should be of (N, 2) shape and contain coordinates of reference traces.
        bad_traces : ndarray
            2D matrix of traces where the metric should not be computed.
        safe_strip : int
            Margin for computing metrics safely.
        device : str
            Device specificator. Can be either string (`cpu`, `gpu:4`) or integer (`4`).
        seed : int, optional
            Seed the random numbers generator.
        kwargs : dict
            Additional keyword arguments for the :meth:`.HorizonMetrics.evaluate`.
        """
        xp = cp if (CUPY_AVAILABLE and device != 'cpu') else np

        horizons_bad_traces = []

        # Generate support coordinates
        if isinstance(supports, int):
            # Get bad traces for all compared horizons
            if bad_traces is None:
                bad_traces = xp.zeros(shape=horizons[0].field.spatial_shape, dtype=int)
            else:
                bad_traces = to_device(bad_traces.copy(), device)

            for horizon in horizons:
                horizon_bad_traces = (horizon.full_matrix == horizon.FILL_VALUE).astype(int)
                horizon_bad_traces = to_device(horizon_bad_traces, device)

                horizons_bad_traces.append(horizon_bad_traces)
                bad_traces |= horizon_bad_traces

        support_coords = BaseMetrics.find_supports(supports=supports, bad_traces=bad_traces,
                                                   safe_strip=safe_strip, carcass_mode=False,
                                                   horizon=None, device=device, seed=seed)

        metrics = []

        for horizon in horizons:
            horizon_metric = cls(horizon).evaluate(metric=metric, supports=support_coords, horizon=horizon, **kwargs)
            metrics.append(horizon_metric)

        return metrics


    def get_plot_defaults(self):
        """ Axis labels and horizon/cube names in the title. """
        title = f'horizon `{self.name}` on cube `{self.horizon.field.short_name}`'
        return title, {
            'xlabel': self.horizon.field.axis_names[0],
            'ylabel': self.horizon.field.axis_names[1],
        }

    @property
    def data(self):
        """ Create `data` attribute at the first time of evaluation. """
        if self._data is None:
            self._data = self.horizon.get_cube_values(window=self.window, offset=self.offset,
                                                      chunk_size=self.chunk_size)
            self._data[self._data == Horizon.FILL_VALUE] = np.nan
        return self._data

    @property
    def bad_traces(self):
        """ Traces to fill with `nan` values. """
        if self._bad_traces is None:
            self._bad_traces = self.horizon.field.dead_traces_matrix.copy()
            self._bad_traces[self.horizon.full_matrix == Horizon.FILL_VALUE] = 1
        return self._bad_traces


    def compare(self, *others, clip_value=7, ignore_zeros=False, enlarge=True, width=9, printer=print,
                visualize=True, hist_kwargs=None, show=True, savepath=None, **kwargs):
        """ Compare `self` horizon against the closest in `others`.
        Print textual and show graphical visualization of differences between the two.
        Returns dictionary with collected information: `closest` and `proximity_info`.

        Parameters
        ----------
        clip_value : number
            Clip for differences graph and histogram
        ignore_zeros : bool
            Whether to ignore zero-differences on histogram.
        enlarge : bool
            Whether to enlarge the difference matrix, if one of horizons is a carcass.
        width : int
            Enlarge width. Works only if `enlarge` is True.
        printer : callable, optional
            Function to use to print textual information
        visualize : bool
            Whether to plot the graph
        hist_kwargs, kwargs : dict
            Parameters for histogram / main graph visualization.
        show : bool
            Whether to show created plot or not.
        savepath : str
            Path to save the plot to.
        """
        closest, proximity_info = other, oinfo = self.horizon.find_closest(*others)
        returns = {'closest': closest, 'proximity_info': proximity_info}

        msg = f"""
        Comparing horizons:
        {self.horizon.short_name.rjust(45)}
        {other.short_name.rjust(45)}
        {'—'*45}
        Rate in 5ms:                         {oinfo['window_rate']:8.3f}
        Mean / std of errors:          {oinfo['difference_mean']:+6.2f} / {oinfo['difference_std']:5.2f}
        Mean / std of abs errors:       {oinfo['abs_difference_mean']:5.2f} / {oinfo['abs_difference_std']:5.2f}
        Max abs error:                           {oinfo['abs_difference_max']:4.0f}
        Accuracy@0:                             {oinfo['accuracy@0']:4.3f}
        Accuracy@1:                             {oinfo['accuracy@1']:4.3f}
        Accuracy@2:                             {oinfo['accuracy@2']:4.3f}
        {'—'*45}
        Lengths of horizons:               {len(self.horizon):10,}
                                           {       len(other):10,}
        {'—'*45}
        Average depths of horizons:          {self.horizon.d_mean:8.2f}
                                             {       other.d_mean:8.2f}
        {'—'*45}
        Coverage of horizons:                {self.horizon.coverage:8.4f}
                                             {       other.coverage:8.4f}
        {'—'*45}
        Number of holes in horizons:         {self.horizon.number_of_holes:8}
                                             {       other.number_of_holes:8}
        {'—'*45}
        Additional traces labeled:           {oinfo['present_at_1_absent_at_2']:8}
        (present in one, absent in other)    {oinfo['present_at_2_absent_at_1']:8}
        {'—'*45}
        """
        msg = dedent(msg)

        if printer is not None:
            printer(msg)

        if visualize:
            # Prepare data
            matrix = proximity_info['difference_matrix'].copy()
            if enlarge and (self.horizon.is_carcass or other.is_carcass):
                matrix = self.horizon.matrix_enlarge(matrix, width=width)

            # Field boundaries
            bounds = self.horizon.field.dead_traces_matrix

            # Main plot: differences matrix
            kwargs = {
                'title': (f'Depth comparison\n'
                          f'`self={self.horizon.short_name}` and `other={closest.short_name}`'),
                'suptitle': '',
                'cmap': ['seismic', 'lightgray'],
                'mask_color': ['black', (0, 0, 0, 0)],
                'colorbar': True,
                'vmin': [-clip_value, 0],
                'vmax': [+clip_value, 1],

                'xlabel': self.horizon.field.index_headers[0],
                'ylabel': self.horizon.field.index_headers[1],

                'ncols': 2,
                'augment_mask': True,
                **kwargs,
            }

            plotter = plot([matrix, bounds], show=False, **kwargs)

            legend_kwargs = {
                'color': ('white', 'blue', 'red', 'black', 'lightgray'),
                'label': ('self.depths = other.depths',
                          'self.depths < other.depths',
                          'self.depths > other.depths',
                          'unlabeled traces',
                          'dead traces'),
                'size': 20,
                'loc': 10,
            }

            plotter[1].add_legend(**legend_kwargs)

            # Histogram and labels
            hist_kwargs = {
                'xlabel': 'difference values',
                'ylabel': 'counts',
                'title': 'Histogram of horizon depth differences',
                'ncols': 2,
                **(hist_kwargs or {}),
            }

            graph_msg = '\n'.join(msg.replace('—', '').split('\n')[5:-7])
            graph_msg = graph_msg.replace('\n' + ' '*20, ', ').replace('\t', ' ')
            graph_msg = ' '.join(item for item in graph_msg.split('  ') if item).strip('\n')

            matrix = proximity_info['difference_matrix'].copy()
            hist_data = np.clip(matrix, -clip_value, clip_value)
            hist_data = hist_data[~np.isnan(hist_data)]

            if ignore_zeros:
                zero_mask = hist_data == 0.0
                # Data can be empty in case of two identical horizons
                if zero_mask.sum() != hist_data.size:
                    # pylint: disable=invalid-unary-operand-type
                    hist_data = hist_data[~zero_mask]

                graph_msg += f'\nNumber of zeros in histogram: {zero_mask.sum()}'

            hist_plotter = plot(hist_data, mode='histogram', show=show, **hist_kwargs)
            hist_plotter[1].add_text(graph_msg, size=15)

            if savepath is not None:
                savepath = self.horizon.field.make_path(savepath, name=self.name)
                plotter.save(savepath=savepath)
                hist_plotter.save(savepath=savepath.replace('.', '_histogram.'))

            returns['plotter'] = plotter

        return returns

    Horizon.compare.__doc__ = compare.__doc__

    @staticmethod
    def compute_prediction_std(horizons):
        """ Compute std along depth axis of `horizons`. Used as a measurement of stability of predicitons. """
        field = horizons[0].field
        fill_value = horizons[0].FILL_VALUE

        mean_matrix = np.zeros(field.spatial_shape, dtype=np.float32)
        std_matrix = np.zeros(field.spatial_shape, dtype=np.float32)
        counts_matrix = np.zeros(field.spatial_shape, dtype=np.int32)

        for horizon in horizons:
            fm = horizon.full_matrix
            mask = fm != fill_value

            mean_matrix[mask] += fm[mask]
            std_matrix[mask] += fm[mask] ** 2
            counts_matrix[mask] += 1

        mean_matrix[counts_matrix != 0] /= counts_matrix[counts_matrix != 0]
        mean_matrix[counts_matrix == 0] = fill_value

        std_matrix[counts_matrix != 0] /= counts_matrix[counts_matrix != 0]
        std_matrix -= mean_matrix ** 2
        std_matrix[std_matrix < 0] = 0
        std_matrix = np.sqrt(std_matrix)
        std_matrix[counts_matrix == 0] = np.nan

        return std_matrix



class FaultsMetrics:
    """ Faults metric class. """
    SHIFTS = [-20, -15, -5, 5, 15, 20]

    def similarity_metric(self, semblance, masks, threshold=None):
        """ Compute similarity metric for faults mask. """
        if threshold:
            masks = masks > threshold
        if semblance.ndim == 2:
            semblance = np.expand_dims(semblance, axis=0)
        if semblance.ndim == 3:
            semblance = np.expand_dims(semblance, axis=0)

        if masks.ndim == 2:
            masks = np.expand_dims(masks, axis=0)
        if masks.ndim == 3:
            masks = np.expand_dims(masks, axis=0)

        res = []
        m = self.sum_with_axes(masks * (1 - semblance), axes=[1,2,3])
        weights = np.ones((len(self.SHIFTS), 1))
        weights = weights / weights.sum()
        for i in self.SHIFTS:
            random_mask = self.make_shift(masks, shift=i)
            rm = self.sum_with_axes(random_mask * (1 - semblance), axes=[1,2,3])
            ratio = m/rm
            res += [np.log(ratio)]
        res = np.stack(res, axis=0)
        res = (res * weights).sum(axis=0)
        res = np.clip(res, -2, 2)
        return res

    def sum_with_axes(self, array, axes=None):
        """ Sum for several axes. """
        if axes is None:
            return array.sum()
        if isinstance(axes, int):
            axes = [axes]
        res = array
        axes = sorted(axes)
        for i, axis in enumerate(axes):
            res = res.sum(axis=axis-i)
        return res

    def make_shift(self, array, shift=20):
        """ Make shifts for mask. """
        result = np.zeros_like(array)
        for i, _array in enumerate(array):
            if shift > 0:
                result[i][:, shift:] = _array[:, :-shift]
            elif shift < 0:
                result[i][:, :shift] = _array[:, -shift:]
            else:
                result[i] = _array
        return result


class FaciesMetrics:
    """ Evaluate facies metrics.
    To get the value of a particular metric, use :meth:`.evaluate`::
        FaciesMetrics(horizon, true_label, pred_label).evaluate('dice')

    Parameters
    horizons : :class:`.Horizon` or sequence of :class:`.Horizon`
        Horizon(s) to use as base labels that contain facies.
    true_labels : :class:`.Horizon` or sequence of :class:`.Horizon`
        Facies to use as ground-truth labels.
    pred_labels : :class:`.Horizon` or sequence of :class:`.Horizon`
        Horizon(s) to use as predictions labels.
    """
    def __init__(self, horizons, true_labels=None, pred_labels=None):
        self.horizons = to_list(horizons)
        self.true_labels = to_list(true_labels or [])
        self.pred_labels = to_list(pred_labels or [])


    @staticmethod
    def true_positive(true, pred):
        """ Calculate correctly classified facies pixels. """
        return np.sum(true * pred)

    @staticmethod
    def true_negative(true, pred):
        """ Calculate correctly classified non-facies pixels. """
        return np.sum((1 - true) * (1 - pred))

    @staticmethod
    def false_positive(true, pred):
        """ Calculate misclassified facies pixels. """
        return np.sum((1 - true) * pred)

    @staticmethod
    def false_negative(true, pred):
        """ Calculate misclassified non-facies pixels. """
        return np.sum(true * (1 - pred))

    def sensitivity(self, true, pred):
        """ Calculate ratio of correctly classified facies points to ground-truth facies points. """
        tp = self.true_positive(true, pred)
        fn = self.false_negative(true, pred)
        return tp / (tp + fn)

    def specificity(self, true, pred):
        """ Calculate ratio of correctly classified non-facies points to ground-truth non-facies points. """
        tn = self.true_negative(true, pred)
        fp = self.false_positive(true, pred)
        return tn / (tn + fp)

    def dice(self, true, pred):
        """ Calculate the similarity of ground-truth facies mask and preditcted facies mask. """
        tp = self.true_positive(true, pred)
        fp = self.false_positive(true, pred)
        fn = self.false_negative(true, pred)
        return 2 * tp / (2 * tp + fp + fn)


    def evaluate(self, metrics):
        """ Calculate desired metric and return a dataframe of results.

        Parameters
        ----------
        metrics : str or list of str
            Name of metric(s) to evaluate.
        """
        metrics = [getattr(self, fn) for fn in to_list(metrics)]
        names = [fn.__name__ for fn in metrics]
        rows = []

        for horizon, true_label, pred_label in zip_longest(self.horizons, self.true_labels, self.pred_labels):
            kwargs = {}

            if true_label is not None:
                true = true_label.mask[horizon.mask]
                kwargs['true'] = true

            if pred_label is not None:
                pred = pred_label.mask[horizon.mask]
                kwargs['pred'] = pred

            values = [fn(**kwargs) for fn in metrics]

            index = pd.MultiIndex.from_arrays([[horizon.field.short_name], [horizon.short_name]],
                                              names=['field_name', 'horizon_name'])
            data = dict(zip(names, values))
            row = pd.DataFrame(index=index, data=data)
            rows.append(row)

        df = pd.concat(rows)
        return df
