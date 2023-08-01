""" !!. """
from copy import deepcopy
import numpy as np
import pandas as pd

from scipy.signal import ricker, find_peaks, fftconvolve
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from sklearn.linear_model import Ridge

from ..geometry import array_to_segy
from ..plotters import plot




class WellSeismicMatcher:
    """ Controller class for the process of well-seismic tie.

    At initialization, we extract `seismic_trace` from the provided `field` and process `well` in a way to obtain
    values of `well_times`, `well_impedance` and `well_reflectivity`: only this information is required from input data.
    An initial estimate of a wavelet is computed by :meth:`process_well`.

    After that, we use the concept of `state` to describe the state of well-seismic tie at each given step.
    Essentially, each state is a dict with required `well_times` and `wavelet` keys: they completely
    identify matching at any given stage and allow to visualize it or introspect in any other way.
    Other keys can be used to store important info about operation that created this state.

    The first state should be created manually by :meth:`init_state` that takes an original `well_times`,
    inferred at initialization, and a wavelet (most probably computed by :meth:`extract_wavelet`).
    Methods that alter well-seismic tie automatically store their updated state to the `states` attribute,
    and each subsequent call works of the selected (usually, the last) previous state.

    The usual pipeline of using this class looks like this:
        - instance initialization, call of :meth:`process_well` with required parameters
        - wavelet computation: either manual or :meth:`extract_wavelet`
        - starting state initialization: :meth:`init_state` call with the computed wavelet
        - t0 computation: :meth:`compute_t0` can be used to analytically compute t0 from elevations and speeds
        or to guess it by optimizing cross correlation function between synthetic and real traces
        - stretching and squeezing of `well_times`: by using either method of optimization, we change `well times` to
        better tie the synthetic and real seismic traces.
        Currently available methods are :meth:`optimize_extremas` and :meth:`optimize_well_times_pytorch`.

    The last three items create one or more states, that are stored in the instance.
    At any point, states can be exported or visualized by `save_*` and `show_*` methods.

    Implementation details
    ----------------------
    Instance attributes do not change: `well_times` attribute always points to the well times from the LAS file itself.
    Everything time-related is stored in seconds. Other logs are unchanged.
    Most of the utility methods are written in a way to accept `(**state)` as an argument.

    TODO: DTW optimization method; rethink resampling; better deterministic wavelet;

    Attributes
    ----------
    well_times : np.array
        Initial well times in seconds.
    well_reflectivity : np.array
        Well reflectivity values, indexed by MD.
    seismic_times : np.array
        Time values in seismic data in seconds.
    seismic_trace : np.array
        Seismic amplitudes, indexed by seismic times.
    states : list
        States describing well matching after given operation.
        Each state is a dict with required `well_times` and `wavelet` keys.
    """
    def __init__(self, well, field, coordinates):
        # Well data
        self.well = well
        self.well_bounds = None
        self.well_times = None
        self.well_impedance = None
        self.well_reflectivity = None

        # Seismic data: extract trace at well location
        self.field = field
        self.seismic_times = np.arange(0, field.depth, dtype=np.float32) * field.sample_interval * 1e-3 # seconds
        self.seismic_trace = None
        self.coordinates = None

        if coordinates is not None:
            if isinstance(coordinates, dict):
                coordinates = coordinates[self.well.name]
            self.extract_seismic_trace(coordinates)

        self.states = []


    def extract_seismic_trace(self, coordinates):
        """ Get data from a SEG-Y seismic data.
        Should be overriden for custom algorithm of seismic times / seismic trace acquisition.
        """
        # TODO: add averaging (copy from 2d matcher), add inclinometry warning
        trace = self.field.geometry[coordinates[0], coordinates[1], :]
        self.seismic_trace = trace
        self.coordinates = coordinates


    # Extended initialization
    def process_well(self, impedance_log=None, recompute_ai=False, recompute_rhob=False,
                     filter_ai=False, filter_dt=False, filter_rhob=False):
        """ Get data from a well with optional filtration.
        Should be overriden for custom algorithm of well times / well reflectivity acquisition.

        Reflectivity is computed from an AI log.
        If AI   log is unavailable, it is recomputed from RHOB and DT logs.
        If RHOB log is unavailable, it is recomputed from the DT log by Gardner's equation.

        Parameters
        ----------
        impedance_log : str or None
            If provided, then directly used for computation of reflectivity.
        filter_* : bool, dict
            If bool, then whether to apply filtration to a given log.
            If dict, then parameters of the filtration.
            `fs` parameter defaults to well sampling frequency.
        recompute_* : bool
            Whether to force recompute a given log.
        """
        #pylint: disable=protected-access
        fs = self.well.compute_sampling_frequency()

        # Sonic log, optionally filtered
        dt_values = self.well.DT.values                                                                  # us/ft

        if filter_dt:
            filtration_parameters = {
                'order': 4, 'frequency': 30, 'btype': 'lowpass', 'fs': fs,
                **(filter_dt if isinstance(filter_dt, dict) else {}),
            }
            dt_values = self.well._compute_filtered_log(dt_values, **filtration_parameters)
            dt_values = np.nan_to_num(dt_values)
            self.well['DT_FILTERED'] = dt_values

        # Prepare impedance log
        if impedance_log is not None:
            pass
        elif 'AI' in self.well.keys and not recompute_ai:
            impedance_log = 'AI'
        else:
            if recompute_rhob or 'RHOB' not in self.well.keys:
                # Gardner's equation
                vp = (0.3048 / dt_values) * 1e6                                                          # m/s
                self.well['RHOB_RECOMPUTED'] = 310 * (vp ** 0.25)                                        # g/cm3
                rhob_log = 'RHOB_RECOMPUTED'
            else:
                rhob_log = 'RHOB'

            # Density values, optionally filtered
            rhob_values = self.well[rhob_log].values
            if filter_rhob:
                filtration_parameters = {
                    'order': 4, 'frequency': 30, 'btype': 'lowpass', 'fs': fs,
                    **(filter_rhob if isinstance(filter_rhob, dict) else {}),
                }
                rhob_values = self.well._compute_filtered_log(rhob_values, **filtration_parameters)
                self.well['RHOB_FILTERED'] = rhob_values

            # Recomputed AI. Can omit unit conversions as they dont influence the reflectivity
            self.well['AI_RECOMPUTED'] = rhob_values / (dt_values * 1e-6 / 0.3048)                       # kPa.s/m
            impedance_log = 'AI_RECOMPUTED'

        # Filter impedance log
        if filter_ai:
            filtration_parameters = {
                'order': 4, 'frequency': 30, 'btype': 'lowpass', 'fs': fs,
                **(filter_ai if isinstance(filter_ai, dict) else {}),
            }
            self.well['AI_FILTERED'] = self.well._compute_filtered_log(self.well[impedance_log],
                                                                       **filtration_parameters)
            impedance_log = 'AI_FILTERED'

        self.well.compute_reflectivity(impedance_log=impedance_log, name='R_RECOMPUTED')

        bounds = self.well.get_bounds(dt_values)
        self.well_bounds = slice(bounds[0]-1, bounds[1])
        self.well_times = np.cumsum(dt_values[self.well_bounds]) * 1e-6                                  # seconds
        self.well_impedance = self.well[impedance_log].values[self.well_bounds]
        self.well_reflectivity = self.well['R_RECOMPUTED'].values[self.well_bounds]


    def extract_wavelet(self, method='statistical', window=(1, 1), normalize=False, limits=slice(None),
                        taper=True, wavelet_length=61, state=-1,
                        smoothing=False, smoothing_length=7, smoothing_order=3, **kwargs):
        """ Compute a wavelet by a chosen method.
        Available methods are:
            - `ricker` creates a fixed Ricker wavelet. Additional parameters are `a` for width.
            - `ricker_f` creates a fixed Ricker wavelet. Additional parameters are `f` for peak frequency.
            - `stats1` creates a wavelet with the same power spectrum as the one in seismic trace.
            - `stats2` creates a wavelet with the same power spectrum as the one in autocorrelation of seismic trace.
            - `stats3` creates a wavelet with the same power spectrum as the one in autocorrelation of seismic trace
            with additional tapering.
            - `division` creates a wavelet with the spectrum of divised spectras of reflectivity and seismic trace.
            - `lstsq` computes an optimal wavelet by solving system of linear equations (~Wiegner).

        The last two wavelets should be used after initial well-seismic tie is already performed.

        TODO: add better deterministic wavelets.

        Parameters
        ----------
        method : str
            Which method to use for wavelet extraction.
        normalize : bool
            Whether to normalize output wavelet so that max value equals to 1.
        taper : bool
            Whether to apply taper to seismic trace / reflectivity before computations.
        wavelet_length : int
            Size of the wavelet.
        smoothing : bool
            Whether to apply smoothing to the power spectrum before IRFFT for statistical wavelets.
        smoothing_length : int
            Length of the smoothing kernel.
        smoothing_order : int
            Order of polynomial used for smoothing.
        window : tuple of ints
            Number of traces along each (inline/crossline) direction to use.
            (1, 1) means that only the trace directly at well coordinates is used.
            (3, 3) means that total of 9 traces centering at well coordinates are used.
        limits : slice or None
            If provided, then used to slice seismic trace(s) along depth dimension.
        state : int, dict
            If int, then the index of previous state to use.
            If dict, then a state directly.
        """
        # Prepare traces
        c0, c1 = self.coordinates
        window = np.clip(window, 0, self.field.spatial_shape)
        k0, k1 = window
        traces = self.field.geometry[c0 - k0//2 : c0 + (k0 - k0//2), c1 - k1//2 : c1 + (k1 - k1//2)]
        traces = traces.reshape(-1, traces.shape[-1])
        traces = traces[:, limits]
        trace = np.mean(traces, axis=0)

        if taper:
            traces *= np.blackman(len(trace)) # TODO: taper selection, maybe?
        lenhalf, wlenhalf, wlenflag = len(trace) // 2, wavelet_length // 2, wavelet_length % 2

        # Optionally, prepare reflectivity
        if method in {'lstsq', 'division'}:
            state = state if isinstance(state, dict) else self.states[state]
            reflectivity = self.resample_to_seismic(seismic_times=self.seismic_times,
                                                    well_times=state['well_times'],
                                                    well_data=np.nan_to_num(self.well_reflectivity))[limits]
            if taper:
                reflectivity *= np.blackman(len(trace))

        # Create wavelet
        if method == 'deterministic':
            ...
        elif method == 'ricker':
            # Given `frequency`, one can compute width: a = geometry.sample_rate / (frequency * np.sqrt(2) * np.pi)
            kwargs = {'points': wavelet_length, 'a': 4.5, **kwargs}
            wavelet = ricker(**kwargs)

        elif method == 'ricker_f':
            # ...or just use this method for wavelet creation
            kwargs = {'f': 25, **kwargs}
            f = kwargs['f']
            t = np.arange(wavelet_length) * self.field.sample_interval * 1e-3
            t -= t[wlenhalf]
            pft2 = (np.pi * f * t) ** 2
            wavelet = (1 - 2 * pft2) * np.exp(-pft2)

        elif method == 'lstsq':
            # Fit wavelet coeffs to the current reflectivity / seismic trace
            projection = np.zeros((reflectivity.size, reflectivity.size))
            projection[:wavelet_length, :wavelet_length] = np.eye(wavelet_length)

            reflectivity_toeplitz = toeplitz(reflectivity)
            operator  = np.dot(reflectivity_toeplitz, projection)

            # wavelet = np.linalg.lstsq(op, trace)[0]
            model = Ridge(alpha=0.1, fit_intercept=False)
            model.fit(operator, trace)
            wavelet = model.coef_[:wlenhalf + wlenflag]
            wavelet = np.concatenate((wavelet[::-1], wavelet[wlenflag:]), axis=0)

        else:
            # Compute power spectrum by different algorithms
            if method in {'stats1', 'statistical'}:
                power_spectrum = np.abs(np.fft.rfft(trace))

            elif method in {'stats2', 'autocorrelation'}:
                autocorrelation = fftconvolve(traces, traces[:, ::-1], mode='same', axes=-1)
                autocorrelation = autocorrelation[:, lenhalf - wlenhalf : lenhalf + wlenhalf + wlenflag]
                power_spectrum = np.sqrt(np.abs(np.fft.rfft(autocorrelation, axis=-1))).mean(axis=0)

            elif method in {'stats3'}:
                autocorrelation = fftconvolve(traces, traces[:, ::-1], mode='same', axes=-1)
                autocorrelation = autocorrelation[:, lenhalf - wlenhalf : lenhalf + wlenhalf + wlenflag]
                autocorrelation *= np.hanning(autocorrelation.shape[1])
                power_spectrum = np.sqrt(np.abs(np.fft.rfft(autocorrelation, axis=-1))).mean(axis=0)

                # frequencies = np.fft.rfftfreq(len(autocorrelation), d=self.field.sample_interval * 1e-3)
                # power_spectrum[0] = 0.0

            elif method in {'division'}:
                power_spectrum = np.fft.rfft(trace) / np.fft.rfft(reflectivity)

            if smoothing:
                from scipy.signal import savgol_filter # pylint: disable=import-outside-toplevel
                power_spectrum = savgol_filter(power_spectrum, smoothing_length, smoothing_order)

            # from scipy.signal import hilbert
            # minphase = hilbert(np.log(power_spectrum))

            wavelet = np.real(np.fft.irfft(power_spectrum)[:wlenhalf + wlenflag])
            wavelet = np.concatenate((wavelet[::-1], wavelet[wlenflag:]), axis=0)

        if normalize:
            wavelet /= wavelet.max()
        # if True: #post_taper
        #     wavelet *= np.hanning(wavelet.size)
        return wavelet


    def init_state(self, wavelet=None, state=-1):
        """ Make the first state. By default, uses provided `wavelet` and takes original `well_times`. """
        if isinstance(state, dict):
            previous_state = state
        elif self.states:
            previous_state = self.states[state]
        else:
            previous_state = {
                'well_times': self.well_times,
            }

        state = {
            'type': 'init',
            'well_times': previous_state['well_times'],
            'wavelet': wavelet,
        }
        state['correlation'] = self.compute_metric(**state)
        self.states.append(state)


    def change_wavelet(self, keep_bounds=True,
                       method='statistical', normalize=False, taper=True, wavelet_length=61, state=-1, **kwargs):
        """ Create a state with the computed wavelet.
        Essentially, an alias to a combination of :meth:`extract_wavelet` and :meth:`init_state`.
        """
        previous_state = state if isinstance(state, dict) else self.states[state]
        wavelet = self.extract_wavelet(method=method, normalize=normalize, taper=taper,
                                       wavelet_length=wavelet_length, state=state, **kwargs)
        state = {
            'type': 'change_wavelet',
            'well_times': previous_state['well_times'],
            'wavelet': wavelet
        }
        if keep_bounds and 'bounds' in previous_state:
            state['bounds'] = previous_state['bounds']
        state['correlation'] = self.compute_metric(**state)
        self.states.append(state)


    # Helper functions
    def compute_resampled_synthetic(self, well_times=None, wavelet=None, limits=None, multiply=False, **kwargs):
        """ Compute synthetic trace in seismic times.

        Uses the following process:
            reflectivity -> reflectivity_resampled -> synthetic_trace.

        Other way to compute the synthetic trace may be:
            impedance -> impedance_resampled -> reflectivity -> synthetic_trace
        but it is less stable computationally.

        TODO: make two methods of computation a parameter

        Parameters
        ----------
        limits : slice or None
            If provided, then used to slice both seismic trace and synthetic trace to a given range.
        multiply : bool
            Whether to adjust the mean abs value of synthetic trace to that of seismic trace.
        """
        #pylint: disable=protected-access
        _ = kwargs
        limits = limits if limits is not None else slice(None)

        # # Alternative way of computations
        # impedance_resampled = self.resample_to_seismic(seismic_times=self.seismic_times[limits],
        #                                                well_times=well_times,
        #                                                well_data=self.well_impedance)
        # reflectivity_resampled = self.well._compute_reflectivity(impedance_resampled)
        # synthetic_trace = self.well._compute_synthetic(reflectivity_resampled, wavelet=wavelet)

        reflectivity_resampled = self.resample_to_seismic(seismic_times=self.seismic_times[limits],
                                                          well_times=well_times,
                                                          well_data=self.well_reflectivity)
        synthetic_trace = self.well._compute_synthetic(reflectivity_resampled, wavelet=wavelet)

        if multiply:
            synthetic_trace *= self.compute_multiplier(self.seismic_trace[limits], synthetic_trace)
        return synthetic_trace

    @staticmethod
    def resample_to_seismic(seismic_times, well_times, well_data):
        """ Resample `well_data`, indexed by `well_times`, to `seismic_times`.
        TODO: better interpolation
        """
        return np.interp(x=seismic_times, xp=well_times, fp=well_data)
        # return interp1d(x=well_times, y=well_data, kind='slinear',
        #                 bounds_error=False, fill_value=(well_data[0], well_data[-1]))(seismic_times)

    @staticmethod
    def compute_multiplier(seismic_trace, synthetic_trace):
        """ Compute multiplicative difference between abs mean values. """
        return np.abs(seismic_trace).mean() / np.abs(synthetic_trace).mean()

    def compute_metric(self, metric='correlation', synthetic_trace=None,
                       well_times=None, wavelet=None, limits=None, **kwargs):
        """ Compute a given metric between real and synthetic seismic traces.

        Parameters
        ----------
        limits : slice or None
            If provided, then used to slice both seismic trace and synthetic trace to a given range.
        """
        _ = kwargs
        limits = limits if limits is not None else slice(None)

        if synthetic_trace is None:
            synthetic_trace = self.compute_resampled_synthetic(well_times=well_times, wavelet=wavelet, limits=limits)

        if metric == 'correlation':
            value = self.correlation(self.seismic_trace[limits], synthetic_trace)
        return value

    @staticmethod
    def correlation(array_0, array_1):
        """ Compute correlation coefficient between two arrays. """
        return ((array_0 - array_0.mean()) * (array_1 - array_1.mean())).mean() / (array_0.std() * array_1.std())


    # t0 optimization
    def compute_t0(self, ranges=(-0.5, +1.5), n=1000, limits=None, state=-1, index=0):
        """ Compute t0.

        Under the hood, uses either analytic formula to estimate t0 by elevations and their velocities or
        just takes a given extrema on cross-correlation functions between synthetic and real seismic traces.

        TODO: better way to signal that analytic computation is needed; maybe, dont always compute cross correlations;

        Parameters
        ----------
        ranges : tuple
            Ranges of tested shifts for cross-correlation computation.
        n : int
            Number of shifts tested.
        index : int
            Index of the extrema to take.
            If equals to -1, then analytic formula is used instead.
        limits : slice or None
            If provided, then used to slice both seismic trace and synthetic trace to a given range.
        state : int, dict
            If int, then the index of previous state to use.
            If dict, then a state directly.
        """
        previous_state = state if isinstance(state, dict) else self.states[state]
        limits = limits if limits is not None else slice(None)
        well_times = previous_state['well_times']
        wavelet = previous_state['wavelet']

        # Compute correlation values. TODO: can be massively speed up by vectorization (the same as in 2d matching)
        shifts = np.linspace(*ranges, n)
        values = [self.compute_metric(well_times=well_times+shift, wavelet=wavelet, limits=limits)
                  for shift in shifts]
        values = np.array(values)

        # Compute peaks and their corresponding metric values
        peak_indices = find_peaks(values, distance=10, prominence=0.1)[0]
        peak_values = values[peak_indices]

        argsort = np.argsort(peak_values)[::-1]
        peak_indices = peak_indices[argsort]
        peak_shifts = shifts[peak_indices]
        peak_values = values[peak_indices]

        # Select the best t0
        if index == -1:
            t0 = 2 * self.well.index[0] * np.diff(self.well_times)[0] / 0.3048
        else:
            t0 = peak_shifts[index]

        new_well_times = self.well_times + t0
        correlation = self.compute_metric(well_times=new_well_times, wavelet=wavelet)

        # Save state
        state = {
            'type': 'compute_t0',
            'well_times': new_well_times,
            'wavelet': wavelet,
            't0': t0, 'correlation': correlation,
            'shifts': shifts, 'values': values,
            'peak_shifts': peak_shifts, 'peak_values': peak_values,
        }
        self.states.append(state)


    def optimize_t0(self, state=-1, limits=None, **kwargs):
        """ Optimize the position of t0.
        Directly minimizes metric in a small neighbourhood of the current (previous state) t0.

        Parameters
        ----------
        limits : slice or None
            If provided, then used to slice both seismic trace and synthetic trace to a given range.
        state : int, dict
            If int, then the index of previous state to use.
            If dict, then a state directly.
        """
        kwargs = {
            'method': 'SLSQP',
            'options': {'maxiter': 100, 'ftol': 1e-3, 'eps': 1e-6},
            **kwargs
        }
        previous_state = state if isinstance(state, dict) else self.states[state]
        limits = limits if limits is not None else slice(None)
        wavelet = previous_state['wavelet']

        # Select t0
        if 't0' in previous_state:
            t0_start = previous_state['t0']
        else:
            t0_start = previous_state['well_times'][0] - self.well_times[0]

        # Actual optimization
        def minimization_proxy(x):
            return -self.compute_metric(well_times=self.well_times+x, wavelet=wavelet, limits=limits)

        optimization_results = minimize(minimization_proxy, x0=t0_start, **kwargs)
        t0 = optimization_results['x']

        # Save state
        state = {
            'type': 'optimize_t0',
            'well_times': self.well_times + t0,
            'wavelet': previous_state['wavelet'],
            't0': t0, 'correlation': -optimization_results['fun'],
            'optimization_results': optimization_results,
        }
        self.states.append(state)


    # Extrema optimization
    @staticmethod
    def stretch_well_times(well_times, position, alpha, left_bound, right_bound, **kwargs):
        """ Stretch the `well_times` around `position` by a factor of `alpha`, while having `bounds` fixed.

        Stretching by a factor of `alpha` is applied to the left segment (segment x).
        By having fixed left/right bounds, we can compute stretch factor `beta` for the right segment (segment y),
        so that the time at the `right_bound` is unchanged.
        |           `left_bound`                              `position`                        `right_bound`
        |----------------|-----------------------------------------o----------------------------------|---------------|
        |                               <segment x>                            <segment y>
        |                           <stretched by alpha>                   <stretched by alpha>

        This way, the entire stretching process depends only on `alpha` and the positions of fixed/moved points.
        Out of them only `alpha` should/can be optimized.
        """
        _ = kwargs

        # Maybe, add taper (~blackman)?
        dt = np.diff(well_times, prepend=0)
        x = dt[left_bound:position]
        y = dt[position:right_bound]

        beta = 1 + (1 - alpha) * x.sum() / y.sum()

        new_dt = dt.copy()
        new_dt[left_bound:position] *= alpha
        new_dt[position:right_bound] *= beta
        return np.cumsum(new_dt)

    def optimize_extrema(self, topk=20, threshold_max=0.050, threshold_min=0.001, threshold_nearest=0.010,
                         threshold_iv_max=500, alpha_bounds=(0.9, 1.1), state=-1, **kwargs):
        """ Optimize `well_times` by stretching it around some point.
        Points to stretch about are selected as extremas of synthetic trace: we take `topk` of them based on metric.
        Each of those extremas is tested against the stretching process: we then select the best one.

        Thresholds regulate how much we allow to stretch each extrema and how close they should be to
        the already-stretched ones.

        After successfull stretching, we store the `position` to keep it fixed in the next iterations of stretching.
        This way, already moved extrema points stay in the the positions.

        TODO: explicitly add `limits`; add different strategies for extrema choice;

        Parameters
        ----------
        topk : int
            Number of extremas to test.
        threshold_max : number
            Max amount to shift the extrema position in seconds. Useful to constrain the optimization process.
        threshold_min : number
            Min amount to shift the extrema position in seconds. Useful to skip tiny shifts.
        threshold_nearest : number
            Min distance to already-shifted extrema in seconds. Useful to disallow stretching the same peak twice.
        threshold_iv_max : number
            Max difference in interval velocities after the shift in m/s.
        alpha_bounds : tuple of two numbers
            Maximum stretch/squeeze allowed.
        limits : slice or None
            If provided, then used to slice both seismic trace and synthetic trace to a given range.
        state : int, dict
            If int, then the index of previous state to use.
            If dict, then a state directly.
        """
        kwargs = {
            'method': 'SLSQP',
            'options': {'maxiter': 100, 'ftol': 1e-3, 'eps': 1e-6},
            **kwargs
        }

        # Retrieve from previous state
        previous_state = state if isinstance(state, dict) else self.states[state]
        well_times, wavelet = previous_state['well_times'], previous_state['wavelet']
        bounds = previous_state.get('bounds', [1, len(well_times) - 1])

        # Compute current state of synthetic; find extremas on it # TODO: maybe, use peaks of envelope?
        synthetic_trace = self.compute_resampled_synthetic(**previous_state)
        dt = np.diff(well_times, prepend=0)

        values = np.abs(synthetic_trace)
        peak_indices = find_peaks(values, distance=5)[0]
        peak_indices = peak_indices[np.argsort(values[peak_indices])[::-1]] # sort peaks by extrema value

        # For each extrema, check the potential correlation gain by stretching left/right side of it
        results = []
        for index in range(min(topk, len(peak_indices))):
            #pylint: disable=cell-var-from-loop
            # Locate extreme in well times
            peak_index = peak_indices[index]
            peak_time = self.seismic_times[peak_index]
            peak_position = np.searchsorted(well_times, peak_time)

            # Select left/right segments. Potentially early-stop
            bounds_idx = np.searchsorted(bounds, peak_position)

            if peak_position <= bounds[0] or peak_position >= bounds[-1]:
                continue
            left_bound, right_bound = bounds[bounds_idx-1], bounds[bounds_idx]

            if min(abs(well_times[peak_position] - well_times[left_bound]),
                   abs(well_times[peak_position] - well_times[right_bound])) <= threshold_nearest:
                continue

            x = dt[left_bound:peak_position]
            y = dt[peak_position:right_bound]
            xsum , ysum = x.sum(), y.sum()

            # Optimize via adjusting left-stretches `alpha`
            def minimization_proxy(alpha):
                beta = 1 + (1 - alpha) * xsum / ysum

                new_dt = dt.copy()
                new_dt[left_bound:peak_position] *= alpha
                new_dt[peak_position:right_bound] *= beta

                new_well_times = np.cumsum(new_dt)
                # TODO: can add limits=(left bounds, right bounds)
                return -self.compute_metric(well_times=new_well_times, wavelet=wavelet)

            # Prepare bounds and early stop, if too restrictive
            s = threshold_max / xsum
            tmax = min(x.min() * threshold_iv_max / 0.3048, 0.5)
            optimization_bounds = (max(1 - s, alpha_bounds[0]),
                                   min(1 + s, 1 / (1 - tmax), alpha_bounds[1]))
            if optimization_bounds[0] > optimization_bounds[1]:
                continue

            # Actual optimization
            optimization_results = minimize(minimization_proxy, x0=1., bounds=[optimization_bounds], **kwargs)

            # Check if the stretch on either side is too small / too big
            alpha = optimization_results['x'].item()
            beta = 1 + (1 - alpha) * xsum / ysum
            if (1 - threshold_min / xsum) <= alpha <= (1 + threshold_min / ysum) or \
               (1 - threshold_min / xsum) <= beta  <= (1 + threshold_min / ysum) or \
                beta <= alpha_bounds[0] or beta >= alpha_bounds[1]:
                continue

            iv_diffs_x = 0.3048 * (alpha - 1) / (alpha * x)
            iv_diffs_y = 0.3048 * ( beta - 1) / ( beta * y)
            if np.abs(iv_diffs_x).max() > threshold_iv_max or np.abs(iv_diffs_y).max() > threshold_iv_max:
                continue

            results.append({
                'position': peak_position,
                'alpha': alpha, 'beta': beta,
                'left_bound': left_bound, 'right_bound': right_bound,
                'correlation': -optimization_results['fun'],
                'xsum': xsum, 'ysum': ysum,
                'optimization_bounds': optimization_bounds,
                'optimization_results': optimization_results,
            })

        if len(results) == 0:
            return False

        # Select the best extrema to stretch about
        metrics = [item['correlation'] for item in results]
        index = np.argmax(metrics)
        state = results[index]
        position = state['position']
        new_well_times = self.stretch_well_times(well_times, **state)
        time_shift = well_times[position] - new_well_times[position]

        state.update({
            'type': 'optimize_extrema',
            'well_times': new_well_times,
            'wavelet': wavelet,
            'time_shift': time_shift,
            'time_before': well_times[position], 'time_after': new_well_times[position],
            'correlation_delta': state['correlation'] - previous_state['correlation'],
            'bounds': sorted(bounds + [position]),
        })
        self.states.append(state)
        return True

    def optimize_extremas(self, steps=20, threshold_delta=0.01, verbose=True,
                          topk=20, threshold_max=0.050, threshold_min=0.001, threshold_nearest=0.010,
                          threshold_iv_max=500, alpha_bounds=(0.9, 1.1), **kwargs):
        """ Optimize `well_times` by stretching multiple times about extrema positions.
        Simply runs :meth:`optimize_extrema` in a loop with an early-stopping condition.

        Parameters
        ----------
        steps : int
            Number of steps of individual stretching to take.
        threshold_delta : number
            Early stop if the metric improves by less than this amount.
        verbose : bool
            Whether to print metric values and other info on each step.
        other parameters : dict
            Directly passed to :meth:`optimize_extrema`.
        """
        for i in range(steps):
            success = self.optimize_extrema(topk=topk, threshold_max=threshold_max, threshold_min=threshold_min,
                                            threshold_nearest=threshold_nearest,
                                            alpha_bounds=alpha_bounds, threshold_iv_max=threshold_iv_max, **kwargs)
            if not success:
                if verbose:
                    print('Early break: no good adjustment found!')
                break

            state = self.states[-1]
            correlation_delta = state['correlation_delta']

            if verbose:
                correlation = self.compute_metric(**state)
                time_shift = state['time_shift'] * 1000
                alpha, beta = state['alpha'], state['beta']
                print(f'{i:3} :: {correlation=:3.5f} :: {correlation_delta=:3.5f} :: {time_shift=:>+7.4} ms'
                      f'      ||      {alpha=:3.3f} :: {beta=:3.3f}')

            if correlation_delta < threshold_delta:
                if verbose:
                    print('Early break: correlation is no longer increasing!')
                break


    # Pytorch well times optimization
    @staticmethod
    def compute_resampled_synthetic_pytorch(well_times, well_reflectivity, seismic_times, wavelet):
        """ Compute synthetic trace in seismic times.
        Same as :meth:`compute_resampled_synthetic`, but with `PyTorch` operations instead.

        TODO: replace the interpolation function: the current one is flawed / repo is broken;
        """
        #pylint: disable=import-outside-toplevel
        import torch
        from xitorch.interpolate import Interp1D
        reflectivity_resampled = Interp1D(well_times, well_reflectivity,
                                          method='linear', assume_sorted=True, extrap=0.0)(seismic_times)

        synthetic_trace = torch.nn.functional.conv1d(input=reflectivity_resampled.reshape(1, 1, -1),
                                                    weight=wavelet.reshape(1, 1, -1),
                                                    padding='same')
        return synthetic_trace.reshape(-1)


    def optimize_well_times_pytorch(self, n_segments=100, n_iters=1000,
                                    optimizer_params=None, regularization_params=None, bounds=None,
                                    limits=None, pbar='t', state=-1):
        """ Optimize well times by adjusting time values directly.
        Originally, we allow for each element of `well_times` to be multiplied by a value.
        Values are computed by optimizing the vector of multipliers with a usual PyTorch training loop.

        To constrain the amount of multipliers, we split the `well_times` into `n_segments`: each segment uses the same
        multiplier. This way, the number of perturbations is much smaller and, essentially, regularized.

        Parameters
        ----------
        n_segments : int
            Number of segments to split `well_times` into.
        n_iters : int
            Number of optimization iterations.
        optimizer_params : dict
            Parameters for optimizer initialization.
        regularization_params : dict
            Regularization parameters: used keys are 'l1', 'l2', 'dl1', 'dl2'.
        pbar : bool, str
            If bool, then whether to display progress bar.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        limits : slice or None
            If provided, then used to slice both seismic trace and synthetic trace to a given range.
        state : int, dict
            If int, then the index of previous state to use.
            If dict, then a state directly.
        """
        #pylint: disable=import-outside-toplevel
        import torch
        from batchflow import Notifier

        limits = limits if limits is not None else slice(None)

        # Retrieve from previous state
        previous_state = state if isinstance(state, dict) else self.states[state]
        well_times, wavelet = previous_state['well_times'], previous_state['wavelet']

        # Prepare variables for optimization: one multiplier for each segment
        # TODO: figure out a better way to multiplicate values instead of `torch.repeat_interleave`
        if n_segments == len(well_times) or n_segments == 'well':
            multipliers = torch.ones(len(well_times), dtype=torch.float32, requires_grad=True)
            segment_size = 1
        else:
            if isinstance(n_segments, str) and n_segments.startswith('top'):
                # TODO: very dirty, refactor
                n_segments = int(n_segments[3:])

                synthetic_trace = self.compute_resampled_synthetic(**previous_state)
                dt = np.diff(well_times, prepend=0)

                values = np.abs(synthetic_trace)
                peak_indices = find_peaks(values, distance=5)[0]
                peak_indices = peak_indices[np.argsort(values[peak_indices])[::-1]]
                peak_positions = []
                for index in range(n_segments):
                    peak_index = peak_indices[index]
                    peak_time = self.seismic_times[peak_index]
                    peak_position = np.searchsorted(well_times, peak_time)
                    peak_positions.append(peak_position)
                peak_positions = np.sort(np.array(peak_positions))

                segment_size = np.diff(peak_positions, prepend=0)
                segment_size[-1] += len(well_times) - segment_size.sum()
                segment_size = torch.from_numpy(segment_size)
            else:
                segment_size = len(well_times) // n_segments + 1
            multipliers = torch.ones(n_segments, dtype=torch.float32, requires_grad=True)


        # Convert data to PyTorch. Clone everything, as CPU tensors share data with numpy arrays
        seismic_times = torch.from_numpy(self.seismic_times).float().clone()
        seismic_trace = torch.from_numpy(self.seismic_trace).float().clone()
        well_reflectivity = torch.from_numpy(np.nan_to_num(self.well_reflectivity)).float().clone()

        well_times = torch.from_numpy(well_times).float().clone()
        wavelet = torch.from_numpy(wavelet).float().clone()
        dt = torch.from_numpy(np.diff(well_times, prepend=0)).float().clone()

        # Prepare infrastructure for train
        optimizer_params = {
            'lr': 0.0002,
            **(optimizer_params or {})
        }
        optimizer = torch.optim.AdamW((multipliers,), **optimizer_params)

        regularization_params = {
            'l1': 0.0, 'l2': 0.0,
            'dl1': 0.0, 'dl2': 0.0,
            **(regularization_params or {})
        }

        # Run train loop
        loss_history = []
        notifier = Notifier(pbar, frequency=min(50, n_iters),
                            monitors=[{'source': loss_history, 'format': 'correlation={:5.4f}'}])
        for _ in notifier(n_iters):
            multipliers_ = torch.repeat_interleave(multipliers, segment_size)[:len(well_times)]
            multipliers_[0] = 1.0
            new_well_times = torch.cumsum(dt * multipliers_, dim=0)
            synthetic_trace = self.compute_resampled_synthetic_pytorch(well_times=new_well_times,
                                                                       well_reflectivity=well_reflectivity,
                                                                       seismic_times=seismic_times,
                                                                       wavelet=wavelet)
            loss = -self.correlation(seismic_trace, synthetic_trace)

            # Regularization
            dmultipliers = torch.diff(multipliers)
            regularization = (
                regularization_params['l1'] * torch.abs(multipliers - 1).mean() +
                regularization_params['l2'] * torch.abs((multipliers - 1) ** 2).mean() +
                regularization_params['dl1'] * torch.abs(dmultipliers).mean() +
                regularization_params['dl2'] * torch.abs(dmultipliers ** 2).mean()
            )

            # Update
            (loss + regularization).backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_history.append(-loss.detach().numpy().item())

        # Save state
        multipliers_ = torch.repeat_interleave(multipliers, segment_size)[:len(well_times)]
        multipliers_[0] = 1.0
        new_well_times = torch.cumsum(dt * multipliers_, dim=0).detach().numpy()
        correlation = self.compute_metric(well_times=new_well_times, wavelet=previous_state['wavelet'])
        state = {
            'type': 'optimize_well_times_pytorch',
            'well_times': new_well_times,
            'wavelet': previous_state['wavelet'],
            'correlation': correlation,
            'loss_history': loss_history,
            'multipliers': multipliers_.detach().numpy(),
        }
        self.states.append(state)


    # Wavelet optimization
    def optimize_wavelet(self, limits=None, state=-1, **kwargs):
        """ Optimize wavelet's phase by direct minimization.

        Parameters
        ----------
        limits : slice or None
            If provided, then used to slice both seismic trace and synthetic trace to a given range.
        state : int, dict
            If int, then the index of previous state to use.
            If dict, then a state directly.
        """
        kwargs = {
            'method': 'SLSQP',
            'options': {'maxiter': 100, 'ftol': 1e-4, 'eps': 1e-7},
            **kwargs
        }

        # Retrieve from previous state
        previous_state = state if isinstance(state, dict) else self.states[state]
        well_times, wavelet = previous_state['well_times'], previous_state['wavelet']

        spectrum = np.fft.rfft(wavelet)
        # power_spectrum = np.abs(spectrum)
        # phase_spectrum = np.angle(spectrum)

        def minimization_proxy(phase_shift):
            new_wavelet = np.fft.irfft(spectrum * np.exp(1.0j * phase_shift), n=len(wavelet))
            return -self.compute_metric(well_times=well_times, wavelet=new_wavelet, limits=limits)

        optimization_results = minimize(minimization_proxy, x0=0., bounds=[[-np.pi/2, +np.pi/2]], **kwargs)

        phase_shift = optimization_results['x'].item()
        new_wavelet = np.fft.irfft(spectrum * np.exp(1.0j * phase_shift), n=len(wavelet))

        state = {
            'type': 'optimize_wavelet',
            'well_times': well_times,
            'wavelet': new_wavelet,
            'correlation': -optimization_results['fun'],
            'phase_shift': phase_shift,
            'phase_shift_angles': np.rad2deg(phase_shift),
        }
        self.states.append(state)

    def optimize_wavelet_(self, n=10, delta=1., limits=None, state=-1, **kwargs):
        """ Optimize wavelet's phases by directly changing them to maximize metric.

        TODO: explanation, references?

        Parameters
        ----------
        n : int
            Number of (first) frequency phases to optimize.
        delta : number
            Additive bound to keep the resulting phase values in.
        limits : slice or None
            If provided, then used to slice both seismic trace and synthetic trace to a given range.
        state : int, dict
            If int, then the index of previous state to use.
            If dict, then a state directly.
        """
        kwargs = {
            'method': 'SLSQP',
            'options': {'maxiter': 100, 'ftol': 1e-4, 'eps': 1e-7},
            **kwargs
        }

        # Retrieve from previous state
        previous_state = state if isinstance(state, dict) else self.states[state]
        well_times = previous_state['well_times']
        wavelet = previous_state['wavelet']

        spectrum = np.fft.rfft(wavelet)
        power_spectrum = np.abs(spectrum)
        phase_spectrum = np.angle(spectrum)

        # Optimization objective
        def minimization_proxy(phase_shifts):
            new_phase_spectrum = phase_spectrum.copy()
            new_phase_spectrum[:len(phase_shifts)] = phase_shifts
            new_wavelet = np.fft.irfft(power_spectrum * np.exp(1.0j * new_phase_spectrum), n=len(wavelet))
            return -self.compute_metric(well_times=well_times, wavelet=new_wavelet, limits=limits)

        x0 = phase_spectrum[:n]
        optimization_bounds = np.array([x0-delta, x0+delta]).T
        optimization_bounds = np.clip(optimization_bounds, -np.pi, +np.pi)
        optimization_results = minimize(minimization_proxy, x0=x0, bounds=optimization_bounds, **kwargs)

        # Retrieve solution
        phase_shifts = optimization_results['x']
        new_phase_spectrum = phase_spectrum.copy()
        new_phase_spectrum[:len(phase_shifts)] = phase_shifts
        new_wavelet = np.fft.irfft(power_spectrum * np.exp(1.0j * new_phase_spectrum), n=len(wavelet))

        state = {
            'type': 'optimize_wavelet',
            'well_times': well_times,
            'wavelet': new_wavelet,
            'correlation': -optimization_results['fun'],
            'phase_shifts': phase_shifts,
        }
        self.states.append(state)


    # Metrics
    def evaluate_markers(self, markers, state=-1):
        """ Compare predicted `well_times` on specific horizons to the marked one.
        TODO: refactor to work with more formats;
        """
        state = state if isinstance(state, dict) else self.states[state]

        if isinstance(markers, str):
            markers = pd.read_csv(markers, sep='\t')
        markers = markers[markers['Well'] == self.well.name]

        results_df = []
        for idx, row in markers.iterrows():
            marker_name = row['Top']
            if not marker_name.isupper():
                continue

            marker_depth = row['TVDSS [m]']
            marker_time = row['Time [s] X']
            idx = np.searchsorted(self.well.index, marker_depth)
            if self.well.index[idx] - marker_depth > marker_depth - self.well.index[idx-1]:
                idx -= 1
            predicted_time = state['well_times'][idx]

            results_df.append({
                'Top': marker_name,
                'TVDSS [m]': marker_depth,
                'Time [s]': marker_time,
                'Predicted Time [s]': round(predicted_time, 6),
                'Diff Time [s]': round(predicted_time - marker_time, 6),
            })

        return pd.DataFrame(results_df)


    # Export
    def save_well_times(self, path, state=-1):
        """ Save `well_times` as the depth-time table. """
        state = self.states[state]
        path = self.field.make_path(path, name=self.well.name)

        well_times = state['well_times']
        depths = self.well.index.values[self.well_bounds]

        data = np.array([depths, well_times]).T
        df = pd.DataFrame(data=data, columns=['MD, m', 'TWT, s'])

        df.to_csv(path, header=True, index=False, sep=' ')

    def save_wavelet(self, path, state=-1):
        """ Save wavelet as time-value table. """
        state = self.states[state]
        path = self.field.make_path(path, name=self.well.name)

        wavelet = state['wavelet']
        times = np.arange(len(wavelet)) * self.field.sample_interval * 1e-3

        data = np.array([times, wavelet]).T
        df = pd.DataFrame(data=data, columns=['TWT, s', 'VALUE'])

        df.to_csv(path, header=True, index=False, sep=' ')

    def save_las(self, path, state=-1):
        """ Save well information with all used (possibly, recomputed and filtered) logs. """
        state = self.states[-1]
        path = self.field.make_path(path, name=self.well.name)

        well_times = state['well_times']
        dt_optimized = np.diff(well_times, prepend=0) * 1e6
        well_impedance = self.well_impedance

        if well_times.size != self.well.shape[0]:
            pad_width = (self.well_bounds.start, self.well.shape[0]-self.well_bounds.stop)
            dt_optimized = np.pad(dt_optimized, pad_width=pad_width, constant_values=np.nan)
            well_impedance = np.pad(well_impedance, pad_width=pad_width, constant_values=np.nan)

        lasfile = deepcopy(self.well.lasfile)
        lasfile.append_curve('DT_OPTIMIZED', dt_optimized, unit='us/ft', descr='DT_OPTIMIZED')
        lasfile.append_curve('AI_USED', well_impedance, unit='kPa.s/m', descr='AI_USED')

        if 'RHOB_FILTERED' in self.well.keys:
            rhob_values = self.well['RHOB_FILTERED']
            if well_times.size != self.well.shape[0]:
                rhob_values = np.pad(rhob_values, pad_width=pad_width, constant_values=np.nan)
            lasfile.append_curve('RHOB_FILTERED', rhob_values, unit='kg/m^3', descr='RHOB_FILTERED')


        lasfile.write(path, version=2.0)

    def save_synthetic(self, path, state=-1):
        """ Save synthetic trace in SEG-Y format. """
        state = self.states[state]
        path = self.field.make_path(path, name=self.well.name)

        synthetic_trace = self.compute_resampled_synthetic(**state)
        synthetic_trace = synthetic_trace.reshape(1, 1, -1)

        array_to_segy(synthetic_trace, path=path, origin=(*self.coordinates, 0), pbar=False)


    # Visualization
    def show_state(self, state=-1, limits=slice(None), force_dt=False, **kwargs):
        """ Show state.
        Visualizes real-to-synthetic comparison, the wavelet,
        original and state interval velocity, ratio between original and state velocties.

        If `well_times` are unchanged, the last two graphs are not displayed.

        Parameters
        ----------
        force_dt : bool
            Whether to show the last two graphs even if `well_times` are unchanged in the state.
        limits : slice or None
            If provided, then used to slice both seismic trace and synthetic trace to a given range.
        state : int, dict
            If int, then the index of previous state to use.
            If dict, then a state directly.
        kwargs : dict
            Other parameters are directly passed to the plotting function.
        """
        state = len(self.states) - 1 if state == -1 else state
        state_name = '<user_dict>' if isinstance(state, dict) else state
        state = state if isinstance(state, dict) else self.states[state]

        wavelet = state['wavelet']
        synthetic_trace = self.compute_resampled_synthetic(**state, multiply=True)
        correlation = self.compute_metric(synthetic_trace=synthetic_trace)

        dt = np.diff(self.well_times)
        dt_state = np.diff(state['well_times'])

        # Seismic to synthetic comparison; wavelet
        well_times = state['well_times'][1:]
        seismic_times = self.seismic_times[limits]
        wavelet_times = self.seismic_times[:len(wavelet)]
        wavelet_times -= wavelet_times[len(wavelet)//2 + 0]

        data = [[(seismic_times, self.seismic_trace[limits]), (seismic_times, synthetic_trace[limits])],
                [(wavelet_times, wavelet)]]

        # Interval velocities: show only if changed
        if not np.allclose(dt, dt_state) or force_dt:
            iv = 0.3048 / dt
            iv_state = 0.3048 / dt_state
            iv_diff = np.abs(iv - iv_state)
            data.append([(well_times, iv), (well_times, iv_state), (well_times, iv_diff)])

            relative_iv = np.round(dt / dt_state, 6)
            data.append([(well_times, relative_iv)])

        kwargs = {
            'combine': 'overlay',
            'ncols': 2,
            'ratio': 0.3 if len(data) == 2 else 0.5,
            'suptitle': f'Well `{self.well.name}`\nstate={state_name}; {correlation=:3.3f}',
            'title': ['seismic vs synthetic', 'wavelet',
                      'interval velocity', 'relative increase in velocity: dt/dt_state'],
            'xlabel': ['seismic time, s', 'time, s',
                       'well time, s', 'well time, s'],
            'ylabel': ['amplitude', 'amplitude',
                       'velocity, m/s', 'ratio'],
            'label': [['seismic_trace', 'synthetic_trace'], '',
                      ['original IV', 'state IV', 'diff IV'], ''],
            'xlabel_size': 18,
            **kwargs
        }
        plotter = plot(data, mode='curve', **kwargs)
        if len(data) == 4:
            plotter.subplots[-1].ax.axhline(1, linestyle='dashed', alpha=.5, color='sandybrown', linewidth=3)

        return plotter


    def show_wavelet(self, state=-1, **kwargs):
        """ Display wavelet and its power/phase spectra. """
        state = len(self.states) - 1 if state == -1 else state
        state_name = '<user_dict>' if isinstance(state, dict) else state
        state = state if isinstance(state, dict) else self.states[state]
        wavelet = state['wavelet']
        correlation = state['correlation']
        times = np.arange(len(wavelet)) * self.field.sample_interval
        times -= times[-1] / 2

        spectrum = np.fft.rfft(wavelet)
        power_spectrum = 20 * np.log10(np.abs(spectrum))
        phase_spectrum = np.angle(spectrum)
        frequencies = np.fft.rfftfreq(len(wavelet), d=self.field.sample_interval * 1e-3)

        kwargs = {
            'combine': 'separate',
            'ncols': 3,
            'ratio': 0.25,
            'suptitle': f'well `{self.well.name}`\nstate={state_name}; {correlation=:3.3f}',
            'title': ['wavelet', 'power spectrum', 'phase spectrum'],
            'xlabel': ['time, ms', 'Hz', 'Hz'],
            **kwargs
        }
        return plot([(times, wavelet), (frequencies, power_spectrum), (frequencies, phase_spectrum)],
                    mode='curve', **kwargs)


    def show_progress(self, start_idx=1, **kwargs):
        """ Display correlation over the states. """
        data = [[state.get('correlation', self.compute_metric(**state)) for state in self.states[start_idx:]]]

        kwargs = {
            'combine': 'separate',
            'title': ['correlation over states', 'time shifts of states'],
            'xlabel': 'state index',
            'ylabel': ['correlation', 'time shift (ms)'],
            **kwargs
        }

        # Extrema optimization states
        states = [state for state in self.states if state['type'] == 'optimize_extrema']
        if states:
            time_shifts = [state['time_shift'] * 1000 for state in states]
            data.append(time_shifts)

        plotter = plot(data, mode='curve', ncols=2 if states else 1, **kwargs)

        if states:
            plotter[1].ax.lines.pop(0)
            colors = np.where(np.array(time_shifts) > 0, 'r', 'b')
            plotter[1].ax.bar(range(len(states)), time_shifts, color=colors)
        return plotter


    def show_crosscorrelation(self, state=1, n_peaks=3, **kwargs):
        """ Display cross-correlation function between real and synthetic trace.
        Requires for the state to be created by :meth:`compute_t0`.
        """
        state = state if isinstance(state, dict) else self.states[state]
        if state['type'] not in {'compute_t0'}:
            raise TypeError('State type should be `compute_t0`.')

        kwargs = {
            'title': 'correlation VS shift of well data',
            'xlabel': 'shift, seconds', 'ylabel': 'correlation',
            'size': 18, 'title_size': 22,
            **kwargs
        }

        plotter = plot((state['shifts'], state['values']), mode='curve', **kwargs)
        plotter[0].ax.scatter(state['peak_shifts'], state['peak_values'], s=15, c='r', marker='8')

        for idx in range(n_peaks):
            shift = state['peak_shifts'][idx]
            correlation = state['peak_values'][idx]
            plotter[0].ax.axvline(shift, correlation, linestyle='dashed', color='orange', alpha=0.9,
                                  label=f'{shift=:+2.3f} {correlation=:2.3f}')
        plotter[0].ax.legend(prop={'size': 14})
        return plotter

    def show_time_shifts(self, zoom=None, **kwargs):
        """ Display applied stretches. """
        zoom = zoom if zoom is not None else (0, self.seismic_times[-1])
        kwargs = {
            'title': 'Extrema shift visualization',
            'xlim': zoom,
            'ylabel': '', 'ylim': (0, 1), 'ytick_labels': '',
            **kwargs
        }

        plotter = plot((self.seismic_times, self.seismic_times * np.nan), mode='curve', **kwargs)

        states = [state for state in self.states if state['type'] == 'optimize_extrema']
        for i, state in enumerate(states):
            time_before, time_after = state['time_before'], state['time_after']
            plotter[0].ax.axvline(time_before, linestyle='--', alpha=0.8, color='blue')
            plotter[0].ax.axvline(time_after, linestyle='solid', alpha=1, color='green')

            text = f'{state["time_shift"] * 1000:+2.3f} ms'
            plotter[0].ax.annotate(text, xy=(max(time_before, time_after), (i + 1) / len(states)), size=14)
        return plotter
