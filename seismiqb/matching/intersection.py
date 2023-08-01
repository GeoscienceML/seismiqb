""" Intersections between fields. """
from textwrap import dedent

import numpy as np
import scipy
import matplotlib.pyplot as plt

from .functional import compute_correlation, compute_r2, modify_trace, minimize_proxy, compute_shifted_traces
from ..plotters import plot


def prepare_field(field):
    """ Cache CDP values in a field as np.ndarray. """
    if getattr(field, 'cdp_values', None) is None:
        field.cdp_values = field.geometry.headers[['CDP_X', 'CDP_Y']].values
    return field


class Intersection:
    """ Base class to describe an intersection between two fields.

    The usual workflow is to create an instance by :meth:`.new`, which finds one or multiple intersections between
    two fields, then match extracted traces (that are padded / aggregated across multiple close indices) with either
    analytic or optimization algorithms. Finally, use visualization tools to display interesting properties.
    """
    @classmethod
    def new(cls, field_0, field_1, limits=slice(None), pad_width=0, threshold=10, use_std=False, unwrap=True):
        """ Create one or more instances of intersection classes with automatic class selection.
        Preferred over directly instantiating objects.
        """
        is_2d_0 = 1 in field_0.spatial_shape
        is_2d_1 = 1 in field_1.spatial_shape
        if is_2d_0 and is_2d_1:
            return Intersection2d2d.new(field_0, field_1, limits=limits, pad_width=pad_width,
                                        threshold=threshold, use_std=use_std, unwrap=unwrap)

        return Intersection2d3d(field_0, field_1)


class Intersection2d2d:
    """ Intersection between two 2D fields. """
    @classmethod
    def new(cls, field_0, field_1, limits=slice(None), pad_width=20, threshold=10, use_std=False, unwrap=True):
        """ Create one or more instances of intersection.
        Preferred over directly instantiating objects.
        """
        prepare_field(field_0)
        prepare_field(field_1)

        values_0 = field_0.cdp_values
        values_1 = field_1.cdp_values

        bbox_0 = np.sort(values_0[[0, -1]].T, axis=-1)
        bbox_1 = np.sort(values_1[[0, -1]].T, axis=-1)

        overlap = np.maximum(bbox_0[:, 0], bbox_1[:, 0]), np.minimum(bbox_0[:, 1], bbox_1[:, 1])
        if (overlap[1] - overlap[0]).min() < 0:
            return False if unwrap else []

        # pylint: disable=import-outside-toplevel
        from shapely import LineString, MultiLineString, MultiPoint, GeometryCollection
        # TODO: improve and describe edge cases
        line_0 = LineString(values_0)
        line_1 = LineString(values_1)

        intersection = line_0.intersection(line_1)
        if isinstance(intersection, (MultiLineString, MultiPoint, GeometryCollection)):
            # intersection = intersection.geoms[0]
            points = [list(zip(*geometry.xy)) for geometry in intersection.geoms]
            points = sum(points, [])
        else:
            points = list(zip(*intersection.xy))

        result = []
        for point in points:
            trace_idx_0 = ((values_0 - point) ** 2).sum(axis=-1).argmin()
            trace_idx_1 = ((values_1 - point) ** 2).sum(axis=-1).argmin()

            trace_idx_0 = cls.adjust_index(field_0, trace_idx_0, use_std=use_std)
            trace_idx_1 = cls.adjust_index(field_1, trace_idx_1, use_std=use_std)
            if trace_idx_0 is False or trace_idx_1 is False:
                continue

            instance = cls(field_0=field_0, field_1=field_1,
                           trace_idx_0=trace_idx_0, trace_idx_1=trace_idx_1,
                           limits=limits, pad_width=pad_width)

            if not any(other.is_similar(instance, threshold=threshold) for other in result):
                result.append(instance)

        if not result:
            return False if unwrap else []
        return result[0] if len(result) == 1 and unwrap else result

    @staticmethod
    def adjust_index(field, trace_idx, n=3, use_std=False):
        """ Move the trace index to one of the neighboring in case of dead trace at the `trace_idx`. """
        nhalf = (n - 1) // 2
        indices = list(range(max(0, trace_idx - nhalf), min(trace_idx + nhalf + 1, field.n_traces)))

        if use_std:
            trace_data = field.geometry.load_by_indices(indices)
            trace_std = trace_data.std(axis=1)
            argmax_std = np.argmax(trace_std)

            if trace_std[argmax_std] == 0:
                return False
            return indices[argmax_std]

        dead_traces = field.geometry.dead_traces_matrix.reshape(-1)[indices]
        argmin = np.argmin(dead_traces)
        if dead_traces[argmin] is True:
            return False
        return indices[argmin]


    def __init__(self, field_0, field_1, trace_idx_0, trace_idx_1,
                 limits=slice(None), pad_width=0, n=1, transform=None):
        prepare_field(field_0)
        prepare_field(field_1)

        self.field_0, self.field_1 = field_0, field_1
        self.trace_idx_0, self.trace_idx_1 = trace_idx_0, trace_idx_1
        self.limits = limits
        self.pad_width = pad_width
        self.n = n
        self.transform = transform
        self.max_depth = int(max(field_0.depth * field_0.sample_interval + field_0.delay,
                                 field_1.depth * field_1.sample_interval + field_1.delay) + 1)

        # Compute distance
        self.coordinates_0 = field_0.cdp_values[trace_idx_0]
        self.coordinates_1 = field_1.cdp_values[trace_idx_1]
        self.distance = ((self.coordinates_0 - self.coordinates_1).astype(np.float64) ** 2).sum() ** (1 / 2)

        self.matching_results = None

    def is_similar(self, other, threshold=10):
        """ Check if other intersection is close index-wise. """
        if abs(self.trace_idx_0 - other.trace_idx_0) <= threshold and \
            abs(self.trace_idx_1 - other.trace_idx_1) <= threshold:
            return True
        return False


    def to_dict(self, precision=3):
        """ Represent intersection parameters (including computed matching values) as a dictionary. """
        intersection_dict = {
            'field_0_name': self.field_0.name,
            'field_1_name': self.field_1.name,
            'distance': self.distance,
        }

        if self.matching_results is not None:
            intersection_dict.update(self.matching_results)

        for key, value in intersection_dict.items():
            if isinstance(value, (float, np.floating)):
                intersection_dict[key] = round(value, precision)
        return intersection_dict


    # Data
    def prepare_traces(self, limits=None, index_shifts=(0, 0), pad_width=None, n=1, transform=None):
        """ Prepare traces from both intersecting fields.
        Under the hood, we load traces, pad to max depth, slice with `limits` and add additional `pad_width`.
        Also, we average over `n` traces at loading to reduce noise.
        """
        limits = limits if limits is not None else self.limits
        pad_width = pad_width if pad_width is not None else self.pad_width
        n = n if n is not None else self.n
        transform = transform if transform is not None else self.transform

        trace_0 = self._prepare_trace(self.field_0, index=self.trace_idx_0 + index_shifts[0],
                                      limits=limits, pad_width=pad_width, n=n, transform=transform)
        trace_1 = self._prepare_trace(self.field_1, index=self.trace_idx_1 + index_shifts[1],
                                      limits=limits, pad_width=pad_width, n=n, transform=transform)
        return trace_0, trace_1

    def _prepare_trace(self, field, index, limits=None, pad_width=None, n=1, transform=None):
        # TODO: add taper
        # Load data
        nhalf = (n - 1) // 2
        indices = list(range(index - nhalf, index + nhalf + 1))
        traces = field.geometry.load_by_indices(indices)
        trace = np.mean(traces, axis=0)

        # Resample to ms
        arange = np.arange(trace.size, dtype=np.float32)
        arange_ms = np.arange(trace.size, step=(1 / field.sample_interval), dtype=np.float32)
        trace = np.interp(arange_ms, arange, trace, left=0, right=0)

        # Adjust for field delay: move the start of the trace to 0ms level
        trace = np.pad(trace, (field.delay, 0)) if field.delay >= 0 else trace[-field.delay:]

        # Pad/slice
        if trace.size < self.max_depth:
            trace = np.pad(trace, (0, self.max_depth - trace.size))
        trace = trace[limits]
        if pad_width > 0:
            trace = np.pad(trace, pad_width)

        if transform is not None:
            trace = transform(trace)
        return trace


    def prepare_horizons(self):
        """ Prepare mappings from horizon name to its depth on intersection for both fields. """
        return (self._prepare_horizons(field=self.field_0, coordinates=self.coordinates_0),
                self._prepare_horizons(field=self.field_1, coordinates=self.coordinates_1))

    @staticmethod
    def _prepare_horizons(field, coordinates):
        horizon_to_depth = {}
        for horizon_name, horizon_ixd in field.horizons.items():
            distances = ((horizon_ixd[:, :2] - coordinates) ** 2).sum(axis=1) ** (1 / 2)
            idx = np.argmin(distances)
            horizon_to_depth[horizon_name] = horizon_ixd[idx, -1]
        return horizon_to_depth


    # Matching algorithms
    def match_traces(self, method='analytic', **kwargs):
        """ Selector for matching method.
        Refer to the documentation of :meth:`match_traces_analytic` and :meth:`match_traces_optimize` for details.

        TODO: add `mixed` mode, where we select the initial point by `analytic` method and then use optimization
        procedure to find the exact location.
        """
        if method in {'analytic'}:
            matching_results = self.match_traces_analytic(**kwargs)
        elif method in {'optimize'}:
            matching_results = self.match_traces_optimize(**kwargs)
        else:
            matching_results = self.match_on_horizon(**kwargs)

        matching_results['petrel_corr'] = (matching_results['corr'] + 1) / 2
        if getattr(self, 'key'):
            matching_results['key'] = self.key
        self.matching_results = matching_results
        return matching_results


    def match_traces_optimize(self, limits=None, index_shifts=(0, 0), pad_width=None, n=1, transform=None,
                              init_shifts=range(-100, +100), init_angles=(0,), metric='r2',
                              bounds_shift=(-150, +150), bounds_angle=None, bounds_gain=(0.9, 1.1),
                              maxiter=100, eps=1e-6, **kwargs):
        """ Match traces by iterative optimization of the selected loss function.
        Slower, than :meth:`match_traces_analytic`, but allows for finer control.

        We use every combination of parameters in `init_shifts` and `init_angles` as
        the starting point for optimization. This way, we try to avoid local minima, improving the result by a lot.
        The optimization is bounded: `bounds_*` parameters allow to control the spread of possible values.
        """
        _ = kwargs

        # Load data
        trace_0, trace_1 = self.prepare_traces(limits=limits, index_shifts=index_shifts,
                                               pad_width=pad_width, n=n, transform=transform)

        # For each element in init, perform optimize
        minimize_results = []
        for init_shift in init_shifts:
            for init_angle in init_angles:
                bounds_angle_ = bounds_angle or (init_angle-eps, init_angle+eps)
                minimize_result = scipy.optimize.minimize(fun=minimize_proxy,
                                                          x0=np.array([init_shift, init_angle, 1.0]),
                                                          args=(trace_0, trace_1, metric),
                                                          bounds=(bounds_shift, bounds_angle_, bounds_gain),
                                                          method='SLSQP',
                                                          options={'maxiter': maxiter,
                                                                   'ftol': 1e-6, 'eps': 1e-3})
                minimize_results.append(minimize_result)
        minimize_results = np.array([(item.fun, *item.x) for item in minimize_results])

        # Find the best result
        argmin = np.argmin(minimize_results[:, 0])
        best_loss, best_shift, best_angle, best_gain = minimize_results[argmin]
        if metric == 'correlation':
            best_gain = self.compute_gain(data_0=trace_0, data_1=trace_1)

        best_corr = compute_correlation(trace_0,
                                        modify_trace(trace_1, shift=best_shift, angle=best_angle, gain=best_gain))

        return {
            'corr': best_corr,
            'shift': best_shift,
            'angle': best_angle,
            'gain': best_gain,
            'loss': best_loss,
        }


    def match_traces_analytic(self, limits=None, index_shifts=(0, 0), pad_width=None, n=1, transform=None,
                              twostep=False, twostep_margin=10,
                              max_shift=100, resample_factor=10, taper=True,
                              apply_correction=False, correction_step=3, return_intermediate=False, **kwargs):
        """ Match traces by using analytic formulae.
        Bishop, Nunns "`Correcting amplitude, time, and phase mis-ties in seismic data
        <https://www.researchgate.net/publication/249865260>`_"
        Fast, but rather unflexible.

        Under the hood, the algorithm works as follows:
            - we compute possible shifts with possibly non-whole numbers (`resample_factor`)
            - compute correlation for each possible shift, resulting in cross-correlation function
            - compute envelope and instantaneous phase of the cross-correlation
            - argmax of the envelope is the optimal shift, and the phase at this shift is the optimal angle.
            Essentially, this is equivalent to finding the best combination of the trace and its analytic counterpart,
            which conveniently coincide with vertical and phase shifts.
        """
        _ = kwargs

        # Load data
        trace_0, trace_1 = self.prepare_traces(limits=limits, index_shifts=index_shifts,
                                               pad_width=pad_width, n=n, transform=transform)

        # Prepare array of tested shifts
        if twostep:
            # Compute approximate `shift` to narrow the interval
            shifts = np.linspace(-max_shift, max_shift, 2*max_shift + 1, dtype=np.float32)
            shift = self._match_traces_analytic(trace_0=trace_0, trace_1=trace_1, shifts=shifts, taper=taper,
                                                apply_correction=apply_correction, correction_step=correction_step,
                                                return_intermediate=False)['shift']
            shifts = np.linspace(shift - twostep_margin, shift + twostep_margin,
                                 2*twostep_margin*resample_factor + 1, dtype=np.float32)
        else:
            shifts = np.linspace(-max_shift, max_shift, 2*max_shift*resample_factor + 1, dtype=np.float32)

        # Compute `shift` with required precision
        matching_results = self._match_traces_analytic(trace_0=trace_0, trace_1=trace_1, shifts=shifts,
                                                       taper=taper, apply_correction=apply_correction,
                                                       correction_step=correction_step,
                                                       return_intermediate=return_intermediate)

        matching_results['corr'] = self.evaluate(shift=matching_results['shift'],
                                                 angle=matching_results['angle'],
                                                 gain=matching_results['gain'],
                                                 pad_width=pad_width, limits=limits, n=n)
        return matching_results

    def _match_traces_analytic(self, trace_0, trace_1, shifts,
                               taper=True, apply_correction=False, correction_step=3, return_intermediate=False):
        # Compute metrics for each shift on a resampled grid
        # TODO: fix nan/inf values in case of short windows and large `max_shift``
        shifted_traces = compute_shifted_traces(trace=trace_1, shifts=shifts)
        metrics = (trace_0 * shifted_traces).mean(axis=1) / (trace_0.std() * shifted_traces.std(axis=1))

        # Compute envelope and phase of metrics
        analytic_signal = scipy.signal.hilbert(metrics)
        envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        instantaneous_phase = np.rad2deg(instantaneous_phase)

        # Optional taper
        if taper:
            taper = 0.1 if taper is True else taper
            lentaper = int(taper * envelope.size)
            taper = np.hanning(lentaper)
            envelope[:lentaper // 2] *= taper[:lentaper // 2]
            envelope[-lentaper // 2:] *= taper[-lentaper // 2:]
            # envelope *= np.hanning(envelope.size)

        # Find the best shift and compute its relative quality
        idx = np.argmax(envelope)

        # Optional correction: parabolic interpolation in the neighborhood of a maxima
        if apply_correction is False:
            shift = shifts[idx]
            angle = instantaneous_phase[idx]
        else:
            # TODO: refactor / rethink
            correction = ((metrics[idx-correction_step] - metrics[idx+correction_step]) /
                          (2*metrics[idx-correction_step] - 4*metrics[idx] + 2*metrics[idx+correction_step]))
            quality = metrics[idx]\
                      - 0.25 * (((metrics[idx-correction_step] - metrics[idx+correction_step]) * correction) /
                                (np.linalg.norm(trace_0) * np.linalg.norm(trace_1)))
            _ = quality

            # Shift: correct according to values to the sides of maximum
            corrected_idx = int(idx + correction)
            shift = shifts[corrected_idx]

            # Angle: correct according to values to the sides of maximum
            p0 = instantaneous_phase[idx]
            p1 = instantaneous_phase[idx+correction_step] if correction >= 0 else \
                 instantaneous_phase[idx-correction_step]
            if p1 - p0 > 180:
                p1 = p1 - 360
            elif p1 - p0 < -180:
                p1 = p1 + 360
            angle = p0 + ((p1 - p0) * correction if correction >= 0 else (p0 - p1) * correction)

        gain = self.compute_gain(data_0=trace_0, data_1=trace_1)

        matching_results = {
            'shift': shift,
            'angle': angle,
            'gain': gain,
        }

        if return_intermediate:
            matching_results.update({
                'trace_0': trace_0,
                'trace_1': trace_1,
                'shifts': shifts,
                'metrics': metrics,
                'envelope': envelope,
                'instantaneous_phase': instantaneous_phase,
            })
        return matching_results


    def match_on_horizon(self, horizon_name=None, **kwargs):
        """ Use horizon picks to determine necessary corrections on intersection.
        Note that it actually computes only the vertical mistie; angle is left at 0 and gain is the RMS ratio.
        """
        horizon_to_depth_0, horizon_to_depth_1 = self.prepare_horizons()

        if horizon_name is None:
            common_horizons = set(horizon_to_depth_0.keys()) & set(horizon_to_depth_1.keys())
            if len(common_horizons) == 1:
                horizon_name = common_horizons.pop()
            else:
                raise ValueError('Provide horizon name for matching!')

        depth_0 = horizon_to_depth_0[horizon_name]
        depth_1 = horizon_to_depth_1[horizon_name]

        shift = depth_0 - depth_1
        angle = 0.0
        gain = self.compute_gain()
        corr = self.evaluate(shift=shift, angle=angle, gain=gain)
        return {
            'shift': shift,
            'angle': angle,
            'gain': gain,
            'corr': corr
        }

    def compute_gain(self, data_0=None, data_1=None, **kwargs):
        """ Compute gain by as ratio between RMS data values. """
        if data_0 is None and data_1 is None:
            data_0, data_1 = self.prepare_traces(**kwargs)
        return (data_0**2).mean() ** (1/2) / (data_1**2).mean() ** (1/2)


    def evaluate(self, shift=0, angle=0, gain=1, metric='correlation',
                 pad_width=None, limits=None, n=1, transform=None, **kwargs):
        """ Compute provided metric with a given mistie parameters. """
        trace_0, trace_1 = self.prepare_traces(pad_width=pad_width, limits=limits, n=n, transform=transform)
        metric_function = compute_correlation if metric == 'correlation' else compute_r2
        return metric_function(trace_0, modify_trace(trace_1, shift=shift, angle=angle, gain=gain))

    def compute_horizon_metric(self, horizon_name=None, shift=0, **kwargs):
        """ Compute the difference between horizon matching and suggested shift. """
        horizon_matching = self.match_on_horizon(horizon_name=horizon_name)
        return np.abs(horizon_matching['shift'] - shift)

    def get_correction(self):
        """ Get corrections from field. Used to see how intersection would behave after field corrections applied. """
        dict_0 = self.field_0.correction_results
        dict_1 = self.field_1.correction_results

        shift = dict_0['shift'] - dict_1['shift']
        angle = dict_0['angle'] - dict_1['angle']
        gain = np.exp(np.log(dict_0['gain']) - np.log(dict_1['gain']))
        corr = self.evaluate(shift=shift, angle=angle, gain=gain)
        return {
            'shift': shift,
            'angle': angle,
            'gain': gain,
            'corr': corr
        }


    # Visualization
    def __repr__(self):
        return (f'<Intersection of "{self.field_0.short_name}.sgy"'
                f' and "{self.field_1.short_name}.sgy" at {hex(id(self))}>')

    def __str__(self):
        return dedent(f"""
        Intersection of "{self.field_0.short_name}.sgy" and "{self.field_1.short_name}.sgy"
        distance                     {self.distance:4.2f} m
        trace_idx_0                  {self.trace_idx_0}
        trace_idx_1                  {self.trace_idx_1}
        coordinates_0                {self.coordinates_0.tolist()}
        coordinates_1                {self.coordinates_1.tolist()}
        key                          {getattr(self, 'key', None)}
        """).strip()

    def show_curves(self, method='analytic', limits=None, index_shifts=(0, 0), pad_width=None, n=1, transform=None,
                    shift=0, angle=0, gain=1,
                    max_shift=100, resample_factor=10, apply_correction=False, n_plots=3, **kwargs):
        """ Display traces, cross-correlation vs shift and phase vs shift graphs. """
        # Get matching results with all the intermediate variables
        matching_results = self.match_traces(method=method,
                                             limits=limits, index_shifts=index_shifts,
                                             pad_width=pad_width, n=n, transform=transform,
                                             max_shift=max_shift, resample_factor=resample_factor,
                                             apply_correction=apply_correction,
                                             return_intermediate=True)

        trace_0, trace_1 = matching_results['trace_0'], matching_results['trace_1']
        trace_1 = modify_trace(trace_1, shift=shift, angle=angle, gain=gain)
        shifts, metrics, envelope, instantaneous_phase = (matching_results['shifts'],
                                                          matching_results['metrics'],
                                                          matching_results['envelope'],
                                                          matching_results['instantaneous_phase'])

        # Prepare plotter parameters
        limits = limits or self.limits
        pad_width = pad_width or self.pad_width
        start_tick = (limits.start or 0) - pad_width
        ticks = np.arange(start_tick, start_tick + len(trace_0))

        kwargs = {
            'title': [f'traces of "{self.field_0.short_name}.sgy" x "{self.field_1.short_name}.sgy"'
                      f'\n{shift=:3.3f}  {angle=:3.3f}  {gain=:3.3f}',
                      'cross-correlation', 'instantaneous phase'],
            'label': [['trace_0', 'trace_1'], ['crosscorrelation', 'envelope'], 'instant phases'],
            'xlabel': ['depth', 'shift', 'shift'], 'xlabel_size': 16,
            'ylabel': ['amplitude', 'metric', 'phase (degrees)'],
            'xlim': [(start_tick, start_tick + len(trace_0)),
                     (-max_shift, +max_shift),
                     (-max_shift, +max_shift)],
            'ratio': 0.8,
            **kwargs
        }

        plotter = plot([[(ticks, trace_0), (ticks, trace_1)],
                        [(shifts, metrics), (shifts, envelope)],
                        [(shifts, instantaneous_phase)]][:n_plots],
                       mode='curve', **kwargs)

        # Add more annotations
        shift = matching_results['shift']
        angle = matching_results['angle']
        corr = matching_results['corr']
        if n_plots >= 2:
            plotter[1].ax.axvline(shift, linestyle='--', alpha=0.9, color='green')
            plotter[1].add_legend(mode='curve', label=f'optimal shift: {shift:4.3f}',
                                alpha=0.9, color='green')
            plotter[1].ax.axhline(corr, linestyle='--', alpha=0.9, color='red')
            plotter[1].add_legend(mode='curve', label=f'max correlation: {corr:4.3f}',
                                alpha=0.9, color='red')
        if n_plots >= 3:
            plotter[2].ax.axvline(shift, linestyle='--', alpha=0.9, color='green')
            plotter[2].add_legend(mode='curve', label=f'optimal angle: {angle:4.3f}')
        return plotter


    def show_lines(self, figsize=(14, 8), colors=('b', 'r'), arrow_step=20, arrow_size=30, savepath=None, show=True):
        """ Display shot lines on a 2d graph in CDP coordinates. """
        fig, ax = plt.subplots(figsize=figsize)

        # Data
        for field, color in zip([self.field_0, self.field_1], colors):
            values = field.geometry.headers[['CDP_X', 'CDP_Y']].values
            x, y = values[:, 0], values[:, 1]
            ax.plot(x, y, color, label=field.short_name)

            idx = x.size // 2
            ax.annotate('', size=arrow_size,
                        xytext=(x[idx-arrow_step], y[idx-arrow_step]),
                            xy=(x[idx+arrow_step], y[idx+arrow_step]),
                        arrowprops=dict(arrowstyle="->", color=color))

        # Annotations
        ax.set_title(f'"{self.field_0.short_name}.sgy" and "{self.field_1.short_name}.sgy"', fontsize=26)
        ax.set_xlabel('CDP_X', fontsize=22)
        ax.set_ylabel('CDP_Y', fontsize=22)
        ax.legend(prop={'size' : 22})
        ax.grid()
        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=100, facecolor='white')
        if show:
            fig.show()
        else:
            plt.close(fig)


    def show_metric_surface(self, metric='correlation',
                            limits=None, index_shifts=(0, 0), pad_width=None, n=1, transform=None,
                            shifts=range(-20, +20+1, 1), angles=range(-180, +180+1, 30),
                            figsize=(14, 8), cmap='seismic', levels=7, grid=True):
        """ Display metric values as a function of shift and angle. """
        # Compute metric matrix: metric value for each combination of shift and angle
        trace_0, trace_1 = self.prepare_traces(limits=limits, index_shifts=index_shifts,
                                               pad_width=pad_width, n=n, transform=transform)
        metric_function = compute_correlation if metric == 'correlation' else compute_r2

        metric_matrix = np.empty((len(shifts), len(angles)))
        for i, shift in enumerate(shifts):
            for j, angle in enumerate(angles):
                modified_trace_1 = modify_trace(trace_1, shift=shift, angle=angle)

                metric_matrix[i, j] = metric_function(trace_0, modified_trace_1)

        # Show contourf and contour
        fig, ax = plt.subplots(1, figsize=figsize)
        img = ax.contourf(angles, shifts, metric_matrix, cmap=cmap, levels=levels)
        fig.colorbar(img)

        contours = ax.contour(angles, shifts, metric_matrix, levels=levels, colors='k', linewidths=0.4)
        ax.clabel(contours, contours.levels, inline=True, fmt=lambda x: f'{x:2.1f}', fontsize=10)

        ax.set_title('METRIC SURFACE', fontsize=20)
        ax.set_xlabel('PHASE (DEGREES)', fontsize=16)
        ax.set_ylabel('SHIFT (MS)', fontsize=16)
        ax.grid(grid)
        fig.show()


    def show_neighborhood(self, max_index_shift=7, limits=None, pad_width=None, n=1, transform=None,
                          max_shift=10, resample_factor=10):
        """ Compute matching on all neighboring traces. """
        # Prepare data
        k = max_index_shift * 2 + 1
        matrix = np.empty((k, k), dtype=np.float32)
        iterator = range(-max_index_shift, max_index_shift + 1)
        for i, index_shift_1 in enumerate(iterator):
            for j, index_shift_2 in enumerate(iterator):
                matching_results = self.match_traces_analytic(index_shifts=(index_shift_1, index_shift_2),
                                                              limits=limits, pad_width=pad_width, n=n,
                                                              transform=transform,
                                                              max_shift=max_shift, resample_factor=resample_factor)
                matrix[i, j] = matching_results['corr']

        # Visualize
        value = matrix[max_index_shift, max_index_shift]
        delta = max(matrix.max() - value, value - matrix.min())
        vmin, vmax = value - delta, value + delta

        return plot(matrix, colorbar=True, cmap='seismic',
                    title='Correlation values for neighbouring indices of intersection',
                    vmin=vmin, vmax=vmax,
                    extent=(-max_index_shift, +max_index_shift,
                            -max_index_shift, +max_index_shift))


    def show_composite_slide(self, sides=(0, 0), horizon_width=3,
                             limits=None, gap_width=1, gap_value=None, pad_width=None, transform=None,
                             shift=0, angle=0, gain=1, width='auto', title_prefix='', **kwargs):
        """ Display sides of shot lines on one plot. """
        limits = limits if limits is not None else self.limits
        pad_width = pad_width if pad_width is not None else self.pad_width
        transform = transform if transform is not None else self.transform

        # Make combined slide
        combined_slide, slide_0, slide_1 = self._prepare_combined_slide(sides=sides, data='field',
                                                                        shift=shift, angle=angle, gain=gain,
                                                                        limits=limits, gap_width=gap_width,
                                                                        gap_value=gap_value, width=width,
                                                                        pad_width=pad_width, transform=transform)
        data = [combined_slide]
        cmap = ['Greys_r']

        mask_slide = self._prepare_combined_slide(sides=sides, data='labels',
                                                  shift=shift, angle=0, gain=1.,
                                                  horizon_width=horizon_width,
                                                  limits=limits, gap_width=gap_width, gap_value=gap_value, width=width,
                                                  pad_width=pad_width, transform=transform)[0].astype(np.bool_)
        data.append(mask_slide)
        cmap.append('magenta')

        # Compute correlation on traces
        correlation = kwargs.get('corr', compute_correlation(slide_0[-1], slide_1[0]))

        # Prepare plotter parameters
        start_tick = (limits.start or 0) - pad_width
        extent = (0, combined_slide.shape[0], start_tick + combined_slide.shape[1], start_tick)

        title = (f'"{self.field_0.short_name}.sgy":{sides[0]} x "{self.field_1.short_name}.sgy":{sides[1]}\n'
                 f'{shift=:3.2f}  {angle=:3.1f}  {gain=:3.3f}\n'
                 f'{correlation=:3.2f}  corrected_correlation={(1 + correlation)/2:3.2f}')
        if title_prefix is not None:
            title = title_prefix + '\n' + title

        kwargs = {
            'cmap': cmap,
            'colorbar': True,
            'title': title, 'title_fontsize': 14,
            'extent': extent,
            'augment_mask': True,
            'labelright': False, 'labeltop': False,
            **kwargs
        }
        return plot(data, **kwargs)

    def _prepare_combined_slide(self, sides, data='field', horizon_width=5, shift=0, angle=0, gain=1,
                                limits=None, width='auto', gap_width=1, gap_value=None, pad_width=None,
                                n=1, transform=None):
        # Load data and orient it in a correct way
        slide_0 = self._prepare_slide(self.field_0, self.trace_idx_0, sides[0], horizon_width=horizon_width,
                                      data=data, limits=limits, pad_width=pad_width, transform=transform)
        slide_1 = self._prepare_slide(self.field_1, self.trace_idx_1, sides[1], horizon_width=horizon_width,
                                      data=data, limits=limits, pad_width=pad_width, transform=transform)

        if sides == (0, 0):
            slide_1 = slide_1[::-1]
        elif sides == (0, 1):
            pass
        elif sides == (1, 0):
            slide_0 = slide_0[::-1]
            slide_1 = slide_1[::-1]
        elif sides == (1, 1):
            slide_0 = slide_0[::-1]

        # Apply modifications to the right side
        for c in range(slide_1.shape[0]):
            slide_1[c] = modify_trace(slide_1[c], shift=shift, angle=angle, gain=gain)

        # Slice to limits. Done after the modification so that there is no empty areas on top/bottom of the image
        slide_0 = slide_0[:, limits]
        slide_1 = slide_1[:, limits]

        # Combine slides into one composite
        width = slide_0.shape[1] if width == 'auto' else width
        halfwidth = width//2 if width is not None else max(len(slide_0), len(slide_1))
        fv = gap_value if gap_value is not None else min(slide_0.min(), slide_1.min())

        combined_slide = np.concatenate([slide_0[-halfwidth:],
                                         np.full((gap_width, slide_0.shape[1]), fill_value=fv, dtype=np.float32),
                                         slide_1[:+halfwidth]], axis=0)
        return combined_slide, slide_0, slide_1

    def _prepare_slide(self, field, trace_idx, side, data='field', horizon_width=5,
                       limits=None, pad_width=None, n=1, transform=None):
        # Load data
        if data == 'field':
            slide = field.load_slide(0)
        else:
            slide = np.zeros(field.shape[1:], dtype=np.float32)
            for horizon in getattr(field, 'horizon_instances', {}).values():
                slide += horizon.load_slide(0, width=horizon_width)

        slide = slide[:trace_idx + 1] if side == 0 else slide[trace_idx:]

        # Resample to ms
        arange = np.arange(slide.shape[1], dtype=np.float32)
        arange_ms = np.arange(slide.shape[1], step=(1 / field.sample_interval), dtype=np.float32)
        arange_ms += field.delay / field.sample_interval if data != 'field' else 0
        interpolator = lambda trace: np.interp(arange_ms, arange, trace, left=0, right=0)
        slide = np.apply_along_axis(interpolator, 1, slide)

        # Pad to adjust for field delays
        slide = np.pad(slide, ((0, 0), (field.delay, 0))) if field.delay > 0 else slide[:, -field.delay:]

        # Pad to the same depth
        slide = np.pad(slide, ((0, 0), (0, self.max_depth - slide.shape[1])))


        # Additional padding
        slide = np.pad(slide, ((0, 0), (pad_width, pad_width)))
        return slide

    def compare_composite_slides(self, sides=(0, 0), horizon_width=5,
                                 limits=None, gap_width=1, gap_value=None, pad_width=None, transform=None,
                                 shift=0, angle=0, gain=1, width='auto', figsize=(14, 20), **kwargs):
        """ Display composite slides over intersection with and without corrections side-by-side. """
        _, ax = plt.subplots(1, 2, figsize=figsize)
        self.show_composite_slide(sides=sides, horizon_width=horizon_width,
                                  limits=limits, gap_width=gap_width, gap_value=gap_value,
                                  pad_width=pad_width, transform=transform,
                                  width=width, axes=ax[0], adjust_figsize=False, **kwargs)

        self.show_composite_slide(sides=sides, horizon_width=horizon_width,
                                  limits=limits, gap_width=gap_width, gap_value=gap_value,
                                  pad_width=pad_width, transform=transform,
                                  shift=shift, angle=angle, gain=gain,
                                  width=width, axes=ax[1], adjust_figsize=False, **kwargs)
        plt.show()

    def compare_methods(self, sides=(0, 0), horizon_width=5, limits=None, gap_width=1, gap_value=None,
                        analytic_kwargs=None, optimize_kwargs=None,
                        pad_width=None, transform=None, width='auto', layout=None, figsize=(14, 20), **kwargs):
        """ !!. """
        if getattr(self.field_0, 'correction_results', None) and getattr(self.field_1, 'correction_results', None):
            n_plots = 4
        else:
            n_plots = 3

        layout = layout if layout is not None else (1, n_plots)
        _, ax = plt.subplots(*layout, figsize=figsize)
        ax = np.array(ax).flatten()

        # Graph 1: original slides
        self.show_composite_slide(sides=sides, horizon_width=horizon_width,
                                  limits=limits, gap_width=gap_width, gap_value=gap_value,
                                  pad_width=pad_width, transform=transform,
                                  title_prefix='original data',
                                  width=width, axes=ax[0], adjust_figsize=False, **kwargs)

        # Graph 2: analytic method
        analytic_kwargs = analytic_kwargs if analytic_kwargs is not None else {}
        matching_dict = self.match_traces_analytic(**analytic_kwargs)
        self.show_composite_slide(sides=sides, horizon_width=horizon_width,
                                  limits=limits, gap_width=gap_width, gap_value=gap_value,
                                  pad_width=pad_width, transform=transform,
                                  **matching_dict, title_prefix='method=`analytic`',
                                  width=width, axes=ax[1], adjust_figsize=False, **kwargs)

        # Graph 3: optimize method
        optimize_kwargs = optimize_kwargs if optimize_kwargs is not None else {}
        matching_dict = self.match_traces_optimize(**optimize_kwargs)
        self.show_composite_slide(sides=sides, horizon_width=horizon_width,
                                  limits=limits, gap_width=gap_width, gap_value=gap_value,
                                  pad_width=pad_width, transform=transform,
                                  **matching_dict, title_prefix='method=`optimize`',
                                  width=width, axes=ax[2], adjust_figsize=False, **kwargs)

        if n_plots == 4:
            matching_dict = self.get_correction()
            self.show_composite_slide(sides=sides, horizon_width=horizon_width,
                                    limits=limits, gap_width=gap_width, gap_value=gap_value,
                                    pad_width=pad_width, transform=transform,
                                    **matching_dict, title_prefix='method=`corrections`',
                                    width=width, axes=ax[3], adjust_figsize=False, **kwargs)

        plt.show()




class Intersection2d3d:
    """ TODO. """
