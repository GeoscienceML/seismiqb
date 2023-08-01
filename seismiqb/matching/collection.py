""" Collection of multiple intersecting fields. """
import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from numba import njit

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from batchflow.notifier import Notifier
from .intersection import Intersection
from .functional import modify_trace

from ..field import Field
from ..labels import Horizon
from ..plotters import plot
from ..geometry.memmap_loader import MemmapLoader



class FieldCollection:
    """ Collection of 2D fields and their intersections. """
    def __init__(self, fields, limits=slice(None), pad_width=0, threshold=10,
                 n_intersections=np.inf, geometry_kwargs=None):
        self.fields = self.load_fields(fields, geometry_kwargs=geometry_kwargs)
        self.n_fields = len(self.fields)

        self.intersections = self.compute_intersections(limits=limits, pad_width=pad_width,
                                                        threshold=threshold, n_intersections=n_intersections)
        self.horizons = {}

        self.corrections = {}

    # Instance initialization
    DEFAULT_GEOMETRY_KWARGS = {
        'index_headers': ['FieldRecord', 'CDP'],
        'additional_headers': ['CDP', 'CDP_X', 'CDP_Y'],
        'collect_stats': False, 'collect_stats_params': {'pbar': False},
        'dump_headers': True, 'dump_meta': True
    }

    def load_fields(self, fields, geometry_kwargs=None):
        """ Load field instances from their paths.
        If an element of `fields` is already an instance of Field, it is left untouched.
        """
        # TODO: try to remove ~duplicate fields?
        if isinstance(fields, str):
            fields = sorted(list(glob(fields)))

        geometry_kwargs = geometry_kwargs if geometry_kwargs is not None else self.DEFAULT_GEOMETRY_KWARGS
        return [Field(item, geometry_kwargs=geometry_kwargs) if isinstance(item, str) else item for item in fields]

    def compute_intersections(self, limits=slice(None), pad_width=0, threshold=10, n_intersections=np.inf):
        """ Compute intersections over the present fields. """
        result = {}
        for i, field_0 in enumerate(self.fields[:-1]):
            for j, field_1 in enumerate(self.fields[i+1:], start=i+1):
                intersections = Intersection.new(field_0=field_0, field_1=field_1,
                                                 limits=limits, pad_width=pad_width, threshold=threshold,
                                                 unwrap=False)

                for k, intersection in enumerate(intersections):
                    if k > n_intersections:
                        break
                    key = (i, j, k)
                    result[key] = intersection
                    intersection.key = (i, j, k)

        return result

    def load_horizon(self, path, add_instances=True, verbose=True):
        """ Load horizon: save into dict for each field. """
        horizon_name = os.path.basename(path)

        df = pd.read_csv(path, sep=r'\s+', index_col=False, skiprows=[0],
                        names=['FIELD_NAME', '_', 'CDP_X', 'CDP_Y', 'DEPTH', '__'],
                        dtype={'FIELD_NAME': str, })
        self.horizons[horizon_name] = df
        unique_field_names = df['FIELD_NAME'].unique()

        for field in self.fields:
            if field.short_name not in unique_field_names:
                if verbose:
                    print(f'Field "{field.short_name}" is not labeled in horizon')
                continue
            subdf = df[df['FIELD_NAME'] == field.short_name]

            # Prepare horizon points: (N, 3) array
            horizon_ixd = subdf[['CDP_X', 'CDP_Y', 'DEPTH']].values
            if not getattr(field, 'horizons'):
                field.horizons = {horizon_name : horizon_ixd}
            else:
                field.horizons[horizon_name] = horizon_ixd

            # Prepare horizon instances
            if add_instances: #TODO: refactor
                indices = []
                for value in horizon_ixd[:, :2]:
                    index = np.argmin(np.abs(field.cdp_values - value).sum(axis=1))
                    indices.append(index)
                indices = np.array(indices)

                depths = (subdf['DEPTH'].values - field.delay) / field.sample_interval
                points = np.array([np.zeros(len(indices), dtype=np.int32),
                                   indices,
                                   np.round(depths).astype(np.int32)]).T
                horizon_instance = Horizon(points, field=field, name=horizon_name)

                if not hasattr(field, 'horizon_instances'):
                    field.horizon_instances = {horizon_name : horizon_instance}
                else:
                    field.horizon_instances[horizon_name] = horizon_instance


    # Work with intersections
    def find_intersection(self, name_0, name_1):
        """ Find intersection by names of shot lines. """
        for intersection in self.intersections.values():
            if (name_0 in intersection.field_0.name and name_1 in intersection.field_1.name) or \
                (name_1 in intersection.field_0.name and name_0 in intersection.field_1.name):
                return intersection
        raise KeyError(f'No intersection of `{name_0}` and `{name_1}` in collection!')

    def find_intersections(self, name):
        """ Find all intersections of a given show with other lines. """
        return [intersection for intersection in self.intersections.values()
                if name in intersection.field_0.name or name in intersection.field_1.name]

    def match_intersections(self, pbar='t', method='analytic', limits=None, pad_width=None, n=1, transform=None,
                            **kwargs):
        """ Match traces on each intersection. """
        for intersection in Notifier(pbar)(self.intersections.values()):
            intersection.match_traces(method=method, limits=limits, pad_width=pad_width, n=n, transform=transform,
                                      **kwargs)

    def match_intersections_p(self, pbar='t', method='analytic', limits=None, pad_width=None, n=1, transform=None,
                              max_workers=8, **kwargs):
        """ Match traces on each intersection. """
        with Notifier(pbar, total=len(self.intersections)) as progress_bar:
            def callback(future):
                matching_results = future.result()
                key = matching_results.pop('key')
                self.intersections[key].matching_results = matching_results
                progress_bar.update(1)

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for intersection in self.intersections.values():
                    future = executor.submit(intersection.match_traces,
                                             method=method, limits=limits, pad_width=pad_width,
                                             n=n, transform=transform, **kwargs)
                    future.add_done_callback(callback)

    def get_matched_value(self, key):
        """ Get required `key` value from each of the intersections. """
        return [intersection.matching_results[key] for intersection in self.intersections.values()]

    def intersections_df(self, errors=False, corrections=False, indices=False):
        """ Dataframe with intersections: each row describes quality of matching, mis-tie parameters for every crossing.
        If corrections are available, also use them.
        """
        df = []
        for key, intersection in self.intersections.items():
            intersection_dict = {'key': key, **intersection.to_dict()}
            df.append(intersection_dict)
        df = pd.DataFrame(df)
        df.set_index('key', inplace=True)

        # Corrections are distributed
        if self.corrections:
            shifts_errors = np.abs(self.corrections['shift']['errors'])
            gains_errors = np.abs(self.corrections['gain']['errors'])
            angles_errors = np.abs(self.corrections['angle']['errors'])
            suspicious = ((shifts_errors > shifts_errors.mean() + 3 * shifts_errors.std()) +
                          (gains_errors > gains_errors.mean() + 3 * gains_errors.std()) +
                          (angles_errors > angles_errors.mean() + 3 * angles_errors.std()))

            if errors:
                df['shifts_errors'] = shifts_errors
                df['gains_errors'] = gains_errors
                df['angles_errors'] = angles_errors
            df['suspicious'] = suspicious

            if corrections:
                idx_0 = [key[0] for key in self.intersections]
                df['shift_correction'] = self.corrections['shift']['x'][idx_0]
                df['angle_correction'] = self.corrections['angle']['x'][idx_0]
                df['gain_correction'] = self.corrections['gain']['x'][idx_0]

        columns = [
            'field_0_name', 'field_1_name', 'distance', 'suspicious',
            'corr', 'petrel_corr',
            'shift', 'shift_correction', 'angle', 'angle_correction', 'gain', 'gain_correction',
        ]
        columns = [c for c in columns if c in df.columns.values]
        columns += list(set(df.columns.values) - set(columns))
        df = df[columns]
        return df

    def compute_horizon_metric(self, horizon_name=None):
        """ Compute the difference between horizon and proposed shifts.
        The first metric adds no shift, measuring the difference on horizon picks in the original file.
        The second uses field corrections, and the last applies shifts from the intersections.
        """
        shifts = self.corrections['shift']['x']

        metrics = []
        for key, intersection in self.intersections.items():
            metric_0 = intersection.compute_horizon_metric(horizon_name=horizon_name, shift=0)
            metric_1 = intersection.compute_horizon_metric(horizon_name=horizon_name,
                                                           shift=shifts[key[0]] - shifts[key[1]])
            metric_2 = intersection.compute_horizon_metric(horizon_name=horizon_name,
                                                           shift=intersection.matching_results['shift'])

            metrics.append((metric_0, metric_1, metric_2))

        names = ['mean_horizon_shift', 'mean_horizon_to_correction_shift', 'mean_horizon_to_intersection_shift']
        values = np.array(metrics).mean(axis=0).round(4)
        return dict(zip(names, values))


    # Work with fields
    def distribute_corrections(self, skip_index=-1, max_iters=100, alpha=0.75, tolerance=0.00001):
        """ Distribute computed mis-ties from each intersection to fields.
        Under the hood, we iteratively optimize mis-ties of every type with respect to ~MSE loss.

        For the phase corrections, we also add phase unwrapping. Refer to the original article for details.
        Bishop, Nunns "`Correcting amplitude, time, and phase mis-ties in seismic data
        <https://www.researchgate.net/publication/249865260>`_"
        """
        a = np.array([key[:2] for key in self.intersections])
        n = self.n_fields

        # Shift
        b = np.array(self.get_matched_value('shift'))
        x, loss = distribute_misties(a=a, b=b, n=n, skip_index=skip_index,
                                     max_iters=max_iters, alpha=alpha, tolerance=tolerance)
        xk, xl = x[a[:, 0]], x[a[:, 1]]
        errors = b - (xk - xl)
        self.corrections['shift'] = {'x': x, 'errors': errors, 'loss': loss, 'b': b}

        # Gain
        b = np.array(self.get_matched_value('gain'))
        b = np.log(b)
        x, loss = distribute_misties(a=a, b=b, n=n, skip_index=skip_index,
                                     max_iters=max_iters, alpha=alpha, tolerance=tolerance)
        xk, xl = x[a[:, 0]], x[a[:, 1]]
        errors = b - (xk - xl)
        self.corrections['gain'] = {'x': np.exp(x), 'errors': errors, 'loss': loss, 'b': b}

        # Angle
        b = np.array(self.get_matched_value('angle'))
        x, loss = distribute_misties(a=a, b=b, n=n, skip_index=skip_index,
                                     max_iters=max_iters, alpha=alpha, tolerance=tolerance)

        b_arange = np.arange(len(b))

        for _ in range(0, len(self.intersections)):
            xk, xl = x[a[:, 0]], x[a[:, 1]]

            b_unwrapped = np.repeat(b[:, np.newaxis], 3, axis=-1) + [-360, 0, +360]
            errors_unwrapped = np.abs(b_unwrapped - (xk - xl).reshape(-1, 1))
            argmins = np.argmin(errors_unwrapped, axis=-1)

            # Stop condition: no phase unwrapping required
            if (argmins == 1).all():
                # TODO: add one-time forced perturbation
                break

            b = b_unwrapped[b_arange, argmins]
            x, loss = distribute_misties(a=a, b=b, n=n, skip_index=skip_index,
                                         max_iters=max_iters, alpha=alpha, tolerance=tolerance)

        errors = b - (xk - xl)
        self.corrections['angle'] = {'x': x, 'errors': errors, 'loss': loss, 'b': b}
        # TODO: add return with info about the process

        # Store correction info in fields as well
        for i, field in enumerate(self.fields):
            field.correction_results = {
                'shift': self.corrections['shift']['x'][i],
                'angle': self.corrections['angle']['x'][i],
                'gain': self.corrections['gain']['x'][i],
            }

        return (self.corrections['shift']['loss'][-1],
                self.corrections['angle']['loss'][-1],
                self.corrections['gain']['loss'][-1])


    def compute_suspicious_intersections(self):
        """ For each intersection, compute whether it is suspicious.
        # TODO: add more checks
        """
        if self.corrections is None:
            return [False] * len(self.intersections)
        shifts_errors = np.abs(self.corrections['shift']['errors'])
        gains_errors = np.abs(self.corrections['gain']['errors'])
        angles_errors = np.abs(self.corrections['angle']['errors'])
        suspicious = ((shifts_errors > shifts_errors.mean() + 3 * shifts_errors.std()) +
                      (gains_errors > gains_errors.mean() + 3 * gains_errors.std()) +
                      (angles_errors > angles_errors.mean() + 3 * angles_errors.std()))
        return suspicious

    def remove_intersections(self, indices=None, skip_index=-1, max_iters=100, alpha=0.75, tolerance=0.00001):
        """ Remove all suspicious intersections and re-distribute corrections. """
        if indices is None:
            suspicious = self.compute_suspicious()
            indices = np.nonzero(suspicious)[0]
        keys = list(self.intersections.keys())
        for idx in np.sort(indices)[::-1]:
            key = keys[idx]
            self.intersections.pop(key)

        self.distribute_corrections(skip_index=skip_index, max_iters=max_iters, alpha=alpha, tolerance=tolerance)
        return indices

    def remove_fields(self, indices):
        """ Remove fields (and their intersections) by indices.
        TODO: test
        """
        for idx in np.sort(indices)[::-1]:
            self.fields.pop(idx)

        new_intersections = {}
        for key, intersection in self.intersections.items():
            if key[0] in indices or key[1] in indices:
                continue

            new_key = (key[0] - (indices > key[0]).sum(),
                        key[1] - (indices > key[1]).sum(),
                        key[2])
            intersection.key = new_key
            new_intersections[new_key] = intersection
        self.intersections = new_intersections


    def fields_df(self):
        """ Dataframe with fields: each row describes a field with computed mis-ties. """
        shifts = self.corrections['shift']['x']
        gains = self.corrections['gain']['x']
        angles = self.corrections['angle']['x']

        df, bad_intersections_keys = [], []
        for i, field in enumerate(self.fields):
            intersections = []
            recomputed_corrs = []

            for key, intersection in self.intersections.items():
                if i in key[:2]:
                    recomputed_corr = intersection.evaluate(shift=shifts[key[0]] - shifts[key[1]],
                                                            angle=angles[key[0]] - angles[key[1]],)
                    recomputed_corr = (recomputed_corr + 1) / 2
                    recomputed_corrs.append(recomputed_corr)

                    intersections.append(intersection)

            if intersections:
                min_, mean_, std_ = np.min(recomputed_corrs), np.mean(recomputed_corrs), np.std(recomputed_corrs)

                for j, intersection in enumerate(intersections):
                    recomputed_corr = recomputed_corrs[j]
                    if recomputed_corr < 0.5 or abs(mean_ - recomputed_corr) > 0.25:
                        bad_intersections_keys.append(intersection.key)

                # Stats on intersections: no distribution of corrections
                dicts = [intersection.matching_results for intersection in intersections]
                corrs = [d['petrel_corr'] for d in dicts]
                mean_intersection, std_intersection = np.mean(corrs), np.std(corrs)
            else:
                # field has no intersections
                min_ = mean_ = std_ = mean_intersection = std_intersection = np.float64(-1)


            correction_results = {
                'name': field.name,
                'shift': shifts[i],
                'angle': angles[i],
                'gain': gains[i],

                'mean_recomputed_corr': mean_.round(3),
                'std_recomputed_corr': std_.round(3),
                'min_recomputed_corr': min_.round(3),

                'n_intersections': len(intersections),
                'mean_corr_intersections': mean_intersection.round(3),
                'std_corr_intersections': std_intersection.round(3),
            }

            field.correction_results = correction_results
            df.append(correction_results)

        # Compute potential bad fields
        n_bad_intersections = np.zeros(self.n_fields, dtype=np.int8)
        if bad_intersections_keys:
            bad_fields = np.array(bad_intersections_keys)[:, :2].flatten()
            u, c = np.unique(bad_fields, return_counts=True)
            argsort = np.argsort(c)[::-1]
            u, c = u[argsort], c[argsort]
            n_bad_intersections[u] = c

        df = pd.DataFrame(df).set_index('name')
        df['n_bad_intersections'] = n_bad_intersections

        columns = [
            'shift', 'angle', 'gain', 'n_bad_intersections',
            'mean_recomputed_corr', 'min_recomputed_corr',# 'std_recomputed_corr',
            'n_intersections', 'mean_corr_intersections',# 'std_corr_intersections'
        ]

        return df[columns]


    # Export: SEG-Y
    def export_segy(self, path, method='traces', apply_angle=True, apply_gain=True, pad_width=10, pbar='t'):
        """ Export present fields with suggested corrections.
        Uses either trace headers or trace values to introduce corrections.
        In the first case, only vertical and gain corrections are applied.
        The second one uses interpolation and FFT-shift under the hood.

        `path` can contain the '$' symbol, which is replaced by the field name. Useful to save a lot of files.
        """
        for field in Notifier(pbar, desc='Exporting SEG-Y files')(self.fields):
            self._export_segy(field=field, path=path, method=method,
                              apply_angle=apply_angle, apply_gain=apply_gain, pad_width=pad_width)

    @staticmethod
    def _export_segy(field, path, method='traces', apply_angle=True, apply_gain=True, pad_width=10):
        """ Export one SEG-Y file. """
        #pylint: disable=protected-access
        # Prepare correction
        shift = -field.correction_results['shift']
        angle = -field.correction_results['angle'] if apply_angle else 0.0
        gain = 1/field.correction_results['gain'] if apply_gain else 1.0

        # Make a copy of a file. To make sure that it is not IBM floats, we copy data and headers separately
        path = field.make_path(path, name=field.name)
        FieldCollection._copy_segy(field, path)

        # Prepare dst memory map
        dst_loader = MemmapLoader(path)
        mmap_trace_headers_dtype = dst_loader._make_mmap_headers_dtype(['DelayRecordingTime'],
                                                                       endian_symbol=dst_loader.endian_symbol)
        mmap_trace_dtype = np.dtype([*mmap_trace_headers_dtype,
                                     ('data', dst_loader.mmap_trace_data_dtype, dst_loader.mmap_trace_data_size)])

        mmap = np.memmap(filename=path, mode='r+', shape=dst_loader.n_traces,
                         offset=dst_loader.file_traces_offset, dtype=mmap_trace_dtype)

        # Modify copied file
        if method == 'headers':
            added_delay = np.round(shift).astype(np.int16)
            mmap['DelayRecordingTime'] += added_delay
            mmap['data'] *= gain

        elif method == 'traces':
            data = field.load_slide(0)

            # Resample to MS
            arange = np.arange(data.shape[1], dtype=np.float32)
            arange_ms = np.arange(data.shape[1], step=(1 / field.sample_interval), dtype=np.float32)
            interpolator = lambda trace: np.interp(arange_ms, arange, trace)
            data = np.apply_along_axis(interpolator, 1, data)

            # Apply modifications
            data = np.pad(data, ((0, 0), (pad_width, pad_width)))
            for c in range(data.shape[0]):
                data[c] = modify_trace(data[c], shift=shift, angle=angle, gain=gain)
            data = data[:, pad_width:-pad_width]

            # Resample back to samples
            interpolator = lambda trace: np.interp(arange, arange_ms, trace)
            data = np.apply_along_axis(interpolator, 1, data)
            mmap['data'] = data
        return path

    @staticmethod
    def _copy_segy(field, path):
        """ Copy the data of SEG-Y file in float32 format, then copy trace headers. """
        data = field.geometry[:, :, :]
        field.geometry.array_to_segy(data, path=path, format=5, pbar=False)

        src_loader = field.geometry.loader
        src_mmap = np.memmap(field.path, mode='r', shape=src_loader.n_traces,
                             offset=src_loader.file_traces_offset, dtype=src_loader.mmap_trace_dtype)

        dst_loader = MemmapLoader(path)
        dst_mmap = np.memmap(path, mode='r+', shape=dst_loader.n_traces,
                             offset=dst_loader.file_traces_offset, dtype=dst_loader.mmap_trace_dtype)

        dst_mmap['headers'] = src_mmap['headers']
        return path

    # Export: horizons
    def export_horizons(self, path):
        """ Save horizons with applied corrections. """
        for horizon_name in self.horizons:
            self.export_horizon(horizon_name=horizon_name, path=path)

    def export_horizon(self, path, horizon_name=None, encoding=None):
        """ Save one horizon with applied corrections. """
        path = path.replace('$', horizon_name)
        df = self.horizons[horizon_name]

        depth_column = df['DEPTH'].copy()
        for field in self.fields:
            mask = df['FIELD_NAME'] == field.short_name
            depth_column[mask] += -field.correction_results['shift'] # TODO: adjust z-shift on angle

        out_df = df.copy()
        out_df['DEPTH'] = depth_column

        with open(path, 'w+', encoding=encoding) as file:
            file.write(horizon_name + '\n')
        out_df.to_csv(path, mode='a', index=False, header=False, sep='\t', encoding=encoding)
        return path


    # Visualize
    def show_lines(self, arrow_step=10, arrow_size=20, annotate_index=True, annotate_name=False):
        """ Display annotated shot lines on a 2d graph in CDP coordinates. """
        fig, ax = plt.subplots(figsize=(14, 8))

        depths = np.array([field.depth for field in self.fields])
        colors = ['black', 'firebrick', 'gold', 'limegreen', 'magenta'] * 25
        depth_to_color = dict(zip(sorted(np.unique(depths)), colors))

        # Data
        for i, field in enumerate(self.fields):
            color = depth_to_color[field.depth]
            values = field.geometry.headers[['CDP_X', 'CDP_Y']].values
            x, y = values[:, 0], values[:, 1]
            ax.plot(x, y, color)

            idx = x.size // 2
            arrow_step_ = min(arrow_step, idx - 0, x.size - idx - 1)
            ax.annotate('', size=arrow_size,
                        xytext=(x[idx-arrow_step_], y[idx-arrow_step_]),
                            xy=(x[idx+arrow_step_], y[idx+arrow_step_]),
                        arrowprops=dict(arrowstyle="->", color=color))

            if annotate_index or annotate_name:
                annotation = field.short_name if annotate_name else i
                ax.annotate(annotation, xy=(x[0], y[0]), size=12)

        # Annotations
        ax.set_title('2D profiles', fontsize=26)
        ax.set_xlabel('CDP_X', fontsize=22)
        ax.set_ylabel('CDP_Y', fontsize=22)
        ax.grid()
        fig.show()


    def show_bubblemap(self, savepath=None):
        """ Display annotated shot lines and their intersections on a 2d interactive graph in CDP coordinates. """
        fig = go.Figure()

        depths = np.array([field.depth for field in self.fields])
        colors = ['black', 'firebrick', 'gold', 'limegreen', 'magenta'] * 25
        depth_to_color = dict(zip(sorted(np.unique(depths)), colors))

        intersections_df = self.intersections_df()

        # Line for each SEG-Y
        for i, field in enumerate(self.fields):
            correction_results = field.correction_results
            values = field.geometry.headers[['CDP_X', 'CDP_Y']].values
            color_ = depth_to_color[field.depth]

            name_ = f'{i} : "{field.short_name}.sgy"'
            hovertemplate_ = (f' #{i} <br>'
                              f' FIELD : "{field.short_name}.sgy" <br>'
                              f' DEPTH : {field.depth} <br>'
                               ' CDP_X : %{x:,d} <br>'
                               ' CDP_Y : %{y:,d} <br>'
                               ' TSF : %{customdata} <br>'
                              f' MEAN INTERSECTION CORR : {correction_results["mean_corr_intersections"]:3.3f} <br>'
                              f' MEAN RECOMPUTED CORR : {correction_results["mean_recomputed_corr"]:3.3f}'
                               '<extra></extra>')

            step = 30
            fig.add_trace(go.Scatter(x=values[::step, 0], y=values[::step, 1],
                                     customdata=field.geometry.headers['TRACE_SEQUENCE_FILE'][::step],
                                     name=name_, hovertemplate=hovertemplate_,
                                     mode='lines',
                                     line=dict(color=color_, width=2)))

        # Markers on intersections
        for key, intersection in self.intersections.items():
            # Retrieve data
            i, j = key[:2]
            field_0, field_1 = intersection.field_0, intersection.field_1
            x, y = (intersection.coordinates_0 + intersection.coordinates_1) // 2

            matching_results = intersection.matching_results
            corr, shift = matching_results['corr'], matching_results['shift']
            angle, gain = matching_results['angle'], matching_results['gain']

            # HTML things
            name_ = f'"{field_0.short_name}.sgy" X "{field_1.short_name}.sgy"'
            hovertemplate_ = (f' ({i}, {j}) <br>'
                              f' {name_} <br>'
                               ' CDP_X : %{x:,d} <br>'
                               ' CDP_Y : %{y:,d} <br>'
                              f' BEST_CORR   : {corr:3.3f} <br>'
                              f' BEST_PCORR  : {(1 + corr)/2:3.3f} <br>'
                              f' SHIFT       : {shift:3.3f} <br>'
                              f' ANGLE       : {angle:3.3f} <br>'
                              f' GAIN        : {gain:3.3f} <extra></extra>')

            size_ = 4 + (1 - corr) * 5
            color_ = 'red' if intersections_df.loc[[key]]['suspicious'].all() else 'green'

            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                     name=name_, hoverlabel={},
                                     hovertemplate=hovertemplate_,
                                     showlegend=False,
                                     marker=dict(size=size_, color=color_)))


        fig.update_layout(title=f'2D SEG-Y<br>{len(self.intersections)} intersections',
                          xaxis_title='CDP_X', yaxis_title='CDP_Y',
                          width=1200, height=500, margin=dict(l=10, r=10, t=40, b=10))
        fig.show()

        if savepath is not None:
            fig.write_html(savepath)

    def show_histogram(self, keys=('corr', 'shift', 'angle', 'gain'), **kwargs):
        """ Display histogram of mis-tie values across all intersections. """
        data = [np.array(self.get_matched_value(key)) for key in keys]
        data = [item[~np.isnan(item)] for item in data]

        kwargs = {
            'title': list(keys),
            'xlabel': list(keys),
            'combine': 'separate',
            'ncols': 4,
            **kwargs
        }
        return plot(data, mode='histogram', **kwargs)



@njit
def distribute_misties(a, b, n, skip_index=-1, max_iters=100, alpha=0.75, tolerance=0.00001):
    """ Distribute misties `b` on intersections `a` over `n` fields by a iterative optimization procedure.
    Bishop, Nunns "`Correcting amplitude, time, and phase mis-ties in seismic data
    <https://www.researchgate.net/publication/249865260>`_"

    Probably, not as fast as highly-optimized linear solvers, but allows for more flexibility.
    Also, usual run time is less than 1ms, so it is fast enough anyways.

    Parameters
    ----------
    a : np.ndarray
        (M, 2)-shaped matrix that describes geometry of intersections.
        Each row is a pair of indices of intersecting lines.
    b : np.ndarray
        (M,)-shaped vector with misties on each intersection.
    n : int
        Number of lines in the intersections.
        For each of them, we compute a distributed mistie as a result of this function.

    Example
    -------
    To reproduce example from the original paper, one can use::
        a = np.array([
            [1, 2],
            [0, 2],
            [0, 1],
            [0, 3],
            [1, 3],
            [2, 3],
        ])
        b = np.array([21, 1, -19, 1, 20, -2])
        n = 4
    """
    x = np.zeros(n)
    errors = np.empty(max_iters)

    for iteration in range(max_iters):
        # Stop condition: no further decrease in error
        xk, xl = x[a[:, 0]], x[a[:, 1]]
        errors[iteration] = ((b - (xk - xl)) ** 2).mean() ** (1 / 2)
        if iteration != 0:
            stop_condition = (errors[iteration - 1] - errors[iteration]) / errors[iteration - 1]
            if stop_condition < tolerance or errors[iteration] == 0.0:
                break
        else:
            if errors[0] == 0:
                break

        # Compute next iteration of solution
        x_next = x.copy()

        for j in range(n):
            if j == skip_index:
                continue

            d, s = 0, 0.0 # number of intersections / sum of discrepancies
            for i, idx in enumerate(a):
                k, l = idx

                if k == j:
                    s += b[i] - (x[k] - x[l])
                    d += 1
                if l == j:
                    s += (x[k] - x[l]) - b[i]
                    d += 1

            if d != 0:
                x_next[j] += (alpha / d) * s

        x = x_next

    return x, errors[:iteration+1]
