""" Well class to describe core logs. """
import os

import numpy as np
import pandas as pd
import scipy

from lasio import LASFile

from ...plotters import plot



class Well:
    """ A class to hold information about core logs and perform simple processing operations.
    Main idea is to initialize the well instance from either LAS file or checkshot, and then combine multiple instances
    into one with all (possibly, interpolated) available logs.

    TODO: write more docs
    """
    def __init__(self, storage, field=None, name=None, **kwargs):
        self.path = None
        self.name = name
        self.field = field

        # Attributes, available for wells, matched with seismic
        self.vertical = None
        self.location = None
        self.points = None
        self.bboxes = {} # log name to its bbox

        # pylint: disable=import-outside-toplevel
        from ...field import Field

        if isinstance(storage, str):
            path = storage
            self.path = path
            self.name = os.path.basename(path).split('.')[0]

            with open(path, mode='r', encoding='utf-8') as file:
                line = file.readline()

            if 'LAS' in line or 'version' in line.lower():
                # LAS format: independent of software written by
                self.from_las(path, **kwargs)
                self.format = 'las'
            elif 'Petrel checkshots format' in line:
                # Checkshots can be saved by different software products and have different types
                self.from_petrel_checkshot(path, **kwargs)
                self.format = 'petrel_checkshot'
            else:
                # TODO: add more types of supported well files
                raise TypeError(f'Unknown type of file! first line is: {line}')
        elif isinstance(storage, pd.DataFrame):
            self.data = storage
            self.format = 'dataframe'
        elif isinstance(storage, Field):
            self.name = name or f'seismicdata on {storage.short_name}'
            self.from_field(storage, **kwargs)
            self.field = storage
            self.format = 'field'

    @property
    def keys(self):
        """ Available logs. """
        return list(self.data.columns)

    @property
    def n_logs(self):
        """ Number of available logs. """
        return len(self.keys)


    # Initialization from different containers
    def from_las(self, path, **kwargs):
        """ Initialize instance from LAS file. """
        self.lasfile = LASFile(path)
        self.data = self.lasfile.df().rename_axis('DEPTH')

    def from_petrel_checkshot(self, path, **kwargs):
        """ Initialize instance from a checkshot, saved by Petrel software. """
        #pylint: disable='anomalous-backslash-in-string
        self.data = pd.read_csv(path, skiprows=14, header=None, sep='\s+',
                                names=['X', 'Y', 'Z', 'TWT', 'MD', 'Well', 'AvgV', 'IV'],
                                usecols=['X', 'Y', 'Z', 'TWT', 'MD', 'AvgV', 'IV'])
        self.data = self.data.set_index('MD')
        self.data['TWT'] = -self.data['TWT']
        self.data['Z'] = -self.data['Z']

    def from_field(self, field, location=None, column_name='DATA', **kwargs):
        """ Initialize instance from a known field geometry: available pseudo-logs are `TIME` and `SAMPLES`. """
        samples = np.arange(0, field.depth, 1, dtype=np.int32)
        seismic_time = samples * field.sample_interval
        data = {'SAMPLES': samples, 'TIME': seismic_time}

        if location is not None:
            data[column_name] = field.geometry[location[0], location[1], :]

        self.data = pd.DataFrame(data).set_index('SAMPLES')


    # Redefined protocols
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        # TODO: add bbox computation, if matched to seismic
        self.data.__setitem__(key, value)

    def __getattr__(self, key):
        if not key.endswith('state__'):
            return getattr(self.data, key)
        raise AttributeError


    # Methods to combine multiple wells into one
    def merge_data(self, other, key_self='index', key_other='index', inplace=True, kind='linear', prefix=''):
        """ Merge data from other well instance or dataframe.
        Resamples the data of `other` to match points in `self` in one of the columns.
        """
        self_df = self.data.copy() if inplace is False else self.data
        self_x = self_df.index if key_self=='index' else self_df[key_self]

        other_df = other.data if isinstance(other, Well) else other
        other_x = other_df.index if key_other=='index' else other_df[key_other]
        other_df.reset_index(inplace=True)

        for column in set(other_df.columns) - {key_other}:
            if column in self.keys:
                if not prefix:
                    raise ValueError('!!.')
                column_ = prefix + column
            else:
                column_ = column

            self_df[column_] = np.interp(self_x, other_x, other_df[column])

        if inplace:
            return self
        return Well(self_df, name='+'+self.name, field=self.field)


    def match_to_seismic(self, columns=('X', 'Y', 'TWT'), field=None, well=None):
        """ Match `self` to a provided `well` or to default seismic pseudo-well. """
        field = field or self.field
        assert field is not None, 'Making points requires field reference in either `Well` instance or method call!'

        #
        cdp_xy_values = self.data[list(columns[:2])].values
        ordinal_xy_values = field.lines_to_ordinals(field.cdp_to_lines(cdp_xy_values).astype(np.int32))

        data = self.data.copy()
        data[['INLINE_3D', 'CROSSLINE_3D']] = ordinal_xy_values

        #
        matched_well = well or Well(storage=field, location=ordinal_xy_values[0], name=f'matched_{self.name}')
        matched_well.merge_data(other=data, key_self='TIME', key_other=columns[-1])
        matched_well.data = matched_well.data.astype({'INLINE_3D': np.int32, 'CROSSLINE_3D': np.int32})

        uniques = np.unique(ordinal_xy_values, axis=0)
        matched_well.vertical = bool(len(uniques) == 1)
        matched_well.location = ordinal_xy_values[0]

        points = matched_well.data.reset_index()[['INLINE_3D', 'CROSSLINE_3D', 'SAMPLES']]
        matched_well.points = points.values.astype(np.int32)

        for column in matched_well.keys:
            i_min, x_min = np.min(ordinal_xy_values, axis=0)
            i_max, x_max = np.max(ordinal_xy_values, axis=0)
            d_min, d_max = self.get_bounds(matched_well.data[column])

            matched_well.bboxes[column] = np.array([[i_min, i_max],
                                                    [x_min, x_max],
                                                    [d_min, d_max]],
                                                   dtype=np.int32)
        return matched_well


    # Work with present logs
    def add_seismic_trace(self, field, name):
        """ Add seismic trace as a pseudo-log. """
        seismic_trace = field.mmap[self.points[:, 0], self.points[:, 1], self.points[:, 2]]
        seismic_trace = seismic_trace.astype(np.float32)
        self.data[name] = seismic_trace

    def compute_reflectivity(self, impedance_log='AI', name='R'):
        """ Compute reflectivity from available impedance log. """
        impedance = self.data[impedance_log].values
        reflectivity = self._compute_reflectivity(impedance)
        self.data[name] = reflectivity

    @staticmethod
    def _compute_reflectivity(impedance, fill_value=0):
        reflectivity = impedance.copy()
        reflectivity[1:] = ((impedance[1:] - impedance[:-1]) /
                            (impedance[1:] + impedance[:-1]))
        reflectivity[0:1] = fill_value
        return reflectivity

    def compute_synthetic(self, wavelet, reflectivity_log='R', name='SYNTHETIC'):
        """ Compute synthetic trace from available reflectivity log and provided wavelet. """
        reflectivity = self.data[reflectivity_log].values
        synthetic = self._compute_synthetic(reflectivity, wavelet)
        self.data[name] = synthetic

    @staticmethod
    def _compute_synthetic(reflectivity, wavelet):
        reflectivity = np.nan_to_num(reflectivity, nan=0.0)
        synthetic = np.convolve(reflectivity, wavelet, mode='same')
        return synthetic

    def compute_filtered_log(self, log, order=5, frequency=60, btype='lowpass', name=None):
        """ Apply filtration to a given log. """
        array = self.data[log].values
        filtered_array = self._compute_filtered_log(array, order=order, frequency=frequency,
                                                    btype=btype, fs=self.field.sample_rate)
        self.data[name] = filtered_array

        if log in self.bboxes:
            bbox = self.bboxes[log].copy()
            bbox[-1] += [1, -1]
            filtered_array[:bbox[-1][0]] = np.nan
            filtered_array[bbox[-1][1]:] = np.nan
            self.bboxes[name] = bbox

    @staticmethod
    def _compute_filtered_log(array, despike=None, order=5, frequency=60, btype='lowpass', fs=500):
        if despike is not None:
            array = scipy.signal.medfilt(array, despike)

        sosfilt = scipy.signal.butter(order, frequency, btype=btype, fs=fs, output='sos')
        filtered_array = scipy.signal.sosfiltfilt(sosfilt, np.nan_to_num(array))
        filtered_array[filtered_array <= np.nanmin(array)] = np.nan
        return filtered_array

    def compute_sampling_frequency(self):
        """ Sampling frequency of the well data. Useful for frequency-domain filtrations. """
        return (1e6 * 0.3048) / self.DT.mean()


    # Methods for Batch loads
    def compute_overlap_mask(self, mask_bbox, log='AI'):
        """ Compute a depth-wise mask of overlap between well log and a given mask bbox. """
        log_bbox = self.bboxes[log]
        overlap_min = np.maximum(mask_bbox[:, 0], log_bbox[:, 0])
        overlap_max = np.minimum(mask_bbox[:, 1], log_bbox[:, 1] + 1)

        if not (overlap_max - overlap_min > 0).all():
            return None

        mask = (self.points[:, -1] >= mask_bbox[-1][0]) & (self.points[:, -1] < mask_bbox[-1][1])
        if self.vertical is False:
            mask &= (self.points[:, 0] >= mask_bbox[0][0]) & (self.points[:, 0] < mask_bbox[0][1])
            mask &= (self.points[:, 1] >= mask_bbox[1][0]) & (self.points[:, 1] < mask_bbox[1][1])
        return mask

    def compute_overlap_size(self, mask_bbox, log='AI'):
        """ Compute the number of pixels in a well within a given mask bbox. """
        overlap_mask = self.compute_overlap_mask(mask_bbox=mask_bbox, log=log)
        return 0 if overlap_mask is None else overlap_mask.sum()

    def add_to_mask(self, mask, locations, log='AI', **kwargs):
        """ Add values from log to a mask in a given location. """
        mask_bbox = np.array([[slc.start, slc.stop] for slc in locations], dtype=np.int32)
        overlap_mask = self.compute_overlap_mask(mask_bbox=mask_bbox, log=log)

        if overlap_mask is not None:
            points = self.points[overlap_mask] - mask_bbox[:, 0]
            mask[points[:, 0], points[:, 1], points[:, 2]] = self.data[log].values[overlap_mask]
        return mask


    # Depth-wise filtration of logs
    @staticmethod
    def get_bounds(array):
        """ Return the index of the first and the last meaningful elements in array.
        Meaningful means non-nan and non-constant.
        """
        diff = np.diff(array, prepend=array[0])
        diff = np.nan_to_num(diff, copy=False, nan=0.0)
        mask = diff != 0
        return np.argmax(mask), len(array) - np.argmax(mask[::-1])

    def filter(self, exclude=('INLINE_3D', 'CROSSLINE_3D', 'TWT', 'DEPTH', 'TIME')):
        """ Fill insignificant values on left/right bounds of each log with nans. """
        for column in set(self.keys) - set(exclude):
            d_min, d_max = self.get_bounds(self.data[column])
            self.data[column][:d_min] = np.nan
            self.data[column][d_max:] = np.nan


    # Visualization
    @property
    def short_name(self):
        """ Name without extension. """
        if self.name is not None:
            return self.name.split('.')[0]
        return None

    def __repr__(self):
        return f"""<Well `{self.name}` for `{self.field.short_name}` at {hex(id(self))}>"""

    def plot(self, logs='all', layout='horizontal', dropkeys=None, zoom=None, combine='separate', **kwargs):
        """ Show log curves. """
        horizontal_layout = layout.startswith('h')

        # Parse logs to use
        logs = self.keys if logs == 'all' else logs
        logs = [logs] if isinstance(logs, str) else logs
        for key in dropkeys or []:
            logs.remove(key)
        keys = logs
        n_subplots = len(logs) if combine == 'separate' else 1

        # Parse limits
        index_label = self.data.index.name
        if zoom is None:
            index_lim = [0, self.index[-1]]
        else:
            index_lim = [zoom.start, zoom.stop]

        # Default plot parameters
        kwargs = {
            'suptitle': f'Well `{self.name}`',
            'title': keys,
            'mode': 'curve',
            'combine': combine,
            'label': keys if combine == 'overlay' else None,
            **kwargs
        }

        # Build layout
        if horizontal_layout:
            data = [(self.index, self[key]) for key in keys]
            line_method = 'axvline'

            kwargs = {
                'xlabel': index_label,
                'xlabel_size': 20,
                'xlim': [index_lim] * n_subplots if combine == 'separate' else index_lim,
                'ylabel': '',
                'window': 100, 'alpha': 0.7,
                'ratio': 0.6,
                'ncols': min(2, n_subplots),
                **kwargs
            }
        else:
            data = [(self[key], self.index) for key in keys]
            line_method = 'axhline'

            kwargs = {
                'ylabel': index_label,
                'ylabel_size': 20,
                'ylim': [index_lim] * n_subplots if combine == 'separate' else index_lim,
                'xlabel': '',
                'ratio': 0.4,
                'ncols': n_subplots,
                **kwargs
            }


        plotter = plot(data, **kwargs)
        for (column, subplot) in zip(keys, plotter.subplots):
            ax = subplot.ax
            if ax.axison is False:
                continue
            if not horizontal_layout:
                ax.invert_yaxis()

            getattr(ax, line_method)(self.index[ 0], linestyle='--', color='orange')
            getattr(ax, line_method)(self.index[-1], linestyle='--', color='orange')

            if combine == 'separate' and self.bboxes.get(column) is not None:
                d_min, d_max = self.bboxes[column][-1]
                getattr(ax, line_method)(d_min, linestyle='--', color='red')
                getattr(ax, line_method)(d_max, linestyle='--', color='red')
        return plotter


class MatchedWell(Well):
    """ Automatic combination of multiple data sources.
    TODO: extend the documentation
    """
    def __init__(self, storage, additional_paths, field=None, name=None, data_name='DATA', **kwargs):
        super().__init__(storage=field, field=field, name=name, **kwargs)

        main_well = Well(storage, field=field, **kwargs)
        main_well.data['X'] //= 2 # bug in labeling
        main_well.data['Y'] //= 2 # bug in labeling

        additional_paths = [additional_paths] if isinstance(additional_paths, str) else additional_paths
        additional_wells = [Well(path, field=field) for path in additional_paths]
        for well in additional_wells:
            main_well.merge_data(well, key_self='index', key_other='index', inplace=True)

        main_well.match_to_seismic(field=field, well=self)
        main_well.add_trace(field=field, name=data_name)
