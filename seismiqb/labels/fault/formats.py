""" Mixins to deal with fault storing files. """

import os
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .postprocessing import split_array
from ...utils import CharismaMixin, SQBStorage, make_interior_points_mask

class FaultSticksMixin(CharismaMixin):
    """ Mixin to load, process and dump FaultSticks files. """
    FAULT_STICKS_SPEC = ['inline_marker', 'INLINE_3D', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y', 'DEPTH', 'name', 'number']
    REDUCED_FAULT_STICKS_SPEC = ['INLINE_3D', 'CROSSLINE_3D', 'DEPTH', 'name', 'number']

    @classmethod
    def read_df(cls, path):
        """ Automatically detect format of csv-like file and create pandas.DataFrame from FaultSticks/CHARISMA file. """
        with open(path, encoding='utf-8') as file:
            line_len = len([item for item in file.readline().split(' ') if len(item) > 0])

        if line_len == 0:
            return pd.DataFrame({})

        if line_len == 3:
            names = cls.REDUCED_CHARISMA_SPEC
        elif line_len == 5:
            names = cls.REDUCED_FAULT_STICKS_SPEC
        elif line_len == 8:
            names = cls.FAULT_STICKS_SPEC
        elif line_len >= 9:
            names = cls.CHARISMA_SPEC
        else:
            raise ValueError('Fault labels must be in FAULT_STICKS, CHARISMA or REDUCED_CHARISMA format.')

        return pd.read_csv(path, sep=r'\s+', names=names)

    def split_df_into_sticks(self, df, return_direction=False):
        """ Group nodes in FaultSticks dataframe into sticks.

        Parameters
        ----------
        df : pandas.DataFrame
            FaultSticks
        return_direction : bool, optional
            Whether return direction of fault, by default False.

        Returns
        -------
        pandas.Series or (pandas.Series, int)
            Sequence of stick nodes and (optionally) direction of the fault.
        """
        col, direction = None, None

        ilines_diff = sum(df['INLINE_3D'][1:].values - df['INLINE_3D'][:-1].values == 0)
        xlines_diff = sum(df['CROSSLINE_3D'][1:].values - df['CROSSLINE_3D'][:-1].values == 0)
        if ilines_diff > xlines_diff: # Use iline as an index
            col = 'INLINE_3D'
            direction = 0
        else: # Use xline as an index
            col = 'CROSSLINE_3D'
            direction = 1

        if 'number' in df.columns: # Dataframe has stick index
            col = 'number'

        if col is None:
            raise ValueError('Wrong format of sticks: there is no column to group points into sticks.')

        df = df.sort_values('DEPTH')
        sticks = df.groupby(col).apply(lambda x: x[self.COLUMNS].values).reset_index(drop=True)

        return (sticks, direction) if return_direction else sticks

    def remove_broken_sticks(self, sticks):
        """ Remove sticks with one node and remove sticks from fault with one stick. """
        # Remove sticks with one node.
        mask = sticks.apply(len) > 1
        if not mask.all():
            warnings.warn(f'{self.name}: Fault has one-point sticks.')
        sticks = sticks.loc[mask]

        # Filter faults with one stick.
        if len(sticks) == 1:
            warnings.warn(f'{self.name}: Fault has an only one stick')
            sticks = pd.Series()
        elif len(sticks) == 0:
            warnings.warn(f'{self.name}: Empty file')
            sticks = pd.Series()

        return sticks

    def load_fault_sticks(self, path, transform=True, verify=True,
                          recover_lines=True, remove_broken_sticks=False, **kwargs):
        """ Get sticks from FaultSticks file.

        Parameters
        ----------
        path : str
            Path to file.
        transform : bool, optional
            Whether transform from cubic coordinates to line or not, by default True
        verify : bool, optional
            Filter points outside of the cube, by default True
        recover_lines : bool, optional
            Fill broken iline/crossline coordinate (extremely large values) from CDP, by default True
        remove_broken_sticks : bool, optional
            Whether remove sticks with one node and remove sticks from fault with one stick,
            by default False
        """
        if isinstance(path, str):
            df = self.read_df(path)
        else:
            df = path

        if len(df) == 0:
            self._sticks = [[]]
            self.direction = 1
            return

        if recover_lines and 'CDP_X' in df.columns:
            df = self.recover_lines_from_cdp(df)

        points = df[self.REDUCED_CHARISMA_SPEC].values

        if transform:
            points = self.field_reference.geometry.lines_to_ordinals(points)
        df[self.REDUCED_CHARISMA_SPEC] = np.round(points).astype(np.int32)

        if verify:
            mask = make_interior_points_mask(points, self.field_reference.shape)
            df = df.iloc[mask]

        if len(df) == 0:
            self._sticks = None
            return

        sticks, direction = self.split_df_into_sticks(df, return_direction=True)
        if remove_broken_sticks:
            sticks = self.remove_broken_sticks(sticks)

        # Order sticks with respect of fault direction. Is necessary to perform following triangulation.
        if len(sticks) > 1:
            pca = PCA(1)
            coords = pca.fit_transform(np.array([stick[0][:2] for stick in sticks.values]))
            indices = np.array([i for _, i in sorted(zip(coords, range(len(sticks))))])
            sticks = sticks.iloc[indices]

        self._sticks = sticks.values

        # fix several slides sticks
        if direction is not None:
            ptp = np.array([np.ptp(stick[:, direction]) for stick in self.sticks])
            if (ptp > 2).any():
                warnings.warn(f"{self.name}: there sticks on several slides in both directions")

            for stick in self.sticks[np.logical_and(ptp > 0, ptp <= 2)]:
                stick[:, direction] = stick[0, direction]

        self.direction = direction
        self.stick_orientation = 2

    def dump_fault_sticks(self, path):
        """ Dump fault sticks into FaultSticks format. """
        path = self.field.make_path(path, name=self.field.short_name, makedirs=False)

        sticks_df = []
        for stick_idx, stick in enumerate(self.sticks):
            stick = self.field.geometry.ordinals_to_lines(stick).astype(int)
            cdp = self.field.geometry.lines_to_cdp(stick[:, :2])
            df = {
                'inline_marker': 'INLINE-',
                'INLINE_3D': stick[:, 0],
                'CROSSLINE_3D': stick[:, 1],
                'CDP_X': cdp[:, 0],
                'CDP_Y': cdp[:, 1],
                'DEPTH': stick[:, 2],
                'name': os.path.basename(path),
                'number': stick_idx
            }
            sticks_df.append(pd.DataFrame(df))
        sticks_df = pd.concat(sticks_df)
        sticks_df.to_csv(path, header=False, index=False, sep=' ')

    def show_file(self):
        """ Show content of the initial FaultSticks file as a text. """
        with open(self.path, encoding='utf-8') as f:
            print(f.read())

    @classmethod
    def check_format(cls, path, verbose=False):
        """ Find errors in fault file.

        Parameters
        ----------
        path : str
            Path to file or glob expression
        verbose : bool
            Response if file is successfully read.
        """
        for filename in glob.glob(path):
            if os.path.splitext(filename)[1] == '.dvc':
                continue
            try:
                df = cls.read_df(filename)
                sticks = cls.split_df_into_sticks(cls, df)
            except ValueError:
                print(filename, ': wrong format')
            else:
                if 'name' in df.columns and len(df.name.unique()) > 1:
                    print(filename, ': file must be splitted.')
                    continue

                if len(sticks) == 1:
                    print(filename, ': fault has an only one stick')
                    continue

                if any(len(item) == 1 for item in sticks):
                    print(filename, ': fault has one point stick')
                    continue
                mask = sticks.apply(lambda x: len(np.unique(np.array(x)[:, 2])) == len(x))
                if not mask.all():
                    print(filename, ': fault has horizontal parts of sticks.')
                    continue

                if verbose:
                    print(filename, ': OK')

    @classmethod
    def split_charisma(cls, path):
        """ Split file with multiple faults (indexed by 'name' column) into separate dataframes. """
        df = cls.read_df(path)
        if 'name' in df.columns:
            return dict(list(df.groupby('name')))
        return {path: df}

    @classmethod
    def _fault_to_csv(cls, df, dst):
        """ Save the fault to csv. """
        df.to_csv(os.path.join(dst, df.name), sep=' ', header=False, index=False)


class FaultSerializationMixin:
    """ Mixin for npy/npz storage of fault components (points, sticks, nodes, simplices). """
    def load_npz(self, path, transform=False):
        """ Load fault points, nodes and sticks from npz file. """
        npzfile = np.load(path, allow_pickle=False)

        sticks = npzfile.get('sticks')
        sticks_labels = npzfile.get('sticks_labels')

        self.from_dict({
            'points': npzfile.get('points'),
            'nodes': npzfile.get('nodes'),
            'simplices': npzfile.get('simplices'),
            'sticks': self._labeled_array_to_sticks(sticks, sticks_labels) if sticks is not None else None,
        }, transform=transform)

        direction = npzfile.get('direction')
        if direction is not None:
            direction = int(direction)
        self.direction = direction

    def load_npy(self, path):
        """ Load fault points from npy file. """
        points = np.load(path, allow_pickle=False)
        self._points = points

    def dump_npz(self, path, attributes_to_create=None):
        """ Dump fault to npz. """
        path = self.field.make_path(path, name=self.short_name, makedirs=False)

        if attributes_to_create:
            if isinstance(attributes_to_create, str):
                attributes_to_create = [attributes_to_create]
            for item in attributes_to_create:
                getattr(self, item)

        kwargs = {'direction': self.direction}
        if self.has_component('sticks'):
            sticks, sticks_labels = self._sticks_to_labeled_array(self.sticks)
            kwargs['sticks'] = sticks
            kwargs['sticks_labels'] = sticks_labels

        for item in ['points', 'nodes', 'simplices']:
            if self.has_component(item):
                kwargs[item] = getattr(self, item)

        np.savez(path, **kwargs)


    def load_sqb(self, path):
        """ Load fault from SQB file. """
        storage = SQBStorage(path)
        if storage.get('type') != 'fault':
            raise TypeError('SQB storage is not marked as fault!')

        self.from_dict({key : storage[key] for key in ['points', 'nodes', 'simplices', 'sticks']})
        self.direction = storage['direction']

    def dump_sqb(self, path):
        """ Dump fault to SQB file. """
        storage = SQBStorage(path)
        storage.update({
            'type': 'fault',
            'points': self.points,
            'nodes': self.nodes,
            'simplices': self.simplices,
            'sticks': self.sticks,
            'direction': self.direction
        })


    def _sticks_to_labeled_array(self, sticks):
        """ Auxilary method to dump fault into npz with allow_pickle=False. """
        labels = sum([[i] * len(item) for i, item in enumerate(sticks)], [])
        return np.concatenate(sticks), labels

    def _labeled_array_to_sticks(self, sticks, labels):
        """ Auxilary method to dump fault into npz with allow_pickle=False. """
        sticks = split_array(sticks, labels)
        array = np.empty(len(sticks), dtype=object)
        for i, item in enumerate(sticks):
            array[i] = item
        return array
