""" Charisma mixin for saving and loading data in CHARISMA-compatible format. """
import os

import numpy as np
import pandas as pd

from .functions import make_interior_points_mask

class CharismaMixin:
    """ Methods for saving and loading data in CHARISMA-compatible format. """
    #pylint: disable=redefined-builtin

    # CHARISMA: default seismic format of storing surfaces inside the 3D volume
    CHARISMA_SPEC = ['inline_marker', '_', 'INLINE_3D', 'xline_marker', '__', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y', 'DEPTH']

    # REDUCED_CHARISMA: CHARISMA without redundant columns
    REDUCED_CHARISMA_SPEC = ['INLINE_3D', 'CROSSLINE_3D', 'DEPTH']

    @property
    def field_reference(self):
        """ Reference to Field for applying methods. """
        return self.field if hasattr(self, 'field') else self

    # Load and save data in charisma-compatible format
    def load_charisma(self, path, dtype=np.int32, format='points', fill_value=np.nan,
                      transform=True, verify=True, recover_lines=False, **kwargs):
        """ Load data from path to either CHARISMA or REDUCED_CHARISMA csv-like file.

        Parameters
        ----------
        path : str
            Path to a file to import data from.
        dtype : data-type
            Output dtype.
        format : str
            Output array format, can be 'points' or 'matrix'.
            If format is 'points' then return data as ndarray of (ilines_len, xlines_len, depth) with shape (N, 3).
            If format is 'matrix' then return data as ndarray of (ilines_len, xlines_len) shape.
        fill_value : int or float
            Value to place into blank spaces.
        transform : bool
            Whether transform from line coordinates (ilines, xlines) to cubic system.
        verify : bool
            Whether to remove points outside of the cube range.
        """
        _ = kwargs
        path = self.field_reference.make_path(path, makedirs=False)

        # Load data as a points array from a file
        with open(path, encoding='utf-8') as file:
            line_len = len(file.readline().split())
        if line_len == len(self.REDUCED_CHARISMA_SPEC):
            names = self.REDUCED_CHARISMA_SPEC
        elif line_len >= len(self.CHARISMA_SPEC):
            names = self.CHARISMA_SPEC
        else:
            raise ValueError('Data must be in CHARISMA or REDUCED_CHARISMA format.')

        df = pd.read_csv(path, sep=r'\s+', names=names, usecols=self.REDUCED_CHARISMA_SPEC)
        if recover_lines:
            df = self.recover_lines_from_cdp(df)
        df.sort_values(self.REDUCED_CHARISMA_SPEC, inplace=True)
        points = df.values

        # Transform and verify points
        if transform:
            points = self.field_reference.geometry.lines_to_ordinals(points)

        if verify:
            mask = make_interior_points_mask(points, self.field_reference.shape)
            points = points[mask]

        # Set datatype
        if np.issubdtype(dtype, np.integer):
            points = np.round(points)

        points = points.astype(dtype)

        if format == 'points':
            return points

        # Make a matrix from points and return
        matrix = np.full(shape=self.field_reference.shape[:2], fill_value=fill_value, dtype=dtype)
        matrix[points[:, 0].astype(np.int32), points[:, 1].astype(np.int32)] = points[:, 2]

        return matrix

    def dump_charisma(self, data, path, format='points', name=None, transform=None):
        """ Save data as (N, 3) array of points to a disk in CHARISMA-compatible format.

        Parameters
        ----------
        data : ndarray
            Array of (N, 3) shape or (i_lines, x_lines) shape.
        path : str
            Path to a file to save array to.
        format : str
            Input array format, can be 'points' or 'matrix'.
            If format is 'points' then input data is a ndarray of (ilines_len, xlines_len, depth) with shape (N, 3).
            If format is 'matrix' then input data is a ndarray of (ilines_len, xlines_len) shape.
        name : str
            Dumped object name.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        """
        path = self.field_reference.make_path(path, name=name or self.name)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if format != 'points':
            # Convert data to points array
            idx = np.nonzero(~np.isnan(data))

            data = np.hstack([idx[0].reshape(-1, 1),
                                idx[1].reshape(-1, 1),
                                data[idx[0], idx[1]].reshape(-1, 1)])

        points = self.field_reference.geometry.ordinals_to_lines(data)

        # Additional transform
        points = points if transform is None else transform(points)

        # Dump a charisma file
        df = pd.DataFrame(points, columns=self.REDUCED_CHARISMA_SPEC)
        df.sort_values(['INLINE_3D', 'CROSSLINE_3D'], inplace=True)
        df = df.astype({'INLINE_3D': np.int32, 'CROSSLINE_3D': np.int32, 'DEPTH': np.float32})
        df.to_csv(path, sep=' ', columns=self.REDUCED_CHARISMA_SPEC, index=False, header=False)

    @classmethod
    def is_charisma_like(cls, path, bad_extensions=None, size_threshold=100):
        """ Check if the path looks like the charisma file.

        Parameters
        ----------
        path : str
            Path of file to check.
        bad_extensions : list, optional
            If provided, then list of extensions to consider file not charisma-like.
        size_threshold : number
            If file size in kilobytes is less, than the threshold, then file is considered not charisma-like.
        """
        bad_extensions = bad_extensions or []
        bad_extensions.extend(['.py', '.ipynb', '.ckpt',
                               '.png', '.jpg',
                               '.log', '.txt', '.torch'])

        try:
            if os.path.isdir(path):
                return False

            if max(path.endswith(ext) for ext in bad_extensions):
                return False

            if (os.path.getsize(path) / 1024) < size_threshold:
                return False

            with open(path, encoding='utf-8') as file:
                line = file.readline()
                n = len(line.split(' '))

            is_reduced_charisma = (n == len(cls.REDUCED_CHARISMA_SPEC))
            is_charisma = (n >= len(cls.CHARISMA_SPEC) and 'INLINE' in line)
            return is_reduced_charisma or is_charisma

        except UnicodeDecodeError:
            return False


    def recover_lines_from_cdp(self, df):
        """ Fix broken iline and crossline coordinates.
        If coordinates are out of the cube, 'iline' and 'xline' will be infered from 'cdp_x' and 'cdp_y'. """
        i_bounds = [self.field.shifts[0], self.field.shifts[0] + self.field.shape[0]]
        x_bounds = [self.field.shifts[1], self.field.shifts[1] + self.field.shape[1]]

        i_mask = np.logical_or(df['INLINE_3D'] < i_bounds[0], df['INLINE_3D'] >= i_bounds[1])
        x_mask = np.logical_or(df['CROSSLINE_3D'] < x_bounds[0], df['CROSSLINE_3D'] >= x_bounds[1])

        _df = df[np.logical_or(i_mask, x_mask)]

        coords = np.rint(self.field.geometry.cdp_to_lines(_df[['CDP_X', 'CDP_Y']].values)).astype(np.int32)
        df.loc[np.logical_or(i_mask, x_mask), ['INLINE_3D', 'CROSSLINE_3D']] = coords

        return df
