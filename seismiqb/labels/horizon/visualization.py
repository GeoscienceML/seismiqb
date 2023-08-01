""" Mixin for horizon visualization. """
from copy import copy
from textwrap import dedent

import numpy as np
from scipy.spatial import Delaunay

from ..mixins import VisualizationMixin
from ...plotters import show_3d
from ...utils import AugmentedList, DelegatingList, filter_simplices



class HorizonVisualizationMixin(VisualizationMixin):
    """ Methods for textual and visual representation of a horizon. """
    #pylint: disable=protected-access
    def __repr__(self):
        return f"""<Horizon `{self.name}` for `{self.field.short_name}` at {hex(id(self))}>"""

    def __str__(self):
        msg = f"""
        Horizon {self.name} for {self.field.short_name} loaded from {self.format}
        Ilines range:      {self.i_min} to {self.i_max}
        Xlines range:      {self.x_min} to {self.x_max}
        Depth range:       {self.d_min} to {self.d_max}
        Depth mean:        {self.d_mean:.6}
        Depth std:         {self.d_std:.6}

        Length:            {len(self)}
        Perimeter:         {self.perimeter}
        Coverage:          {self.coverage:3.5}
        Solidity:          {self.solidity:3.5}
        Num of holes:      {self.number_of_holes}
        """

        if self.is_carcass:
            msg += f"""
        Unique ilines:     {self.carcass_ilines}
        Unique xlines:     {self.carcass_xlines}
        """
        return dedent(msg)


    # 2D
    def find_self(self):
        """ Get reference to the instance in a field.
        If it was loaded/added correctly, then it should be one of `loaded_labels`.
        Otherwise, we add it in a fake attribute and remove later.
        """
        for src in self.field.loaded_labels:
            labels = getattr(self.field, src)

            if isinstance(labels, list):
                for idx, label in enumerate(labels):
                    if label is self:
                        return f'{src}:{idx}'

        # Instance is not attached to a field: add it temporarily (clean-up when finish plot creation)
        self.field._unknown_label = AugmentedList([self])
        self.field.loaded_labels.append('_unknown_label')
        return '_unknown_label:0'

    @staticmethod
    def _show_add_prefix(attribute, prefix=None):
        if isinstance(attribute, str):
            attribute = ('/'.join([prefix, attribute])).replace('//', '/')
        elif isinstance(attribute, dict):
            attribute['src'] = ('/'.join([prefix, attribute['src']])).replace('//', '/')
        return attribute


    def show(self, attributes='depths', mode='image', show=True, **kwargs):
        """ Field visualization with custom naming scheme. """
        attributes = DelegatingList(attributes)
        attributes = attributes.map(lambda item: copy(item) if isinstance(item, dict) else item)
        attributes = attributes.map(self._show_add_prefix, prefix=self.find_self())

        kwargs = {
            'suptitle': f'`{self.name}` on field `{self.field.short_name}`',
            **kwargs
        }
        plotter = self.field.show(attributes=attributes, mode=mode, show=show, **kwargs)

        # Clean-up
        if self.field.loaded_labels[-1] == '_unknown_label':
            delattr(self.field, '_unknown_label')
            self.field.loaded_labels.pop(-1)

        return plotter

    def compute_auto_zoom(self, index, axis=None, zoom_margin=100):
        """ Get slice around the horizon without zero-traces on bounds. """
        bounds = self.field.geometry.compute_auto_zoom(index, axis)[0]
        return (bounds, slice(self.d_min - zoom_margin, self.d_max + zoom_margin))

    # 3D
    def show_3d(self, n_points=100, threshold=100., z_ratio=1., zoom=None, show_axes=True,
                width=1200, height=1200, margin=(0, 0, 100), savepath=None, **kwargs):
        """ Interactive 3D plot. Roughly, does the following:
            - select `n` points to represent the horizon surface
            - triangulate those points
            - remove some of the triangles on conditions
            - use Plotly to draw the tri-surface

        Parameters
        ----------
        n_points : int
            Number of points for horizon surface creation.
            The more, the better the image is and the slower it is displayed.
        threshold : int
            Threshold to remove triangles with bigger depth differences in vertices.
        z_ratio : int
            Aspect ratio between height axis and spatial ones.
        zoom : tuple of slices
            Crop from cube to show.
        show_axes : bool
            Whether to show axes and their labels.
        width, height : int
            Size of the image.
        margin : int
            Added margin from below and above along depth axis.
        savepath : str
            Path to save interactive html to.
        kwargs : dict
            Other arguments of plot creation.
        """
        title = f'Horizon `{self.short_name}` on `{self.field.short_name}`'
        aspect_ratio = (self.i_length / self.x_length, 1, z_ratio)
        axis_labels = (self.field.index_headers[0], self.field.index_headers[1], 'DEPTH')
        if zoom is None:
            zoom = [slice(0, i) for i in self.field.shape]
        zoom[-1] = slice(self.d_min, self.d_max)

        x, y, z, simplices = self.make_triangulation(n_points, threshold, zoom)

        show_3d(x, y, z, simplices, title, zoom, None, show_axes, aspect_ratio,
                axis_labels, width, height, margin, savepath, **kwargs)

    def make_triangulation(self, n_points, threshold, slices, **kwargs):
        """ Create triangultaion of horizon.

        Parameters
        ----------
        n_points: int
            Number of points for horizon surface creation.
            The more, the better the image is and the slower it is displayed.
        slices : tuple
            Region to process.

        Returns
        -------
        x, y, z, simplices
            `x`, `y` and `z` are np.ndarrays of triangle vertices, `simplices` is (N, 3) array where each row
            represent triangle. Elements of row are indices of points that are vertices of triangle.
        """
        _ = kwargs
        weights_matrix = self.full_matrix.astype(np.float32)

        grad_i = np.diff(weights_matrix, axis=0, prepend=0)
        grad_x = np.diff(weights_matrix, axis=1, prepend=0)
        weights_matrix = (grad_i + grad_x) / 2
        weights_matrix[np.abs(weights_matrix) > 100] = np.nan

        idx = np.stack(np.nonzero(self.full_matrix > 0), axis=0)
        mask_1 = (idx <= np.array([slices[0].stop, slices[1].stop]).reshape(2, 1)).all(axis=0)
        mask_2 = (idx >= np.array([slices[0].start, slices[1].start]).reshape(2, 1)).all(axis=0)
        mask = np.logical_and(mask_1, mask_2)
        idx = idx[:, mask]

        probs = np.abs(weights_matrix[idx[0], idx[1]].flatten())
        probs[np.isnan(probs)] = np.nanmax(probs)
        indices = np.random.choice(len(probs), size=n_points, p=probs / probs.sum())

        # Convert to meshgrid
        ilines = self.points[mask, 0][indices]
        xlines = self.points[mask, 1][indices]
        ilines, xlines = np.meshgrid(ilines, xlines)
        ilines = ilines.flatten()
        xlines = xlines.flatten()

        # Remove from grid points with no horizon in it
        depths = self.full_matrix[ilines, xlines]
        mask = (depths != self.FILL_VALUE)
        x = ilines[mask]
        y = xlines[mask]
        z = depths[mask]

        # Triangulate points and remove some of the triangles
        tri = Delaunay(np.vstack([x, y]).T)
        simplices = filter_simplices(simplices=tri.simplices, points=tri.points,
                                     matrix=self.full_matrix, threshold=threshold)
        return x, y, z, simplices
