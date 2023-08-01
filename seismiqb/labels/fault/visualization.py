""" Fault visualization mixin. """

import numpy as np

from ..mixins import VisualizationMixin
from ...plotters import show_3d
from ...utils import make_slices

class FaultVisualizationMixin(VisualizationMixin):
    """ Mixin to visualize fault. """
    def __repr__(self):
        return f"""<Fault `{self.name}` for `{self.field.short_name}` at {hex(id(self))}>"""

    def show_slide(self, index, **kwargs):
        """ Show slides from seismic with fault. """
        defaults = {'cmap': ['Greys_r', 'red'], 'width': 5}
        kwargs = {**defaults, **kwargs}
        return super().show_slide(index, **kwargs)

    def compute_auto_zoom(self, index, axis, zoom_margin=20):
        """ Get center slice of the fault. """
        _ = index
        return tuple(slice(max(self.bbox[i][0]-zoom_margin, 0),
                           min(self.bbox[i][1]+zoom_margin, self.field.shape[i]))
                     for i in range(3) if i != axis)

    def show(self, axis=0, zoom='auto', **kwargs):
        """ Show center of fault for different axes. """
        return self.show_slide(index=int(np.mean(self.bbox[axis])), zoom=zoom, axis=axis, **kwargs)

    def show_3d(self, sticks_step=None, stick_nodes_step=None, z_ratio=1., colors='green',
                zoom=None, margin=20, sticks=False, **kwargs):
        """ Interactive 3D plot. Roughly, does the following:
            - select `n` points to represent the horizon surface
            - triangulate those points
            - remove some of the triangles on conditions
            - use Plotly to draw the tri-surface

        Parameters
        ----------
        sticks_step : int or None
            Number of slides between sticks. If None, fault triangulation (nodes and simplices) will be used.
        stick_nodes_step : int or None
            Distance between stick nodes. If None, fault triangulation (nodes and simplices) will be used.
        z_ratio : int
            Aspect ratio between height axis and spatial ones.
        zoom : tuple of slices or None.
            Crop from cube to show. If None, the whole cube volume will be shown.
        show_axes : bool
            Whether to show axes and their labels, by default True
        width, height : int
            Size of the image, by default 1200, 1200
        margin : int
            Added margin from below and above along height axis, by default 20
        savepath : str
            Path to save interactive html to.
        sticks : bool
            If True, show fault sticks. If False, show interpolated surface.
        kwargs : dict
            Other arguments of plot creation.
        """
        title = f'Fault `{self.name}` on `{self.field.short_name}`'
        aspect_ratio = (self.i_length / self.x_length, 1, z_ratio)
        axis_labels = (self.field.index_headers[0], self.field.index_headers[1], 'DEPTH')

        zoom = make_slices(zoom, self.field.shape)

        margin = [margin] * 3 if isinstance(margin, int) else margin
        x, y, z, simplices = self.make_triangulation(zoom, sticks_step, stick_nodes_step, sticks=sticks)
        if isinstance(colors, str):
            colors = [colors for _ in simplices]

        show_3d(x, y, z, simplices, title=title, zoom=zoom, aspect_ratio=aspect_ratio,
                axis_labels=axis_labels, margin=margin, colors=colors, **kwargs)

    def make_triangulation(self, slices=None, sticks_step=None, stick_nodes_step=None,
                           stick_orientation=None, sticks=False, **kwargs):
        """ Return triangulation of the fault. It will created if needed. """
        # pylint: disable=too-many-boolean-expressions
        if ((sticks_step is not None and sticks_step != self.sticks_step) or
            (stick_nodes_step is not None and stick_nodes_step != self.stick_nodes_step) or
            (stick_orientation is not None and stick_orientation != self.stick_orientation)):
            # sticks must be recreated with new parameters from points

            temporary_fault = type(self)({'points': self.points}, field=self.field, direction=self.direction)
            temporary_fault.points_to_sticks(slices, sticks_step or 10, stick_nodes_step or 10,
                                             stick_orientation=stick_orientation)
            return temporary_fault.make_triangulation(slices, sticks=sticks, **kwargs)

        if sticks:
            sticks = self.sticks
            faults = [
                type(self)({'sticks': [stick]}, direction=self.direction, stick_orientation=self.stick_orientation,
                           field=self.field, name=(self.short_name or "") + '_' + str(i))
                for i, stick in enumerate(sticks)
            ]
            x, y, z, simplices = [], [], [], []
            n_points = 0
            for fault in faults:
                triangulation = fault.make_triangulation(slices)
                if len(triangulation[3]) > 0:
                    simplices.append(triangulation[3] + n_points)
                x.append(triangulation[0])
                y.append(triangulation[1])
                z.append(triangulation[2])
                n_points += len(triangulation[0])
            return [np.concatenate(data) if len(data) > 0 else data for data in [x, y, z, simplices]]

        if len(self.simplices) > 0:
            return self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2], self.simplices
        if len(self.sticks) == 1 and len(self.sticks[0]) > 1:
            fake_fault = get_fake_one_stick_fault(self)
            return fake_fault.make_triangulation(slices, sticks_step, stick_nodes_step,  **kwargs)

        return [], [], [], []

def get_fake_one_stick_fault(fault):
    """ Create fault with additional shifted stick to visualize one stick faults. """
    stick = fault.sticks[0]

    fake_fault = type(fault)({'sticks': np.array([stick, stick + 1])}, stick_orientation=fault.stick_orientation,
                             direction=fault.direction, field=fault.field)

    return fake_fault
