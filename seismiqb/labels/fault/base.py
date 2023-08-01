""" Fault class and processing methods. """

import os
import numpy as np
import pandas as pd

from .triangulation import sticks_to_simplices, triangle_rasterization
from .approximation import points_to_sticks
from .visualization import FaultVisualizationMixin, get_fake_one_stick_fault
from .formats import FaultSticksMixin, FaultSerializationMixin
from ...utils import insert_points_into_mask, take_along_axis

class Fault(FaultSticksMixin, FaultSerializationMixin, FaultVisualizationMixin):
    """ Class to represent Fault object.

    Initialized from `storage` and `field`, where storage can be one of:
        - csv-like file in FAULT_STICKS format.
        - npy file with ndarray of (N, 3) shape or array itself.
        - npz file with 'points', 'nodes', 'simplices' and 'sticks' or dict with the same keys.

    Each fault has 3 representations:
        - points : cloud of surface points. The most accurate way to define surface but
                   not so handy for manual editing and occupies the most memory. Is needed
                   to create masks.
        - sticks : polylines that approximate fault surface. Usually are placed on a sequence
                   of ilines or crosslines. The most common result of the experts labeling but
                   is not enough flexible.
        - nodes and simplices : approximation of the surface by triangulation. Is needed to
                                approximate arbitrary surface.

    All representations can be converted to each other:
            sticks -------> (nodes, simplices)
               ^                  |
               └--- points < -----┘

    Convertion from sticks to nodes/simplices is simply concating and triangles creation.
    To convert triangulation (nodes and simplices) to points, we rasterize each triangle.
    Convertion from points to sticks is more difficult and assumes that points are on almost
    flat 3d plane. Note that convertion from points to sticks leads to loss of information
    due to the approximation.

    Parameters
    ----------
    storage : str, numpy.ndarray or dict
        str - path to file (FaultSticks or npy/npz)
        numpy.ndarray of (N, 3) shape - array of fault points
        dict - fault data: points, sticks, nodes and/or simplices. Can include one of them.
    field : Field

    name : str, optional
        fault name, by default None
    direction : int or None, optional
        direction of the fault surface, by default None
    """

    # Columns used from the file
    COLUMNS = ['INLINE_3D', 'CROSSLINE_3D', 'DEPTH']

    def __init__(self, storage, field, name=None, direction=None, stick_orientation=None, **kwargs): #pylint: disable=super-init-not-called
        self.name = name
        self.field = field

        self.short_name = name
        self._points = None
        self._sticks = None
        self._nodes = None
        self._simplices = None
        self.direction = None
        self.stick_orientation = stick_orientation
        self.sticks_step = None
        self.stick_nodes_step = None

        if isinstance(storage, str):
            source = 'file'
        elif isinstance(storage, np.ndarray):
            source = 'points'
        elif isinstance(storage, dict):
            source = 'dict'
        elif isinstance(storage, pd.DataFrame):
            source = 'df'
        getattr(self, f'from_{source}')(storage, **kwargs)

        self.create_stats()

        if len(self) > 0 and self.direction is None:
            self.set_direction(direction)

    def interpolate(self):
        """ Create points of fault surface from sticks or nodes and simplices. """
        _ = self.points

    def has_component(self, component):
        """ Check if faults has points, sticks, simplices or nodes. """
        return getattr(self, '_'+component) is not None

    def create_stats(self):
        """ Compute fault stats (bounds, bbox, etc.) """
        if self.has_component('points'):
            data = self.points
        elif self.has_component('nodes'):
            data = self.nodes
        elif self.has_component('sticks'):
            data = np.concatenate(self.sticks)
        else:
            self.bbox = None
            return

        if len(data) == 0: # It can be for empty fault file.
            data = np.zeros((1, 3))

        i_min, x_min, d_min = np.min(data, axis=0)
        i_max, x_max, d_max = np.max(data, axis=0)

        self.d_min, self.d_max = int(d_min), int(d_max)
        self.i_min, self.i_max, self.x_min, self.x_max = int(i_min), int(i_max), int(x_min), int(x_max)

        self.i_length = (self.i_max - self.i_min) + 1
        self.x_length = (self.x_max - self.x_min) + 1
        self.bbox = np.array([[self.i_min, self.i_max],
                              [self.x_min, self.x_max],
                              [self.d_min, self.d_max]],
                             dtype=np.int32)

    def set_direction(self, direction):
        """ Find azimuth of the fault. """
        if self.direction is not None:
            return
        if direction is None:
            if self.has_component('sticks') and len(self.sticks) > 0:
                ptp = np.abs([item[:, :2].ptp(axis=0) for item in self.sticks]) # pylint: disable=invalid-sequence-index
                self.direction = int((ptp == 0).sum(axis=0).argmax())
            else:
                if self.has_component('points') and len(self.points) > 0:
                    data = self.points
                else:
                    data = self.nodes
                mean_depth = np.argsort(data[:, 2])[len(data[:, 2]) // 2]
                depth_slice = data[data[:, 2] == data[:, 2][mean_depth]]
                self.direction = 0 if depth_slice[:, 0].ptp() > depth_slice[:, 1].ptp() else 1
        elif isinstance(direction, int):
            self.direction = direction
        elif isinstance(direction[self.field.short_name], int):
            self.direction = direction[self.field.short_name]
        else:
            self.direction = direction[self.field.short_name][self.name]

    def reset_storage(self, storage):
        """ Clear 'points', 'sticks', 'nodes' or 'simplices' storage. """
        setattr(self, '_' + storage, None)

    @classmethod
    def load(cls, path, field, name=None, interpolate=False, **kwargs):
        """ Load faults. """
        if not isinstance(path, str) or os.path.splitext(path)[1][1:] not in ['char', '']:
            faults = [cls(path, field=field, name=name, **kwargs)]
        else:
            faults = [cls(df, field=field, name=name, **kwargs) for name, df in cls.split_charisma(path).items()]

        if interpolate:
            for fault in faults:
                fault.interpolate()

        return faults

    def from_points(self, points, transform=False, **kwargs):
        """ Initialize points cloud. """
        if transform:
            points = self.field.geometry.lines_to_cubic(points)
        self._points = points
        self.short_name = self.name

    def from_file(self, path, **kwargs):
        """ Init from path to either FAULT_STICKS csv-like file or from npy/npz. """
        path = self.field.make_path(path, makedirs=False)
        self.path = path

        self.name = self.name or os.path.basename(path)
        self.short_name = self.short_name or os.path.splitext(path)[0]

        ext = os.path.splitext(path)[1][1:]

        if ext == 'npz':
            self.load_npz(path, **kwargs)
            self.format = 'file-npz'
        elif ext == 'npy':
            self.load_npy(path, **kwargs)
            self.format = 'file-npy'
        elif ext == 'sqb':
            self.load_sqb(path, **kwargs)
            self.format = 'file-sqb'
        else:
            self.load_fault_sticks(path, **kwargs)
            self.format = 'file-sticks'

    def from_dict(self, storage, transform=False, **kwargs):
        """ Load fault from dict with 'points', 'nodes', 'simplices' and 'sticks'. """
        for key in ['points', 'nodes']:
            data = storage.get(key)
            if data is not None and transform:
                data = self.field.geometry.lines_to_cubic(data)
            setattr(self, '_' + key, data)

        sticks = storage.get('sticks')
        if sticks is not None and transform:
            sticks = [self.field.geometry.lines_to_cubic(item) for item in sticks]
        setattr(self, '_sticks', sticks)

        setattr(self, '_simplices', storage.get('simplices'))

    def from_df(self, storage, **kwargs):
        """ Load fault sticks. """
        self.load_fault_sticks(storage, **kwargs)

    # Transformation of attributes: sticks -> (nodes, simplices) -> points -> sticks

    @property
    def simplices(self):
        """ Approximation of the surface by triangulation. Is needed to approximate arbitrary surface.
        Exists in pair with nodes.
        """
        if self._simplices is None:
            if self._points is None and self._sticks is None:
                raise AttributeError("'simplices' can't be created ('points' and 'sticks' don't exist)")

            self.sticks_to_simplices()

        return self._simplices

    @property
    def nodes(self):
        """ Approximation of the surface by triangulation. Is needed to approximate arbitrary surface.
        Exists in pair with simplices.
        """
        if self._nodes is None:
            if self._points is None and self._sticks is None:
                raise AttributeError("'nodes' can't be created ('points' and 'sticks' don't exist)")

            self.sticks_to_simplices()

        return self._nodes

    @property
    def points(self):
        """ Cloud of surface points. The most accurate way to define surface but not so handy
        for manual editing and occupies the most memory. Is needed to create masks.
        """
        if self._points is None:
            if self._simplices is None and self._sticks is None:
                raise AttributeError("'points' can't be created ('nodes'/'simplices' and 'sticks' don't exist)")
            if len(self.simplices) > 1:
                self.simplices_to_points()
            elif len(self.nodes) > 0:
                fake_fault = get_fake_one_stick_fault(self)
                points = fake_fault.points
                self._points = points[points[:, self.direction] == self.sticks[0][0, self.direction]]

        return self._points

    @property
    def sticks(self):
        """ Polylines that approximate fault surface. Usually are placed on a sequence of ilines or crosslines.
        The most common result of the experts labeling but is not enough flexible.
        """
        if self._sticks is None:
            if self._simplices is None and self._points is None:
                raise AttributeError("'sticks' can't be created ('nodes'/'simplices' and 'points' don't exist)")
            self.points_to_sticks()

        return self._sticks

    def simplices_to_points(self, width=1):
        """ Interpolate triangulation.

        Parameters
        ----------
        simplices : numpy.ndarray
            Array of shape (n_simplices, 3) with indices of nodes to connect into triangle.
        nodes : numpy.ndarray
            Array of shape (n_nodes, 3) with coordinates.
        width : int, optional
            Thickness of the simplex to draw, by default 1.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_points, 3)
        """
        points = []
        for triangle in self.simplices:
            points.append(triangle_rasterization(self.nodes[triangle].astype('float32'), width))
        self._points = np.concatenate(points, axis=0).astype('int32')

    def points_to_sticks(self, slices=None, sticks_step=10, stick_nodes_step=10, stick_orientation=2,
                         nodes_threshold=5, move_bounds=False):
        """ Create sticks from fault points. """
        points = self.points.copy()
        if slices is not None:
            for i in range(3):
                points = points[points[:, i] <= slices[i].stop]
                points = points[points[:, i] >= slices[i].start]
        stick_orientation = stick_orientation if stick_orientation is not None else 2
        self._sticks = points_to_sticks(points=points, sticks_step=sticks_step, nodes_step=stick_nodes_step,
                                        fault_orientation=self.direction, stick_orientation=stick_orientation,
                                        threshold=nodes_threshold, move_bounds=move_bounds)
        self.stick_orientation = stick_orientation
        self.sticks_step = sticks_step
        self.stick_nodes_step = stick_nodes_step

    def sticks_to_simplices(self, max_simplices_depth=None, max_nodes_distance=None):
        """ Create nodes/simplices from fault sticks. """
        self._simplices, self._nodes = sticks_to_simplices(
            self.sticks, self.direction, max_simplices_depth, max_nodes_distance
        )


    def add_to_mask(self, mask, locations=None, width=1, axis=None, sparse=False, alpha=1, **kwargs):
        """ Add fault to background.

        Parameters
        ----------
        mask : ndarray
            Background to add fault to.
        locations : ndarray
            Where the fault is located.
        width : int
            Width of an added fault.
        axis : int or None, optional
            Orientation of the crop to insert fault, by default None (unknown or crop is 3D)
        sparse : bool, optional
            Whether create sparse mask (only on labeled slides) or not, by default False
        """
        _ = kwargs

        if axis is not None and axis not in (2, self.direction):
            return mask

        mask_bbox = np.array([[locations[0].start, locations[0].stop],
                              [locations[1].start, locations[1].stop],
                              [locations[2].start, locations[2].stop]],
                             dtype=np.int32)
        points = self.points

        if (self.bbox[:, 1] < mask_bbox[:, 0]).any() or (self.bbox[:, 0] >= mask_bbox[:, 1]).any():
            return mask

        if sparse and self.has_component('sticks'):
            loc = np.unique(self.nodes[:, self.direction])
            loc = loc[np.logical_and(mask_bbox[self.direction, 0] <= loc, loc < mask_bbox[self.direction, 1])]

            points = points[np.isin(points[:, self.direction], loc)]

            unlabeled_slides = take_along_axis(mask, loc - mask_bbox[self.direction, 0], self.direction)
            unlabeled_slides = loc[unlabeled_slides[:, 0, 0] == -1]

            slices = [slice(None)] * 3
            slices[self.direction] = unlabeled_slides - mask_bbox[self.direction, 0]
            mask[tuple(slices)] = 0

        insert_points_into_mask(mask, points, mask_bbox, width=width, axis=1-self.direction, alpha=alpha)
        return mask

    def __len__(self):
        """ The size of the fault. """
        if self.bbox is None:
            return 0
        return self.bbox[2].ptp() * (self.bbox[self.direction].ptp() + 1)

    def __add__(self, other):
        points = np.concatenate([self.points, other.points])
        return type(self)({'points': points}, field=self.field, name=f"{self.name}+{other.name}",
                          direction=self.direction)
