""" Generator of predetermined locations based on field or current state of labeled surface. Mainly used for inference.

Locations describe the cube and the exact place to load from in the following format:
(field_id, label_id, orientation, i_start, x_start, d_start, i_stop, x_stop, d_stop).

Locations are passed to `make_locations` method of `SeismicCropBatch`, which
transforms them into 3D slices to index the data and other useful info like origin points, shapes and orientation.

Each of the classes provides:
    - `call` method (aliased to either `sample` or `next_batch`), that generates given amount of locations
    - `to_names` method to convert the first two columns of sampled locations into string names of field and label
    - convenient visualization to explore underlying `locations` structure
"""
import numpy as np
from numba import njit

from .utils import make_ranges



class BaseGrid:
    """ Deterministic generator of crop locations. """
    def __init__(self, crop_shape=None, batch_size=64,
                 locations=None, orientation=None, origin=None, endpoint=None, field=None, label_name='unknown',
                 **kwargs):
        self._iterator = None
        self.crop_shape = np.array(crop_shape)
        self.batch_size = batch_size

        if locations is None:
            self._make_locations()
        else:
            self.locations = locations
            self.orientation = orientation
            self.origin = origin
            self.endpoint = endpoint
            self.ranges = np.array([origin, endpoint]).T
            self.shape = endpoint - origin
            self.field = field
            self.field_name = field.short_name
            self.label_name = label_name

    def _make_locations(self):
        raise NotImplementedError('Must be implemented in sub-classes')

    def to_names(self, id_array):
        """ Convert the first two columns of sampled locations into field and label string names. """
        return np.array([(self.field_name, self.label_name) for ids in id_array])

    # Iteration protocol
    @property
    def iterator(self):
        """ Iterator that generates batches of locations. """
        if self._iterator is None:
            self._iterator = self.make_iterator()
        return self._iterator

    def make_iterator(self):
        """ Iterator that generates batches of locations. """
        return (self.locations[i:i+self.batch_size] for i in range(0, len(self), self.batch_size))

    def __call__(self, batch_size=None):
        _ = batch_size
        return next(self.iterator)

    def next_batch(self, batch_size=None):
        """ Yield the next batch of locations. """
        _ = batch_size
        return next(self.iterator)

    def __len__(self):
        """ Total number of locations to be generated. """
        return len(self.locations)

    @property
    def length(self):
        """ Total number of locations to be generated. """
        return len(self.locations)

    @property
    def n_iters(self):
        """ Total number of iterations. """
        return np.ceil(len(self) / self.batch_size).astype(np.int32)

    # Concatenate multiple grids into one
    def join(self, other):
        """ Update locations of a current grid with locations from other instance of BaseGrid. """
        if not isinstance(other, BaseGrid):
            raise TypeError('Other should be an instance of `BaseGrid`')
        if self.field_name != other.field_name:
            raise ValueError('Grids should be for the same field!')

        locations = np.concatenate([self.locations, other.locations], axis=0)
        locations = np.unique(locations, axis=0)
        batch_size = min(self.batch_size, other.batch_size)

        if self.orientation == other.orientation:
            orientation = self.orientation
        else:
            orientation = 2

        self_origin = self.origin if isinstance(self, RegularGrid) else self.actual_origin
        other_origin = other.origin if isinstance(other, RegularGrid) else other.actual_origin
        origin = np.minimum(self_origin, other_origin)

        self_endpoint = self.endpoint if isinstance(self, RegularGrid) else self.actual_endpoint
        other_endpoint = other.endpoint if isinstance(other, RegularGrid) else other.actual_endpoint
        endpoint = np.maximum(self_endpoint, other_endpoint)

        label_name = other.label_name if isinstance(other, ExtensionGrid) else self.label_name

        return BaseGrid(locations=locations, batch_size=batch_size, orientation=orientation,
                        origin=origin, endpoint=endpoint, field=self.field, label_name=label_name)

    def __add__(self, other):
        return self.join(other)

    def __and__(self, other):
        return self.join(other)


    # Useful info
    def __repr__(self):
        return f'<BaseGrid for {self.field_name}: origin={tuple(self.origin)}, endpoint={tuple(self.endpoint)}>'

    @property
    def original_crop_shape(self):
        """ Original crop shape. """
        return self.crop_shape if self.orientation == 0 else self.crop_shape[[1, 0, 2]]

    @property
    def actual_origin(self):
        """ The upper leftmost point of all locations. """
        return self.locations[:, [3, 4, 5]].min(axis=0).astype(np.int32)

    @property
    def actual_endpoint(self):
        """ The lower rightmost point of all locations. """
        return self.locations[:, [6, 7, 8]].max(axis=0).astype(np.int32)

    @property
    def actual_shape(self):
        """ Shape of the covered by the grid locations. """
        return self.endpoint - self.origin

    @property
    def actual_ranges(self):
        """ Ranges of covered by the grid locations. """
        return np.array(tuple(zip(self.origin, self.endpoint)))

    def show(self, grid=True, markers=False, n_patches=None, **kwargs):
        """ Display the grid over field overlay.

        Parameters
        ----------
        grid : bool
            Whether to show grid lines.
        markers : bool
            Whether to show markers at location origins.
        n_patches : int
            Number of locations to display with overlayed mask.
        kwargs : dict
            Other parameters to pass to the plotting function.
        """
        n_patches = n_patches or int(np.sqrt(len(self))) // 5
        plotter = self.field.geometry.show('dead_traces_matrix', cmap='Gray', colorbar=False, **kwargs)
        ax = plotter[0].ax

        if grid:
            spatial = self.locations[:, [3, 4]]
            for i in np.unique(spatial[:, 0]):
                sliced = spatial[spatial[:, 0] == i][:, 1]
                ax.vlines(i, sliced.min(), sliced.max(), colors='pink')

            spatial = self.locations[:, [3, 4]]
            for x in np.unique(spatial[:, 1]):
                sliced = spatial[spatial[:, 1] == x][:, 0]
                ax.hlines(x, sliced.min(), sliced.max(), colors='pink')

        if markers:
            ax.scatter(self.locations[:, 3], self.locations[:, 3], marker='x', linewidth=0.1, color='r')

        overlay = np.zeros_like(self.field.dead_traces_matrix)
        for n in range(0, len(self), len(self)//n_patches - 1):
            slc = tuple(slice(o, e) for o, e in zip(self.locations[n, [3, 4]], self.locations[n, [6, 7]]))
            overlay[slc] = 1
            ax.scatter(*self.locations[n, [3, 4]], marker='x', linewidth=3, color='g')

        kwargs = {
            'cmap': 'green',
            'alpha': 0.3,
            'colorbar': False,
            'matrix_name': 'Grid visualization',
            'ax': ax,
            **kwargs,
        }
        self.field.geometry.show(overlay, **kwargs)


    def to_chunks(self, size, overlap=0.05, orientation=None):
        """ Split the current grid into chunks along `orientation` axis.

        Parameters
        ----------
        size : int
            Length of one chunk along the splitting axis.
        overlap : number
            If integer, then number of slices for overlapping between consecutive chunks.
            If float, then proportion of `size` to overlap between consecutive chunks.

        Returns
        -------
        iterator with instances of `RegularGrid`.
        """
        if orientation is None:
            if self.orientation != 2:
                orientation = self.orientation

        if not isinstance(size, (tuple, list)):
            size_ = [None, None, None]

            if orientation is not None:
                orientation = self.field.geometry.parse_axis(orientation)
                size_[orientation] = size
        else:
            size_ = size

        if not isinstance(overlap, (tuple, list)):
            overlap = [overlap, overlap, overlap]
        return RegularGridChunksIterator(grid=self, size=size_, overlap=overlap)


class RegularGrid(BaseGrid):
    """ Regular grid over the selected `ranges` of cube, covering it with overlapping locations.
    Filters locations with less than `threshold` meaningful traces.

    Parameters
    ----------
    field : Field
        Field to create grid for.
    ranges : sequence
        Nested sequence, where each element is either None or sequence of two ints.
        Defines ranges to create grid for inline, crossline and depths.
    crop_shape : sequence
        Shape of crop locations to generate.
    orientation : int
        Either 0 or 1. Defines orientation of a grid. Used in `locations` directly.
    threshold : number
        Minimum amount of non-dead traces in a crop to keep it in locations.
        If number in 0 to 1 range, then used as percentage.
    strides : sequence, optional
        Strides between consecutive crops. Only one of `strides`, `overlap` or `overlap_factor` should be specified.
    overlap : sequence, optional
        Overlaps between consecutive crops. Only one of `strides`, `overlap` or `overlap_factor` should be specified.
    overlap_factor : sequence, optional
        Ratio of overlap between consecutive crops.
        Only one of `strides`, `overlap` or `overlap_factor` should be specified.
    batch_size : int
        Number of batches to generate on demand.
    field_id, label_id : int
        Used as the first two columns of sampled values.
    label_name : str, optional
        Name of the inferred label.
    locations : np.array, optional
        Pre-defined locations. If provided, then directly stored and used as the grid coordinates.
    """
    def __init__(self, field, ranges, crop_shape, orientation=0, strides=None, overlap=None, overlap_factor=None,
                 filtering_matrix='dead_traces_matrix', threshold=0, batch_size=64,
                 field_id=-1, label_id=-1, label_name='unknown', locations=None, **kwargs):
        # Make correct crop shape
        orientation = field.geometry.parse_axis(orientation)
        crop_shape = np.array(crop_shape)
        crop_shape = crop_shape if orientation == 0 else crop_shape[[1, 0, 2]]

        if strides is not None:
            strides = np.array(strides)
            strides = strides if orientation == 0 else strides[[1, 0, 2]]

        # Make ranges
        ranges = make_ranges(ranges, field.shape)
        ranges = np.array(ranges)
        self.ranges = ranges

        # Infer from `ranges`
        self.origin = ranges[:, 0]
        self.endpoint = ranges[:, 1]
        self.shape = ranges[:, 1] - ranges[:, 0]

        # Make `strides`
        if (strides is not None) + (overlap is not None) + (overlap_factor is not None) > 1:
            raise ValueError('Only one of `strides`, `overlap` or `overlap_factor` should be specified!')
        overlap_factor = [overlap_factor] * 3 if isinstance(overlap_factor, (int, float)) else overlap_factor

        if strides is not None:
            strides = strides if isinstance(strides, (tuple, list, np.ndarray)) else [strides] * 3
            if np.issubdtype(np.array(strides).dtype, np.floating):
                strides = np.array(crop_shape) * np.array(strides)
            strides = np.maximum(strides, 1).astype(int)
        else:
            if overlap is not None:
                strides = [c - o for c, o in zip(crop_shape, overlap)]
            elif overlap_factor is not None:
                strides = [max(1, c // f) for c, f in zip(crop_shape, overlap_factor)]
            else:
                strides = crop_shape
        self.strides = np.array(strides)

        # Update threshold: minimum amount of non-empty traces
        if isinstance(filtering_matrix, str):
            filtering_matrix = getattr(field, filtering_matrix)
        self.filtering_matrix = filtering_matrix

        if 0 < threshold < 1:
            threshold = int(threshold * crop_shape[0] * crop_shape[1])
        self.threshold = threshold

        self.field_id = field_id
        self.label_id = label_id
        self.orientation = orientation
        self.field = field
        self.field_name = field.short_name
        self.label_name = label_name
        self.unfiltered_length = None
        super().__init__(crop_shape=crop_shape, batch_size=batch_size, locations=locations, field=field,
                         orientation=self.orientation, origin=self.origin, endpoint=self.endpoint)

    @staticmethod
    def _arange(start, stop, stride, limit):
        grid = np.arange(start, stop, stride, dtype=np.int32)
        grid = np.unique(np.clip(grid, 0, limit))
        return np.sort(grid)

    def _make_locations(self):
        # Ranges for each axis
        i_args, x_args, d_args = tuple(zip(self.ranges[:, 0],
                                           self.ranges[:, 1],
                                           self.strides,
                                           self.field.shape - self.crop_shape))
        i_grid = self._arange(*i_args)
        x_grid = self._arange(*x_args)
        d_grid = self._arange(*d_args)
        self.unfiltered_length = len(i_grid) * len(x_grid) * len(d_grid)
        self._i_grid, self._x_grid, self._d_grid = i_grid, x_grid, d_grid

        # Create points: origins for each crop. Keep only those that produce crops with more than `threshold` points
        order = (1, 2, 3, 0) if self.orientation == 0 else (2, 1, 3, 0)
        points = np.array(np.meshgrid(i_grid, x_grid, d_grid, indexing='ij')).transpose(order).reshape(-1, 3)

        if self.filtering_matrix is not None:
            points = filter_points(points, self.filtering_matrix, self.crop_shape, self.threshold)

        # Buffer: (cube_id, i_start, x_start, d_start, i_stop, x_stop, d_stop)
        buffer = np.empty((len(points), 9), dtype=np.int32)
        buffer[:, 0] = self.field_id
        buffer[:, 1] = self.label_id
        buffer[:, 2] = self.orientation
        buffer[:, [3, 4, 5]] = points
        buffer[:, [6, 7, 8]] = points
        buffer[:, [6, 7, 8]] += self.crop_shape
        self.locations = buffer

    def __repr__(self):
        return f'<RegularGrid for {self.field.short_name}: '\
               f'origin={tuple(self.origin)}, endpoint={tuple(self.endpoint)}, crop_shape={tuple(self.crop_shape)}, '\
               f'orientation={self.orientation}>'


@njit
def filter_points(points, filtering_matrix, crop_shape, threshold):
    """ Remove locations covering less than `threshold` of present traces. """
    mask = np.ones(len(points), dtype=np.bool_)

    for j, (i, x, _) in enumerate(points):
        sliced = filtering_matrix[i:i+crop_shape[0], x:x+crop_shape[1]]
        n_alive_traces = sliced.size - sliced.sum()
        if n_alive_traces <= threshold:
            mask[j] = False
    return points[mask]


class RegularGridChunksIterator:
    """ Split regular grid into chunks along `orientation` axis. Supposed to be iterated over.

    Parameters
    ----------
    grid : BaseGrid
        Grid to split into chunks.
    size : int or None or tuple of two ints or Nones
        Length of chunks along corresponding axes. `None` indicates that there is no chunking along the axis.
    overlap : number
        If integer, then number of slices for overlapping between consecutive chunks.
        If float, then proportion of `size` to overlap between consecutive chunks.
    """
    def __init__(self, grid, size, overlap):
        self.grid = grid

        size_i, size_x, size_d = size
        overlap_i, overlap_x, overlap_d = overlap

        if size_i is not None:
            step_i = int(size_i*(1 - overlap_i)) if np.issubdtype(type(overlap_i), np.floating) else size_i - overlap_i
        else:
            step_i = size_i = self.grid.shape[0]

        if size_x is not None:
            step_x = int(size_x*(1 - overlap_x)) if np.issubdtype(type(overlap_x), np.floating) else size_x - overlap_x
        else:
            step_x = size_x = self.grid.shape[1]

        if size_d is not None:
            step_d = int(size_d*(1 - overlap_d)) if np.issubdtype(type(overlap_d), np.floating) else size_d - overlap_d
        else:
            step_d = size_d = self.grid.shape[2]

        self.size_i, self.size_x, self.size_d = size_i, size_x, size_d
        self.step_i, self.step_x, self.step_d = step_i, step_x, step_d

        self._iterator = None

    @property
    def iterator(self):
        """ Cached sequence of chunks. """
        # pylint: disable=protected-access
        if self._iterator is None:
            iterator = []
            grid = self.grid

            grid_i = RegularGrid._arange(*grid.ranges[0], self.step_i, max(0, grid.endpoint[0] - self.size_i))
            grid_x = RegularGrid._arange(*grid.ranges[1], self.step_x, max(0, grid.endpoint[1] - self.size_x))
            grid_d = RegularGrid._arange(*grid.ranges[2], self.step_d, max(0, grid.endpoint[2] - self.size_d))

            for start_i in grid_i:
                stop_i = start_i + self.size_i
                for start_x in grid_x:
                    stop_x = start_x + self.size_x
                    for start_d in grid_d:
                        stop_d = start_d + self.size_d

                        chunk_origin = np.array([start_i, start_x, start_d])
                        chunk_endpoint = np.array([stop_i, stop_x, stop_d])

                        # Filter points beyond chunk ranges along `orientation` axis
                        mask = ((grid.locations[:, 3] >= start_i) &
                                (grid.locations[:, 6] <= stop_i)  &
                                (grid.locations[:, 4] >= start_x) &
                                (grid.locations[:, 7] <= stop_x) &
                                (grid.locations[:, 5] >= start_d) &
                                (grid.locations[:, 8] <= stop_d))
                        chunk_locations = grid.locations[mask]

                        if len(chunk_locations):
                            chunk_grid = BaseGrid(field=grid.field, locations=chunk_locations,
                                                  origin=chunk_origin, endpoint=chunk_endpoint,
                                                  batch_size=grid.batch_size)
                            iterator.append(chunk_grid)
            self._iterator = iterator
        return self._iterator

    def __iter__(self):
        for chunk_grid in self.iterator:
            yield chunk_grid

    def __len__(self):
        return len(self.iterator)


class ExtensionGrid(BaseGrid):
    """ Generate locations to enlarge horizon from its boundaries both inwards and outwards.

    For each point on the boundary of a horizon, we test 4 possible directions and pick `top` best of them.
    Each location is created so that the original point is `stride` units away from the left/right edge of a crop.
    Only the locations that would potentially add more than `threshold` pixels remain.

    Refer to `_make_locations` method and comments for more info about inner workings.

    Parameters
    ----------
    horizon : Horizon
        Surface to extend.
    crop_shape : sequence
        Shape of crop locations to generate. Note that both iline and crossline orientations are used.
    stride : int
        Overlap with already known horizon for each location.
    threshold : int
        Minimum amount of potentially added pixels for each location.
    randomize : bool
        Whether to randomize the loop for computing the potential of each location.
    batch_size : int
        Number of batches to generate on demand.
    mode : {'best_for_each_independent', 'up', 'down', 'left', 'right',
            'vertical', 'horizontal', 'best_for_all', 'best_for_each'}
        Mode for directions of locations to generate.
        If mode is 'up', 'down', 'left' or 'right', then use only that direction.
        If 'vertical' ('horizontal'), then use up and down (right and left) directions.

        If 'best_for_all',  then select one direction for all points, based on total potentially added points.

        If 'best_for_each', then select direction for each point individually, based on total potentially added points.
        The potential of locations is computed sequentially: if one of the previous locations already covers area,
        it is considered to covered for all of the next potentials.

        If 'best_for_each_independent', then select direction for each point individually, based on total potentially
        added points. The potential of locations is computed independently of other locations.

    top : int
        Number of the best directions to keep for each point. Relevant only in `best_*` modes.
    """
    def __init__(self, horizon, crop_shape, stride=16, batch_size=64,
                 top=1, threshold=4, prior_threshold=8, randomize=True, mode='best_for_each', **kwargs):
        self.top = top
        self.stride = stride
        self.threshold = threshold
        self.prior_threshold = prior_threshold
        self.randomize = randomize
        self.mode = mode

        self.horizon = horizon
        self.field = horizon.field
        self.field_name = horizon.field.short_name
        self.label_name = horizon.short_name

        self.uncovered_before = None
        self.locations_stats = {}

        allowed_directions = ['up', 'down', 'left', 'right']

        if self.mode in ['best_for_all', 'best_for_each', 'best_for_each_independent']:
            self.directions = allowed_directions
        elif self.mode in allowed_directions:
            self.directions = [self.mode]
        elif self.mode == 'vertical':
            self.directions = ['up', 'down']
        elif self.mode == 'horizontal':
            self.directions = ['left', 'right']
        else:
            raise ValueError('Provided wrong `mode` argument, for possible options look at the docstring.')

        super().__init__(crop_shape=crop_shape, batch_size=batch_size)


    def _make_locations(self):
        # Get border points (N, 3)
        # Create locations for all four possible directions, stack into (4*N, 6)
        # Compute potential added area for each of the locations, while also updating coverage matrix
        # For each point, keep `top` of the best (potentially add more points) locations
        # Keep only those locations that potentially add more than `threshold` points
        #pylint: disable=too-many-statements

        crop_shape = self.crop_shape
        crop_shape_t = crop_shape[[1, 0, 2]]

        # True where dead trace / already covered
        coverage_matrix = self.field.dead_traces_matrix.copy().astype(np.bool_)
        coverage_matrix[self.horizon.full_matrix > 0] = True
        self.uncovered_before = coverage_matrix.size - coverage_matrix.sum()

        # Compute boundary points of horizon: both inner and outer borders
        border_points = np.stack(np.where(self.horizon.boundaries_matrix), axis=-1)
        depths = self.horizon.matrix[border_points[:, 0], border_points[:, 1]]

        # Shift depths up
        border_points += (self.horizon.i_min, self.horizon.x_min)
        depths -= crop_shape[2] // 2

        # Buffer for locations (orientation, i_start, x_start, d_start, i_stop, x_stop, d_stop).
        buffer = np.empty((len(border_points), 7), dtype=np.int32)
        buffer[:, 0] = 0
        buffer[:, [1, 2]] = border_points
        buffer[:, 3] = depths
        buffer[:, [4, 5]] = border_points
        buffer[:, 6] = depths

        # Repeat the same data along new 0-th axis: shift origins/endpoints
        n_directions = len(self.directions)
        buffer = np.repeat(buffer[np.newaxis, ...], n_directions, axis=0)
        directions_iterator = 0

        if 'up' in self.directions:
            # Crops with fixed INLINE, moving CROSSLINE: [-stride:-stride + shape]
            buffer[directions_iterator, :, [2, 5]] -= self.stride

            np.clip(buffer[directions_iterator, :, 2], 0, self.field.shape[1],
                    out=buffer[directions_iterator, :, 2])
            np.clip(buffer[directions_iterator, :, 5], 0, self.field.shape[1],
                    out=buffer[directions_iterator, :, 5])

            buffer[directions_iterator, :, [4, 5, 6]] += crop_shape.reshape(-1, 1)
            directions_iterator += 1

        if 'down' in self.directions:
            # Crops with fixed INLINE, moving CROSSLINE: [-shape + stride:+stride]
            buffer[directions_iterator, :, [2, 5]] -= (crop_shape[1] - self.stride)

            np.clip(buffer[directions_iterator, :, 2], 0, self.field.shape[1] - crop_shape[1],
                    out=buffer[directions_iterator, :, 2])
            np.clip(buffer[directions_iterator, :, 5], 0, self.field.shape[1] - crop_shape[1],
                    out=buffer[directions_iterator, :, 5])

            buffer[directions_iterator, :, [4, 5, 6]] += crop_shape.reshape(-1, 1)
            directions_iterator += 1

        if 'left' in self.directions:
            # Crops with fixed CROSSLINE, moving INLINE: [-stride:-stride + shape]
            buffer[directions_iterator, :, [1, 4]] -= self.stride

            np.clip(buffer[directions_iterator, :, 1], 0, self.field.shape[0],
                    out=buffer[directions_iterator, :, 1])
            np.clip(buffer[directions_iterator, :, 4], 0, self.field.shape[0],
                    out=buffer[directions_iterator, :, 4])

            buffer[directions_iterator, :,  [4, 5, 6]] += crop_shape_t.reshape(-1, 1)
            buffer[directions_iterator, :, 0] = 1
            directions_iterator += 1

        if 'right' in self.directions:
            # Crops with fixed CROSSLINE, moving INLINE: [-shape + stride:+stride]
            buffer[directions_iterator, :, [1, 4]] -= (crop_shape[1] - self.stride)

            np.clip(buffer[directions_iterator, :, 1], 0, self.field.shape[0] - crop_shape[1],
                    out=buffer[directions_iterator, :, 1])
            np.clip(buffer[directions_iterator, :, 4], 0, self.field.shape[0] - crop_shape[1],
                    out=buffer[directions_iterator, :, 4])

            buffer[directions_iterator, :,  [4, 5, 6]] += crop_shape_t.reshape(-1, 1)
            buffer[directions_iterator, :, 0] = 1
            directions_iterator += 1

        update_coverage_matrix = self.mode not in ['best_for_all', 'best_for_each_independent']
        if self.randomize and update_coverage_matrix:
            buffer = buffer[np.random.permutation(n_directions)]

        # Array with locations for each of the directions
        # Each 4 consecutive rows are location variants for each point on the boundary
        buffer = buffer.transpose((1, 0, 2)).reshape(-1, 7)
        self.locations_stats['possible'] = buffer.shape[0]

        # Compute potential addition for each location
        # for 'best_for_all' and 'best_for_each_independent' modes potential calculated independently
        potential = compute_potential(locations=buffer, coverage_matrix=coverage_matrix,
                                      shape=crop_shape, stride=self.stride, prior_threshold=self.prior_threshold,
                                      update_coverage_matrix=update_coverage_matrix)

        if self.mode in ['best_for_each', 'best_for_each_independent']:
            # For each trace get the most potential direction index
            # Get argsort for each group of four
            argsort = potential.reshape(-1, n_directions).argsort(axis=-1)[:, -self.top:].reshape(-1)

            # Shift argsorts to original indices
            shifts = np.repeat(np.arange(0, len(buffer), n_directions, dtype=np.int32), self.top)
            indices = argsort + shifts

        elif self.mode == 'best_for_all':
            # Get indices of locations corresponding to the best direction
            # The best direction is a direction ('left', 'right', 'up' or 'down') with maximal potentially added traces
            positive_potential = potential.copy()
            positive_potential[positive_potential < 0] = 0

            best_direction_idx = np.argmax(positive_potential.reshape(-1, n_directions).sum(axis=0))
            indices = range(best_direction_idx, len(buffer), n_directions)

        else:
            indices = slice(None)

        # Keep only top locations; remove locations with too small potential if needed
        potential = potential[indices]
        buffer = buffer[indices, :]

        # Drop locations duplicates
        buffer, unique_locations_indices = np.unique(buffer, axis=0, return_index=True)
        potential = potential[unique_locations_indices]

        self.locations_stats['top_locations'] = buffer.shape[0]

        mask = potential > self.threshold
        buffer = buffer[mask]
        potential = potential[mask]
        self.locations_stats['selected'] = buffer.shape[0]

        # Correct the depth
        np.clip(buffer[:, 3], 0, self.field.depth - crop_shape[2], out=buffer[:, 3])
        np.clip(buffer[:, 6], 0 + crop_shape[2], self.field.depth, out=buffer[:, 6])

        locations = np.empty((len(buffer), 9), dtype=np.int32)
        locations[:, [0, 1]] = -1
        locations[:, 2:9] = buffer
        self.locations = locations
        self.potential = potential.reshape(-1, 1)

        if update_coverage_matrix:
            self.uncovered_best = coverage_matrix.size - coverage_matrix.sum()
        else:
            # In the 'best_for_all' and 'best_for_each_independent' we don't update the `coverage_matrix``
            self.uncovered_best = self.uncovered_after


    @property
    def uncovered_after(self):
        """ Number of points not covered in the horizon, if all of the locations would
        add their maximum potential amount of pixels to the labeling.
        """
        coverage_matrix = self.field.dead_traces_matrix.copy().astype(np.bool_)
        coverage_matrix[self.horizon.full_matrix > 0] = True

        for (i_start, x_start, _, i_stop, x_stop, _) in self.locations[:, 3:]:
            coverage_matrix[i_start:i_stop, x_start:x_stop] = True
        return coverage_matrix.size - coverage_matrix.sum()

    def show(self, markers=False, overlay=True, frequency=1, **kwargs):
        """ Display the grid over horizon overlay.

        Parameters
        ----------
        markers : bool
            Whether to show markers at location origins.
        overlay : bool
            Whether to show overlayed mask for locations.
        frequency : int
            Frequency of shown overlayed masks.
        kwargs : dict
            Other parameters to pass to the plotting function.
        """
        hm = self.horizon.full_matrix.astype(np.float32)
        hm[hm < 0] = np.nan
        plotter = self.field.geometry.show(hm, cmap='Depths', colorbar=False, **kwargs)
        ax = plotter[0].ax

        self.field.geometry.show('dead_traces_matrix', ax=ax, cmap='Grey', colorbar=False, **kwargs)

        if markers:
            ax.scatter(self.locations[:, 3], self.locations[:, 4], marker='x', linewidth=0.1, color='r')

        if overlay:
            overlay = np.zeros_like(self.field.dead_traces_matrix)
            for n in range(0, len(self), frequency):
                slc = tuple(slice(o, e) for o, e in zip(self.locations[n, [3, 4]], self.locations[n, [6, 7]]))
                overlay[slc] = 1

            kwargs = {
                'cmap': 'blue',
                'alpha': 0.3,
                'colorbar': False,
                'title': f'Extension Grid on `{self.label_name}`',
                'ax': ax,
                **kwargs,
            }
            self.field.geometry.show(overlay, **kwargs)

@njit
def compute_potential(locations, coverage_matrix, shape, stride, prior_threshold, update_coverage_matrix=True):
    """ For each location, compute the amount of points it would potentially add to the labeling.
    If the shape of a location is not the same, as requested at grid initialization, we place `-1` value instead:
    that is filtered out later. That is used to filter locations out of field bounds.

    For each location, we also check whether one of its sides (left/right/up/down) contains more
    than `prior_threshold` covered points.
    """
    area = shape[0] * shape[1]
    buffer = np.empty((len(locations)), dtype=np.int32)

    for i, (orientation, i_start, x_start, _, i_stop, x_stop, _) in enumerate(locations):
        sliced = coverage_matrix[i_start:i_stop, x_start:x_stop]

        if sliced.size == area:

            if orientation == 0:
                left, right = sliced[:stride, :].sum(), sliced[:-stride, :].sum()
                prior = max(left, right)
            elif orientation == 1:
                up, down = sliced[:, :stride].sum(), sliced[:, :-stride].sum()
                prior = max(up, down)

            if prior >= prior_threshold:
                covered = sliced.sum()
                buffer[i] = area - covered

                if update_coverage_matrix:
                    coverage_matrix[i_start:i_stop, x_start:x_stop] = True
            else:
                buffer[i] = -1
        else:
            buffer[i] = -1

    return buffer

class LocationsPotentialContainer:
    """ Container for saving history of `ExtensionGrid`.

    It saves locations and their potential from each grid provided in the method `update_grid`.
    Also, it removes repetitions from the grid locations and potentials.
    """
    def __init__(self, locations=None, potential=None):
        if locations is None:
            locations = np.empty(shape=(0, 9), dtype=np.int32)
        if potential is None:
            potential = np.empty(shape=(0, 1), dtype=np.int32)

        ncols = locations.shape[1]

        self.initial_dtype = locations.dtype
        self.locations_dtype = {'names': [f'col_{i}' for i in range(ncols)],
                                          'formats': ncols * [self.initial_dtype]}

        self.locations = locations.view(self.locations_dtype)
        self.potential = potential

        self.stats = {
            'repeated_locations': [],
            'total_repeated_locations': 0
        }

    def update_grid(self, grid):
        """ Update grid and container locations and potential.

        For the container, we update potentials for existing locations and safe new locations and their potentials.
        For the grid, we remove locations and potentials pairs that are saved in the container. It helps reduce
        locations amount and avoid repetitive locations processing such as model inference on these locations.
        """
        # Choose locations and potential pairs that are not in the container
        grid_locations = grid.locations.view(self.locations_dtype)

        repeated_locations = np.in1d(grid_locations, self.locations)
        repeated_potential = np.in1d(grid.potential, self.potential)
        repeated_locations_potential = repeated_locations & repeated_potential

        new_locations = grid_locations[~repeated_locations_potential]
        new_potential = grid.potential[~repeated_locations_potential]

        # Safe stats
        repeated_locations = len(grid_locations) - len(new_locations)
        self.stats['repeated_locations'].append(repeated_locations)
        self.stats['total_repeated_locations'] += repeated_locations

        # Update container: save new potentials for old locations and save new locations with their potential
        if len(new_locations) > 0:
            if len(self.locations) > 0:
                repeated_locations_history = np.in1d(self.locations, new_locations)
                locations = self.locations[~repeated_locations_history]
                potential = self.potential[~repeated_locations_history]

                self.locations = np.vstack([locations, new_locations])
                self.potential = np.vstack([potential, new_potential])
            else:
                self.locations = new_locations
                self.potential = new_potential

            new_locations = new_locations.view(self.initial_dtype).reshape(-1, grid.locations.shape[1])
        else:
            new_locations = np.empty(shape=(0, grid.locations.shape[1]))
            new_potential = np.empty(shape=(0, 1))

        # Update grid: set locations and grid with values that are not in the container
        grid.locations, grid.potential = new_locations, new_potential
