""" Faults extractor from point cloud. """
import numpy as np

from cc3d import connected_components
from scipy.ndimage import find_objects

from batchflow import Notifier

from .base import Fault
from .postprocessing import skeletonize
from .coords_utils import (bboxes_adjacent, bboxes_embedded, bboxes_intersected, compute_distances, dilate_coords,
                           find_contour, restore_coords_from_projection)
from ...utils import groupby_min, groupby_max, make_ranges, int_to_proba



class FaultExtractor:
    """ Extract fault surfaces from a skeletonized or smoothed probabilities array.

    Main naming rules, which help to understand what's going on:
    - Component is a 2D connected component on slide (corresponds to :class:`~.Component` instance).
    - Prototype is a 3D points cloud of merged components (corresponds to :class:`~.FaultPrototype` instance).
    Instances of :class:`~.FaultPrototype` are essentially the same as :class:`~.Fault` instances,
    but with their own processing methods such as concat, split, etc.
    - `coords` are spatial coordinates ndarray in format (iline, xline, depth) with (N, 3) shape.
    - `points` are coords and probabilities values ndarray in format (iline, xline, depth, proba) with (N, 4) shape.
    Note, that probabilities can be converted into (0, 255) values for applying integer storage for points.

    Implementation details
    ----------------------
    The extraction algorithm is:

    0) Label connected components for each 2D slide of the input array.

    1) Create prototypes.
    We extract prototype approximations as a set of similar components on neighboring slides on `direction` axis:
        - first, we select one of the unmerged 2D component, prioritizing the longest component
        - find the closest one on the next slide, and save them into one prototype.
        We repeat this until we fail to find close enough objects.

    Distance between components is computed axis-wise and further optimized by early exits on thresholds.

    We can have a situation, where two components are considered to be close, but have different lengths:
    in this case, we split (depth-wise) each component into up to three parts:
        - one on the overlap with the second component
        - one above the overlap and one below it: may be absent if not required
    The overlapping parts are then merged as usual.

    For more, see the :meth:`~.extract_one_prototype`.

    2) Merge connected prototypes.
    As we potentially did some splitting of components during the prototype creation,
    we concat them back where we need.

    For this, we find prototypes which are connected as puzzle details.
    For more, see the :meth:`~.concat_connected_prototypes`.

    This operation is recommended to be repeated for both `depth` and `self.direction` axes,
    and also for multiple prototype overlap thresholds.
    You can see the recommended operations sequence in the :meth:`~.run`.

    3) Merge embedded prototypes.
    We can have a situation where one prototype is completely inside the other.
    That is caused by prototypes concatenation order.

    For more, see the :meth:`~.concat_embedded_prototypes`.
    -------------------------------------------------------


    To sum up, the algorithm is:
    0) Initialize container with smoothed probabilities predictions.
    1) Extract first prototype approximations with :meth:`~.extract_prototypes`.
    2) Iteratively concat close prototypes with :meth:`~.concat_connected_prototypes`.
    3) Concat internal prototypes pieces with :meth:`~.concat_embedded_prototypes`.
    As an example of the overall pipeline, see the :meth:`~.run`.


    Parameters
    ----------
    data : np.ndarray or :class:`~.Geometry` instance, optional
        A 3D volume with smoothed or skeletonized predictions. By default we assume the data to be already skeletonized.
    ranges : sequence, optional
        Nested sequence, where each element is either None or sequence of two ints.
        Defines data ranges for faults extraction.
    do_skeletonize : bool, optional
        Whether the `data` argument needs to be skeletonized.
        Should be True, if the data is smoothed model output.
    direction : {0, 1}
        Extraction direction, 0 for ilines and 1 for crosslines.
        It is the same as the prediction direction.
    component_len_threshold : int, optional
        Threshold to filter out too small connected components on data slides.
        If 0, then no filter applied (recommended for higher accuracy).
        If more than 0, then extraction will be faster but some small prototypes can be not extracted.
    shape : sequence of three ints, optional
        Field shape.
    """
    # pylint: disable=protected-access
    def __init__(self, data=None, ranges=None, do_skeletonize=False, direction=0,
                 component_len_threshold=0, shape=None):
        # Data parameters
        self.shape = data.shape if data is not None else shape

        self.direction = direction
        self.orthogonal_direction = 1 - self.direction

        self.proba_transform = None

        # Make ranges
        ranges = make_ranges(ranges=ranges, shape=self.shape)
        ranges = np.array(ranges)
        self.ranges = ranges

        self.origin = ranges[:, 0]

        # Internal parameters
        self._dilation = 3 # constant for internal operations
        self.component_len_threshold = component_len_threshold

        self._unprocessed_slide_idx = self.origin[self.direction] # first index of the slide with unmerged components

        # Containers
        self.prototypes_queue = [] # prototypes for extraction
        self.prototypes = [] # extracted prototypes

        if data is not None:
            self.container = self._init_container(data=data, do_skeletonize=do_skeletonize)
        else:
            self.container = None

    def _init_container(self, data, do_skeletonize=False):
        """ Extract connected components on each slide and save them into container.

        Returns
        -------
        container : dict
            Dicts where keys are slide indices and values are dicts in the following format:
            {'components' : list of :class:`.Component` instances,
             'lengths' : list of corresponding lengths}.
        """
        container = {}

        # Process data slides: extract connected components and their info
        for slide_idx in Notifier('t')(range(*self.ranges[self.direction])):
            # Get skeletonized slide
            slide = data.take(slide_idx, axis=self.direction)
            slide = slide[slice(*self.ranges[self.orthogonal_direction]), slice(*self.ranges[2])]

            if do_skeletonize:
                skeletonized_slide = skeletonize(slide, width=3).astype(bool)
            else:
                skeletonized_slide = slide > np.min(slide) # for signed dtypes

            # Extract connected components from the slide
            labeled_slide = connected_components(skeletonized_slide)
            objects = find_objects(labeled_slide)

            # Get components info
            components, lengths = [], []

            for idx, object_bbox in enumerate(objects, start=1):
                # Extract component mask
                object_mask = labeled_slide[object_bbox] == idx

                # Filter by proba
                object_proba = slide[object_bbox][object_mask].max().astype(data.dtype) # TODO: think about percentile

                if np.issubdtype(data.dtype, np.integer):
                    object_proba = int_to_proba(object_proba)

                if object_proba < 0.1: # TODO: think about more appropriate threshold
                    continue

                # Check length
                length = np.count_nonzero(object_mask)

                if length <= self.component_len_threshold:
                    continue

                lengths.append(length)

                # Extract 3D coords and probabilities
                coords_2D = np.nonzero(object_mask)
                coords = np.zeros((len(coords_2D[0]), 3), dtype=np.int32)

                coords[:, self.direction] = slide_idx
                coords[:, self.orthogonal_direction] = coords_2D[0].astype(np.int32) + object_bbox[0].start + \
                                                       self.origin[self.orthogonal_direction]
                coords[:, 2] = coords_2D[1].astype(np.int32) + object_bbox[1].start + self.origin[2]

                probas = slide[object_bbox][coords_2D[0], coords_2D[1]]

                # Convert probas to integer values for saving them in points array with 3D-coordinates
                if not np.issubdtype(data.dtype, np.integer):
                    probas = np.round(probas * 255)
                    self.proba_transform = lambda x: x / 255
                if probas.dtype != coords.dtype:
                    probas = probas.astype(coords.dtype)

                points = np.hstack((coords, probas.reshape(-1, 1)))

                # Bbox
                bbox = np.empty((3, 2), dtype=np.int32)

                bbox[self.direction, :] = slide_idx

                bbox[self.orthogonal_direction, 0] = object_bbox[0].start + self.origin[self.orthogonal_direction]
                bbox[self.orthogonal_direction, 1] = object_bbox[0].stop + self.origin[self.orthogonal_direction] - 1

                bbox[2, 0] = object_bbox[1].start + self.origin[2]
                bbox[2, 1] = object_bbox[1].stop + self.origin[2] - 1

                # Save component
                component = Component(points=points, slide_idx=slide_idx, bbox=bbox)
                components.append(component)

            container[slide_idx] = {
                'components': components,
                'lengths': lengths
            }

        return container

    @classmethod
    def from_prototypes(cls, prototypes, shape):
        """ Initialize extractor from prototypes.

        Useful for applying operations on prototypes from different data chunks.

        Parameters
        ----------
        prototypes : list of :class:`~.FaultPrototype` instances, optional
            Prototypes for applying :class:`~.FaultExtractor` methods on.
        shape : sequence of three ints, optional
            Field shape from which the `prototypes` were extracted.
        """
        instance = cls(direction=prototypes[0].direction, shape=shape)

        instance.prototypes = prototypes
        instance.shape = shape
        return instance

    # Prototypes extraction from the data volume
    def extract_prototypes(self):
        """ Extract all fault prototypes from the point cloud.

        Returns
        -------
        prototypes: list of the :class:`~.FaultPrototype` instances
            Prototypes extracted from the data volume.
        """
        prototype = self.extract_one_prototype()

        while prototype is not None:
            self.prototypes.append(prototype)
            prototype = self.extract_one_prototype()

        return self.prototypes

    def extract_one_prototype(self):
        """ Extract one fault prototype from the point cloud.

        Under the hood, we find unmerged 2D component and find the closest one on the next slide.
        If components are close enough, they are merged into one 3D surface - fault prototype.
        Merging repeats until we are unable to find close enough components on next slides.

        Returns
        -------
        prototype: :class:`~.FaultPrototype` instance
            Prototype extracted from the data volume.
        """
        # Get intial 2D component and init prototype (or get from queue)
        if len(self.prototypes_queue) == 0:
            component, component_idx = self._find_unmerged_component()

            if component is None: # Nothing to merge
                return None

            self.container[component.slide_idx]['lengths'][component_idx] = -1 # Mark component as merged

            prototype = FaultPrototype(points=component.points, direction=self.direction, last_component=component,
                                       proba_transform=self.proba_transform)
        else:
            prototype = self.prototypes_queue.pop(0)
            component = prototype.last_component

        # Find closest components on next slides
        for next_slide in range(component.slide_idx + 1, self.ranges[self.direction][1]):
            # Find the closest component on the slide_idx_ to the current
            component, split_indices = self._find_closest_component(component=component, slide_idx=next_slide)

            # Postprocess prototype - it need to be splitted if it is out of component ranges
            if component is not None:
                prototype, new_prototypes = prototype.split(split_indices=split_indices, axis=2)
                self.prototypes_queue.extend(new_prototypes)

                prototype.append(component)
            else:
                break

        return prototype

    def _find_unmerged_component(self):
        """ Find the longest unmerged component on the first slide with unmerged components.
        Under the hood, we start from the very first slide, use all of its components, and then move to the next slides
        while keeping track of the index of slide with not all merged components.

        Returns
        -------
        component : :class:`.Component` instance or None
            First unmerged component in the data container.
            Can be None if there are no suitable components in the container.
        component_idx : int or None
            The index of the found component.
        """
        for slide_idx in range(self._unprocessed_slide_idx, self.ranges[self.direction][1]):
            slide_info = self.container[slide_idx]

            if len(slide_info['lengths']) > 0:
                component_idx = np.argmax(slide_info['lengths'])

                if slide_info['lengths'][component_idx] != -1:
                    self._unprocessed_slide_idx = slide_idx
                    component = self.container[slide_idx]['components'][component_idx]
                    return component, component_idx

        return None, None

    def _find_closest_component(self, component, slide_idx, distances_threshold=None,
                                depth_iteration_step=10, depths_threshold=20, distance_neighborhood=3):
        """ Find the closest component to the provided on next slide, get splitting indices for prototype.

        Parameters
        ----------
        component : instance of :class:`~.Component`
            Component for which find the closest one on the next slide.
        slide_idx : int
            Slide num on which to find the closest component.
        distances_threshold : int, optional
            Threshold for the max possible axis-wise distance between components,
            where axis is `self.orthogonal_direction`.
        depth_iteration_step : int
            The depth iteration step to find distances between components.
            Value 1 is recommended for higher accuracy.
            Value more than 1 is less accurate but speeds up this method.
        depths_threshold : int
            Depth-length threshold to decide to split closest component or prototype.
            If one component is longer than another more than on depths_threshold,
            then we need to split the longest one into parts:
             - one part is the closest component;
             - another parts corresponds to the other components,
             which are not allowed to merge into the current prototype.
        distance_neighborhood : int
            Area in which to find close components to choose the closest and longest one.
            For example, if we have two close enough components with distances 0 and 0 + distance_neighborhood and
            the second is longer than the closest, then we will choose it.
            We make this because one component on the next slide can be splitted into two parts and we want to concat it
            with the longest one. In this case distance_neighborhood is allowable error for finding the close enough
            components.

        Returns
        -------
        closest_component : :class:`.Component` instance
            The next slide closest component to the provided one.
        prototype_split_indices : list of two ints or Nones
            Depth coordinates for splitting extracted prototype (if needed).
            Indices are evaluated as components overlap range by the depth axis.
        """
        # Dilate component bbox for detecting close components: component on next slide can be shifted
        dilated_bbox = component.bbox.copy()
        dilated_bbox[self.orthogonal_direction, :] += (-self._dilation // 2, self._dilation // 2)
        dilated_bbox[self.orthogonal_direction, 0] = max(0, dilated_bbox[self.orthogonal_direction, 0])
        dilated_bbox[self.orthogonal_direction, 1] = min(dilated_bbox[self.orthogonal_direction, 1],
                                                         self.shape[self.orthogonal_direction])

        min_distance = distances_threshold if distances_threshold is not None else 100

        # Init returned values
        closest_component = None
        selected_component_length = -1
        prototype_split_indices = [None, None]
        component_split_indices = [None, None]

        # Iter over components and find the closest one
        for other_component_idx, other_component in enumerate(self.container[slide_idx]['components']):
            if self.container[slide_idx]['lengths'][other_component_idx] == -1:
                continue

            # Check bboxes intersection
            if not bboxes_intersected(dilated_bbox, other_component.bbox, axes=(self.orthogonal_direction, 2)):
                continue

            # Check closeness of some points (as depth-wise distances)
            # Faster then component overlap, but not so accurate
            overlap_depths = (max(component.bbox[2, 0], other_component.bbox[2, 0]),
                              min(component.bbox[2, 1], other_component.bbox[2, 1]))

            step = min(depth_iteration_step, (overlap_depths[1]-overlap_depths[0])//3)
            step = max(step, 1)

            indices_1 = np.in1d(component.coords[:, -1], np.arange(overlap_depths[0], overlap_depths[1]+1, step))
            indices_2 = np.in1d(other_component.coords[:, -1], np.arange(overlap_depths[0], overlap_depths[1]+1, step))

            coords_1 = component.coords[indices_1, self.orthogonal_direction]
            coords_2 = other_component.coords[indices_2, self.orthogonal_direction]

            components_distances = compute_distances(coords_1, coords_2,
                                                     max_threshold=min_distance+distance_neighborhood)

            if (components_distances[0] == -1) or (components_distances[0] > distance_neighborhood):
                # Components are not close
                continue

            if components_distances[1] >= min_distance + distance_neighborhood:
                # `other_component` is not close enough
                continue

            # The most depthwise distant points in components are close enough -> we can combine components
            # Also, we want to find the longest close enough component
            if selected_component_length < len(other_component):
                min_distance = components_distances[1]
                selected_component_length = len(other_component)
                closest_component = other_component
                merged_idx = other_component_idx
                overlap_borders = overlap_depths

        if closest_component is not None:
            # Process (split if needed) founded component and get split indices for prototype
            self.container[closest_component.slide_idx]['lengths'][merged_idx] = -1 # mark component as merged

            # Get prototype split indices:
            # check that the new component is smaller than the previous one (for each border)
            if overlap_borders[0] - component.bbox[2, 0] > depths_threshold:
                prototype_split_indices[0] = overlap_borders[0]

            if component.bbox[2, 1] - overlap_borders[1] > depths_threshold:
                prototype_split_indices[1] = overlap_borders[1]

            # Split new component: check that the new component is bigger than the previous one (for each border)
            # Create splitted items and save them as new elements for merge
            if overlap_borders[0] - closest_component.bbox[2, 0] > depths_threshold:
                component_split_indices[0] = overlap_borders[0]

            if closest_component.bbox[2, 1] - overlap_borders[1] > depths_threshold:
                component_split_indices[1] = overlap_borders[1]

            closest_component, new_components = closest_component.split(split_indices=component_split_indices)
            self._add_new_components(new_components)

        return closest_component, prototype_split_indices

    def _add_new_components(self, components):
        """ Add new components into the container.

        New items are created after splitting.
        """
        for component in components:
            if len(component) > self.component_len_threshold:
                self.container[component.slide_idx]['components'].append(component)
                self.container[component.slide_idx]['lengths'].append(len(component))


    # Prototypes concatenation
    def concat_connected_prototypes(self, overlap_ratio_threshold=None, axis=2,
                                    border_threshold=20, width_split_threshold=100):
        """ Concat prototypes which are connected as puzzle details.

        Under the hood, we compare prototypes with each other and find connected pairs.
        For this, we get neighboring borders and compare them:
        if they are almost overlapped after spatial shift then we merge corresponding prototypes.

        Parameters
        ----------
        overlap_ratio_threshold : float or None
            Prototypes borders overlap ratio to decide that prototypes are not close.
            Possible values are float numbers in the (0, 1] interval or None.
            If None, then default values are used: 0.5 for the depth axis and 0.9 for others.
        axis : {0, 1, 2}
            Axis along which to find prototype borders connections.
            Recommended values are 2 (for depths) and `self.direction`.
        border_threshold : int
            Minimal amount of points out of borders overlap to decide that prototypes are not close.
        width_split_threshold : int or None
            Merging prototypes width (along `self.direction` axis) difference threshold to decide
            that they need to be splitted.
            If value is None, then no splitting applied. But there are the risk of interpenetration of
            triangulated surfaces in this case.
            With lower values more splitting applied and smaller prototypes are extracted.
            If you want more detailing, then provide smaller `width_split_threshold` (near to 10).
            If you want to extract bigger surfaces, then provide higher `width_split_threshold` (near to 100 or None).

        Returns
        -------
        prototypes: list of the :class:`~.FaultPrototype` instances
            Prototypes instances after concatenation.
        """
        #pylint: disable=too-many-branches
        margin = 1 # local constant for code prettifying

        if overlap_ratio_threshold is None:
            overlap_ratio_threshold = 0.5 if axis in (-1, 2) else 0.9

        overlap_axis = self.direction if axis in (-1, 2) else 2

        # Under the hood, we check borders connectivity (as puzzles)
        borders_to_check = ('up', 'down') if axis in (-1, 2) else ('left', 'right')

        # Presort objects by overlap axis for early stopping
        sort_axis = overlap_axis
        prototypes_starts = np.array([prototype.bbox[sort_axis, 0] for prototype in self.prototypes])
        prototypes_order = np.argsort(prototypes_starts)
        reodered_prototypes = [self.prototypes[idx] for idx in prototypes_order]

        new_prototypes = []

        for i, prototype_1 in enumerate(reodered_prototypes):
            prototype_for_merge = None
            best_overlap = -1

            for prototype_2 in reodered_prototypes[i+1:]:
                # Exit if we out of sort_axis ranges for prototype_1
                if (prototype_1.bbox[sort_axis, 1] < prototype_2.bbox[sort_axis, 0]):
                    break

                adjacent_borders = bboxes_adjacent(prototype_1.bbox, prototype_2.bbox)

                if adjacent_borders is None:
                    continue

                # Check that bboxes overlap is enough
                overlap_threshold = min(prototype_1.bbox[overlap_axis, 1] - prototype_1.bbox[overlap_axis, 0],
                                        prototype_2.bbox[overlap_axis, 1] - prototype_2.bbox[overlap_axis, 0])
                overlap_threshold *= overlap_ratio_threshold

                overlap_length = adjacent_borders[overlap_axis][1] - adjacent_borders[overlap_axis][0]

                if overlap_length < overlap_threshold:
                    continue

                # Find object borders on close borders
                is_first_upper = prototype_1.bbox[axis, 0] < prototype_2.bbox[axis, 0]

                border_1 = prototype_1.get_border(border=borders_to_check[is_first_upper],
                                                  projection_axis=self.orthogonal_direction)
                border_2 = prototype_2.get_border(border=borders_to_check[~is_first_upper],
                                                  projection_axis=self.orthogonal_direction)

                # Get objects width in area near to overlap for intersection threshold
                # to avoid concatenation of objects with too little overlap
                neighborhood_range = (min(adjacent_borders[axis]) - 20, max(adjacent_borders[axis]) + 20)

                neighboring_border_1 = border_1[(border_1[:, axis] >= neighborhood_range[0]) & \
                                                (border_1[:, axis] <= neighborhood_range[1])]
                neighboring_border_2 = border_2[(border_2[:, axis] >= neighborhood_range[0]) & \
                                                (border_2[:, axis] <= neighborhood_range[1])]

                if len(neighboring_border_1) == 0 or len(neighboring_border_2) == 0:
                    continue

                width_neighboring_1 = np.ptp(neighboring_border_1[:, overlap_axis])
                width_neighboring_2 = np.ptp(neighboring_border_2[:, overlap_axis])

                # TODO: think about more appropriate criteria than proportion
                overlap_threshold = 0.5*max(width_neighboring_1, width_neighboring_2)

                # Get borders in the area of interest
                overlap_range = (min(adjacent_borders[axis]) - margin, max(adjacent_borders[axis]) + margin)

                border_1 = border_1[(border_1[:, axis] >= overlap_range[0]) & \
                                    (border_1[:, axis] <= overlap_range[1])]
                border_2 = border_2[(border_2[:, axis] >= overlap_range[0]) & \
                                    (border_2[:, axis] <= overlap_range[1])]

                # If one data border is much longer than other, then we can't connect them as puzzle details
                if len(border_1) == 0 or len(border_2) == 0:
                    continue

                length_ratio = min(len(border_1), len(border_2)) / max(len(border_1), len(border_2))

                if length_ratio < overlap_ratio_threshold:
                    continue

                # Correct border_threshold for too short borders
                if (1 - overlap_ratio_threshold) * min(len(border_1), len(border_2)) < border_threshold:
                    corrected_border_threshold = min(2*margin, border_threshold)
                else:
                    corrected_border_threshold = border_threshold

                # Shift one of the objects, making their borders intersected
                shift = 1 if is_first_upper else -1
                border_1[:, axis] += shift

                # Check that one component border is inside another (for both)
                border_1_width = np.ptp(border_1[:, overlap_axis])
                border_2_width = np.ptp(border_2[:, overlap_axis])

                if border_1_width <= border_2_width:
                    overlap_range = self._borders_overlap(border_1, border_2,
                                                          border_threshold=corrected_border_threshold,
                                                          overlap_threshold=overlap_threshold,
                                                          overlap_axis=overlap_axis)
                else:
                    overlap_range = self._borders_overlap(border_2, border_1,
                                                          border_threshold=corrected_border_threshold,
                                                          overlap_threshold=overlap_threshold,
                                                          overlap_axis=overlap_axis)

                if overlap_range is None:
                    continue

                # Select the best one prototype for merge
                border_length = border_1[:, axis].max() - border_1[:, axis].min() + 1
                overlap_ratio = (overlap_range[1] - overlap_range[0]) / border_length

                if best_overlap < overlap_ratio:
                    best_overlap_range = overlap_range
                    best_overlap = overlap_ratio
                    prototype_for_merge = prototype_2

                    best_border_1_width = border_1_width
                    best_border_2_width = border_2_width

                    is_first_upper_than_best = is_first_upper

            # Split for avoiding wrong prototypes shapes:
            # - concat only overlapped parts
            # - lower part can't be wider then upper;
            # - if one prototype is much bigger than it should be splitted.
            if prototype_for_merge is None:
                continue

            width_threshold = min(best_border_1_width, best_border_2_width) - 2*margin

            if (best_overlap_range[1] - best_overlap_range[0]) < width_threshold:
                prototype_for_merge, new_prototypes_ = prototype_for_merge.split(best_overlap_range,
                                                                                 axis=self.direction)
                new_prototypes.extend(new_prototypes_)

                prototype_1, new_prototypes_ = prototype_1.split(best_overlap_range, axis=self.direction)
                new_prototypes.extend(new_prototypes_)

            elif axis in (-1, 2):
                width_diff = 5

                lower_is_wider = (is_first_upper_than_best and \
                                  prototype_for_merge.width - best_border_1_width > width_diff) or \
                                 (not is_first_upper_than_best and prototype_1.width - best_border_2_width > width_diff)

                too_big_width_diff = (width_split_threshold is not None) and \
                                     (np.abs(prototype_1.width - prototype_for_merge.width) > width_split_threshold)

                if (lower_is_wider or too_big_width_diff):
                    if is_first_upper_than_best:
                        prototype_for_merge, new_prototypes_ = prototype_for_merge.split(best_overlap_range,
                                                                                         axis=self.direction)
                    else:
                        prototype_1, new_prototypes_ = prototype_1.split(best_overlap_range, axis=self.direction)

                    new_prototypes.extend(new_prototypes_)

            prototype_for_merge.concat(prototype_1)
            prototype_1._already_merged = True

        self.prototypes = [prototype for prototype in self.prototypes
                           if not getattr(prototype, '_already_merged', False)]
        self.prototypes.extend(new_prototypes)
        return self.prototypes

    def _borders_overlap(self, border_1, border_2, border_threshold, overlap_axis, overlap_threshold=0):
        """ Check that `border_1` is almost inside the dilated `border_2` and return their overlap range.

        We apply dilation for `border_2` because the fault can be shifted on neighboring slides.

        Parameters
        ----------
        border_1, border_2 : np.ndarrays of (N, 3) shape
            Contours coordinates for check.
        border_threshold : int
            Minimal amount of points out of overlap to decide that `border_1` is not inside `border_2`.
        overlap_threshold : int
            Minimal amount of overlapped points to decide that borders are overlapping.

        Returns
        -------
        overlap_range : tuple of two ints or None
            The longest overlap range on the `overlap_axis` for provided borders.
            None, if there are no overlap.
            Note, that two borders can have more than one overlapping area, we choose the longest one.
        """
        border_1_set = set(tuple(x) for x in border_1)

        # Objects can be shifted on `self.orthogonal_direction`, so apply dilation for coords
        border_2_dilated = dilate_coords(coords=border_2, dilate=self._dilation,
                                         axis=self.orthogonal_direction,
                                         max_value=self.shape[self.orthogonal_direction])

        border_2_dilated = set(tuple(x) for x in border_2_dilated)

        overlap = border_1_set.intersection(border_2_dilated)
        borders_overlapped = len(overlap) > overlap_threshold

        if borders_overlapped and (len(border_1_set - border_2_dilated) < border_threshold):
            return get_range(overlap, axis=overlap_axis)

        return None

    def concat_embedded_prototypes(self, border_threshold=100):
        """ Concat embedded prototypes (with 2 or more close borders).

        Under the hood, we compare different prototypes to find pairs in which one prototype is inside another.
        If more than two borders of internal prototype is connected with other prototype, then we merge them.

        Internal logic looks similar to `.concat_connected_prototypes`,
        but now we find embedded bboxes and need two borders coincidence instead of one.

        Embedded prototypes examples:

        ||||||  or  |||||||  or  ||||||  etc.
        ...|||      |...|||      |||...
           |||      |||||||
        ||||||

         - where | means one prototype points, and . - other prototype points.

        Parameters
        ----------
        border_threshold : int
            Minimal amount of points out of borders overlap to decide that prototypes are not close.

        Returns
        -------
        prototypes: list of the :class:`~.FaultPrototype` instances
            Prototypes after concatenation.
        """
        # Presort objects by other valuable axis for early stopping
        sort_axis = self.direction
        prototypes_starts = np.array([prototype.bbox[sort_axis, 0] for prototype in self.prototypes])
        prototypes_order = np.argsort(prototypes_starts)
        reodered_prototypes = [self.prototypes[idx] for idx in prototypes_order]

        margin = 3 # local constant

        for i, prototype_1 in enumerate(reodered_prototypes):
            for prototype_2 in reodered_prototypes[i+1:]:
                # Check that prototypes are embedded
                if (prototype_1.bbox[sort_axis, 1] < prototype_2.bbox[sort_axis, 0]):
                    break

                is_embedded, swap = bboxes_embedded(prototype_1.bbox, prototype_2.bbox, margin=margin)

                if not is_embedded:
                    continue

                coords = prototype_1.coords if swap is False else prototype_2.coords
                other = prototype_2 if swap is False else prototype_1

                # Check borders connections
                close_borders_counter = 0

                for border_position in ('up', 'down', 'left', 'right'): # TODO: get more optimal order depend on bboxes
                    # Find internal object border
                    border = other.get_border(border=border_position, projection_axis=self.orthogonal_direction)
                    border = border.copy() # will be shifted

                    # Shift border to make it intersected with another object
                    shift = -1 if border_position in ('up', 'left') else 1
                    shift_axis = self.direction if border_position in ('left', 'right') else 2
                    border[:, shift_axis] += shift

                    # Get main object coords in the area of the interest for speeding up evaluations
                    slices = other.bbox.copy()
                    slices[:, 0] -= margin
                    slices[:, 1] += margin

                    coords_sliced = coords[(coords[:, 0] >= slices[0, 0]) & (coords[:, 0] <= slices[0, 1]) & \
                                           (coords[:, 1] >= slices[1, 0]) & (coords[:, 1] <= slices[1, 1]) & \
                                           (coords[:, 2] >= slices[2, 0]) & (coords[:, 2] <= slices[2, 1])]

                    # Check that the shifted border is inside the main_object area
                    corrected_border_threshold = min(border_threshold, len(border)//2)

                    overlap_axis = self.direction if border in ('up', 'down') else 2

                    if  self._borders_overlap(border, coords_sliced,
                                              border_threshold=corrected_border_threshold,
                                              overlap_axis=overlap_axis) is not None:
                        close_borders_counter += 1

                    if close_borders_counter >= 2:
                        break

                # If objects have more than 2 closed borders then they are parts of the same prototype -> merge them
                if close_borders_counter >= 2:
                    prototype_2.concat(prototype_1)
                    prototype_1._already_merged = True
                    break

        self.prototypes = [prototype for prototype in self.prototypes
                           if not getattr(prototype, '_already_merged', False)]
        return self.prototypes


    def split_horseshoe(self, height_ratio_threshold=0.7, height_diff_threshold=30, axis=2, frequency=5):
        """ Split prototypes which looks like horseshoe.

        Under the hood, we iterate over prototype components to find sharp drop in their
        height and after that a sharp increase. For example:

        |||||||||||
        |||    ||||
        |||    ||||

        Parameters
        ----------
        height_ratio_threshold : float in [0, 1]
            Heigts ratio to decide that one component is much bigger than other.
        height_diff_threshold : int
            Minimal difference between heights to check that one is much bigger than other.
            We have no need in splitting very small objects.
        axis : {0, 1, 2}
            Axis along which to check heights changing.
        frequency : int
            Traces iteration frequency for heights comparison.

        Returns
        -------
        prototypes: list of the :class:`~.FaultPrototype` instances
            Prototypes after splitting.
        """
        traces_axis = 2 if axis == self.direction else self.direction
        new_prototypes = []

        for prototype in self.prototypes:
            # Skip too small prototypes
            if prototype.bbox[axis, 1] - prototype.bbox[axis, 0] <= height_diff_threshold:
                continue

            sign = +1
            previous_height = 0

            for line in range(prototype.bbox[traces_axis, 0], prototype.bbox[traces_axis, 1] + 1, frequency):
                # Compare current and previous heights
                points_ = prototype.points[prototype.points[:, traces_axis] == line, axis]

                if len(points_) == 0:
                    continue

                height = np.ptp(points_)

                height_ratio = min(height, previous_height) / max(height, previous_height)
                height_diff = height - previous_height

                if (height_ratio <= height_ratio_threshold) and (np.abs(height_diff) > height_diff_threshold):
                    #pylint: disable=chained-comparison
                    if sign > 0 and height_diff < 0:
                        sign = -1
                    elif sign < 0 and height_diff > 0:
                        sign = +1

                        # Split prototype because we found height increase after decrease
                        prototype, new_prototypes_ = prototype.split(split_indices=(line-frequency, None),
                                                                     axis=traces_axis)
                        new_prototypes.extend(new_prototypes_)

                previous_height = height

        self.prototypes.extend(new_prototypes)
        return self.prototypes


    # Addons
    def run(self, prolongate_in_depth=False, concat_iters=20, overlap_ratio_threshold=None,
            additional_filters=False, **filtering_kwargs):
        """ Recommended extracting procedure.

        The procedure is:
         - extract prototypes from the point cloud;
         - filter too small prototypes (for speed up, optional);
         - concat connected prototypes (concat by depth axis, concat by `self.direction` axis) `concat_iters` times
        with changed `overlap_ratio_threshold`;
         - filter too small prototypes (for speed up, optional);
         - concat embedded prototypes;
         - filter all unsuitable prototypes (optional).

        Parameters
        ----------
        prolongate_in_depth : bool
            Whether to maximally prolongate faults in depth or not.
            If True, then surfaces will be tall and thin.
            If False, then surfaces will be more longer for `self.direction` than for depth axis.
            This parameter affects concatenation axis order:
             - if True, than we apply concat by depth axis until `overlap_ratio_threshold` reaches the minimal value;
             - otherwise, we alternate concatenation by depth and `self.direction` axes.
        concat_iters : int
            Maximal amount of :meth:`~.concat_connected_prototypes` operations.
            One operation include both concat along the depth and `self.direction` axes.
        overlap_ratio_threshold : dict or None
            Prototype borders overlap ratio to decide that prototypes should be concated into one.
            Note, it is decrementally changed over concatenation iterations.
            Keys are concatenation axes (depth and `self.direction`) and values are in the (start, stop, step) format.
        additional_filters : bool
            Whether to apply additional filtering for speed up.
        filtering_kwargs
            The :meth:`~.filter_prototypes` kwargs.
            These kwargs are applied only in the filtration after whole extraction procedure.

        Returns
        -------
        prototypes : list of the :class:`~.FaultPrototype` instances
            Resulting prototypes.
        stats : dict
            Amount of prototypes after each proceeding.
        """
        stats = {}

        if overlap_ratio_threshold is None:
            overlap_ratio_threshold = {
                self.direction: (0.9, 0.7, 0.05), # (start, stop, step)
                2: (0.9, 0.5, 0.05)
            }

        depth_overlap_threshold = overlap_ratio_threshold[2][0]
        direction_overlap_threshold = overlap_ratio_threshold[self.direction][0]

        # Extract prototypes from data
        if len(self.prototypes) == 0:
            _ = self.extract_prototypes()
        stats['extracted'] = len(self.prototypes)

        # Filter for speed up
        if additional_filters:
            self.prototypes = self.filter_prototypes(min_height=3, min_width=3, min_n_points=10)
            stats['filtered_extracted'] = len(self.prototypes)

        # Concat connected (as puzzles) prototypes
        stats['after_connected_concat'] = {}

        for i in Notifier('t')(concat_iters):
            stats['after_connected_concat'][i] = []
            # Concat by depth axis
            _ = self.concat_connected_prototypes(overlap_ratio_threshold=depth_overlap_threshold,
                                                 axis=2)
            stats['after_connected_concat'][i].append(len(self.prototypes))

            # Concat by direction axis
            if (not prolongate_in_depth) or (depth_overlap_threshold <= overlap_ratio_threshold[2][1]):
                _ = self.concat_connected_prototypes(overlap_ratio_threshold=direction_overlap_threshold,
                                                    axis=self.direction)

                stats['after_connected_concat'][i].append(len(self.prototypes))

            # Early stopping
            if (depth_overlap_threshold <= overlap_ratio_threshold[2][1]) and \
               (direction_overlap_threshold <= overlap_ratio_threshold[self.direction][1]) and \
               (stats['after_connected_concat'][i][-1] == stats['after_connected_concat'][i-1][-1]):
                break

            depth_overlap_threshold = round(depth_overlap_threshold - overlap_ratio_threshold[2][-1], 2)
            depth_overlap_threshold = max(depth_overlap_threshold, overlap_ratio_threshold[2][1])

            if (not prolongate_in_depth) or (depth_overlap_threshold <= overlap_ratio_threshold[2][1]):
                direction_overlap_threshold = round(direction_overlap_threshold - \
                                                         overlap_ratio_threshold[self.direction][-1], 2)
                direction_overlap_threshold = max(direction_overlap_threshold,
                                                       overlap_ratio_threshold[self.direction][1])

        # Filter for speed up
        if additional_filters:
            self.prototypes = self.filter_prototypes(min_height=3, min_width=3, min_n_points=10)
            stats['filtered_connected_concat'] = len(self.prototypes)

        # Concat embedded
        _ = self.concat_embedded_prototypes()
        stats['after_embedded_concat'] = len(self.prototypes)

        # Split wrong objects
        _ = self.split_horseshoe()
        stats['after_split_horseshoe'] = len(self.prototypes)
        return self.prototypes, stats

    def filter_prototypes(self, min_height=40, min_width=20, min_n_points=100):
        """ Filer out unsuitable prototypes.

        min_height : int
            Minimal preferred prototype height (along the depth axis).
        min_width : int
            Minimal preferred prototype width (along the `self.direction` axis).
        min_n_points : int
            Minimal preferred points amount.

        Returns
        -------
        prototypes: list of the :class:`~.FaultPrototype` instances
            Filtered prototypes.
        """
        filtered_prototypes = []

        for prototype in self.prototypes:
            if (prototype.height >= min_height) and (prototype.width >= min_width) and \
               (prototype.n_points >= min_n_points):

                filtered_prototypes.append(prototype)

        return filtered_prototypes

    def prototypes_to_faults(self, field):
        """ Convert all prototypes to faults. """
        faults = [Fault(prototype.coords, field=field) for prototype in self.prototypes]
        return faults


class FromComponentExtractor(FaultExtractor):
    """ Extractor for finding prototypes from provided components.

    All you need is just to run the :meth:`~.extract_from_component`."""
    def extract_one_prototype(self, component, height_threshold=0.6):
        """ Extract one prototype from the point cloud starting from the provided component.

        Similar to the :meth:`.FaultExtractor.extract_one_prototype`, but this one finds prototype components to
        the left and right sides of the provided one, while the original one finds only to the right.

        Parameters
        ----------
        height_threshold : float in [0, 1] range, None
            Neighboring components height ratio threshold.
            If ratio more than threshold, then components are from one prototype. Otherwise not.
            If None, then no threshold is applied.
            On some slides component can be splitted into separate parts and these parts have a significant height
            decrease (most frequently in Y areas).
        """
        component_idx = np.argwhere(np.array(self.container[component.slide_idx]['components']) == component)[0][0]
        self.container[component.slide_idx]['lengths'][component_idx] = -1 # Mark component as merged

        prototype = FaultPrototype(points=component.points, direction=self.direction, last_component=component,
                                   proba_transform=self.proba_transform)

        # Find closest components on further slides
        self._find_prototype_components(prototype=prototype, component=component, slide_step=1,
                                        height_threshold=height_threshold)

        # Find closest components on previous slides
        first_slide_idx = prototype.bbox[self.direction, 0]
        first_slide_points = prototype.points[prototype.points[:, self.direction] == first_slide_idx]
        component = Component(points=first_slide_points, slide_idx=first_slide_idx)

        self._find_prototype_components(prototype=prototype, component=component, slide_step=-1,
                                        height_threshold=height_threshold)

        self.prototypes.append(prototype)
        return prototype

    def _find_prototype_components(self, prototype, component, slide_step, height_threshold=0.6):
        """ Find prototype components starting from the provided and going on `slide_step`.

        Similar to the :meth:`.FaultExtractor.extract_one_prototype`, but without split.

        Parameters
        ----------
        height_threshold : float in [0, 1] range, optional
            Neighboring components height ratio threshold.
            If ratio more than threshold, then components are from one prototype. Otherwise not.
            If None, then no threshold is applied.
            On some slides component can be splitted into separate parts and these parts have a significant height
            decrease (most frequently in Y areas).
        """
        stop_slide = self.ranges[self.direction][0] if slide_step < 0 else self.ranges[self.direction][1]
        previous_height = component.bbox[-2][1] - component.bbox[-2][0] + 1

        for next_slide in range(component.slide_idx + slide_step, stop_slide, slide_step):
            # Find the closest component on the the next slide to the current
            component, _ = self._find_closest_component(component=component, slide_idx=next_slide, depths_threshold=20)

            if component is not None:
                # TODO: think about splitting necessity
                height = component.bbox[-2][1] - component.bbox[-2][0] + 1
                heights_ratio = min(height, previous_height)/max(height, previous_height)

                if (height_threshold is not None) and (heights_ratio > height_threshold):
                    prototype.append(component)
                    previous_height = height
                else:
                    break
            else:
                break

        return prototype

    def find_similar_components(self, component):
        """ Find components similar to the provided one in the data container.

        Similar component is the closest component on the same slide as the provided.

        Similar to the :meth:`.FaultExtractor._find_closest_component`, but finds all close components,
        not the longest one.
        """
        # Dilate component bbox for detecting close components
        dilated_bbox = component.bbox.copy()
        dilated_bbox[self.orthogonal_direction, :] += (-self._dilation // 2, self._dilation // 2)
        dilated_bbox[self.orthogonal_direction, 0] = max(0, dilated_bbox[self.orthogonal_direction, 0])
        dilated_bbox[self.orthogonal_direction, 1] = min(dilated_bbox[self.orthogonal_direction, 1],
                                                         self.shape[self.orthogonal_direction])

        closest_components = []

        # Iter over components and find the closest one
        for other_component in self.container[component.slide_idx]['components']:
            # Check bboxes intersection
            if not bboxes_intersected(dilated_bbox, other_component.bbox, axes=(self.orthogonal_direction, 2)):
                continue

            # Check closeness of some points (as depth-wise distances)
            # Faster then component overlap, but not so accurate
            overlap_depths = (max(component.bbox[2, 0], other_component.bbox[2, 0]),
                              min(component.bbox[2, 1], other_component.bbox[2, 1]))

            # Select valid coords for distances finding
            valid_depths = component.coords[(component.coords[:, -1] >= overlap_depths[0]) & \
                                            (component.coords[:, -1] <= overlap_depths[1]), -1]

            indices_1 = np.in1d(component.coords[:, -1], valid_depths)
            indices_2 = np.in1d(other_component.coords[:, -1], valid_depths)

            coords_1 = component.coords[indices_1, self.orthogonal_direction]
            coords_2 = other_component.coords[indices_2, self.orthogonal_direction]

            components_distances = compute_distances(coords_1, coords_2, max_threshold=100)

            if (components_distances[0] == -1) or (components_distances[0] > 1):
                # Components are not close
                continue

            closest_components.append(other_component)

        return closest_components

    def extract_from_component(self, component):
        """ Extract prototypes which conclude the provided component. """
        prototypes = []
        closest_components = self.find_similar_components(component=component)

        for component_ in closest_components:
            prototype = self.extract_one_prototype(component=component_)
            prototypes.append(prototype)

        return prototypes


class Component:
    """ Extracted 2D connected component.

    Parameters
    ----------
    points : np.ndarray of (N, 4) shape
        Spatial coordinates and probabilities in the (ilines, xlines, depths, proba) format.
        Sorting is not required. Also usually `points` created in :class:`.~FaultExtractor` will be unsorted.
    slide_idx : int
        Index of the slide from which component was extracted.
    bbox : np.ndarray of (3, 2) shape
        3D bounding box.
    length : int
        Component length.
    """
    def __init__(self, points, slide_idx, bbox=None, length=None):
        self.points = points
        self.slide_idx = slide_idx

        self._bbox = bbox


    @property
    def coords(self):
        """ Spatial coordinates in the (ilines, xlines, depths) format."""
        return self.points[:, :3]

    @property
    def bbox(self):
        """ 3D bounding box. """
        if self._bbox is None:
            self._bbox = np.column_stack([np.min(self.coords, axis=0), np.max(self.coords, axis=0)])
        return self._bbox

    def __len__(self):
        """ Number of points in a component. """
        return len(self.points)


    def split(self, split_indices):
        """ Depth-wise component split by indices.

        Parameters
        ----------
        split_indices : sequence of two ints or None
            Depth values (upper and lower) to split component into parts. If None, then no need in split.

        Returns
        -------
        self : `~.Component` instance
            The most closest component to the `self` after split.
        new_components : list of `~.Component` instances
            Components created from splitted parts.
        """
        new_components = []

        # Cut upper part of the component and save it as another item
        if split_indices[0] is not None:
            # Extract closest part
            mask = self.points[:, 2] >= split_indices[0]
            self, new_component = self._split_by_mask(mask=mask)

            new_components.append(new_component)

        # Cut lower part of the component and save it as another item
        if split_indices[1] is not None:
            # Extract closest part
            mask = self.points[:, 2] <= split_indices[1]
            self, new_component = self._split_by_mask(mask=mask)

            new_components.append(new_component)

        return self, new_components

    def _split_by_mask(self, mask):
        """ Split component into two parts by boolean mask.

        Returns
        -------
        self : `~.Component` instance
            The most closest component to the `self` after split.
        new_component : `~.Component` instance
            Component created from splitted part.
        """
        # Create new Component from extra part
        new_component_points = self.points[~mask]
        new_component = Component(points=new_component_points, slide_idx=self.slide_idx)

        # Extract suitable part
        self.points = self.points[mask]
        self._bbox = None
        return self, new_component



class FaultPrototype:
    """ Class for faults prototypes. Provides a necessary API for convenient prototype extraction process.

    Note, the `last_component` parameter is preferred during the extraction from 3D volume and is optional:
    it is used for finding closest components on next slides.

    Parameters
    ----------
    points : np.ndarray of (N, 4) shape
        Prototype coordinates and probabilities.
        Sorting is not required. Also usually `points` created in :class:`.~FaultExtractor` will be unsorted.
    direction : {0, 1}
        Direction along which the prototype is extracted (the same as prediction direction).
    last_component : instance of :class:`~.Component`
        The last added component into prototype. Useful during the extraction from 3D data volume.
    """
    def __init__(self, points, direction, last_component=None, proba_transform=None):
        self.points = points
        self.direction = direction
        self.proba_transform = proba_transform

        self._bbox = None

        self._last_component = last_component

        self._contour = None
        self._borders = {}

    # Properties
    @property
    def coords(self):
        """ Spatial coordinates in (ilines, xlines, depth) format. """
        return self.points[:, :3]

    @property
    def bbox(self):
        """ 3D bounding box. """
        if self._bbox is None:
            self._bbox = np.column_stack([np.min(self.coords, axis=0), np.max(self.coords, axis=0)])
        return self._bbox

    # Stats for filtering
    @property
    def height(self):
        """ Height (along the depth axis). """
        return self.bbox[2, 1] - self.bbox[2, 0]

    @property
    def width(self):
        """ Width (along the `self.direction` axis). """
        return self.bbox[self.direction, 1] - self.bbox[self.direction, 0]

    @property
    def n_points(self):
        """ Amount of the surface points. """
        return len(self.points)

    @property
    def proba(self):
        """ 90% percentile of approximate proba values in [0, 1] interval. """
        proba_value = np.percentile(self.points[:, 3], 90) # is integer value from 0 to 255
        if self.proba_transform is not None:
            proba_value = self.proba_transform(proba_value)
        return proba_value

    @property
    def max_proba(self):
        """ Maximum of approximate proba values in [0, 1] interval. """
        proba_value = np.max(self.points[:, 3])
        if self.proba_transform is not None:
            proba_value = self.proba_transform(proba_value)
        return proba_value

    # Properties for internal needs
    @property
    def last_component(self):
        """ Last added component. """
        if self._last_component is None:
            last_slide_idx = self.points[:, self.direction].max()

            component_points = self.points[self.points[:, self.direction] == last_slide_idx]
            self._last_component = Component(points=component_points, slide_idx=last_slide_idx)
        return self._last_component


    # Contouring
    @property
    def contour(self):
        """ Contour of 2D projection on axis, orthogonal to the extraction direction.

        Note, output projection axis coordinates are zeros.
        """
        if self._contour is None:
            projection_axis = 1 - self.direction
            self._contour = find_contour(coords=self.coords, projection_axis=projection_axis)
        return self._contour

    def get_border(self, border, projection_axis):
        """ Get contour border.

        Parameters
        ----------
        border : {'up', 'down', 'left', 'right'}
            Which object border to get.
        projection_axis : {0, 1}
            Which projection is used to get the 2D contour coordinates.

        Returns
        -------
        border : np.ndarray of (N, 3) shape
            Sorted coordinates of the requested border.
        """
        if border not in self._borders:
            # Delete extra border from contour
            # For border removing we apply groupby which works only for the last axis, so we swap axes coords
            if border in ('left', 'right'):
                border_coords = self.contour.copy()
                border_coords[:, [-1, 1-projection_axis]] = border_coords[:, [1-projection_axis, -1]]
                border_coords = border_coords[border_coords[:, 1-projection_axis].argsort()] # Groupby needs sorted data
            else:
                border_coords = self.contour

            # Delete border by applying groupby
            if border not in ('up', 'left'):
                border_coords = groupby_max(border_coords)
            else:
                border_coords = groupby_min(border_coords)

            # Restore 3D coordinates
            projection_axis = 1 - self.direction

            if border in ('left', 'right'):
                border_coords[:, [-1, 1-projection_axis]] = border_coords[:, [1-projection_axis, -1]]

            border_coords = restore_coords_from_projection(coords=self.coords, projection_buffer=border_coords,
                                                           axis=projection_axis)
            self._borders[border] = border_coords

        return self._borders[border]


    # Extension operations
    def append(self, component):
        """ Append new component into prototype.

        Parameters
        ----------
        component : instance of :class:`~.Component`
            Component to add into the prototype.
        """
        self.points = np.vstack([self.points, component.points])

        self._contour = None
        self._borders = {}

        self._last_component = component

        self._bbox = self._concat_bbox(component.bbox)

    def concat(self, other):
        """ Concatenate two prototypes. """
        self.points = np.vstack([self.points, other.points])

        self._bbox = self._concat_bbox(other.bbox)

        self._contour = None
        self._borders = {}

        self._last_component = None

    def _concat_bbox(self, other_bbox):
        """ Concat bboxes of two objects into one. """
        bbox = np.empty((3, 2), np.int32)
        bbox[:, 0] = np.min((self.bbox[:, 0], other_bbox[:, 0]), axis=0)
        bbox[:, 1] = np.max((self.bbox[:, 1], other_bbox[:, 1]), axis=0)
        return bbox


    # Split operations
    def split(self, split_indices, axis=2):
        """ Axis-wise prototype split by indices.

        Parameters
        ----------
        split_indices : sequence of two ints or None
            Axis values (upper and lower) to split prototype into parts. If None, then no need in split.

        Returns
        -------
        prototype : `~.FaultPrototype` instance
            The closest prototype to the `self` after splitting.
        new_prototypes : list of `~.FaultPrototype` instances
            Prototypes created from splited parts.
        """
        new_prototypes = []

        # No splitting applied
        if (split_indices[0] is None) and (split_indices[1] is None):
            return self, new_prototypes

        objects_separating_axis = self.direction if axis in (-1, 2) else 2

        # Cut upper part and separate disconnected objects
        if (split_indices[0] is not None) and \
           (np.min(self.points[:, axis]) < split_indices[0] < np.max(self.points[:, axis])):
            mask = self.points[:, axis] >= split_indices[0]

            self, new_prototypes_ = self._split_by_mask(mask=mask, objects_separating_axis=objects_separating_axis)
            new_prototypes.extend(new_prototypes_)

        # Cut lower part and separate disconnected objects
        if (split_indices[1] is not None) and \
           (np.min(self.points[:, axis]) < split_indices[1] < np.max(self.points[:, axis])):
            mask = self.points[:, axis] <= split_indices[1]

            self, new_prototypes_ = self._split_by_mask(mask=mask, objects_separating_axis=objects_separating_axis)
            new_prototypes.extend(new_prototypes_)

        new_prototypes.extend(self._separate_objects(self.points, axis=objects_separating_axis))

        # Update self
        self.points = new_prototypes[-1].points
        self._bbox = new_prototypes[-1].bbox
        self._contour = None
        self._borders = {}
        self._last_component = None
        return self, new_prototypes[:-1]

    def _split_by_mask(self, mask, objects_separating_axis):
        """ Split prototype into parts by boolean mask.

        Returns
        -------
        prototype : `~.FaultPrototype` instance
            The closest prototype to the `self` after splitting.
        new_prototypes : list of `~.FaultPrototype` instances
            Prototypes created from splited parts.
        """
        # Create new prototypes from extra part
        new_prototype_points = self.points[~mask]

        if len(new_prototype_points) > 0:
            new_prototypes = self._separate_objects(new_prototype_points, axis=objects_separating_axis)

        # Extract suitable part
        self.points = self.points[mask]
        return self, new_prototypes

    def _separate_objects(self, points, axis):
        """ Separate points into different objects depend on their connectedness.

        After split we can have the situation when splitted part has more than one connected item.
        This method separate disconnected parts into different prototypes.

        Returns
        -------
        prototypes: list of the :class:`~.FaultPrototype` instances
            Prototypes created from the separated parts.
        """
        # Get coordinates along the axis
        unique_direction_points = np.unique(points[:, axis])

        # Slides distance more than 1 -> different objects
        split_indices = np.nonzero(unique_direction_points[1:] - unique_direction_points[:-1] > 1)[0]

        if len(split_indices) == 0:
            return [FaultPrototype(points=points, direction=self.direction, proba_transform=self.proba_transform)]

        # Separate disconnected objects and create new prototypes instances
        start_indices = unique_direction_points[split_indices + 1]
        start_indices = np.insert(start_indices, 0, 0)

        end_indices = unique_direction_points[split_indices]
        end_indices = np.append(end_indices, unique_direction_points[-1])

        prototypes = []

        for start_idx, end_idx in zip(start_indices, end_indices):
            points_ = points[(start_idx <= points[:, axis]) & (points[:, axis] <= end_idx)]
            prototype = FaultPrototype(points=points_, direction=self.direction, proba_transform=self.proba_transform)
            prototypes.append(prototype)

        return prototypes



# Helpers
def get_range(coords, axis, diff_threshold=2):
    """ Get the longest sequential range of coords on axis.

    Helper for the :meth:`~.FaultExtractor._borders_overlap`.

    Parameters
    ----------
    coords : set of tuples of three ints
        Coordinates in (iline, xline, depth) format.
    axis : {0, 1, 2}
        Axis on which to get coordinates range.
    diff_threshold : int
        Maximal possible difference between points values in one sequential range.

    Returns
    -------
    overlap_range : tuple of two ints
        The longest sequential range on the `axis` for provided coords.
        Sequential range is the orderly sequence of coords with difference in values not more than `diff_threshold`.
    """
    # Extract values
    values = list(set(elem[axis] for elem in coords))
    values.sort()

    # Find sequential ranges
    diff = np.diff(values)

    split_indices = np.argwhere(diff > diff_threshold).reshape(-1)

    if len(split_indices) == 0:
        return (np.min(values), np.max(values))

    ranges = [split_indices[0] + 1, *np.diff(split_indices), len(values) - split_indices[-1] - 1]

    # Find the longest range
    longest_range_idx = np.argmax(ranges)

    if longest_range_idx == len(ranges) - 1:
        range_ = (split_indices[-1] + 1, len(values) - 1)
    elif longest_range_idx == 0:
        range_ = (0, split_indices[0])
    else:
        range_ = (split_indices[longest_range_idx-1] + 1, split_indices[longest_range_idx])

    return (values[range_[0]], values[range_[1]])
