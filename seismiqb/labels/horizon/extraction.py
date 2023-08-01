""" Method for horizon extraction from 3D volume and their merging. """
from enum import IntEnum
from time import perf_counter
from operator import attrgetter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from cv2 import dilate
from numba import njit

from cc3d import connected_components
from scipy.ndimage import find_objects

from ...utils import MetaDict, groupby_all

class MergeStatus(IntEnum):
    """ Possible outcomes of the `:meth:~ExtractionMixin.verify_merge`.
    Values describe the relative position of two horizons.
    """
    # Definetely not mergeable
    DEPTH_SEPARATED = 0
    SPATIALLY_SEPARATED = 1
    TOO_SMALL_OVERLAP = 2

    # Maybe can merge
    SPATIALLY_ADJACENT = 3

    # Definetely merge
    OVERLAPPING = 4

    # Too big of an overlap: no reason to merge
    TOO_BIG_OVERLAP = 5



class ExtractionMixin:
    """ Methods for horizon extraction from 3D volumes and their later merge. """
    #pylint: disable=too-many-statements, too-many-nested-blocks, line-too-long, protected-access
    @classmethod
    def extract_from_mask(cls, mask, field=None, origin=None, minsize=1000,
                          prefix='extracted', verbose=False, max_iters=999):
        """ Extract separate horizon instances from subvolume.

        Basic idea is to find all the connected regions inside the subvolume and mark them as individual horizons.
        Some of the surfaces touch (either because of being too close or accidentally). To separate them we need to:
            - if the connected point cloud has multiple labeled points at each trace, then we split it into three parts.
            One is the minimum envelope of point cloud, the other is the maximum envelope,
            and the third consists of the points where each trace contains only one point.
            - each of the three point clouds is split into connected regions itself, which are considered to be
            individual horizons. We need this step to separate surfaces that are connected by the third point cloud.

        After extracting some points as horizon instance, we remove those points from subvolume.
        The above is repeated until `max_iters` is reached or no points remain in the original array.

        Returned list of horizons is sorted by length of horizons.
        The entire procedure is heavily logged, providing timings and statistics in a separate returned dictionary.

        Parameters
        ----------
        field : Field
            Horizon parent field.
        origin : sequence
            The upper left coordinate of a `mask` in the cube coordinates.
        minsize : int
            Minimum length of a horizon to be extracted.
        prefix : str
            Name of horizon to use.
        verbose : bool
            Whether to print some of the intermediate statistics.
        max_iters : int
            Maximum number of outer iterations (re-labeling the whole cube, extracting surfaces, deleting points).

        Returns
        -------
        (list_of_horizons, stats_dict)
            Tuple with the list of extracted instances as the first element and logging stats as the second.
        """
        mask = mask.copy()
        total_points = int(mask.sum())

        # `num` prefix for horizon count in category, `total` prefix for number of points in category
        stats = MetaDict({
            "measurement_timings" : [],
            "iteration_timings" : [],

            "num_objects" : [],
            "num_deleted" : [],
            "num_extracted_easy" : [],
            "num_extracted_hard" : [],

            "num_easy" : 0,
            "num_hard" : 0,
            "num_hard_separated" : 0,
            "num_hard_joined" : 0,
            "num_extracted_from_easy" : 0,
            "num_extracted_from_hard" : 0,

            "total_remaining_points" : [f'{total_points:,}'],
            "total_deleted_points" : [],
        })

        if verbose:
            print(f'Starting from {total_points:,} points in the mask')

        horizons = []
        for _ in range(max_iters):
            start_timing = perf_counter()
            num_deleted = 0
            num_extracted_easy = 0
            num_extracted_hard = 0
            total_deleted_points = 0
            total_extracted_points = 0

            # Label connected entities
            labeled = connected_components(mask)
            objects = find_objects(labeled)
            stats['measurement_timings'].append(round(perf_counter() - start_timing, 2))
            stats['num_objects'].append(len(objects))

            for i, slices in enumerate(objects):
                # Point cloud in `slices` coordinates
                indices = np.nonzero(labeled[slices] == i + 1)
                points = np.vstack(indices).T

                # iline, crossline, occurencies_ix, min_ix, max_ix, mean_ix
                grouped_points = groupby_all(points)

                if len(points) == len(grouped_points):
                    # Point-cloud extracted cleanly, with no ambiguous points
                    stats['num_easy'] += 1

                    # Remove from the original mask
                    horizon_points = points + [slc.start or 0 for slc in slices]
                    mask[horizon_points[:, 0], horizon_points[:, 1], horizon_points[:, 2]] = 0
                    num_deleted += 1
                    total_deleted_points += len(horizon_points)

                    # Horizon points validation: can be expanded
                    if len(points) < minsize:
                        continue

                    horizons.append(horizon_points)
                    num_extracted_easy += 1
                    total_extracted_points += len(horizon_points)
                    stats['num_extracted_from_easy'] += 1

                else:
                    # Point-cloud contains multiple points at some traces
                    # Extract min/max envelopes, remove them from original mask, repeat the process
                    # We use separate indexing for spatial/depth coordinates to avoid advanced indexing for columns
                    stats['num_hard'] += 1

                    mask_min_max = (grouped_points[:, 3] == grouped_points[:, 4])
                    if mask_min_max.any():
                        # Joined surface: min==max
                        at_joined = grouped_points[mask_min_max]
                        ix_joined = at_joined[:, :2]
                        h_joined = at_joined[:, 3]

                        # Two separate surfaces: min < max
                        # Can't be empty: if that is the case, surfaces would be identical -> extracted cleanly
                        at_separated = grouped_points[~mask_min_max]
                        ix_separated = at_separated[:, :2]
                        min_separated = at_separated[:, 3]
                        max_separated = at_separated[:, 4]

                        surfaces = [(ix_joined, h_joined),
                                    (ix_separated, min_separated),
                                    (ix_separated, max_separated)]
                        stats['num_hard_joined'] += 1
                    else:
                        # Two separate surfaces: min < max
                        ix = grouped_points[:, :2]
                        min_ = grouped_points[:, 3]
                        max_ = grouped_points[:, 4]

                        surfaces = [(ix, min_),
                                    (ix, max_)]
                        stats['num_hard_separated'] += 1

                    # Put each surface on the blank array and extract connected components
                    # Then, remove corresponding points from the original array
                    shape = tuple(slc.stop - slc.start for slc in slices)

                    for ix_coords, h_coords in surfaces:
                        background = np.zeros(shape, dtype=np.int8)
                        background[ix_coords[:, 0], ix_coords[:, 1], h_coords] = 1

                        inner_labeled = connected_components(background)
                        inner_objects = find_objects(inner_labeled)

                        for inner_i, inner_slices in enumerate(inner_objects):
                            inner_indices = np.nonzero(background[inner_slices] == inner_i + 1)
                            inner_points = np.vstack(inner_indices).T

                            # Remove from the original mask
                            horizon_points = inner_points + [(slc.start or 0) + (inner_slc.start or 0)
                                                             for slc, inner_slc in zip(slices, inner_slices)]
                            mask[horizon_points[:, 0], horizon_points[:, 1], horizon_points[:, 2]] = 0
                            num_deleted += 1
                            total_deleted_points += len(horizon_points)

                            # Horizon points validation: can be expanded
                            if len(inner_points) < minsize:
                                continue

                            horizons.append(horizon_points)

                            num_extracted_hard += 1
                            total_extracted_points += len(horizon_points)
                            stats['num_extracted_from_hard'] += 1

            # Log iteration stats
            total_points -= total_deleted_points
            stats['iteration_timings'].append(round(perf_counter() - start_timing, 2))

            stats['num_deleted'].append(num_deleted)
            stats['num_extracted_easy'].append(num_extracted_easy)
            stats['num_extracted_hard'].append(num_extracted_hard)

            stats['total_remaining_points'].append(f'{total_points:,}')
            stats['total_deleted_points'].append(total_deleted_points)

            if verbose:
                print(f'Remaining points in the mask: {total_points:,}, num deleted: {num_deleted:>5}, '
                      f'total extracted points {total_extracted_points:>7,}, '
                      f'extracted easy: {num_extracted_easy:>3}, extracted hard: {num_extracted_hard:>3}')

            if num_deleted == 0:
                break

        # Make `Horizon` instances
        horizons.sort(key=len)
        horizons = [cls(horizon_points + origin, field=field, name=f'{prefix}_{i}')
                    for i, horizon_points in enumerate(horizons)]
        return horizons, stats


    def verify_merge(self, other, mean_threshold=1.0, max_threshold=2, adjacency=0,
                     min_size_threshold=1, max_size_threshold=None):
        """ Compute the relative position of two horizons, based on the thresholding parameters.

        Comparison is split into multiple parts:
            - the simplest check is to look at horizons bounding boxes.
            This check is performed with the `adjacency` in mind: by using it, one can allow touching horizons or
            with gaps of multiple pixels. This parameter's detailed description is below.
            If the bboxes are too far along ilines/crosslines, then horizons are SPATIALLY_SEPARATED

            - otherwise, we compare horizon matrices on overlap of their bboxes.
                - if the size of actual overlap is 0, then horizons are SPATIALLY_ADJACENT

                - if the size of actual overlap is lower than the `min_size_threshold`:
                    - if the mean difference on overlap is lower  than the `mean_threshold`, then horizons are TOO_SMALL_OVERLAP
                    - if the mean difference on overlap is bigger than the `mean_threshold`, then horizons are DEPTH_SEPARATED

                - if the size of actual overlap is bigger than the `max_size_threshold`:
                    - if the mean difference on overlap is lower  than the `mean_threshold`, then horizons are TOO_BIG_OVERLAP
                    - if the mean difference on overlap is bigger than the `mean_threshold`, then horizons are DEPTH_SEPARATED

                - otherwise, the size of overlap is within allowed bounds:
                    - if the mean difference on overlap is lower  than the `mean_threshold`, then horizons are OVERLAPPING
                    - if the mean difference on overlap is bigger than the `mean_threshold`, then horizons are DEPTH_SEPARATED

        Parameters
        ----------
        self, other
            Horizons to compare.
        adjacency : int or sequence of ints
            Allowed size of the gaps between bounding boxes along each of the axis:
            - adjacency = +0 means trying to merge horizons with overlap    of 1 pixel
            - adjacency = +1 means trying to merge horizons with touching   boundaries
            - adjacency = +2 means trying to merge horizons with gap        of 1 pixel between bboxes

            By using negative values, one can require the overlap of bboxes:
            - adjacency = -1 means trying to merge horizons with overlap    of 2 pixel
            If the integer is passed, then the same adjacency rules apply along both iline and crossline directions.

        mean_threshold : number
            Allowed mean difference on horizons overlap.
        max_threshold : number
            Allowed max difference on horizons overlap.
        min_size_threshold : int
            Minimum allowed size of the horizons overlap.
        max_size_threshold : int
            Maximum allowed size of the horizons overlap. Used only if explicitly passed.
            Can be used to refrain from merging `almost completely the same horizons`.

        Returns
        -------
        MergeStatus to describe relative positions of `self` and `other` horizons.
        """
        # Adjacency parsing
        adjacency = adjacency if isinstance(adjacency, tuple) else (adjacency, adjacency)
        adjacency_i, adjacency_x = adjacency

        # Overlap bbox
        overlap_i_min, overlap_i_max = max(self.i_min, other.i_min), min(self.i_max, other.i_max) + 1
        overlap_x_min, overlap_x_max = max(self.x_min, other.x_min), min(self.x_max, other.x_max) + 1

        overlap_size_i = overlap_i_max - overlap_i_min
        overlap_size_x = overlap_x_max - overlap_x_min

        # Simplest possible check: horizon bboxes are too far from each other
        if overlap_size_i < 1 - adjacency_i or overlap_size_x < 1 - adjacency_x:
            status = MergeStatus.SPATIALLY_SEPARATED
        else:
            status = MergeStatus.SPATIALLY_ADJACENT


        # Compare matrices on overlap without adjacency:
        if status != 1 and overlap_size_i > 0 and overlap_size_x > 0:
            self_overlap = self.matrix[overlap_i_min - self.i_min:overlap_i_max - self.i_min,
                                       overlap_x_min - self.x_min:overlap_x_max - self.x_min]
            other_overlap = other.matrix[overlap_i_min - other.i_min:overlap_i_max - other.i_min,
                                        overlap_x_min - other.x_min:overlap_x_max - other.x_min]

            mean_on_overlap, size_of_overlap = intersect_matrix(self_overlap, other_overlap, max_threshold)

            if size_of_overlap == 0:
                # bboxes are overlapping, but horizons are not
                status = MergeStatus.SPATIALLY_ADJACENT

            elif size_of_overlap < min_size_threshold:
                # the overlap is too small
                if mean_on_overlap < mean_threshold:
                    status = MergeStatus.TOO_SMALL_OVERLAP
                else:
                    status = MergeStatus.DEPTH_SEPARATED
            elif max_size_threshold is not None and size_of_overlap > max_size_threshold:
                if mean_on_overlap < mean_threshold:
                    status = MergeStatus.TOO_BIG_OVERLAP
                else:
                    status = MergeStatus.DEPTH_SEPARATED
            else: # min_size_threshold <= size_of_overlap <= max_size_threshold
                if mean_on_overlap <= mean_threshold:
                    status = MergeStatus.OVERLAPPING
                else:
                    status = MergeStatus.DEPTH_SEPARATED

        return status


    def overlap_merge(self, other, inplace=False):
        """ Merge two horizons into one. Values on overlap are the floored average from the `self` and `other` values.
        Can either merge horizons in-place of the first one (`self`), or create a new instance.

        TODO: this function is optimized to use only `matrix` storage from both of the horizons,
        which is the most optimal in current paradigm. Implement the other ways to merge horizons
        to use them accordingly, i.e. `merge_to_matrix_by_points`, `merge_to_points_by_points` methods.
        """
        # Create shared background for both horizons
        shared_i_min, shared_i_max = min(self.i_min, other.i_min), max(self.i_max, other.i_max)
        shared_x_min, shared_x_max = min(self.x_min, other.x_min), max(self.x_max, other.x_max)

        background = np.zeros((shared_i_max - shared_i_min + 1, shared_x_max - shared_x_min + 1),
                              dtype=np.int32)

        # Coordinates inside shared for `self` and `other`
        shared_self_i_min, shared_self_x_min = self.i_min - shared_i_min, self.x_min - shared_x_min
        shared_other_i_min, shared_other_x_min = other.i_min - shared_i_min, other.x_min - shared_x_min

        # Add both horizons to the background
        background[shared_self_i_min:shared_self_i_min+self.i_length,
                   shared_self_x_min:shared_self_x_min+self.x_length] += self.matrix

        background[shared_other_i_min:shared_other_i_min+other.i_length,
                   shared_other_x_min:shared_other_x_min+other.x_length] += other.matrix

        # Correct overlapping points
        overlap_i_min, overlap_i_max = max(self.i_min, other.i_min), min(self.i_max, other.i_max) + 1
        overlap_x_min, overlap_x_max = max(self.x_min, other.x_min), min(self.x_max, other.x_max) + 1

        overlap_i_min -= shared_i_min
        overlap_i_max -= shared_i_min
        overlap_x_min -= shared_x_min
        overlap_x_max -= shared_x_min

        overlap = background[overlap_i_min:overlap_i_max, overlap_x_min:overlap_x_max]
        mask = overlap >= 0
        overlap[mask] //= 2
        overlap[~mask] -= self.FILL_VALUE
        background[overlap_i_min:overlap_i_max, overlap_x_min:overlap_x_max] = overlap

        background[background == 0] = self.FILL_VALUE
        length = len(self) + len(other) - mask.sum()

        # Create new instance or change `self`
        if inplace:
            # Change `self` inplace, mark `other` as merged into `self`
            self.from_matrix(background, i_min=shared_i_min, x_min=shared_x_min,
                             d_min=min(self.d_min, other.d_min),
                             d_max=max(self.d_max, other.d_max), length=length)
            self.reset_storage('points', reset_cache=False)
            other.already_merged = id(self)
            merged = True
        else:
            # Return a new instance of horizon
            merged = type(self)(storage=background, field=self.field, name=self.name,
                                i_min=shared_i_min, x_min=shared_x_min,
                                d_min=min(self.d_min, other.d_min),
                                d_max=max(self.d_max, other.d_max), length=length)
        return merged


    def adjacent_merge(self, other, mean_threshold=3.0, adjacency=3, inplace=False):
        """ Check if adjacent merge (that is merge with some margin) is possible, and, if needed, merge horizons.
        Can either merge horizons in-place of the first one (`self`), or create a new instance.

        TODO: this function may be outdated and should be used with caution.

        Parameters
        ----------
        self, other : :class:`.Horizon` instances
            Horizons to merge.
        mean_threshold : number
            Depth threshold for mean distances.
        adjacency : int
            Margin to consider horizons close (spatially).
        inplace : bool
            Whether to create new instance or update `self`.
        """
        # Adjacency parsing
        adjacency = adjacency if isinstance(adjacency, tuple) else (adjacency, adjacency)
        adjacency_i, adjacency_x = adjacency

        # Simplest possible check: horizons are too far away from one another (depth-wise)
        overlap_d_min, overlap_d_max = max(self.d_min, other.d_min), min(self.d_max, other.d_max)
        if overlap_d_max - overlap_d_min < 0:
            return False

        # Create shared background for both horizons
        shared_i_min, shared_i_max = min(self.i_min, other.i_min), max(self.i_max, other.i_max)
        shared_x_min, shared_x_max = min(self.x_min, other.x_min), max(self.x_max, other.x_max)

        background = np.zeros((shared_i_max - shared_i_min + 1, shared_x_max - shared_x_min + 1), dtype=np.int32)

        # Coordinates inside shared for `self` and `other`
        shared_self_i_min, shared_self_x_min = self.i_min - shared_i_min, self.x_min - shared_x_min
        shared_other_i_min, shared_other_x_min = other.i_min - shared_i_min, other.x_min - shared_x_min

        # Put the second of the horizons on background
        background[shared_other_i_min:shared_other_i_min+other.i_length,
                   shared_other_x_min:shared_other_x_min+other.x_length] += other.matrix

        # Enlarge the image to account for adjacency
        kernel = np.ones((1 + 2*adjacency_i, 1 + 2*adjacency_x), np.float32)
        dilated_background = dilate(background.astype(np.float32), kernel).astype(np.int32)

        # Make counts: number of horizons in each point; create indices of overlap
        counts = (dilated_background > 0).astype(np.int32)
        counts[shared_self_i_min:shared_self_i_min+self.i_length,
               shared_self_x_min:shared_self_x_min+self.x_length] += (self.matrix > 0).astype(np.int32)
        counts_idx = counts == 2

        # Determine whether horizon can be merged (adjacent and depth-close) or not
        mergeable = False
        if counts_idx.any():
            # Put the first horizon on dilated background, compute mean
            background[shared_self_i_min:shared_self_i_min+self.i_length,
                       shared_self_x_min:shared_self_x_min+self.x_length] += self.matrix

            # Compute diffs on overlap
            diffs = background[counts_idx] - dilated_background[counts_idx]
            diffs = np.abs(diffs)
            diffs = diffs[diffs < (-self.FILL_VALUE // 2)]

            if len(diffs) != 0 and np.mean(diffs) < mean_threshold:
                mergeable = True

        if mergeable:
            background[(background < 0) & (background != self.FILL_VALUE)] -= self.FILL_VALUE
            background[background == 0] = self.FILL_VALUE

            length = len(self) + len(other) # since there is no direct overlap

            # Create new instance or change `self`
            if inplace:
                # Change `self` inplace, mark `other` as merged into `self`
                self.from_matrix(background, i_min=shared_i_min, x_min=shared_x_min,
                                 d_min=min(self.d_min, other.d_min),
                                 d_max=max(self.d_max, other.d_max), length=length)
                self.reset_storage('points', reset_cache=False)
                other.already_merged = id(self)
                merged = True
            else:
                # Return a new instance of horizon
                merged = type(self)(storage=background, field=self.field, name=self.name,
                                    i_min=shared_i_min, x_min=shared_x_min,
                                    d_min=min(self.d_min, other.d_min),
                                    d_max=max(self.d_max, other.d_max),
                                    length=length)
            return merged
        return False


    def merge_into(self, horizons, mean_threshold=1., max_threshold=1.2, min_size_threshold=1, max_size_threshold=None,
                   adjacency=1, max_iters=999, num_merged_threshold=1):
        """ Try to merge instances from the list of `horizons` into `self`.

        For each horizon in the list, we check the possibility to merge it into the current `self` horizon:
            - first of all, we select candidates to merge by optimized bbox check
            - for each of the candidates:
                - we use the `verify_merge` to better check the relations between `candidate` and `self`
                - depending on the status, merge the candidate into `self` and remove it from the list.
        The above is repeated until `max_iters` is reached or no horizon from the list can be merged to `self`.

        Parameters
        ----------
        horizons : sequence
            Horizons to merge into `self`.
        adjacency, mean_threshold, max_threshold, min_size_threshold, max_size_threshold : number
            Parameters for `:meth:~.verify_merge`.
        max_iters : int
            Maximum number of outer iterations (computing candidates, merging them and deleting from the original list).
        num_merged_threshold : int
            Minimum amount of merged horizons at outer iteration to perform the next iteration.
            If the number of merged instances is less than this threshold, we break out of the outer loop.

        Returns
        -------
        (self, horizons, stats_dict)
            A tuple with:
            - an instance where some of the items in `horizons` were merged to
            - remaining horizons
            - dictionary with timings and statistics
        """
        if isinstance(horizons, (tuple, list)):
            horizons = np.array([horizon for horizon in horizons if not horizon.already_merged])

        # Pre-compute all the bounding boxes
        bboxes = np.array([horizon.raveled_bbox for horizon in horizons], dtype=np.int32).reshape(-1, 6)

        # Adjacency parsing
        adjacency = adjacency if isinstance(adjacency, tuple) else (adjacency, adjacency)
        adjacency_i, adjacency_x = adjacency

        # Keep track of stats
        merge_stats = defaultdict(int)
        merge_stats.update({'iteration_timings' : [],
                            'merge_candidates': [],
                            'merges' : []})

        for _ in range(max_iters):
            start_timing = perf_counter()
            num_merged = 0
            indices_merged = set()

            # Iline-axis
            overlap_min_i = np.maximum(bboxes[:, 0], self.raveled_bbox[0])
            overlap_max_i = np.minimum(bboxes[:, 1], self.raveled_bbox[1]) + 1
            overlap_size_i = overlap_max_i - overlap_min_i
            mask_i = (overlap_size_i >= 1 - adjacency_i)
            indices_i = np.nonzero(mask_i)[0]
            bboxes_i = bboxes[indices_i]

            # Crossline-axis
            overlap_min_x = np.maximum(bboxes_i[:, 2], self.raveled_bbox[2])
            overlap_max_x = np.minimum(bboxes_i[:, 3], self.raveled_bbox[3]) + 1
            overlap_size_x = overlap_max_x - overlap_min_x
            mask_x = (overlap_size_x >= 1 - adjacency_x)
            indices_x = np.nonzero(mask_x)[0]
            bboxes_x = bboxes_i[indices_x]

            # depth-axis: other threshold
            overlap_min_h = np.maximum(bboxes_x[:, 4], self.raveled_bbox[4])
            overlap_max_h = np.minimum(bboxes_x[:, 5], self.raveled_bbox[5]) + 1
            overlap_size_h = overlap_max_h - overlap_min_h
            mask_h = (overlap_size_h >= 1)
            indices_h = np.nonzero(mask_h)[0]
            bboxes_h = bboxes_x[indices_h]

            indices = indices_i[indices_x][indices_h]
            merge_candidates = horizons[indices]
            _ = bboxes_h

            merge_stats['merge_candidates'].append(len(indices))

            # Merge all possible candidates
            for idx, merge_candidate in zip(indices, merge_candidates):
                # Compute the mergeability
                merge_status = ExtractionMixin.verify_merge(self, merge_candidate,
                                                            mean_threshold=mean_threshold,
                                                            max_threshold=max_threshold,
                                                            min_size_threshold=min_size_threshold,
                                                            max_size_threshold=max_size_threshold,
                                                            adjacency=adjacency)
                merge_stats[merge_status] += 1

                # Merge, if needed
                if merge_status == 4:
                    # Overlapping horizons: definitely merge
                    merged = ExtractionMixin.overlap_merge(self, merge_candidate, inplace=True)

                elif merge_status == 3 and (adjacency_i > 0 or adjacency_x > 0):
                    # Adjacent horizons: maybe we can merge it
                    merged = ExtractionMixin.adjacent_merge(self, merge_candidate, inplace=True,
                                                            mean_threshold=mean_threshold,
                                                            adjacency=adjacency)
                    merge_stats['merged_adjacent'] += (1 if merged else 0)
                else:
                    # Spatially separated or too small of an overlap
                    # Can't merge for now, but maybe will be able later
                    merged = False

                # Keep values for clean-up
                if merged:
                    indices_merged.add(idx)
                    num_merged += 1

            # Once in a while, remove merged horizons from `bboxes` and `horizons` arrays
            if indices_merged:
                indices_merged = list(indices_merged)
                horizons = np.delete(horizons, indices_merged, axis=0)
                bboxes = np.delete(bboxes, indices_merged, axis=0)

                merge_stats['num_deletes'] += 1

            # Global iteration info
            merge_stats['iterations'] += 1
            merge_stats['merges'].append(num_merged)
            merge_stats['iteration_timings'].append(round(perf_counter() - start_timing, 2))

            # Exit condition: merged less horizons then threshold
            if num_merged < num_merged_threshold or len(horizons) == 0:
                break

        return self, horizons, MetaDict(merge_stats)

    @staticmethod
    def merge_list(horizons, mean_threshold=1., max_threshold=2.2,
                   min_size_threshold=1, max_size_threshold=None, adjacency=1,
                   max_iters=999, num_merged_threshold=1, delete_threshold=0.01):
        """ Merge each horizon to each in the `horizons`, until no merges are possible.

        Under the hood, we start by computing the bboxes of all horizons. Then, for each horizon we:
            - first of all, we select candidates to merge by optimized bbox check
            - for each of the candidates:
                - we use the `verify_merge` to better check the relations between `candidate` and `self`
                - depending on the status, merge the candidate into `self`
                - remember the index of the candidate to clean up the `horizons` and `bboxes` arrays later.
        The above is repeated until `max_iters` is reached or no two horizons from the list can be merged.

        We clean-up the `horizons` and `bboxes` only occasionnaly to amortize the costs of the deletion operation.
        Other optimization is to flag horizons as unmerged at outer iteration: if horizon was not merged to any other,
        then it would not be merged at any of the subsequent iterations.

        The entire procedure is heavily logged, providing timings and statistics in a separate returned dictionary.

        Parameters
        ----------
        adjacency, mean_threshold, max_threshold, min_size_threshold, max_size_threshold : number
            Parameters for `:meth:~.verify_merge`.
        max_iters : int
            Maximum number of outer iterations (computing bboxes, merging each horizon with all possible candidates,
            deleting from the original list).
        num_merged_threshold : int
            Minimum amount of merged horizons at outer iteration to perform the next iteration.
            If the number of merged instances is less than this threshold, we break out of the outer loop.

        Returns
        -------
        (horizons, stats_dict)
            A tuple with:
            - remaining horizons
            - dictionary with timings and statistics
        """
        # Adjacency parsing
        adjacency = adjacency if isinstance(adjacency, tuple) else (adjacency, adjacency)
        adjacency_i, adjacency_x = adjacency

        # Flag all horizons. If at some iteration the horizon is not merged to any other,
        # it would not be merged at all (i.e. rejected)
        for horizon in horizons:
            horizon.merge_count = 0
            horizon.id_separated = set()
        horizons = np.array(horizons)
        rejected_horizons = []

        # Keep track of stats. Pretty much no overhead to the procedure
        merge_stats = defaultdict(int)
        merge_stats.update({'global_iteration_timings' : [],
                            'global_merges' : [],
                            'num_rejected_horizons' : []})

        # Global iteration: iterate over the entire list, comparing each horizon to each other
        for _ in range(max_iters):
            start_timing = perf_counter()
            num_merged = 0
            indices_merged = set() # used to periodically clean-up arrays

            # Pre-compute all the bounding boxes
            bboxes = np.array([horizon.raveled_bbox for horizon in horizons], dtype=np.int32)

            # Cycle for the base horizons. As we are removing merged horizons from the list, we iterate with `while`
            i = 0
            while True:
                if i >= len(horizons):
                    break
                if i in indices_merged:
                    i += 1
                    continue

                current_horizon = horizons[i]
                current_bbox = bboxes[i]

                # Filter: keep only overlapping/adjacent horizons
                # Compute bbox overlaps: `overlap_size` > 0 means size of common pixels along the axis
                #                        `overlap_size` = 0 means touching along the axis
                #                        `overlap_size` < 0 means size of gap between horizons along the axis

                # adjacency = -1 -> try to merge horizons with overlap  of 2 pixels
                # adjacency = +0 -> try to merge horizons with overlap  of 1 pixel
                # adjacency = +1 -> try to merge horizons with touching boundaries
                # adjacency = +2 -> try to merge horizons with gap      of 1 pixel between boundaries
                # TODO: check, if using depth as the first mask is faster

                # Iline-axis
                overlap_min_i = np.maximum(bboxes[:, 0], current_bbox[0])
                overlap_max_i = np.minimum(bboxes[:, 1], current_bbox[1]) + 1
                overlap_size_i = overlap_max_i - overlap_min_i
                mask_i = (overlap_size_i >= 1 - adjacency_i)
                indices_i = np.nonzero(mask_i)[0]
                bboxes_i = bboxes[indices_i]

                # Crossline-axis
                overlap_min_x = np.maximum(bboxes_i[:, 2], current_bbox[2])
                overlap_max_x = np.minimum(bboxes_i[:, 3], current_bbox[3]) + 1
                overlap_size_x = overlap_max_x - overlap_min_x
                mask_x = (overlap_size_x >= 1 - adjacency_x)
                indices_x = np.nonzero(mask_x)[0]
                bboxes_x = bboxes_i[indices_x]

                # depth-axis: other threshold
                overlap_min_h = np.maximum(bboxes_x[:, 4], current_bbox[4])
                overlap_max_h = np.minimum(bboxes_x[:, 5], current_bbox[5]) + 1
                overlap_size_h = overlap_max_h - overlap_min_h
                mask_h = (overlap_size_h >= 1)
                indices_h = np.nonzero(mask_h)[0]
                bboxes_h = bboxes_x[indices_h]

                indices = indices_i[indices_x][indices_h]
                merge_candidates = horizons[indices]
                _ = bboxes_h # TODO: can be used to pass already computed overlap sizes

                # Merge all possible candidates
                for idx, merge_candidate in zip(indices, merge_candidates):
                    merge_stats['merge_candidates'] += 1

                    # Conditions to not use the candidate:
                    #   - already merged
                    #   - already verified the impossibility of merge
                    #   - it is the horizon itself
                    if idx in indices_merged:
                        merge_stats['already_merged_hit'] += 1
                        continue
                    if (id(merge_candidate) in current_horizon.id_separated or
                        id(current_horizon) in merge_candidate.id_separated):
                        merge_stats['id_separated_hit'] += 1
                        continue
                    if merge_candidate is current_horizon:
                        # Can move code to the previous clause at the cost of code readability
                        continue

                    # Compute the mergeability
                    merge_stats['verify'] += 1
                    merge_status = ExtractionMixin.verify_merge(current_horizon, merge_candidate,
                                                                mean_threshold=mean_threshold,
                                                                max_threshold=max_threshold,
                                                                min_size_threshold=min_size_threshold,
                                                                max_size_threshold=max_size_threshold,
                                                                adjacency=(adjacency_i, adjacency_x))
                    merge_stats[merge_status] += 1

                    # Merge, if needed
                    if merge_status == 4:
                        # Overlapping horizons: definetely merge
                        merged = ExtractionMixin.overlap_merge(current_horizon, merge_candidate, inplace=True)
                        current_horizon.id_separated = current_horizon.id_separated.union(merge_candidate.id_separated)

                    elif merge_status == 3 and (adjacency_i > 0 or adjacency_x > 0):
                        # Adjacent horizons: maybe we can merge it
                        merged = ExtractionMixin.adjacent_merge(current_horizon, merge_candidate, inplace=True,
                                                                mean_threshold=mean_threshold,
                                                                adjacency=(adjacency_i, adjacency_x))
                        merge_stats['merged_adjacent'] += (1 if merged else 0)

                    elif merge_status == 0:
                        # Depth separated: can't merge and will not be able after other merges
                        current_horizon.id_separated.add(id(merge_candidate))
                        merge_candidate.id_separated.add(id(current_horizon))
                        merged = False

                    else:
                        # Spatially separated or too small of an overlap
                        # Can't merge for now, but maybe will be able later
                        merged = False

                    # Keep values for clean-up
                    if merged:
                        current_horizon.merge_count += 1
                        indices_merged.add(idx)
                        num_merged += 1

                # Update bbox stats
                bboxes[i] = (current_horizon.i_min, current_horizon.i_max,
                             current_horizon.x_min, current_horizon.x_max,
                             current_horizon.d_min, current_horizon.d_max)

                # Once in a while, remove merged horizons from `bboxes` and `horizons` arrays
                if len(indices_merged) > int(delete_threshold * len(horizons)):
                    indices_merged = list(indices_merged)
                    horizons = np.delete(horizons, indices_merged, axis=0)
                    bboxes = np.delete(bboxes, indices_merged, axis=0)
                    i -= sum(1 for idx in indices_merged if idx < i)

                    indices_merged = set()
                    merge_stats['num_deletes'] += 1

                # Move to the next horizon
                i += 1


            # Clean-up at the end of global iteration
            if indices_merged:
                indices_merged = list(indices_merged)
                horizons = np.delete(horizons, indices_merged, axis=0)
                bboxes = np.delete(bboxes, indices_merged, axis=0)
                i -= sum(1 for idx in indices_merged if idx < i)

                merge_stats['num_deletes'] += 1

            # Reject horizons that has not participated in any merges:
            # they will not be merged in the next iterations as well
            if (adjacency_i <= 0 and adjacency_x <= 0):
                rejected_horizons_ = [horizon for horizon in horizons
                                      if horizon.merge_count == 0]
                rejected_horizons.extend(rejected_horizons_)

                horizons = np.array([horizon for horizon in horizons
                                    if horizon.merge_count > 0])
            else:
                rejected_horizons_ = []
            merge_stats['num_rejected_horizons'].append(len(rejected_horizons_))


            # Global iteration info
            merge_stats['global_iterations'] += 1
            merge_stats['global_merges'].append(num_merged)
            merge_stats['global_iteration_timings'].append(round(perf_counter() - start_timing, 2))

            # Exit condition: merged less horizons then threshold
            if num_merged < num_merged_threshold:
                break

        # Get back the rejects
        horizons = list(horizons)
        horizons.extend(rejected_horizons)

        for horizon in horizons:
            delattr(horizon, 'merge_count')
            delattr(horizon, 'id_separated')

        horizons = [horizon for horizon in horizons if not horizon.already_merged]
        return sorted(horizons, key=len), MetaDict(merge_stats)


    @staticmethod
    def merge_list_concurrent(horizons,
                              max_concurrent_iters=2, max_workers=16, min_workers=4, min_length=1000, multiplier=0.8,
                              mean_threshold=1., max_threshold=2.2, adjacency=3, minsize=50, max_iters=1,
                              num_merged_threshold=1, delete_threshold=0.01):
        """ Apply merge procedure in multiple threads.
        Works by splitting the `horizons` list into multiple chunks, merging everything possible in each chunk,
        and then applying one final merge to do cross-chunk merges.

        Parameters
        ----------
        max_concurrent_iters : int
            Number of times to split `horizons` into chunks and processing concurrently.
        max_workers : int
            Maximum number of chunks / workers to use.
        min_workers : int
            Minimum number of chunks / workers. If the optimal amount is lower, then we don't use concurrency at all.
        min_length : int
            If the chunk size is lower, we use fewer workers.
        multiplier : float
            Decrease in number of workers, if needed.
        other parameters : dict
            Passed directly to `merge_list` method.
        """
        for _ in range(max_concurrent_iters):
            threading_flag, num_workers = ExtractionMixin._compute_threading_parameters(length=len(horizons),
                                                                                        min_length=min_length,
                                                                                        max_workers=max_workers,
                                                                                        min_workers=min_workers,
                                                                                        multiplier=multiplier)
            # No more threading is needed
            if threading_flag is False:
                break

            horizons.sort(key=attrgetter('i_min'))

            # Split list into chunks for each worker
            chunk_size = len(horizons) // num_workers
            chunks = [horizons[idx:idx+chunk_size] for idx in range(0, len(horizons), chunk_size)]
            if len(chunks[-1]) < min_length:
                chunks[-2].extend(chunks.pop(-1))

            # Run merging procedure in a separate workers
            with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                function = lambda horizons_list: ExtractionMixin.merge_list(horizons_list,
                                                                            adjacency=adjacency,
                                                                            mean_threshold=mean_threshold,
                                                                            max_threshold=max_threshold,
                                                                            max_iters=max_iters,
                                                                            num_merged_threshold=num_merged_threshold,
                                                                            delete_threshold=delete_threshold)
                processed = list(executor.map(function, chunks))

            horizons = [horizon for chunk, _ in processed for horizon in chunk]

        # One final merge to combine horizons from different chunks
        horizons, counter = ExtractionMixin.merge_list(horizons, adjacency=adjacency,
                                                       mean_threshold=mean_threshold, max_threshold=max_threshold,
                                                       num_merged_threshold=num_merged_threshold,
                                                       delete_threshold=delete_threshold)
        return horizons, counter

    @staticmethod
    def _compute_threading_parameters(length, max_workers, min_workers, min_length, multiplier=0.8):
        """ Compute whether the concurrency is needed.
        Works by computing chunk sizes:
            - if chunks would be smaller than the `min_length`, discount the number of workers by `multiplier`
            - if the number of workers is smaller than `min_workers`, no concurrency is needed
            - otherwise, use current number of workers
        """
        chunk_size = length // max_workers

        # Each chunk is big enough for current `num_workers=max_workers`, so use them all
        if chunk_size >= min_length:
            return True, max_workers

        # Gradually descrease the amount of available workers
        num_workers = int(multiplier * max_workers)
        if num_workers >= min_workers:
            return ExtractionMixin._compute_threading_parameters(length, num_workers,
                                                                 min_workers=min_workers,
                                                                 min_length=min_length)

        # Chunks are too small even for the `num_workers=min_workers`
        return False, None


@njit
def intersect_matrix(first, second, max_threshold):
    """ Given two matrices of equal shapes, compute mean and max differences.
    If the max difference is bigger than `max_threshold`, we break out of the loop early.
    """
    # TODO: return flag of break/nobreak?
    #pylint: disable=consider-using-enumerate
    first = first.ravel()
    second = second.ravel()

    s, c = 0, 0 # running sum and count

    for i in range(len(first)):
        first_h = first[i]
        second_h = second[i]

        # Check that both values are not `Horizon.FILL_VALUE``
        if first_h >= 0:
            if second_h >= 0:
                abs_diff = abs(first_h - second_h)

                if abs_diff > max_threshold: # early stopping on overflow
                    s = 999
                    c = 1
                    break

                s += abs_diff
                c += 1

    if c != 0:
        mean = s / c
    else:
        mean = 999
    return mean, c
