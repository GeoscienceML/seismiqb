""" Extractor of horizon surfaces from a probability array. """
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cc3d import connected_components
from scipy.ndimage import find_objects


from batchflow import Notifier
from .base import Horizon



class HorizonExtractor:
    """ Extractor of horizon surfaces from a probability array.

    The main idea of implementation is:
        - extract connected components on each n-th slice, store them (with their indices) in a container
        Do that for slices along the first and the second axis: usually, these are INLINE_3D and CROSSLINE_3D dims.

        - sample one line as a starter. Usually, we do that by sorting all extracted connected components by length.
        Make a `HorizonPrototype` instance out of it.
        - for that line, find intersections with orthogonal slices.
        On these slices, find exact lines that are intersecting with the current one.
        - Add lines to the prototype instance, until no more lines can be merged in either direction.
        At the end, prototype is basically a collection of lines with known intersections.

        - repeat the last three steps to get more prototypes.
        Optionally, convert prototypes to Horizon instances by iteratively merging lines with additional thresholds.
    """
    def __init__(self, array, origin=None, step=10):
        self.array = array
        self.origin = [0] * 3 if origin is None else origin
        self.step = step

        # Create a mapping:
        # orientation ⟶ {
        #   slide_index ⟶ {
        #       'origin' ⟶ 3D origin of the slide, its upper rightmost point,
        #       'slide' ⟶ view on values in the original probability array,
        #       'labeled_slide' ⟶ labeled connected components on a slide: each region its separate index,
        #       'bboxes' ⟶ for each labeled component, its bounding box,
        #       'lengths' ⟶ for each labeled component, its length,
        #   }
        # }
        self.container = {0: {}, 1: {}}
        self.init_container(array=array)

    def init_container(self, array):
        """ Create a mapping with connected components for a given probability array. """
        for orientation in [0, 1]:
            for slide_idx in Notifier('t')(range(0, array.shape[orientation], self.step)):

                # Compute slide position
                locations = [slice(None)] * 3
                locations[orientation] = slice(slide_idx, slide_idx + 1)

                origin = [0] * 3
                origin[orientation] = slide_idx

                # Label connected components
                slide = array[tuple(locations)]

                # length = 100
                # locations_zfill = [slice(None)] * 3
                # locations_zfill[1-orientation] = slice((slide_idx + orientation * length // 2) % length,
                #                                        None, length)
                # slide = slide.copy()
                # slide[locations_zfill] = 0
                labeled_slide = connected_components(slide >= 0.5, connectivity=26)

                # Extract bboxes and compute length of each point cloud
                bboxes = find_objects(labeled_slide)
                bboxes = dict(enumerate(bboxes, start=1))
                lengths = {item_idx : (item_bbox[1-orientation].stop - item_bbox[1-orientation].start)
                           for item_idx, item_bbox in bboxes.items()}

                self.container[orientation][slide_idx] = {
                    'origin': origin,
                    'slide': slide,
                    'labeled_slide': labeled_slide,
                    'bboxes': bboxes,                          # `item_idx` -> bbox
                    'lengths': lengths,                        # `item_idx` -> length
                    'already_used': defaultdict(bool),         # `item_idx` -> already merged or not
                }

    def get_points(self, orientation, slide_idx, item_idx):
        """ Given orientation, slide index and item index, extract the points of a line. """
        slide_dict = self.container[orientation][slide_idx]
        item_origin = slide_dict['origin']
        item_bbox = slide_dict['bboxes'][item_idx]

        item_indices = (slide_dict['labeled_slide'][item_bbox] == item_idx).nonzero()
        item_points = np.vstack([item_indices[j] + (item_bbox[j].start + item_origin[j])
                                 for j in range(3)]).T
        return item_points

    def point_to_item_idx(self, orientation, point):
        """ Get `item_idx` given point in 3D array and its orientation. """
        point = np.array(point)
        slide_idx = point[orientation]
        point[orientation] = 0

        item_idx = self.container[orientation][slide_idx]['labeled_slide'][point[0], point[1], point[2]]
        return item_idx

    def reset_already_used(self):
        """ Reset `already_used` flag for all lines. """
        for orientation_dict in self.container.values():
            for slide_dict in orientation_dict.values():
                slide_dict['already_used'] = defaultdict(bool)


    # Line sequences: iterator over lines for prototype creation
    def make_line_sequence(self, line_length_threshold=50):
        """ Iterator over lines from entire array, sorted by length in descending order.
        Each element is contains full information about line: ((orientation, slide_index, item_index), line_length).
        """
        iterator = {}
        for orientation, orientation_dict in self.container.items():
            for slide_idx, slide_dict in orientation_dict.items():
                for item_idx, item_length in slide_dict['lengths'].items():
                    if item_length > line_length_threshold and slide_dict['already_used'][item_idx] is False:
                        iterator[(orientation, slide_idx, item_idx)] = item_length

        return sorted(iterator.items(), key=lambda item:item[1], reverse=True)

    def make_slide_line_sequence(self, orientation=0, slide_idx=None, line_length_threshold=50):
        """ Iterator over lines on a given slide, sorted by length in descending order.
        Each element is contains full information about line: ((orientation, slide_index, item_index), line_length).
        """
        slide_idx = slide_idx or (self.array.shape[orientation] // 2)
        slide_idx = slide_idx // self.step * self.step

        iterator = {}
        for item_idx, item_length in self.container[orientation][slide_idx]['lengths'].items():
            if item_length > line_length_threshold:
                iterator[(orientation, slide_idx, item_idx)] = item_length
        return sorted(iterator.items(), key=lambda item:item[1], reverse=True)


    # Prototype extraction
    def make_prototypes(self, sequence=None, n=None, line_length_threshold=50, max_iters=100, pbar='t'):
        """ Make prototypes, starting from lines on a given orientation/slide.
        Lines are sorted in descending order by their length, so we start from bigger lines.
        """
        sequence = sequence or self.make_line_sequence(line_length_threshold=line_length_threshold)
        n = n or len(sequence)

        prototypes = []
        with Notifier(pbar, total=n) as progress_bar:
            for (orientation, slide_idx, item_idx), _ in sequence:
                # Check if the line is already used in other prototype
                if self.container[orientation][slide_idx]['already_used'][item_idx] is True:
                    progress_bar.update()
                    continue

                # Make prototype instance, add all intersecting lines to it
                prototype = self.init_prototype(orientation=orientation, slide_idx=slide_idx, item_idx=item_idx)
                prototype = self.extend_prototype(prototype=prototype, orientation=1-orientation, max_iters=max_iters)
                prototypes.append(prototype)

                progress_bar.update()
                if len(prototypes) == n:
                    progress_bar.close()
                    break

        prototypes.sort(key=len, reverse=True)
        return prototypes


    def init_prototype(self, orientation, slide_idx, item_idx):
        """ Given orientation, slide and item index, extract line points and create a prototype instance out of it. """
        prototype = HorizonPrototype(origin=self.origin)
        points = self.get_points(orientation=orientation, slide_idx=slide_idx, item_idx=item_idx)
        prototype.add_points(points=points, orientation=orientation, slide_idx=slide_idx)
        return prototype

    def extend_prototype(self, prototype, orientation, max_iters=100):
        """ Extend given prototype, alternating between directions. """
        for outer_iter in range(max_iters):
            prev_len = prototype.n_points

            for slide_idx in range(0, self.array.shape[orientation], self.step):
                # Check intersections of current slide and prototype
                intersections = prototype.get_intersections(orientation=orientation,
                                                            slide_idx=slide_idx)
                if len(intersections) == 0:
                    continue

                # `intersections`` are (N, 3) array of coordinates: use them as indexer
                slide_dict = self.container[orientation][slide_idx]
                labeled_slide = slide_dict['labeled_slide']

                intersections[:, orientation] = 0
                item_indices = labeled_slide[intersections[:, 0],
                                             intersections[:, 1],
                                             intersections[:, 2]]
                item_indices = Counter(item_indices)

                # For each item prototype intersected with, add its points
                for item_idx, item_occurencies in item_indices.items():
                    if item_idx == 0:
                        continue
                    if slide_dict['already_used'][item_idx] is True:
                        continue
                    if outer_iter > 0 and item_occurencies < 2:
                        continue

                    # Get line points, merge them into prototype, mark as merged
                    points = self.get_points(orientation=orientation, slide_idx=slide_idx, item_idx=item_idx)
                    prototype.add_points(points=points, orientation=orientation, slide_idx=slide_idx)
                    slide_dict['already_used'][item_idx] = True

            # If no points added, break. Otherwise, change the orientation and repeat
            if prototype.n_points == prev_len:
                break
            orientation = 1 - orientation
        return prototype


    @staticmethod
    def to_horizons(prototypes, field, reduction=7, d_ptp_threshold=20, size_threshold=40, max_iters=100,
                    n=3, pbar=False, max_workers=4):
        """ Convert prototype instances to horizon instances.
        Refer to :meth:`HorizonPrototype.to_horizons` to more details on parameters.
        """
        horizons = []
        with Notifier(pbar, total=len(prototypes)) as progress_bar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                def callback(future):
                    horizons.extend(future.result())
                    progress_bar.update()

                for prototype in prototypes:
                    future = executor.submit(prototype.to_horizons, field=field,
                                             reduction=reduction, d_ptp_threshold=d_ptp_threshold,
                                             size_threshold=size_threshold, max_iters=max_iters, n=n, pbar=False)
                    future.add_done_callback(callback)
        horizons.sort(key=len, reverse=True)
        return horizons


class HorizonPrototype:
    """ Collection of lines along axes. """
    def __init__(self, origin):
        # Mapping:
        # orientation ⟶ {
        #   slide_index ⟶ {
        #       start coordinate ⟶ line points (N, 3) ndarray
        #   }
        # }
        self.origin = origin
        self.container = {0: defaultdict(dict), 1: defaultdict(dict)}

    def add_points(self, points, orientation, slide_idx):
        """ Add `points` to self, given their orientation and slide index. """
        key = points[0][1-orientation]

        slide_dict = self.container[orientation][slide_idx]
        if key not in slide_dict:
            slide_dict[key] = points

    def get_intersections(self, orientation, slide_idx):
        """ Compute coordinates of intersections of lines in `self` with a given slide. """
        intersections = []

        # For each line (represented by its start and points), search `slide_idx`
        for points_dict in self.container[1 - orientation].values():
            for points in points_dict.values():
                if points[0, orientation] > slide_idx or points[-1, orientation] < slide_idx:
                    continue

                insertion_index = np.searchsorted(points[:, orientation], slide_idx)
                # Check that `insertion_index` actually refers to the same value, as `slide_idx`:
                # may not be true if there is no `slide_idx` value in `points`
                if insertion_index < len(points) and points[:, orientation][insertion_index] == slide_idx:
                    intersections.append(points[insertion_index])

                    # TODO: in the perfect case, `break` works.
                    # If multiple lines are intersecting with a slide, it does not.
                    break
        return np.array(intersections)

    # Introspection
    @property
    def n_points(self):
        """ Total number of points in a prototype. """
        counter = 0
        for orientation_dict in self.container.values():
            for points_dict in orientation_dict.values():
                for points_ in points_dict.values():
                    counter += len(points_)
        return counter

    @property
    def flat_points(self):
        """ Points in a prototype, concatenated into one (N, 3) array. """
        buffer = np.empty((self.n_points, 3), dtype=np.int32)
        counter = 0
        for orientation_dict in self.container.values():
            for points_dict in orientation_dict.values():
                for points_ in points_dict.values():
                    buffer[counter:counter + len(points_)] = points_
                    counter += len(points_)
        points = np.array(buffer)
        return points

    @property
    def unique_flat_points(self):
        """ Unique points in a prototype, concatenated into one (N, 3) array. """
        return np.unique(self.flat_points, axis=0)

    @property
    def n_unique_points(self):
        """ Number of unique points in a prototype. """
        return len(self.unique_flat_points)

    def __len__(self):
        return self.n_points

    # Convert to Horizon instance
    def naive_to_horizon(self, field, name='naive_prototype'):
        """ Naive conversion of prototype to horizon instance: no correction on overlapping lines.
        Should not be used other for debugging/introspection purposes.
        """
        points = self.unique_flat_points
        points += self.origin
        return Horizon(self.unique_flat_points, field=field, name=name)

    def to_horizons(self, field, reduction=3, d_ptp_threshold=20, size_threshold=40, max_iters=100, n=3, pbar=False):
        """ Convert Prototype instance to one or more Horizon instances.

        As Prototype is a collection of lines with intersections, we want to avoid depth-wise overlaps.
        To do that, we sequentially merge lines with tight thresholds:
            - at first, we merge lines with less depth variation (low `d_ptp`).
            The threshold is gradually increased from 1 to `d_ptp_threshold`, so that we allow more and more variation.
            - then, we merge lines with high overlap to already merged (big `overlap_size`)
            The threshold is gradually decreased from `size_threshold` to 3, so that we allow less overlaps.
            - repeat the first two steps multiple times, until the horizon size is no longer increasing.

        As some of the lines may overlap depth-wise even where it is not desirable, we use `reduction` to remove
        a few points of either side of the line and produce better extracted horizons with negligible area reduction.

        Parameters
        ----------
        reduction : int
            Number of points on the left/right side of a line to remove.
        d_ptp_threshold : int
            Maximum allowed `d_ptp` of a line to merge during the first step.
        size_threshold : int
            Maxumum minimum overlap size threshold of a line to merge during the second step.
        max_iters : int
            Number of alternating iterations to perform.
        n : int
            Maximum number of horizons to extract.
        pbar : bool, str
            If bool, then whether to display progress bar.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        """
        # Strip the first and the last points of each line in a prototype, convert to Horizon instances
        line_horizons = []
        for orientation, orientation_dict in self.container.items():
            for slide_idx, points_dict in orientation_dict.items():
                for start, points in points_dict.items():
                    if len(points) <= 2*reduction + 1:
                        continue

                    points = points[reduction:-reduction]
                    points += self.origin

                    line_horizon = Horizon(points, field=field,
                                           name=f'line_{orientation}_{slide_idx}_{start}')
                    line_horizons.append(line_horizon)
        line_horizons.sort(key=len, reverse=True)

        horizons = []
        for _ in range(n):
            if len(line_horizons) == 0:
                break
            horizon = line_horizons.pop(0)
            horizons.append(horizon)

            for _ in Notifier(pbar)(range(max_iters)):
                prev_length = len(horizon)

                # Merge with small ptp
                for d_ptp_threshold_ in range(1, d_ptp_threshold, +3):
                    candidates = [line_horizon for line_horizon in line_horizons
                                  if line_horizon.d_ptp <= d_ptp_threshold_]
                    horizon, _, _ = horizon.merge_into(candidates, mean_threshold=0.01, max_threshold=0, adjacency=0)

                # Merge with bigger overlaps
                for size_threshold_ in range(size_threshold, 3, -1):
                    horizon, _, _ = horizon.merge_into(line_horizons, mean_threshold=0.01, max_threshold=0, adjacency=0,
                                                       min_size_threshold=size_threshold_)

                # Remove already merged line horizons. Break, if no progress in horizon size
                line_horizons = [line_horizon for line_horizon in line_horizons
                                if not line_horizon.already_merged]
                if len(horizon) == prev_length or len(line_horizons) < 3:
                    break
        horizons.sort(key=len, reverse=True)
        return horizons

    def to_horizon(self, field, reduction=7, d_ptp_threshold=20, size_threshold=40, max_iters=100, n=3, pbar=False):
        """ Alias for extracting one horizon with the biggest size.
        Refer to :meth:`.to_horizons` for more details on parameters.
        """
        return self.to_horizons(field=field, reduction=reduction,
                                d_ptp_threshold=d_ptp_threshold, size_threshold=size_threshold,
                                max_iters=max_iters, n=n, pbar=pbar)[0]
