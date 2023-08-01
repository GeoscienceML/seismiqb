""" Extractor of fault surfaces from cloud of points. """
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.ndimage import measurements
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components as connected_components_graph

from batchflow import Notifier

from .base import Fault

class FaultExtractor:
    """ Extract separate fault surfaces from array with fault labels (probabilities or binary values).
    It uses assumption that the connected components on each slice in `components_axis` direction
    are in the same direction (`faults_direction`) and do not have branches.

    The algorithm is based on four stages:
        - merge connected components into small patches: sequences of connected components
            (see :class:`.FaultPatch`)
        - group patches around the holes
        - group patches with large intersection of the bound components
        - form separate fault instances from all groups of patches.

    Patches can be organized into directed graph: each patch is connected to the patches which touch its bottom
    component by its top component. Top patches we will call `parents`, the sequent bottom patches we will
    call `children`.

    Parameters
    ----------
    array : str, numpy.ndarray or an object with the same slicing
        Array with fault probabilities
    origin : tuple of int, optional
        Origin (offset) of the cube, is needed to make fault instances with right coordinates, by default (0, 0, 0)
    faults_direction : 0 or 1, optional
        Direction of faults: across ilines or xlines
    components_axis : int, optional
        Axis of cube slices where connected components will be searched
    """
    def __init__(self, array, origin=(0, 0, 0), faults_direction=0, components_axis=2):
        if faults_direction not in (0, 1):
            raise ValueError(f'`faults_direction` must be 0 or 1 but {faults_direction} was given.')

        self.array = array
        self.origin = origin
        self.faults_direction = faults_direction
        self.components_axis = components_axis

        self.labels, self.n_objects, self.objects = self.create_labels()
        self.sizes = self.compute_sizes()

        self.sorted_labels = sorted(self.sizes, key=self.sizes.get, reverse=True)

        self.patchtop_to_patch = {} # key - top patch component, value - FaultPatch instance
        self.patchbottom_to_patch = {} # key - bottom patch component, value - FaultPatch instance

        self.extended_components = {1: set(), -1: set()} # assemble components extended in a given direction

        self._candidates = {1: set(), -1: set()} # components indices to extend
        self._connectivity_matrix = None # binary matrix of (len(self.patchtop_to_patch), len(self.patchtop_to_patch)).
                                         # 1 on (i, j) position means that i-th and j-th patches must be merged.

    def create_labels(self):
        """ Create slice-wise labeled cube. Axis of slices is defined by `self.components_axis`. """
        structure = np.zeros((3, 3, 3))
        slices = [slice(None) for _ in range(3)]
        slices[self.components_axis] = 1
        structure[tuple(slices)] = 1

        labels, n_objects = measurements.label(self.array, structure=structure)
        objects = measurements.find_objects(labels)
        objects = dict(enumerate(objects, start=1))

        return labels, n_objects, objects

    def compute_sizes(self):
        """ Compute sizes of components as a number of points. The zero label (background) is ignored. """
        sizes = {}
        for idx, bbox in self.objects.items():
            mask = (self.labels[bbox] == idx)
            sizes[idx] = mask.sum()
        return sizes

    def idx_to_points(self, idx):
        """ Get component points cloud (taking into account array origin). """
        points = np.stack(np.where(self.labels[self.objects[idx]] == idx), axis=1)
        object_origin = [item.start for item in self.objects[idx]]
        return points + object_origin + self.origin

    def idx_to_bbox(self, idx):
        """ Get component bbox in the array. """
        return self.objects[idx]

    def idx_to_location(self, idx):
        """ Get component slice location. """
        return self.objects[idx][self.components_axis].start

    def get_neighbors(self, idx, direction=1):
        """ Find components on the next slice in the given direction (1 or -1) which touch component
        with label `idx`.
        """
        bbox = list(self.objects[idx])
        location = self.idx_to_location(idx)
        if ((direction ==  1 and location == self.array.shape[self.components_axis] - 1) or \
            (direction == -1 and location == 0)):
            return []

        bbox[self.components_axis] = location
        mask = (self.labels[bbox[0], bbox[1], bbox[2]] == idx)
        bbox[self.components_axis] += direction
        components = np.unique(self.labels[bbox[0], bbox[1], bbox[2]][mask])

        return components[components > 0]

    def compute_intersection_size(self, idx_a, idx_b):
        """ Compute the area (the number of pixels) of ​​contact of two components on different slices. """
        bbox = [
            slice(min(i.start, j.start), max(i.stop, j.stop)) for i, j in zip(self.objects[idx_a], self.objects[idx_b])
        ]

        bbox[self.components_axis] = self.idx_to_location(idx_a)
        a_mask = (self.labels[bbox[0], bbox[1], bbox[2]] == idx_a)

        bbox[self.components_axis] = self.idx_to_location(idx_b)
        b_mask = (self.labels[bbox[0], bbox[1], bbox[2]] == idx_b)

        intersection = np.logical_and(a_mask, b_mask)
        return intersection.sum()

    def create_patches(self, size_threshold=100, pbar='t'):
        """ Create small patches. There are two sources of component indices to initialize patches:
            - list of all connected components sorted by size
            - candidates: components that stopped the extension of previous patches
              (see description of extension of :class:`.FaultPatch`)

        As long as there are candidates they will be extended into patches.
        Otherwise, from the list of components, the next one in size is taken, which has not yet been extended.
        Since there are no candidates at the beginning, the largest component is extended first.

        Components smaller then `size_threshold` are skipped.

        Direction of extension is 0 (both directions) for components from list of all connected components.
        Each candidate has its own direction to extend which depends on the reason why component was rejected in
        patch extension.
        """

        # TODO: rewrite progress bar for used components
        candidates = self._components_to_extend(size_threshold)
        for anchor_idx, direction in Notifier(pbar)(candidates):
            patch = FaultPatch(anchor_idx, direction, extractor=self)

            # Check if patch consists of one component which is a part of the already existed patch
            if len(patch.components) == 1 and (direction == 0 or patch.top in self.extended_components[-direction]):
                continue

            self.patchtop_to_patch[patch.top] = patch
            self.patchbottom_to_patch[patch.bottom] = patch

        # Get mapping of patches to natural enumeration and its reverse version
        self._labels_mapping = dict(enumerate(self.patchtop_to_patch.keys()))
        self._labels_reverse_mapping = {v: k for k, v in self._labels_mapping.items()}
        self._connectivity_matrix = lil_matrix(
            (len(self.patchtop_to_patch), len(self.patchtop_to_patch)), dtype='uint8'
        )

        return self

    def _components_to_extend(self, size_threshold):
        """ Generate components indices and direction to extend into patch. Components smaller
        then `size_threshold` are skipped. """
        components_iter = iter(self.sorted_labels)

        while True:
            if len(self._candidates[1]) > 0:
                direction = 1
            elif len(self._candidates[-1]) > 0:
                direction = -1
            else:
                direction = 0 # extend in both directions

            if direction == 0:
                try:
                    anchor = next(components_iter)
                except StopIteration:
                    break # all components were extended

                if self.sizes[anchor] < size_threshold:
                    break

                # check if anchor was already extended in one of directions
                if anchor in self.extended_components[1]:
                    direction = -1
                elif anchor in self.extended_components[-1]:
                    direction = 1
            else:
                anchor = self._candidates[direction].pop()
                if self.sizes[anchor] < size_threshold:
                    continue # skip patches creation if all components are too small

            if direction != 0 and anchor in self.extended_components[direction]:
                continue

            yield anchor, direction

    def add_candidates(self, candidates_bottom, candidates_top):
        """ Update candidates sets. """
        self._candidates[1].update(candidates_bottom)
        self._candidates[-1].update(candidates_top)

    def find_holes(self, depth=10, threshold=0.9, pbar='t'):
        """ Find holes of fault surfaces and merge patches around them into groups.

        To find holes, we search for a patches which touch more then one other patches (has several components
        in `bottom_rejected`). Then we construct directed graphs of children of that root and check if they has
        common inheritors. If yes, then two such branches from inheritors to root surround the hole and we
        group them.

        Parameters
        ----------
        depth : int, optional
            Maximal depth of the constructed trees, by default 10
        threshold : float, optional
            Two patches can be parent and child if they touch and ratio of intersection of touched components
            to the minimal of them is larger then `threshold`, by default 0.9
        pbar : bool, optional
            Progress bar, by default True
        """
        groups = []
        for idx in Notifier(pbar, desc='Find holes')(self.patchtop_to_patch):
            bottom_rejected = self.patchtop_to_patch[idx].bottom_rejected
            for a, b in combinations(bottom_rejected, 2):
                if a not in self.patchtop_to_patch or b not in self.patchtop_to_patch:
                    continue
                tree_a = self._get_inheritors_tree(a, depth, threshold)
                tree_b = self._get_inheritors_tree(b, depth, threshold)

                # Find common patches in both trees.
                for item in set(tree_a) & set(tree_b):
                    if tree_a[item] == tree_b[item]: # Skip patch if in both trees it has common parents.
                        continue

                    # Find paths from `item` to `idx`
                    parent = tree_a.get(item)
                    path_a = [item]
                    while parent is not None:
                        path_a.append(parent)
                        parent = tree_a.get(parent)

                    parent = tree_b.get(item)
                    path_b = [item]
                    while parent is not None:
                        path_b.append(parent)
                        parent = tree_b.get(parent)

                    if len(set(path_a) & set(path_b)) == 1: # Check if the first common ancestor in both trees is `idx`.
                        groups.append(list(set([idx, *path_a, *path_b])))

            for group in groups:
                idx = self._labels_reverse_mapping.get(group[0])
                if idx is not None:
                    for item in group[1:]:
                        idx_2 = self._labels_reverse_mapping[item]
                        self._connectivity_matrix[idx, idx_2] = 1
                        self._connectivity_matrix[idx_2, idx] = 1

        return self

    def _get_inheritors_tree(self, idx, depth, threshold):
        """ Get tree of connected patches which starts from component `idx` which is no deeper than `depth`.
        Two patches are connected if they touch and ratio of intersection of touched components to the
        minimal of them is larger then `threshold`.
        """
        components = [idx]
        tree = {}
        for _ in range(depth):
            bottom_rejected = set()
            for comp in components:
                for bottom in self.patchtop_to_patch[comp].bottom_rejected:
                    if bottom in self.patchtop_to_patch:
                        leaf = list(self.patchtop_to_patch[comp].components.values())[-1][0]
                        intersection = self.compute_intersection_size(bottom, leaf)
                        a, b = self.sizes[bottom], self.sizes[leaf]
                        if intersection / min(a, b) >= threshold:
                            tree[bottom] = comp
                            bottom_rejected |= {bottom}
            components = bottom_rejected
        return tree


    def merge_patches(self, thresholds, pbar='t'):
        """ Merge patches with large intersections. see :meth:`.FaultPatch.find_largest_intersections`. """
        for idx, patch in Notifier(pbar, desc='Merge patches')(self.patchtop_to_patch.items()):
            for bottom_idx in patch.find_largest_intersections(thresholds=thresholds):
                if bottom_idx in self.patchtop_to_patch:
                    a = self._labels_reverse_mapping[idx]
                    b = self._labels_reverse_mapping[bottom_idx]
                    self._connectivity_matrix[a, b] = 1
                    self._connectivity_matrix[b, a] = 1
        return self

    def extend_patches(self, size_threshold, intersection_threshold, pbar='t'):
        """ Iterative merging of patches. Starts with the largest patch, to which tightly fitting patches are merged
        in succession on both sides. """
        sorted_patches = sorted(self.patchtop_to_patch.values(), key=lambda x: x.size(), reverse=True)
        extended_patches = {-1: [], 1: []}
        for patch in Notifier(pbar, desc='Extend patches')(sorted_patches):
            for direction in [-1, 1]:
                mapping = self.patchtop_to_patch if direction == 1 else self.patchbottom_to_patch
                idx_attr = 'top' if direction == 1 else 'bottom'
                current_patch_idx = getattr(patch, idx_attr)
                top_idx = mapping[current_patch_idx].top

                while True:
                    if top_idx in extended_patches[direction]:
                        break
                    extended_patches[direction].append(top_idx)

                    patch = mapping[current_patch_idx]
                    next_patch_idx = patch.find_largest_neighbor(
                        size_threshold, intersection_threshold, direction=direction
                    )
                    if next_patch_idx is None:
                        break

                    a = self._labels_reverse_mapping[top_idx]
                    if next_patch_idx not in mapping:
                        break

                    if mapping[next_patch_idx].top in extended_patches[-direction]:
                        break
                    b = self._labels_reverse_mapping[mapping[next_patch_idx].top]

                    self._connectivity_matrix[a, b] = 1
                    self._connectivity_matrix[b, a] = 1

                    extended_patches[-direction].append(mapping[next_patch_idx].top)

                    current_patch_idx = next_patch_idx
        return self

    def to_faults(self, field, pbar='t'):
        """ Make Fault instances from groups of patches.

        Parameters
        ----------
        field : Field
            Field instance
        pbar : bool, optional
            Progress bar, by default True

        Returns
        -------
        list
            Fault instances linked with the field
        """
        n_groups, groups = connected_components_graph(self._connectivity_matrix)
        faults = []
        group_sizes = defaultdict(int)

        for idx in Notifier(pbar)(range(n_groups)):
            patches_idx = [self._labels_mapping[item] for item in np.arange(len(self.patchtop_to_patch))[groups == idx]]
            components = [
                list(self.patchtop_to_patch[patch].all_components) + self.patchtop_to_patch[patch].bottom_rejected
                for patch in patches_idx
            ]

            for patch in components:
                points = np.concatenate([self.idx_to_points(i) for i in patch], axis=0)
                fault = Fault({'points': points}, field=field, direction=self.faults_direction)
                fault.short_name = '0'
                fault.group_idx = idx

                faults.append(fault)
                group_sizes[idx] += len(fault)

        for fault in faults:
            fault.group_size = group_sizes[fault.group_idx]
        return faults

class FaultPatch:
    """ A sequence of connected components in labeled array extended from anchor component.

    Each patch starts from anchor in one of the direction: up (-1) or bottom (+1).
    If zero, anchor is extended in both direction and then all components merged into one patch.
    When we say "intersection" of two components on the sequential slides we mean intersection
    of their 2D masks (the word "contact" is more accurate).

    Anchor extension is iterative procedure and stops in 3 cases:
        - in the intersection of the currently extended component with the next depth slide
          there are more then one component
        - the next slide has only one component in intersection but it has other component
          in intersection with the current slide
        - the next slide has only one component but it was already extended in that direction

    Components from intersection on the last step will be added to candidates fot the next fault
    patch anchors.

    Examples (for direction = 1):

                a - anchor
                tr - top_rejected
                t - top
                b - bottom
                br - bottom_rejected
                c - candidate

                                        ----tr---
                                    --------a/t------            ┐
                                  ---------------------          |
                                -----------------------          |  patch
                                  ---------------------          |
                              -------------b------------         ┘
                            ---br/c----         ---br/c---
                            ------                --------
                ===========================================================================
                                    -----tr-----
                          ┌        -----a/t---------
                          |      --------------------
                    patch |      --------------------
                          |      --------------------
                          └    ---------b-------------     ------c-----   candidate
                             ----------br/c---------------------
                ============================================================================
    Parameters
    ----------
    anchor_idx : int
        Index of the component to extend
    extension_direction : -1, 0 or 1
        Direction of the patch extension (increase or decrease location of slice in corresponding axis).
        0 means extension in both directions.
    extractor : FaultExtractor
    """
    def __init__(self, anchor_idx, extension_direction, extractor):
        self.anchor_idx = anchor_idx
        self.extension_direction = extension_direction
        self.extractor = extractor

        self.components = None
        self.top = None # Top component
        self.bottom = None # Bottom component
        self.top_rejected = None
        self.bottom_rejected = None

        if self.extension_direction == 0:
            patch_bottom = FaultPatch(self.anchor_idx, 1, self.extractor)
            patch_top = FaultPatch(self.anchor_idx, -1, self.extractor)

            self.components = {**patch_top.components, **patch_bottom.components}
            self.top_rejected = patch_top.top_rejected
            self.bottom_rejected = patch_bottom.bottom_rejected
        else:
            idx = self.anchor_idx
            components_by_depth = {self.extractor.idx_to_location(idx): np.array([idx])}
            self.extractor.extended_components[self.extension_direction].add(idx)

            while True:
                # Find components on the next slide in the defined direction
                neighbors = self.extractor.get_neighbors(idx, self.extension_direction)

                # Stop extension if there are more then one neighbor or the only one neighbor was already extended
                if len(neighbors) != 1 or neighbors[0] in self.extractor.extended_components[self.extension_direction]:
                    neighbors_ = []
                    break

                # Find other neighbors of the found neighbor on the slide of idx
                neighbors_ = self.extractor.get_neighbors(neighbors[0], -self.extension_direction)

                # Stop extension if there are other neighbors
                if len(neighbors_) > 1:
                    break

                components_by_depth[self.extractor.idx_to_location(neighbors[0])] = neighbors
                self.extractor.extended_components[self.extension_direction].add(idx)
                self.extractor.extended_components[-self.extension_direction].update([idx, neighbors[0]])
                idx = neighbors[0]

            self.components = dict(sorted(components_by_depth.items(), key=lambda x: x[0]))
            self.top_rejected = list(self.extractor.get_neighbors(self.anchor_idx, -self.extension_direction))
            self.bottom_rejected = list(neighbors)

            if self.extension_direction == -1:
                self.top_rejected, self.bottom_rejected = self.bottom_rejected, self.top_rejected
                neighbors, neighbors_ = neighbors_, neighbors

            self.extractor.add_candidates(neighbors, neighbors_)

        self.top = list(self.components.values())[0][0]
        self.bottom = list(self.components.values())[-1][0]

    @property
    def all_components(self):
        """ All components included into patch. """
        return np.concatenate(list(self.components.values())).astype(int)

    def find_largest_intersections(self, thresholds=(0.5, 0.9)):
        """ Find children components that have large intersections with bottom components.

        Parameters
        ----------
        thresholds : tuple, optional
            Thresholds as a ratios of components sizes, by default (0.5, 0.9).
            For each pair of bottom component and bottom rejected the first item is the ratio of the components
            intersection to the size of the smallest item in pair. The second is the ratio of the components
            intersection to the size of the largest item in pair.

        Returns
        -------
        list
            Sorted by size list of components filtered by thresholds.
        """
        merge = {}
        for leaf in self.bottom_rejected:
            if leaf in self.extractor.patchtop_to_patch:
                comp = self.all_components[-1]
                intersection = self.extractor.compute_intersection_size(leaf, comp)
                A, B = self.extractor.sizes[leaf], self.extractor.sizes[comp]
                if len(self.bottom_rejected) == 1 and intersection / (min(A, B)) >= thresholds[1]:
                    merge[leaf] = 0
                else:
                    min_ = min(A, B)
                    max_ = max(A, B)
                    if min_ > 20 and intersection / max_ >= thresholds[0] and intersection / min_ >= thresholds[1]:
                        merge[leaf] = intersection / max_
        return list(sorted(merge, key=lambda x: merge[x]))

    def find_largest_neighbor(self, size_threshold=20, intersection_threshold=0.7,  direction=1):
        """ Find the largest components in the defined direction. """
        neighbors_list = self.bottom_rejected if direction == 1 else self.top_rejected
        comp = self.all_components[-1 if direction == 1 else 0]
        candidates = {}
        for leaf in neighbors_list:
            leaf_size = self.extractor.sizes[leaf]
            comp_size = self.extractor.sizes[comp]
            if leaf_size < size_threshold:
                continue
            intersection = self.extractor.compute_intersection_size(leaf, comp)
            if intersection / min(leaf_size, comp_size) > intersection_threshold:
                candidates[leaf] = leaf_size
        if len(candidates) == 0:
            return None
        return sorted(candidates, key=lambda x: candidates[x], reverse=True)[0] # TODO: remove sorted

    def size(self):
        """ Size of the patch as the number of points. """
        return sum([self.extractor.sizes[item[0]] for item in self.components.values()])

    def __repr__(self):
        if self.top == self.bottom:
            return f"{self.top_rejected} ---> {self.bottom} ---> {self.bottom_rejected}"
        return f"{self.top_rejected} ---> {self.top} -> [{len(self.components)} components] "\
               f"-> {self.bottom} ---> {self.bottom_rejected}"
