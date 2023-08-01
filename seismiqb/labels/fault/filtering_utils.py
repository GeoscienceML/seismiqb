""" Faults utils: filterings, groupings. """
from collections import defaultdict
import numpy as np
from .coords_utils import bboxes_adjacent, dilate_coords


# Filters
# Filter too small faults
def filter_faults(faults, min_length_threshold=2000, min_height_threshold=20, min_n_points_threshold=30,
                  **sticks_kwargs):
    """ Filter too small faults.

    Faults are filtered by amount of points, length and height.
    Used as default filtering after prototypes extraction.

    Parameters
    ----------
    faults : sequence of :class:`~.Fault` instances
        Faults for filtering.
    min_length_threshold : int
        Filter out faults with length less than `min_length_threshold`.
    min_height_threshold : int
        Filter out faults with height less than `min_height_threshold`.
        Note, that height is evaluated from sticks.
    sticks_kwargs : dict, optional
        Arguments for fault conversion into sticks view.
    """
    config_sticks = {
        'sticks_step': 10,
        'stick_nodes_step': 10,
        'move_bounds': False,
        **sticks_kwargs
    }

    filtered_faults = []

    for fault in faults:
        if (len(fault.points) < min_n_points_threshold) or (len(fault) < min_length_threshold):
            continue

        fault.points_to_sticks(sticks_step=config_sticks['sticks_step'],
                               stick_nodes_step=config_sticks['stick_nodes_step'],
                               move_bounds=config_sticks['move_bounds'])

        if len(fault.sticks) <= 2: # two sticks are not enough
            continue

        if np.concatenate([item[:, 2] for item in fault.sticks]).ptp() < min_height_threshold:
            continue

        filtered_faults.append(fault)

    return filtered_faults


# Filter small disconnected faults
def filter_disconnected_faults(faults, direction=0, height_threshold=200, width_threshold=40, **kwargs):
    """ Filter small enough faults without any adjacent neighbors.

    Parameters
    ----------
    faults : sequence of :class:`~.Fault` or :class:`~.FaultPrototype` instances
        Faults for filtering.
    direction : {0, 1}
        Faults direction.
    height_threshold : int
        Filter out disconnected faults with height less than `height_threshold`.
    width_threshold : int
        Filter out disconnected faults with width less than `width_threshold`.
    **kwargs : dict
        Adjacency kwargs for :func:`._group_adjacent_faults`.
    """
    # Create groups of adjacent faults
    groups, _ = _group_adjacent_faults(faults, **kwargs)

    grouped_faults_indices = set(groups.keys())

    for group_members in groups.values():
        grouped_faults_indices = grouped_faults_indices.union(group_members)

    # Filtering
    filtered_faults = []

    for i, fault in enumerate(faults):
        if i in grouped_faults_indices:
            filtered_faults.append(fault)

        else:
            height = fault.bbox[2, 1] - fault.bbox[2, 0]
            width = fault.bbox[direction, 1] - fault.bbox[direction, 0]

            if height > height_threshold or width > width_threshold:
                filtered_faults.append(fault)

    return filtered_faults

def _group_adjacent_faults(faults, adjacency=5, adjacent_points_threshold=5):
    """ Add faults into groups by adjacency criterion.

    Helper for the :func:`~.filter_disconnected_faults`.

    Parameters
    ----------
    faults : sequence of :class:`~.Fault` or :class:`~.FaultPrototype` instances
        Faults for filtering.
    adjacency : int
        Axis-wise distance between two faults to consider them to be grouped.
    adjacent_points_threshold : int
        Minimal amount of points into adjacency area to consider two faults are in one group.
    """
    # Containers for adjacency graph
    groups = {} # group owner -> items
    owners = {} # item -> group owner

    for i, fault_1 in enumerate(faults):
        if i not in owners.keys():
            owners[i] = i

        for j, fault_2 in enumerate(faults[i+1:]):
            adjacent_borders = bboxes_adjacent(fault_1.bbox, fault_2.bbox, adjacency=adjacency)

            if adjacent_borders is None:
                continue

            # Check points amount in the adjacency area
            for fault in (fault_1, fault_2):
                adjacent_points = fault.points[(fault.points[:, 0] >= adjacent_borders[0][0]) & \
                                               (fault.points[:, 0] <= adjacent_borders[0][1]) & \
                                               (fault.points[:, 1] >= adjacent_borders[1][0]) & \
                                               (fault.points[:, 1] <= adjacent_borders[1][1]) & \
                                               (fault.points[:, 2] >= adjacent_borders[2][0]) & \
                                               (fault.points[:, 2] <= adjacent_borders[2][1])]

                if len(adjacent_points) < adjacent_points_threshold:
                    adjacent_borders = None
                    break

            if adjacent_borders is None:
                continue

            # Graph update
            owners[i+1+j] = owners[i]

            if owners[i] not in groups.keys():
                groups[owners[i]] = set()

            groups[owners[i]].add(i+1+j)

    return groups, owners



# Groupings
# Group connected faults
def group_connected_prototypes(prototypes, connectivity_stats=None, ratio_threshold=0.0):
    """ Group connected prototypes.

    Connected prototypes are prototypes where at least one prototype has border overlap
    with another more than `ratio_threshold`.

    Parameters
    ----------
    prototypes : sequence of :class:`~.FaultPrototype` instances
        Prototypes for grouping.
        You can use this method with :class:`~.Fault` instances after conversion.
    connectivity_stats : dict or None, optional
        Output of :meth:`.eval_connectivity_stats`.
        Can be useful for multiple calls with different `ratio_threshold` values.
    ratio_threshold : float
        Overlap ratio to consider that prototypes are connected and can be grouped together.
    """
    # Eval connectivity stats
    if connectivity_stats is None:
        connectivity_stats = eval_connectivity_stats(prototypes)

    # Unpack connectivity graph info
    owners = {} # item -> owner
    groups = defaultdict(set) # owner -> set(items)

    for axis in (2, prototypes[0].direction):
        connectivity_stats_axis = connectivity_stats[axis]

        for prototype_1_idx, connect in connectivity_stats_axis.items():
            for prototype_2_idx, stat_values in connect.items():

                if stat_values['overlap_ratio'] > ratio_threshold:
                    owners, groups = _add_connected_pair(prototype_1_idx, prototype_2_idx, owners=owners, groups=groups)

    # Label prototypes group
    for idx, label in enumerate(prototypes):
        label.group_idx = idx

    for group_owner_idx, group_items in groups.items():
        prototypes[group_owner_idx].group_idx = group_owner_idx

        for item_idx in group_items:
            prototypes[item_idx].group_idx = group_owner_idx

    return prototypes

def eval_connectivity_stats(prototypes):
    """ Evaluation of overlap length and ratio for each prototypes pair.

    Note, zero-overlapping stats are omitted.

    It is a simplified version of `~.FaultExtractor.concat_connected_prototypes`.
    """
    direction = prototypes[0].direction
    orthogonal_direction = 1 - direction

    margin = 1 # local constant for code prettifying
    borders_to_check = {2: ('up', 'down'), direction: ('left', 'right')}

    connectivity_stats = {2: defaultdict(dict), direction: defaultdict(dict)}

    for i, prototype_1 in enumerate(prototypes):
        for j, prototype_2 in enumerate(prototypes[i+1:]):
            # Check prototypes adjacency
            adjacent_borders = bboxes_adjacent(prototype_1.bbox, prototype_2.bbox)

            if adjacent_borders is None:
                continue

            for axis in (2, direction):
                check_borders = borders_to_check[axis]

                # Find object contours on close borders
                is_first_upper = prototype_1.bbox[axis, 0] < prototype_2.bbox[axis, 0]

                contour_1 = prototype_1.get_border(border=check_borders[is_first_upper],
                                                   projection_axis=orthogonal_direction)
                contour_2 = prototype_2.get_border(border=check_borders[~is_first_upper],
                                                   projection_axis=orthogonal_direction)

                # Get border contours in the area of interest
                overlap_range = (min(adjacent_borders[axis]) - margin, max(adjacent_borders[axis]) + margin)

                contour_1 = contour_1[(contour_1[:, axis] >= overlap_range[0]) & \
                                      (contour_1[:, axis] <= overlap_range[1])]
                contour_2 = contour_2[(contour_2[:, axis] >= overlap_range[0]) & \
                                      (contour_2[:, axis] <= overlap_range[1])]

                # If one data contour is much longer than other, then we can't connect them as puzzle details
                if len(contour_1) == 0 or len(contour_2) == 0:
                    continue

                # Shift one of the objects, making their contours intersected
                shift = 1 if is_first_upper else -1
                contour_1[:, axis] += shift

                # Save stats
                overlap_1, overlap_1_ratio = _contours_overlap_stats(contour_1, contour_2,
                                                                     dilation_direction=orthogonal_direction)
                overlap_2, overlap_2_ratio = _contours_overlap_stats(contour_2, contour_1,
                                                                     dilation_direction=orthogonal_direction)

                if overlap_1 > 0:
                    connectivity_stats[axis][i][j+i+1] = {'overlap_length': overlap_1, 'overlap_ratio': overlap_1_ratio}
                    connectivity_stats[axis][j+i+1][i] = {'overlap_length': overlap_2, 'overlap_ratio': overlap_2_ratio}

    return connectivity_stats

def _contours_overlap_stats(contour_1, contour_2, dilation_direction, dilation=3):
    """ Evaluate contours overlap.

    Under the hood, we eval `contour_1` overlap statistics with dilated `contour_2`,
    because we suppsose that connected prototypes can be shifted to each other.
    """
    contour_1_set = set(tuple(x) for x in contour_1)

    # Objects can be shifted on `dilation_direction`, so apply dilation for coords
    contour_2_dilated = dilate_coords(coords=contour_2, dilate=dilation,
                                      axis=dilation_direction)

    contour_2_dilated = set(tuple(x) for x in contour_2_dilated)

    # Eval stats
    overlap = contour_1_set.intersection(contour_2_dilated)
    return len(overlap), len(overlap)/len(contour_1_set)

def _add_connected_pair(prototype_1_idx, prototype_2_idx, owners, groups):
    """ Add prototypes pair into group.

    We save connectivity info into two dicts:
    - owners (item -> owner) - information about which group the item belongs to;
    - groups (owner -> [items]) - information about which items are in the group.
    Items and owners here are prototypes indices.
    """
    if (prototype_1_idx not in owners) and (prototype_2_idx not in owners):
        # Add both, because they are new
        owners[prototype_1_idx] = prototype_1_idx
        owners[prototype_2_idx] = prototype_1_idx

        groups[prototype_1_idx].add(prototype_2_idx)

    elif (prototype_1_idx not in owners) and (prototype_2_idx in owners):
        # Add first into second
        owners[prototype_1_idx] = owners[prototype_2_idx]

        groups[owners[prototype_2_idx]].add(prototype_1_idx)

    elif (prototype_1_idx in owners) and (prototype_2_idx not in owners):
        # Add second into first
        owners[prototype_2_idx] = owners[prototype_1_idx]

        groups[owners[prototype_1_idx]].add(prototype_2_idx)

    else:
        # Merge two groups
        main_owner = owners[prototype_1_idx]
        other_owner = owners[prototype_2_idx]

        if main_owner != other_owner:
            for item in groups[other_owner]:
                owners[item] = main_owner

            groups[main_owner].update(groups[other_owner])
            groups[main_owner].add(other_owner)

            del groups[other_owner]

    return owners, groups


# Group faults with topK biggest faults and filter faults out of groups
def groups_with_biggest_faults(faults, height_threshold=None, groups_num=None,
                                  adjacency=5, adjacent_points_threshold=5):
    """ Get faults which can be merged in groups with the biggest faults.

    The biggest faults are faults with height more than `height_threshold` or
    which have topK-height, where K is `groups_num`.

    Groups are formed with adjacent faults.

    Note, that one of `height_threshold` or `groups_num` should be provided.

    Parameters
    ----------
    faults : sequence of :class:`~.Fault` or :class:`~.FaultPrototype` instances
        Faults for filtering.
    height_threshold : int or None
        Height threshold to consider that fault is big enough for being the biggest fault in the group.
        Group is a set of faults, adjoint with the biggest fault.
    groups_num : int or None
        Amount of groups to return.
        Under the hood, we find `groups_num` biggest faults and use the minimal height as threshold.
    adjacency : int
        Axis-wise distance between two faults to consider them to be grouped.
    adjacent_points_threshold : int
        Minimal amount of fault points into adjacency area to consider that two faults are in one group.
    """
    #pylint: disable=invalid-unary-operand-type
    if (groups_num is None) and (height_threshold is None):
        raise ValueError("One of `groups_num` or `height_threshold` must be not None!")

    # Get height threshold from groups num
    if height_threshold is None:
        heights = [fault.bbox[-1][1]-fault.bbox[-1][0]+1 for fault in faults]
        height_threshold = np.sort(heights)[-groups_num:-groups_num+1]

    # Find neighbors for the biggest faults and faults, that are included in groups with the biggest faults
    filtered_faults = []

    for fault_1 in faults:
        if (fault_1 not in filtered_faults) and (fault_1.bbox[-1][1] - fault_1.bbox[-1][0] + 1 < height_threshold):
            continue

        if fault_1 not in filtered_faults:
            filtered_faults.append(fault_1)

        for fault_2 in faults:
            if fault_2 in filtered_faults:
                continue

            adjacent_borders = bboxes_adjacent(fault_1.bbox, fault_2.bbox, adjacency=adjacency)

            if adjacent_borders is None:
                continue

            # Check points amount in the adjacency area
            for fault in (fault_1, fault_2):
                adjacent_points = fault.points[(fault.points[:, 0] >= adjacent_borders[0][0]) & \
                                               (fault.points[:, 0] <= adjacent_borders[0][1]) & \
                                               (fault.points[:, 1] >= adjacent_borders[1][0]) & \
                                               (fault.points[:, 1] <= adjacent_borders[1][1]) & \
                                               (fault.points[:, 2] >= adjacent_borders[2][0]) & \
                                               (fault.points[:, 2] <= adjacent_borders[2][1])]

                if len(adjacent_points) < adjacent_points_threshold:
                    adjacent_borders = None
                    break

            if adjacent_borders is None:
                continue

            if fault_2 not in filtered_faults:
                filtered_faults.append(fault_2)

    return filtered_faults
