""" Approximation utilities to convert cloud of points to sticks. """

import numpy as np
from sklearn.decomposition import PCA
import cv2
from numba import njit

from .postprocessing import thin_line, split_array


def points_to_sticks(points, sticks_step=10, nodes_step='auto', fault_orientation=None, stick_orientation=2,
                     threshold=5, move_bounds=False):
    """ Get sticks from fault which is represented as a cloud of points.

    Parameters
    ----------
    points : np.ndarray
        Fault points.
    sticks_step : int
        Number of slides between sticks.
    nodes_step : int
        Maximum distance between stick nodes
    fault_orientation : int (0, 1 or 2)
        Direction of the fault
    stick_orientation : int (0, 1 or 2)
        Direction of each stick
    threshold : int
        Threshold to remove nodes which are too close, by default 5. If nodes_step is int, real threshold will be equal
        to `min(threshold, nodes_step // 2)`.
    move_bounds : bool
        Whether to extend fault by moving bound sticks to the nearest slide with index which is a multiple of
        sticks_step.

    Returns
    -------
    numpy.ndarray
        Array of sticks. Each item of array is a stick: sequence of 3D points.
    """
    if fault_orientation is None:
        pca = PCA(1)
        pca.fit(points)
        fault_orientation = 0 if np.abs(pca.components_[0][0]) > np.abs(pca.components_[0][1]) else 1

    if stick_orientation != 2:
        fault_orientation = 2

    points = points[np.argsort(points[:, fault_orientation])]
    if len(points) == 0:
        return []

    slides = split_array(points, points[:, fault_orientation])

    sticks = []

    indices = [i for i, slide_points in enumerate(slides) if slide_points[0, fault_orientation] % sticks_step == 0]

    if move_bounds and 0 not in indices:
        first_stick = slides[0]
        first_stick[:, fault_orientation] = max(
            0, first_stick[0, fault_orientation] - (first_stick[0, fault_orientation] % sticks_step)
        )

    if move_bounds and len(slides)-1 not in indices:
        last_stick = slides[-1]
        shift = sticks_step - last_stick[0, fault_orientation] % sticks_step
        if shift == sticks_step:
            shift = 0
        last_stick[:, fault_orientation] = last_stick[0, fault_orientation] + shift

    indices = [0] + indices + [len(slides) - 1]

    indices = sorted(list(set(indices)))

    for idx in indices:
        slide_points = slides[idx]
        slide_points = slide_points[np.argsort(slide_points[:, stick_orientation])]
        slide_points = thin_line(slide_points, stick_orientation)
        if len(slide_points) > 5:
            nodes = find_stick_nodes(points=slide_points, fault_orientation=fault_orientation,
                                     stick_orientation=stick_orientation, nodes_step=nodes_step,
                                     threshold=threshold).astype('float32')

            # Remove redundant nodes from sticks with the large number of nodes
            if len(nodes) > 4 and nodes_step == 'auto':
                normal = 3 - fault_orientation - stick_orientation
                nodes = nodes[np.unique(remove_redundant_nodes(nodes[:, [normal, stick_orientation]]))]
        else:
            nodes = slide_points[[0, -1]]
        if len(nodes) > 1:
            sticks.append(nodes)

    return sticks


def find_stick_nodes(points, fault_orientation, stick_orientation, nodes_step='auto', threshold=5):
    """ Get sticks from the line (with some width) defined by cloud of points

    Parameters
    ----------
    points : numpy.ndarray
        3D points located on one 2D slide
    fault_orientation : int (0, 1 or 2)
        Direction of the fault
    stick_orientation : int (0, 1 or 2)
        Direction of each stick
    nodes_step : int or 'auto'
        The step between sequent nodes. If 'auto', the optimal number will be chosen.
    threshold : int, optional
        Threshold to remove nodes which are too close, by default 5. If nodes_step is int, real threshold will be equal
        to `min(threshold, nodes_step // 2)`.

    Returns
    -------
    numpy.ndarray
        Stick nodes
    """
    if len(points) <= 2:
        return points

    if nodes_step != 'auto':
        threshold = min(threshold, nodes_step // 2)

    normal = 3 - fault_orientation - stick_orientation

    mask = np.zeros(points.ptp(axis=0)[[normal, stick_orientation]] + 1)
    mask[
        points[:, normal] - points[:, normal].min(),
        points[:, stick_orientation] - points[:, stick_orientation].min()
    ] = 1

    if nodes_step == 'auto':
        line_threshold = cv2.threshold(mask.astype(np.uint8) * 255, 127, 255, 0)[1]
        line_contours = cv2.findContours(line_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)[0]
        nodes = np.unique(np.squeeze(np.concatenate(line_contours)), axis=0) #TODO: unique?
        nodes[:, 0] = nodes[:, 0] + points[:, stick_orientation].min()
        nodes[:, 1] = nodes[:, 1] + points[:, normal].min()
    else:
        indices = list(range(0, len(points), nodes_step))
        if len(points) - 1 not in indices:
            indices = indices + [len(points) - 1]
        nodes = points[indices][:, [stick_orientation, normal]]

    new_points = np.zeros((len(nodes), 3))
    new_points[:, fault_orientation] = points[0, fault_orientation]
    new_points[:, stick_orientation] = nodes[:, 0]
    new_points[:, normal] = nodes[:, 1]
    new_points = new_points[np.argsort(new_points[:, stick_orientation])]

    if threshold > 0:
        # Remove nodes which are too close
        mask = np.concatenate([[True], np.abs(new_points[2:] - new_points[1:-1]).sum(axis=1) > threshold, [True]])
        new_points = new_points[mask]
    return new_points

@njit
def node_deviation(start, end, point):
    """ The distance (in 2D) between `point` and line from `start` to `end`. """
    return np.abs(point[0] - (point[1] - start[1]) / (end[1] - start[1]) * (end[0] - start[0]) - start[0])

@njit
def remove_redundant_nodes(nodes, threshold=1.5):
    """ Remove unnecessary points from stick. """
    nodes_diff = np.ediff1d(nodes[:, 1])
    pos = np.argmax(np.minimum(nodes_diff[:-1], nodes_diff[1:]), axis=0) + 1 # node farthest from neighbors
    filtered_nodes = [pos]
    for direction in [-1, 1]:
        current_pos = pos
        pos_to_check = pos + direction

        while (pos_to_check + direction < len(nodes)) and (pos_to_check + direction >= 0):
            if node_deviation(nodes[current_pos], nodes[pos_to_check + direction], nodes[pos_to_check]) > threshold:
                filtered_nodes += [current_pos]
                current_pos = pos_to_check
                pos_to_check = current_pos + direction
            else:
                pos_to_check += direction

    filtered_nodes += [0, len(nodes)-1]
    return filtered_nodes
