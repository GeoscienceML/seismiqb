""" Faults extraction helpers for 3D coordinates processing. """
import numpy as np
from numba import njit
import cv2 as cv

# Coordinates operations
def dilate_coords(coords, dilate=3, axis=0, max_value=None):
    """ Dilate coordinates with (dilate, 1) structure along the given axis.

    Note, the function returns unique and sorted coords.

    Parameters
    ----------
    coords : np.ndarray of (N, 3) shape
        Coordinates to dilate along the axis. Sorting is not required.
    axis : {0, 1, 2}
        Axis along which to dilate coordinates.
    max_value : None or int, optional
        The maximum possible value for coordinates along the provided axis.
        Used for values clipping into valid range.
    """
    dilated_coords = np.tile(coords, (dilate, 1))

    # Create dilated coordinates
    for i in range(dilate):
        start_idx, end_idx = i*len(coords), (i + 1)*len(coords)
        dilated_coords[start_idx:end_idx, axis] += i - dilate//2

    # Clip to the valid values
    mask = dilated_coords[:, axis] >= 0

    if max_value is not None:
        mask &= dilated_coords[:, axis] < max_value

    dilated_coords = dilated_coords[mask]

    # Get sorted unique values
    dilated_coords = np.unique(dilated_coords, axis=0)
    return dilated_coords


# Distance evaluation
@njit
def bboxes_intersected(bbox_1, bbox_2, axes=(0, 1, 2)):
    """ Check bounding boxes intersection on preferred axes.

    Bboxes are intersected if they have at least 1 overlapping point.

    Parameters
    ----------
    bbox_1, bbox_2 : np.ndarrays of (3, 2) shape.
        Objects bboxes.
    axes : sequence of int values from {0, 1, 2}
        Axes to check bboxes intersection.
    """
    for axis in axes:
        overlap_size = min(bbox_1[axis, 1], bbox_2[axis, 1]) - max(bbox_1[axis, 0], bbox_2[axis, 0]) + 1

        if overlap_size < 1:
            return False
    return True

@njit
def bboxes_adjacent(bbox_1, bbox_2, adjacency=1):
    """ Bounding boxes adjacency ranges.

    Bboxes are adjacent if they are distant not more than on `adjacency` points.

    Parameters
    ----------
    bbox_1, bbox_2 : np.ndarrays of (3, 2) shape.
        Objects bboxes.
    adjacency : int
        Amount of points between two bboxes to decide that they are adjacent.
    """
    borders = np.empty((3, 2), dtype=np.int32)

    for i in range(3):
        borders_i_0 = max(bbox_1[i, 0], bbox_2[i, 0])
        borders_i_1 = min(bbox_1[i, 1], bbox_2[i, 1])

        if borders_i_1 - borders_i_0 < -adjacency:
            return None

        borders[i, 0] = min(borders_i_0, borders_i_1)
        borders[i, 1] = max(borders_i_0, borders_i_1)

    return borders

def bboxes_embedded(bbox_1, bbox_2, margin=3):
    """ Check that one bounding box is inside the other (embedded).

    Parameters
    ----------
    bbox_1, bbox_2 : np.ndarrays of (3, 2) shape.
        Objects bboxes.
    margin : int
        Possible bboxes difference (on each axis) to decide that one is inside another.
    """
    swap = np.count_nonzero(bbox_1[:, 1] >= bbox_2[:, 1]) <= 1 # is second not inside first

    if swap:
        bbox_1, bbox_2 = bbox_2, bbox_1

    for i in range(3):
        is_embedded = (bbox_2[i, 0] >= bbox_1[i, 0] - margin) and (bbox_2[i, 1] <= bbox_1[i, 1] + margin)

        if not is_embedded:
            return is_embedded, swap

    return is_embedded, swap

@njit
def compute_distances(coords_1, coords_2, max_threshold=10000):
    """ Find approximate minimum and maximum distances between two arrays of coordinates.
    We assume coords to have the same length and compare only corresponding points.
    A little bit faster than difference between np.ndarrays with `np.max` and `np.min`.

    Parameters
    ----------
    coords_1, coords_2 : np.ndarrays of (N, 1) shape
        Coords for which find distances.
    max_threshold : int, float or None
        Early stopping: threshold for max distance value.
    """
    min_distance = max_threshold
    max_distance = 0

    for coord_1, coord_2 in zip(coords_1, coords_2):
        distance = np.abs(coord_1 - coord_2)

        if distance >= max_threshold:
            return -1, distance

        if distance > max_distance:
            max_distance = distance

        if distance < min_distance:
            min_distance = distance

    return min_distance, max_distance

def find_contour(coords, projection_axis):
    """ Find closed contour of coords projection.

    Under the hood, we make a 2D coords projection and find its contour.

    Note, returned contour coordinates are equal to 0 for the projection axis.

    Parameters
    ----------
    coords : np.ndarray of (N, 3) shape
        3D object coordinates. Sorting is not required.
    projection_axis : {0, 1}
        Axis for making 2D projection.
        Note, this function doesn't work for axis = 2.
    """
    bbox = np.column_stack([np.min(coords, axis=0), np.max(coords, axis=0)])
    bbox = bbox[(1 - projection_axis, 2), :]

    # Create object image mask
    origin = bbox[:, 0]
    image_shape = bbox[:, 1] - bbox[:, 0] + 1

    mask = np.zeros(image_shape, np.uint8)
    mask[coords[:, 1 - projection_axis] - origin[0], coords[:, 2] - origin[1]] = 1

    # Get only the main object contour: object can contain holes with their own contours
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Extract unique and sorted coords
    contour = contours[0].reshape(len(contours[0]), 2) # Can be non-unique

    contour_coords = np.zeros((len(contour), 3), np.int32)
    contour_coords[:, 1 - projection_axis] = contour[:, 1] + origin[0]
    contour_coords[:, 2] = contour[:, 0] + origin[1]

    contour_coords = np.unique(contour_coords, axis=0) # np.unique is here for sorting and unification
    return contour_coords

@njit
def restore_coords_from_projection(coords, projection_buffer, axis):
    """ Get values along `axis` for 2D projection coordinates from 3D coords.

    Example
    -------
    Useful, where we have subsetted original `coords` and zero-out the result along some axis::
        coords, indices, axis
        subset = coords[indices]
        subset[:, axis] = 0

        restore_coords_from_projection(coords, subset, axis) # change zeros back to original values from `coords`


    Parameters
    ----------
    coords : np.ndarray of (N, 3) shape
        Original coords from which restore the axis values. Sorting is not required.
    projection_buffer : np.ndarray of (N, 3) shape
        Buffer with projection coordinates. Initially, values along `axis` are zeros. Sorting is not required.
        Changed inplace.
    axis : {0, 1, 2}
        Axis for which restore coordinates.
    """
    known_axes = np.array([i for i in range(3) if i != axis])

    for i, buffer_line in enumerate(projection_buffer):
        values =  coords[(coords[:, known_axes[0]] == buffer_line[known_axes[0]]) & \
                         (coords[:, known_axes[1]] == buffer_line[known_axes[1]]),
                         axis]

        projection_buffer[i, axis] = min(values) if len(values) > 0 else -1

    projection_buffer = projection_buffer[projection_buffer[:, axis] != -1]
    return projection_buffer
