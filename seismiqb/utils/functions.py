""" Utility functions. """
import os
import string
import random

import numpy as np
from numba import njit, prange



def file_print(*msg, path, mode='w', **kwargs):
    """ Print to file. """
    with open(path, mode, encoding='utf-8') as file:
        print(*msg, file=file, **kwargs)

def select_printer(printer):
    """ Select printing method. """
    if isinstance(printer, str):
        return lambda *msg, **kwargs: file_print(*msg, path=printer, **kwargs)
    if callable(printer):
        return printer
    return print


def generate_string(size=10, chars=string.ascii_uppercase + string.digits):
    """ Generate random string of given size. """
    return ''.join(random.choice(chars) for _ in range(size))


@njit(parallel=True)
def filtering_function(points, filtering_matrix):
    """ Remove points where `filtering_matrix` is 1. """
    #pylint: disable=consider-using-enumerate, not-an-iterable
    mask = np.ones(len(points), dtype=np.int32)

    for i in prange(len(points)):
        il, xl = points[i, 0], points[i, 1]
        if filtering_matrix[il, xl] == 1:
            mask[i] = 0
    return points[mask == 1, :]


@njit
def filter_simplices(simplices, points, matrix, threshold=5.):
    """ Remove simplices outside of matrix. """
    #pylint: disable=consider-using-enumerate
    mask = np.ones(len(simplices), dtype=np.int32)

    for i in range(len(simplices)):
        tri = points[simplices[i]].astype(np.int32)

        middle_i, middle_x = np.mean(tri[:, 0]), np.mean(tri[:, 1])
        depths = np.array([matrix[tri[0, 0], tri[0, 1]],
                            matrix[tri[1, 0], tri[1, 1]],
                            matrix[tri[2, 0], tri[2, 1]]])

        if matrix[int(middle_i), int(middle_x)] < 0 or np.std(depths) > threshold:
            mask[i] = 0

    return simplices[mask == 1]


def make_bezier_figure(n=7, radius=0.2, sharpness=0.05, scale=1.0, shape=(1, 1),
                       resolution=None, distance=.5, seed=None):
    """ Bezier closed curve coordinates.
    Creates Bezier closed curve which passes through random points.
    Code based on:  https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib

    Parameters
    ----------
    n : int
        Number more than 1 to control amount of angles (key points) in the random figure.
        Must be more than 1.
    radius : float
        Number between 0 and 1 to control the distance of middle points in Bezier algorithm.
    sharpness : float
        Degree of sharpness/edgy. If 0 then a curve will be the smoothest.
    scale : float
        Number between 0 and 1 to control figure scale. Fits to the shape.
    shape : sequence int
        Shape of figure location area.
    resolution : int
        Amount of points in one curve between two key points.
    distance : float
        Number between 0 and 1 to control distance between all key points in a unit square.
    seed: int, optional
        Seed the random numbers generator.
    """
    rng = np.random.default_rng(seed)
    resolution = resolution or int(100 * scale * max(shape))

    # Get key points of figure as random points which are far enough each other
    key_points = rng.random((n, 2))
    squared_distance = distance ** 2

    squared_distances = squared_distance - 1
    while np.any(squared_distances < squared_distance):
        shifted_points = key_points - np.mean(key_points, axis=0)
        angles = np.arctan2(shifted_points[:, 0], shifted_points[:, 1])
        key_points = key_points[np.argsort(angles)]

        squared_distances = np.sum(np.diff(key_points, axis=0)**2, axis=1)
        key_points = rng.random((n, 2))

    key_points *= scale * np.array(shape, float)
    key_points = np.vstack([key_points, key_points[0]])

    # Calculate figure angles in key points
    p = np.arctan(sharpness) / np.pi + .5
    diff_between_points = np.diff(key_points, axis=0)
    angles = np.arctan2(diff_between_points[:, 1], diff_between_points[:, 0])
    angles = angles + 2 * np.pi * (angles < 0)
    rolled_angles = np.roll(angles, 1)
    angles = p * angles + (1 - p) * rolled_angles + np.pi * (np.abs(rolled_angles - angles) > np.pi)
    angles = np.append(angles, angles[0])

    # Create figure part by part: make curves between each pair of points
    curve_segments = []
    # Calculate control points for Bezier curve
    points_distances = np.sqrt(np.sum(diff_between_points ** 2, axis=1))
    radii = radius * points_distances
    middle_control_points_1 = np.transpose(radii * [np.cos(angles[:-1]),
                                                    np.sin(angles[:-1])]) + key_points[:-1]
    middle_control_points_2 = np.transpose(radii * [np.cos(angles[1:] + np.pi),
                                                    np.sin(angles[1:] + np.pi)]) + key_points[1:]
    curve_main_points_arr = np.hstack([key_points[:-1], middle_control_points_1,
                                       middle_control_points_2, key_points[1:]]).reshape(n, 4, -1)

    # Get Bernstein polynomial approximation of each curve
    binom_coefficients = [1, 3, 3, 1]
    for i in range(n):
        bezier_param_t = np.linspace(0, 1, num=resolution)
        current_segment = np.zeros((resolution, 2))
        for point_num, point in enumerate(curve_main_points_arr[i]):
            binom_coefficient = binom_coefficients[point_num]
            polynomial_degree = np.power(bezier_param_t, point_num)
            polynomial_degree *= np.power(1 - bezier_param_t, 3 - point_num)
            bernstein_polynomial = binom_coefficient * polynomial_degree
            current_segment += np.outer(bernstein_polynomial, point)
        curve_segments.extend(current_segment)

    curve_segments = np.array(curve_segments)
    figure_coordinates = np.unique(np.ceil(curve_segments).astype(int), axis=0)
    return figure_coordinates


def trinagular_kernel_1d(length, alpha=.1):
    """ Kernel-function that changes linearly from a center point to alpha on borders. """
    result = np.zeros(length)
    array = np.linspace(alpha, 2, length)
    result[:length // 2] = array[:length // 2]
    result[length // 2:] = 2 + alpha - array[length // 2:]
    return result

def triangular_weights_function_nd(array, alpha=.1):
    """ Weights-function given by a product of 1d triangular kernels. """
    result = 1
    for i, axis_len in enumerate(array.shape):
        if axis_len != 1:
            multiplier_shape = np.ones_like(array.shape)
            multiplier_shape[i] = axis_len
            result = result * trinagular_kernel_1d(axis_len, alpha).reshape(multiplier_shape)
    return result


def to_list(obj):
    """ Cast an object to a list.
    When default value provided, cast it instead if object value is None.
    Almost identical to `list(obj)` for 1-D objects, except for `str` instances,
    which won't be split into separate letters but transformed into a list of a single element.
    """
    return np.array(obj, dtype=object).ravel().tolist()

def make_savepath(path, name, extension=''):
    """ If given replace asterisk in path with label name and create save dir if it does not already exist. """
    if path.endswith('*'):
        path = path.replace('*', f'{name}{extension}')

    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    return path


def make_ranges(ranges, shape):
    """ Make a `ranges` tuple, valid for indexing 3-dimensional arrays:
        - each element is clipped to `(0, shape[i])` range,
        - None elements are changed to `(0, shape[i])`,
        - None at the first place of tuple-element is changed by 0
        - None at the second place of tuple-element is changed by `shape[i]`
        If `ranges` is None, then treated as a tuple of three None's.

    Example
    -------
        None -> (0, shape[0]), (0, shape[1]), (0, shape[2])
        None, None, None -> (0, shape[0]), (0, shape[1]), (0, shape[2])
        (-10, shape[0]+2), (0, 100), (0, None) -> (0, shape[0]), (0, 100), (0, shape[2])
        (10, 20), (10, 20), (10, 20) -> (10, 20), (10, 20), (10, 20)
    """
    if ranges is None:
        ranges = [None, None, None]
    ranges = [(0, c) if item is None else item for item, c in zip(ranges, shape)]
    ranges = [(item[0] or 0, item[1] or c) for item, c in zip(ranges, shape)]
    ranges = [(max(0, item[0]), min(c, item[1])) for item, c in zip(ranges, shape)]
    return tuple(ranges)

def make_slices(slices, shape):
    """ Fill Nones in tuple of slices (analogously to `make_ranges`). """
    if slices is None:
        ranges = None
    else:
        ranges = [None if item is None else (item.start, item.stop) for item in slices]

    ranges = make_ranges(ranges, shape)
    return tuple(slice(*item) for item in ranges)

def make_interior_points_mask(points, cube_shape):
    """ Create mask for points inside of the cube. """
    mask = np.where((points[:, 0] >= 0) &
                    (points[:, 1] >= 0) &
                    (points[:, 2] >= 0) &
                    (points[:, 0] < cube_shape[0]) &
                    (points[:, 1] < cube_shape[1]) &
                    (points[:, 2] < cube_shape[2]))[0]
    return mask


@njit(parallel=True)
def insert_points_into_mask(mask, points, mask_bbox, width, axis, alpha=1):
    """ Add new points into binary mask.

    Parameters
    ----------
    mask : numpy.ndarray
        Array to insert values which correponds to some region in 3d cube (see `mask_bbox` parameter)
    points : numpy.ndarray
        Array of shape `(n_points, 3)` with cube coordinates of points to insert.
    mask_bbox : numpy.ndarray
        Array of shape (3, 2) with postion of the mask in 3d cube
    width : int
        Dilation of the mask along some axis.
    axis : int
        Direction of dilation.
    """
    #pylint: disable=not-an-iterable, too-many-boolean-expressions

    left_margin = [0, 0, 0]
    right_margin = [1, 1, 1]
    left_margin[axis] = width // 2
    right_margin[axis] = width - width // 2

    for i in prange(len(points)):
        point = points[i]
        if ((point[0] >= mask_bbox[0][0] - left_margin[0]) and
            (point[1] >= mask_bbox[1][0] - left_margin[1]) and
            (point[2] >= mask_bbox[2][0] - left_margin[2]) and
            (point[0] <  mask_bbox[0][1] + right_margin[0] - 1) and
            (point[1] <  mask_bbox[1][1] + right_margin[1] - 1) and
            (point[2] <  mask_bbox[2][1] + right_margin[2] - 1)):

            point = point - mask_bbox[:, 0]
            left_bound = max(0, point[axis] - left_margin[axis])
            right_bound = min(mask.shape[axis], point[axis] + right_margin[axis])

            if axis == 0:
                for pos in range(left_bound, right_bound):
                    mask[pos, point[1], point[2]] = alpha
            elif axis == 1:
                for pos in range(left_bound, right_bound):
                    mask[point[0], pos, point[2]] = alpha
            elif axis == 2:
                for pos in range(left_bound, right_bound):
                    mask[point[0], point[1], pos] = alpha


def take_along_axis(array, index, axis):
    """ A functional equivalent of `np.take` which returns a view.
    Unlike `np.take`, should be used only with indices that are ints or slice.
    """
    if axis == 0:
        slide = array[index, :, :]
    elif axis == 1:
        slide = array[:, index, :]
    elif axis == 2:
        slide = array[:, :, index]
    return slide
