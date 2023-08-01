""" Geometric transforms. """
from math import ceil, atan, cos, sin

import numpy as np
import cv2



# Rotate
def rotate_2d(array, angle, adjust=False, fill_value=0):
    """ Rotate an image along the first two axes.
    Assumes `channels_last` format.

    Parameters
    ----------
    angle : number
        Angle of rotation in degrees.
    adjust : bool
        If True, then image is upscaled prior to rotation to avoid padding.
    fill_value : number
        Padding value.
    """
    if adjust:
        array, initial_shape = adjust_shape(array, angle=angle)
    array = _rotate_2d(array=array, angle=angle, fill_value=fill_value)
    if adjust:
        array = center_crop(array, shape=initial_shape)
    return array

def rotate_3d(array, angle, adjust=False, fill_value=0):
    """ Rotate an image in 3D.
    Angles are defined as Tait-Bryan angles and the sequence of extrinsic rotations axes is (axis_2, axis_0, axis_1).

    Parameters
    ----------
    angle : number
        Angle of rotation in degrees.
    fill_value : number
        Padding value.
    """
    if adjust:
        raise NotImplementedError('`adjust` is not implemented for 3D rotation!')

    if angle[0] != 0:
        array = _rotate_2d(array, angle[0], fill_value)
    if angle[1] != 0:
        array = _rotate_2d(array.transpose(1, 2, 0), angle[1], fill_value).transpose(2, 0, 1)
    if angle[2] != 0:
        array = _rotate_2d(array.transpose(2, 0, 1), angle[2], fill_value).transpose(1, 2, 0)
    return array

def _rotate_2d(array, angle, fill_value=0):
    """ Rotate an image along the first two axes. """
    shape = array.shape
    matrix = cv2.getRotationMatrix2D((shape[1]//2, shape[0]//2), angle, 1)
    return cv2.warpAffine(array, matrix, (shape[1], shape[0]), borderValue=fill_value).reshape(shape)

# Resize
def resize(array, size, interpolation=1):
    """ Resize image. """
    # interpolation=1 means bilinear
    if array.shape[0] == 1:
        resized = cv2.resize(src=array.squeeze(), dsize=(size[1], size[0]), interpolation=interpolation)
        resized = resized.reshape(1, *resized.shape)
    else:
        resized = cv2.resize(src=array, dsize=(size[1], size[0]), interpolation=interpolation)
    return resized

# Scale
def scale_2d(array, scale, adjust=False):
    """ Zoom in/out of the image along the first two axes.

    Parameters
    ----------
    scale : tuple or float
        Zooming factor for the first two axis.
    adjust : bool
        If True, then image is upscaled prior to rotation to avoid padding.
    """
    scale = scale if isinstance(scale, (list, tuple, np.ndarray)) else [scale] * 2

    if adjust:
        array, initial_shape = adjust_shape(array, angle=0, scale=(*scale, 1))
    array = _scale_2d(array, scale)
    if adjust:
        array = center_crop(array, shape=initial_shape)
    return array

def scale_3d(array, scale, adjust=False):
    """ Zoom in/out of the image in 3D.

    Parameters
    ----------
    scale : tuple or float
        Zooming factor for the first two axis.
    """
    if adjust:
        raise NotImplementedError('`adjust` is not implemented for 3D rotation!')
    scale = scale if isinstance(scale, (list, tuple)) else [scale] * 3
    array = _scale_2d(array,
                      [scale[0], scale[1]])
    array = _scale_2d(array.transpose(1, 2, 0),
                      [1, scale[-1]]).transpose(2, 0, 1)
    return array

def _scale_2d(array, scale):
    """ Zoom in/out of the image along the first two axes. """
    shape = array.shape
    matrix = np.zeros((2, 3))
    matrix[:, :-1] = np.diag([scale[1], scale[0]])
    matrix[:, -1] = np.array([shape[1], shape[0]]) * (1 - np.array([scale[1], scale[0]])) / 2
    return cv2.warpAffine(array, matrix, (shape[1], shape[0])).reshape(shape)


# Adjust shape
def adjust_shape(array, angle=0, scale=(1, 1, 1)):
    """ Resize image to avoid padding for rotation/scaling operations. """
    initial_shape = array.shape
    adjusted_shape = adjust_shape_3d(shape=array.shape, angle=angle, scale=scale)
    array = cv2.resize(array, dsize=(adjusted_shape[1], adjusted_shape[0])).reshape(*adjusted_shape)
    return array, initial_shape

def adjust_shape_3d(shape, angle=0, scale=(1, 1, 1)):
    """ Compute adjusted 3D crop shape to rotate/scale it and get central crop without padding.

    Adjustments is based on assumption that rotation angles are defined as Tait-Bryan angles and
    the sequence of extrinsic rotations axes is (axis_2, axis_0, axis_1), and scale performed after rotation.

    Parameters
    ----------
    shape : tuple
        Input shape.
    angle : float or tuple of floats
        Rotation angles about each axis.
    scale : int or tuple, optional
        Scale for each axis.

    Returns
    -------
    tuple
        Adjusted shape.
    """
    angle = angle if isinstance(angle, (tuple, list)) else (angle, 0, 0)
    scale = scale if isinstance(scale, (tuple, list)) else (scale, scale, 1)

    shape = np.ceil(np.array(shape) / np.array(scale)).astype(int)
    if angle[2] != 0:
        shape[2], shape[0] = _adjust_shape_for_rotation((shape[2], shape[0]), angle[2])
    if angle[1] != 0:
        shape[2], shape[1] = _adjust_shape_for_rotation((shape[2], shape[1]), angle[1])
    if angle[0] != 0:
        shape[0], shape[1] = _adjust_shape_for_rotation((shape[0], shape[1]), angle[0])
    return tuple(shape)

def _adjust_shape_for_rotation(shape, angle):
    """ Compute adjusted 2D crop shape to rotate it and get central crop without padding. """
    angle = abs(2 * np.pi * angle / 360)
    limit = atan(shape[1] / shape[0])
    x_max, y_max = shape
    if angle != 0:
        if angle < limit:
            x_max = shape[0] * cos(angle) + shape[1] * sin(angle) + 1
        else:
            x_max = (shape[0] ** 2 + shape[1] ** 2) ** 0.5 + 1

        if angle < np.pi / 2 - limit:
            y_max = shape[0] * sin(angle) + shape[1] * cos(angle) + 1
        else:
            y_max = (shape[0] ** 2 + shape[1] ** 2) ** 0.5 + 1
    return (int(ceil(x_max)), int(ceil(y_max)))


# Crops
def center_crop(array, shape):
    """ Cut center crop of given shape. """
    old_shape, new_shape = np.array(array.shape), np.array(shape)
    if (new_shape > old_shape).any():
        raise ValueError(f'Output shape={new_shape} can\'t be larger than input shape={old_shape}!')

    corner = old_shape // 2 - new_shape // 2
    slices = tuple(slice(start, start + length) for start, length in zip(corner, new_shape))
    return array[slices]


# Coordinate transforms
def affine_transform(array, alpha=10, rng=None):
    """ Perspective transform. Moves three points to other locations.
    Guaranteed not to flip image or scale it more than 2 times.

    Parameters
    ----------
    alpha : float
        Maximum distance along each axis between points before and after transform.
    """
    rng = rng or np.random.default_rng(rng)

    shape = np.array(array.shape)[:2]
    alpha = max(alpha, min(shape) // 16)

    center = shape // 2
    square_size = min(shape) // 3

    pts1 = np.float32([center + square_size,
                       center - square_size,
                       [center[0] + square_size, center[1] - square_size]])
    pts2 = pts1 + rng.uniform(-alpha, alpha, size=pts1.shape).astype(np.float32)
    matrix = cv2.getAffineTransform(pts1, pts2)

    return cv2.warpAffine(array, matrix, (shape[1], shape[0])).reshape(array.shape)


def perspective_transform(array, alpha, rng=None):
    """ Perspective transform. Moves four points to other four.
    Guaranteed not to flip image or scale it more than 2 times.

    Parameters
    ----------
    alpha : float
        Maximum distance along each axis between points before and after transform.
    """
    rng = rng or np.random.default_rng(rng)

    shape = np.array(array.shape)[:2]
    alpha = max(alpha, min(shape) // 16)

    center_ = shape // 2
    square_size = min(shape) // 3

    pts1 = np.float32([center_ + square_size,
                       center_ - square_size,
                       [center_[0] + square_size, center_[1] - square_size],
                       [center_[0] - square_size, center_[1] + square_size]])
    pts2 = pts1 + rng.uniform(-alpha, alpha, size=pts1.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    return cv2.warpPerspective(array, matrix, (shape[1], shape[0])).reshape(array.shape)

def elastic_transform(array, alpha=40, sigma=4, rng=None):
    """ Transform indexing grid of the first two axes.

    Parameters
    ----------
    alpha : float
        Maximum shift along each axis.
    sigma : float
        Smoothening factor.
    """
    rng = rng or np.random.default_rng(rng)
    shape_size = array.shape[:2]

    grid_scale = 4
    alpha //= grid_scale
    sigma //= grid_scale
    grid_shape = (shape_size[0]//grid_scale, shape_size[1]//grid_scale)

    blur_size = int(4 * sigma) | 1
    rand_x = cv2.GaussianBlur(rng.random(size=grid_shape, dtype=np.float32) * 2 - 1,
                              ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    rand_y = cv2.GaussianBlur(rng.random(size=grid_shape, dtype=np.float32) * 2 - 1,
                              ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    if grid_scale > 1:
        rand_x = cv2.resize(rand_x, shape_size[::-1])
        rand_y = cv2.resize(rand_y, shape_size[::-1])

    grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
    grid_x = (grid_x.astype(np.float32) + rand_x)
    grid_y = (grid_y.astype(np.float32) + rand_y)

    return cv2.remap(array, grid_x, grid_y,
                     borderMode=cv2.BORDER_REFLECT_101,
                     interpolation=cv2.INTER_LINEAR).reshape(array.shape)
