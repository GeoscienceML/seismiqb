""" Faster version of groupby operations for numpy arrays. """
import numpy as np
from numba import njit



@njit
def groupby_mean(array):
    """ Faster version of mean-groupby of data along the first two columns.
    Input array is supposed to have (N, 3) shape.
    """
    n = len(array)

    output = np.zeros_like(array)
    position = 0

    prev = array[0, :2]
    s, c = array[0, -1], 1

    for i in range(1, n):
        curr = array[i, :2]

        if prev[0] == curr[0] and prev[1] == curr[1]:
            s += array[i, -1]
            c += 1
        else:
            output[position, :2] = prev
            output[position, -1] = round(s / c)
            position += 1

            prev = curr
            s, c = array[i, -1], 1

    output[position, :2] = prev
    output[position, -1] = s / c
    position += 1
    return output[:position]

@njit
def groupby_min(array):
    """ Faster version of min-groupby of data along the first two columns.
    Input array is supposed to have (N, 3) shape.
    """
    n = len(array)

    output = np.zeros_like(array)
    position = 0

    prev = array[0, :2]
    s = array[0, -1]

    for i in range(1, n):
        curr = array[i, :2]

        if prev[0] == curr[0] and prev[1] == curr[1]:
            s = min(s, array[i, -1])
        else:
            output[position, :2] = prev
            output[position, -1] = s
            position += 1

            prev = curr
            s = array[i, -1]

    output[position, :2] = prev
    output[position, -1] = s
    position += 1
    return output[:position]

@njit
def groupby_max(array):
    """ Faster version of max-groupby of data along the first two columns.
    Input array is supposed to have (N, 3) shape.
    """
    n = len(array)

    output = np.zeros_like(array)
    position = 0

    prev = array[0, :2]
    s = array[0, -1]

    for i in range(1, n):
        curr = array[i, :2]

        if prev[0] == curr[0] and prev[1] == curr[1]:
            s = max(s, array[i, -1])
        else:
            output[position, :2] = prev
            output[position, -1] = s
            position += 1

            prev = curr
            s = array[i, -1]

    output[position, :2] = prev
    output[position, -1] = s
    position += 1
    return output[:position]

@njit
def groupby_prob(array, probabilities):
    """ Faster version of weighted mean groupby of data along the first two columns.
    Input array is supposed to have (N, 3) shape.
    """
    n = len(array)

    output = np.zeros_like(array)
    position = 0

    prev = array[0, :2]
    s, c = array[0, -1] * probabilities[-1], probabilities[-1]

    for i in range(1, n):
        curr = array[i, :2]
        probability = probabilities[i]

        if prev[0] == curr[0] and prev[1] == curr[1]:
            s += array[i, -1] * probability
            c += probability
        else:
            output[position, :2] = prev
            output[position, -1] = round(s / c)
            position += 1

            prev = curr
            s, c = array[i, -1] * probability, probability

    output[position, :2] = prev
    output[position, -1] = round(s / c)
    position += 1
    return output[:position]


@njit
def groupby_all(array):
    """ For each trace, compute the number of points on it, min, max and mean values.
    `array` is expected to be of `(N, 3)` shape. Trace is defined by all points with the same first two coordinates.
    """
    # iline, crossline, occurency, min_, max_, mean_
    output = np.zeros((len(array), 6), dtype=np.int32)
    position = 0

    # Initialize values
    previous = array[0, :2]
    min_ = array[0, -1]
    max_ = array[0, -1]
    s = array[0, -1]
    c = 1

    for i in range(1, len(array)):
        current = array[i]

        if previous[1] == current[1] and previous[0] == current[0]:
            # Same iline, crossline: update values
            depth_ = current[-1]
            min_ = min(depth_, min_)
            max_ = max(depth_, max_)
            s += depth_
            c += 1

        else:
            # New iline, crossline: store stats, re-initialize values
            output[position, :2] = previous   # iline, crossline
            output[position, 2] = c           # occurency
            output[position, 3] = min_        # min_
            output[position, 4] = max_        # max_
            output[position, 5] = s / c       # mean_
            position += 1

            depth_ = current[-1]
            previous = current[:2]
            min_ = depth_
            max_ = depth_
            s = depth_
            c = 1

    # The last point
    output[position, :2] = previous   # iline, crossline
    output[position, 2] = c           # occurency
    output[position, 3] = min_        # min_
    output[position, 4] = max_        # max_
    output[position, 5] = s / c       # mean_
    position += 1

    return output[:position]
