""" Utils for faults postprocessing. """
import numpy as np
from numba import njit, prange
from numba.types import bool_

from ...functional import make_gaussian_kernel


@njit(parallel=True)
def skeletonize(slide, width=5, rel_height=0.5, prominence=0.05, threshold=0.05, distance=None, mode=0, axis=1):
    """ Perform skeletonize of faults on 2D slide

    Parameters
    ----------
    slide : numpy.ndarray

    width : int, optional
        width of peaks, by default 5
    rel_height, threshold : float, optional
        parameters of :meth:~.find_peaks`
    prominence : float
        prominence threshold value
    threshold : float
        nullify values below the threshold
    mode : int (from 0 to 4)
        which value to place in the output
        0: ones
        1: peak prominences
        2: values from initial slide
        3: values from initial slide multiplied by prominences
        4: average between values from initial slide and prominences
    Returns
    -------
    numpy.ndarray
        skeletonized slide
    """
    skeletonized_slide = np.zeros_like(slide, dtype='float32')
    for i in prange(slide.shape[axis]): #pylint: disable=not-an-iterable
        x = slide[:, i] if axis == 1 else slide[i]
        peaks, prominences = find_peaks(x, width=width, prominence=prominence,
                                        rel_height=rel_height, threshold=threshold, distance=distance)
        if mode == 0:
            values = np.ones(len(peaks), dtype='float32')
        elif mode == 1:
            values = prominences
        elif mode == 2:
            values = x[peaks]
        elif mode == 3:
            values = x[peaks] * prominences
        elif mode == 4:
            values = (x[peaks] + prominences) / 2

        if axis == 1:
            skeletonized_slide[peaks, i] = values
        else:
            skeletonized_slide[i, peaks] = values
    return skeletonized_slide

@njit
def find_peaks(x, width=5, prominence=0.05, rel_height=0.5, threshold=0.05, distance=None):
    """ See :meth:`scipy.signal.find_peaks`. """
    lmax = (x[1:] - x[:-1] >= 0)
    rmax = (x[:-1] - x[1:] >= 0)
    mask = np.empty(len(x))
    mask[0] = rmax[0]
    mask[-1] = lmax[-1]
    mask[1:-1] = np.logical_and(lmax[:-1], rmax[1:])
    mask = np.logical_and(mask, x >= threshold)
    peaks = np.where(mask)[0]

    if distance is not None:
        keep = _select_by_peak_distance(peaks, x[peaks], distance)
        peaks = peaks[keep]

    prominences, left_bases, right_bases = _peak_prominences(x, peaks, -1)
    widths = _peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases)
    mask = np.logical_and(widths[0] >= width, prominences >= prominence)
    return peaks[mask], prominences[mask]

@njit
def _peak_prominences(x, peaks, wlen):
    prominences = np.empty(peaks.shape[0], dtype=np.float32)
    left_bases = np.empty(peaks.shape[0], dtype=np.intp)
    right_bases = np.empty(peaks.shape[0], dtype=np.intp)

    for peak_nr in range(peaks.shape[0]):
        peak = peaks[peak_nr]
        i_min = 0
        i_max = x.shape[0] - 1

        if wlen >= 2:
            i_min = max(peak - wlen // 2, i_min)
            i_max = min(peak + wlen // 2, i_max)

        # Find the left base in interval [i_min, peak]
        i = left_bases[peak_nr] = peak
        left_min = x[peak]
        while i_min <= i and x[i] <= x[peak]:
            if x[i] < left_min:
                left_min = x[i]
                left_bases[peak_nr] = i
            i -= 1

        i = right_bases[peak_nr] = peak
        right_min = x[peak]
        while i <= i_max and x[i] <= x[peak]:
            if x[i] < right_min:
                right_min = x[i]
                right_bases[peak_nr] = i
            i += 1

        prominences[peak_nr] = x[peak] - max(left_min, right_min)

    return prominences, left_bases, right_bases

@njit
def _peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases):
    widths = np.empty(peaks.shape[0], dtype=np.float64)
    width_heights = np.empty(peaks.shape[0], dtype=np.float64)
    left_ips = np.empty(peaks.shape[0], dtype=np.float64)
    right_ips = np.empty(peaks.shape[0], dtype=np.float64)

    for p in range(peaks.shape[0]):
        i_min = left_bases[p]
        i_max = right_bases[p]
        peak = peaks[p]
        # Validate bounds and order
        height = width_heights[p] = x[peak] - prominences[p] * rel_height

        # Find intersection point on left side
        i = peak
        while i_min < i and height < x[i]:
            i -= 1
        left_ip = i
        if x[i] < height:
            # Interpolate if true intersection height is between samples
            left_ip += (height - x[i]) / (x[i + 1] - x[i])

        # Find intersection point on right side
        i = peak
        while i < i_max and height < x[i]:
            i += 1
        right_ip = i
        if  x[i] < height:
            # Interpolate if true intersection height is between samples
            right_ip -= (height - x[i]) / (x[i - 1] - x[i])

        widths[p] = right_ip - left_ip
        left_ips[p] = left_ip
        right_ips[p] = right_ip

    return widths, width_heights, left_ips, right_ips

@njit
def _select_by_peak_distance(peaks, priority, distance):
    peaks_size = peaks.shape[0]
    distance_ = np.ceil(distance)
    keep = np.ones(peaks_size, bool_)  # Prepare array of flags
    priority_to_position = np.argsort(priority)

    for i in range(peaks_size - 1, -1, -1):
        j = priority_to_position[i]
        if keep[j] == 0:
            continue

        k = j - 1
        while k >= 0 and peaks[j] - peaks[k] < distance_:
            keep[k] = 0
            k -= 1

        k = j + 1
        while k < peaks_size and peaks[k] - peaks[j] < distance_:
            keep[k] = 0
            k += 1
    return keep  # Return as boolean array

def faults_sizes(labels):
    """ Compute sizes of faults.

    Parameters
    ----------
    labels : numpy.ndarray
        array of shape (N, 4) where the first 3 columns are coordinates of points and the last one
        is for labels
    Returns
    -------
    sizes : numpy.ndarray
    """
    sizes = []
    for array in labels:
        i_len = array[:, 0].ptp()
        x_len = array[:, 1].ptp()
        sizes.append((i_len ** 2 + x_len ** 2) ** 0.5)
    return np.array(sizes)

@njit
def split_array(array, labels):
    """ Split (groupby) array by values from labels. Labels must be sorted and all groups must be contiguous. """
    positions = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            positions.append(i)

    return np.split(array, positions)

@njit
def thin_line(points, column=0):
    """ Make thick line. Works with sorted arrays by the axis of interest. """
    line = np.zeros_like(points)
    p = points[0].copy()
    n = 1
    pos = 0
    for i in range(1, len(points)):
        if points[i, column] == points[i-1, column]:
            p += points[i]
            n += 1
        if i == len(points) - 1:
            line[pos] = p / n
            break
        if (points[i, column] != points[i-1, column]):
            line[pos] = p / n
            n = 1
            pos += 1
            p = points[i].copy()

    return line[:pos+1]

# Bilateral filtering
def bilateral_filter(data, kernel_size=3, kernel=None, padding='same', sigma_spatial=None, sigma_range=0.15):
    """ Apply bilateral filtering for data 3d volume.

    Bilateral filtering is an edge-preserving smoothening, which takes special care for areas on faults edges.
    Be careful with `sigma_range` value:
        - The higher the `sigma_range` value, the more 'bilateral' result looks like a 'convolve' result.
        - If the `sigma_range` too low, then no smoothening applied.

    Parameters
    ----------
    data : np.ndarray

    kernel_size : int or sequence of ints
        Size of a created gaussian filter if `kernel` is None.
    kernel : ndarray or None
        If passed, then ready-to-use kernel. Otherwise, gaussian kernel will be created.
    padding : {'valid', 'same'} or sequence of tuples of ints, optional
        Number of values padded to the edges of each axis.
    sigma_spatial : number
        Standard deviation (spread or “width”) for gaussian kernel.
        The lower, the more weight is put into the point itself.
    sigma_range : number
        Standard deviation for additional weight which smooth differences in depth values.
        The lower, the more weight is put into the depths differences between point in a window.
        Note, if it is too low, then no smoothening is applied.
    """
    if kernel is None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        if sigma_spatial is None:
            sigma_spatial = [size//3 for size in kernel_size]

        kernel = make_gaussian_kernel(kernel_size=kernel_size, sigma=sigma_spatial)

    if padding == 'same':
        padding = [(size//2, size - size//2 - 1) for size in kernel_size]
    elif padding == 'valid':
        padding = None

    if padding is not None:
        data = np.pad(data, padding)

    result = _bilateral_filter(src=data, kernel=kernel, sigma_range=sigma_range)

    if padding is not None:
        slices = tuple(slice(size//2, -(size - size//2 - 1)) for size in kernel_size)
        result = result[slices]
    return result


@njit(parallel=True)
def _bilateral_filter(src, kernel, sigma_range=0.15):
    """ Jit-accelerated function to apply 3d bilateral filtering.

    The difference between gaussian smoothing and bilateral filtering is in additional weight multiplier,
    which is a gaussian of difference of convolved elements.
    """
    #pylint: disable=too-many-nested-blocks, consider-using-enumerate, not-an-iterable
    k = [shape//2 for shape in kernel.shape]
    raveled_kernel = kernel.ravel() / np.sum(kernel)
    sigma_squared = sigma_range**2

    i_range, x_range, z_range = src.shape
    dst = src.copy()

    for iline in prange(0, i_range):
        for xline in range(0, x_range):
            for zline in range(0, z_range):
                central = src[iline, xline, zline]

                # Get values in the squared window and apply kernel to them
                element = src[max(0, iline-k[0]):min(iline+k[0]+1, i_range),
                              max(0, xline-k[1]):min(xline+k[1]+1, x_range),
                              max(0, zline-k[2]):min(zline+k[2]+1, z_range)].ravel()

                s, sum_weights = np.float32(0), np.float32(0)
                for item, weight in zip(element, raveled_kernel):
                    # Apply additional weight for values differences (ranges)
                    weight *= np.exp(-0.5*((item - central)**2)/sigma_squared)

                    s += item * weight
                    sum_weights += weight

                if sum_weights != 0.0:
                    dst[iline, xline, zline] = s / sum_weights
    return dst
