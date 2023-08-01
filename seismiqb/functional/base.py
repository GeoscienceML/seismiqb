""" Useful functions. """
import numpy as np
from scipy.ndimage import gaussian_filter


def make_gaussian_kernel(kernel_size=(3, 3), sigma=1.):
    """ Create Gaussian kernel with given parameters: kernel size and std. """
    n = np.zeros(kernel_size)
    n[tuple(np.array(n.shape) // 2)] = 1
    return gaussian_filter(n, sigma=sigma)
