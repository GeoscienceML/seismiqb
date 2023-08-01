""" Functions for geologic transforms. """
from warnings import warn

import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False



# Device management
def to_device(array, device='cpu'):
    """ Transfer array to chosen GPU, if possible.
    If `cupy` is not installed, does nothing.

    Parameters
    ----------
    device : str or int
        Device specificator. Can be either string (`cpu`, `gpu:4`) or integer (`4`).
    """
    if isinstance(device, str) and ':' in device:
        device = int(device.split(':')[1])
    if device in ['cuda', 'gpu']:
        device = 0

    if isinstance(device, int):
        if CUPY_AVAILABLE:
            with cp.cuda.Device(device):
                array = cp.asarray(array)
        else:
            warn('Performance Warning: computing metrics on CPU as `cupy` is not available', RuntimeWarning)
    return array

def from_device(array):
    """ Move the data from GPU, if needed.
    If `cupy` is not installed or supplied array already resides on CPU, does nothing.
    """
    if CUPY_AVAILABLE and hasattr(array, 'device'):
        array = cp.asnumpy(array)
    return array


# Helper functions
def hilbert(array, axis=-1):
    """ Compute the analytic signal, using the Hilbert transform. """
    xp = cp.get_array_module(array) if CUPY_AVAILABLE else np
    N = array.shape[axis]
    fft = xp.fft.fft(array, n=N, axis=axis)

    h = xp.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if array.ndim > 1:
        ind = [xp.newaxis] * array.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]

    result = xp.fft.ifft(fft * h, axis=axis)
    return result

def compute_instantaneous_amplitude(array, axis=-1, analytic=None):
    """ Compute instantaneous amplitude. """
    xp = cp.get_array_module(array) if CUPY_AVAILABLE else np
    analytic = analytic if analytic is not None else hilbert(array, axis=axis)
    amplitude = xp.abs(analytic)
    return amplitude.astype(np.float32)

def compute_instantaneous_phase(array, continuous=False, axis=-1, analytic=None):
    """ Compute instantaneous phase. """
    xp = cp.get_array_module(array) if CUPY_AVAILABLE else np
    analytic = analytic if analytic is not None else hilbert(array, axis=axis)

    phase = xp.angle(analytic) % (2 * xp.pi) - xp.pi
    if continuous:
        phase = xp.abs(phase)
    return phase.astype(np.float32)

def compute_instantaneous_frequency(array, axis=-1, sample_rate=1.0, analytic=None):
    """ Compute instantaneous frequency. """
    iphases = compute_instantaneous_phase(array, axis=axis, analytic=analytic)
    frequency = np.diff(iphases, axis=axis, prepend=0) / (2 * np.pi) * sample_rate
    return frequency.astype(np.float32)

def compute_spectral_decomposition(array, frequencies, wavelet='mexh', sample_rate=1.0, method='fft', axis=-1):
    """ Compute spectral decomposition by convolving data with wavelets at different scales. """
    import pywt #pylint: disable=import-outside-toplevel
    frequencies = np.array(frequencies)
    scales = sample_rate / (frequencies * np.sqrt(2) * np.pi)
    spectral =  pywt.cwt(array, scales=scales, wavelet=wavelet, axis=axis, method=method)[0]
    return spectral
