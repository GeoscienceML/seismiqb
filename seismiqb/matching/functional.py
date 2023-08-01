""" Functions used for matching tasks. """
import numpy as np
from numba import njit



def compute_correlation(array_0, array_1):
    """ Correlation coefficient between two signals. Invariant to signal multiplication. """
    return (array_0 * array_1).mean() / (array_0.std() * array_1.std())

def compute_r2(array_0, array_1):
    """ Determination coefficient between two signals. """
    return 1 - (((array_0 - array_1) ** 2).sum() / ((array_0 - array_0.mean()) ** 2).sum())


def modify_trace(trace, shift=0, angle=0, gain=1):
    """ Add a z-shift, phase shift and a gain multiplier to a trace.

    # TODO: can be optimized by a lot by passing (optional) `fft` of a trace.

    Parameters
    ----------
    trace : np.ndarray
        Signal to modify.
    shift : number
        Vertical shift to apply, measured in samples. Can be floating number: the trace resampled under the hood.
    angle : number
        Phase shift in degrees. Under the hood, the trace is FFT-shifted.
    gain : number
        Multiplier for trace values.
    """
    # Phase shift
    if abs(angle) >= 1:
        fft = np.fft.rfft(trace)
        fft *= np.exp(1.0j * np.deg2rad(angle))
        trace = np.fft.irfft(fft, n=len(trace)).astype(np.float32)

    # Depth shift
    arange = np.arange(len(trace), dtype=np.float32)
    trace = np.interp(arange, arange+shift, trace, left=0, right=0)

    # Gain: multiplier
    trace = trace * gain
    return trace


def minimize_proxy(x, trace_0, trace_1, metric='correlation'):
    """ Proxy function for computing loss for the task of matching two traces with supplied metric. """
    # x is a 3-element array of (shift, angle, gain)
    shift, angle, gain = x
    _ = angle
    modified_trace_1 = modify_trace(trace_1, shift=shift, angle=angle, gain=gain)

    if metric == 'correlation':
        loss = -compute_correlation(trace_0, modified_trace_1)
    elif metric == 'r2':
        loss = -compute_r2(trace_0, modified_trace_1)
    return loss


@njit
def compute_shifted_traces(trace, shifts):
    """ Make an array with shifted `trace` for each `shift`. """
    buffer = np.empty((shifts.size, trace.size), dtype=np.float32)
    arange = np.arange(len(trace), dtype=np.float32)

    for i, shift in enumerate(shifts):
        buffer[i] = np.interp(arange, arange + shift, trace)
    return buffer
