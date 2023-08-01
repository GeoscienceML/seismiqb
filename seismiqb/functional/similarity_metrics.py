""" Metrics for denoising seismic data. """
import numpy as np
from scipy import fftpack

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torchmetrics.functional import structural_similarity_index_measure,\
                                        peak_signal_noise_ratio, mean_squared_error,\
                                        error_relative_global_dimensionless_synthesis, universal_image_quality_index
    METRICS = {
        'ssim': structural_similarity_index_measure,
        'psnr': peak_signal_noise_ratio,
        'ergas': error_relative_global_dimensionless_synthesis,
        'uqi': universal_image_quality_index,
        'mse': mean_squared_error
    }
except ImportError:
    METRICS = {}





def compute_similarity_metric(array1, array2, metrics='all'):
    """ Compute similarity metrics for a pair of images.

    Parameters
    ----------
    array1, array2 : np.ndarray or torch.Tensor
        Source images to evaluate. Works with (B, ...) arrays as well.
    metrics : dict, list or str
        Specifies functions to compute and their parameters.
        Available names are {'ssim', 'psnr', 'ergas', 'uqi', 'mse'}.

        If dict, then should contain metric names as keys and their parameters as values.
        If list, then consists of metric names, and they are evaluated with default parameters.
        If 'all', then evaluate all metrics with default parameters.

    Returns
    -------
    dict
        Dictionary with metric names as keys and computed metrics as values.
    """
    #pylint: disable=not-a-mapping
    if not TORCH_AVAILABLE:
        raise ImportError('Install `torch` library!')
    if not METRICS:
        raise ImportError('Install `torchmetrics` library!')

    # Parse `metrics`
    if metrics == 'all':
        metrics =  list(METRICS.keys())
    if isinstance(metrics, str):
        metrics = [metrics]
    if isinstance(metrics, (tuple, list)):
        metrics = dict.fromkeys(metrics, {})

    for metric_name in metrics.keys():
        if metric_name not in METRICS:
            raise ValueError(f'Incorrect metric name `{metric_name}`!')

    # Convert arrays to tensors. TODO: do that only when the metric requires
    array1, array2 = torch.Tensor(array1), torch.Tensor(array2)

    returns = {}
    for metric_name, metric_kwargs in metrics.items():
        result = METRICS[metric_name](array1, array2, **metric_kwargs)
        returns[metric_name] = result.item() if isinstance(result, torch.Tensor) else result
    return returns



def local_correlation_map(image, prediction, map_to='pred', window_size=9, n_dims=1):
    """ Local correlation map between an image and estimated noise.

    Parameters
    ----------
    window_size : int
        if `n_dims` is 1, correlation is measured between corresponding parts of traces of `window_size` size.
        if `n_dims` is 2, correlation is measured between flattened windows of size (`window_size`, `window_size`).
    n_dims : int
        Number of dimensions for `window_size`.

    Returns
    -------
    np.ndarray
        Array of the same shape.
    """
    image = image.squeeze()
    prediction = prediction.squeeze()
    image_noise = np.abs(image - prediction)
    image = image if map_to == 'image' else prediction
    img_shape = image.shape

    # "same" padding along trace for 1d window or both dims for 2d
    pad = window_size // 2
    pad_width = [[pad, window_size - (1 + pad)], [pad * (n_dims - 1), (window_size - (1 + pad)) * (n_dims - 1)]]

    image = np.pad(image, pad_width=pad_width, mode='mean')
    image_noise = np.pad(image_noise, pad_width=pad_width, mode='mean')

    # Vectorization
    window_shape=[window_size, window_size if n_dims == 2 else 1]
    image_view = np.lib.stride_tricks.sliding_window_view(image, window_shape=window_shape)
    image_noise_view = np.lib.stride_tricks.sliding_window_view(image_noise, window_shape=window_shape)

    straighten = (np.dot(*image_view.shape[:2]), np.dot(*image_view.shape[2:]))
    image_view = image_view.reshape(straighten)
    image_noise_view = image_noise_view.reshape(straighten)

    pearson = _pearson_corr_2d(image_view, image_noise_view).reshape(img_shape)
    return np.nan_to_num(pearson)

def _pearson_corr_2d(x, y):
    """ Squared Pearson correlation coefficient between corresponding rows of 2d input arrays. """
    x_centered = x - x.mean(axis=1, keepdims=True).reshape(-1, 1)
    y_centered = y - y.mean(axis=1, keepdims=True).reshape(-1, 1)
    corr = (x_centered * y_centered).sum(axis=1)
    corr /= np.sqrt((x_centered**2).sum(axis=1) * (y_centered**2).sum(axis=1))
    return corr ** 2


def local_similarity_map(image, prediction, map_to='pred', lamb=0.5, window_size=9, n_dims=1, **kwargs):
    """ Local Similarity Map between an image and estimated noise.
    Chen, Yangkang, and Sergey Fomel. "`Random noise attenuation using local signal-and-noise orthogonalization
    <https://library.seg.org/doi/10.1190/geo2014-0227.1>`_"

    Parameters
    ----------
    lamb : float
        Regularization parameter from 0 to 1.
    window_size : int
        Size of the window for a local similarity estimation.
    n_dims : int
        Number of dimensions for `window_size`.
    tol : float, optional
        Tolerance for `shaping_conjugate_gradient`.
    N : int, optional
        Maximum number of iterations for `shaping_conjugate_gradient`.

    Returns
    -------
    np.ndarray
        Array of the same shape.
    """
    image = image.squeeze()
    prediction = prediction.squeeze()
    image_noise = np.abs(image - prediction)
    image = image if map_to == 'image' else prediction
    img_shape = image.shape

    pad = window_size // 2
    pad_width = [[pad, window_size - (1 + pad)], [pad * (n_dims - 1), (window_size - (1 + pad)) * (n_dims - 1)]]

    image = np.pad(image, pad_width=pad_width, mode='mean')
    image_noise = np.pad(image_noise, pad_width=pad_width, mode='mean')

    window_shape=[window_size, window_size if n_dims == 2 else 1]
    image_view = np.lib.stride_tricks.sliding_window_view(image, window_shape=window_shape)
    image_noise_view = np.lib.stride_tricks.sliding_window_view(image_noise, window_shape=window_shape)

    straighten = (np.dot(*image_view.shape[:2]), np.dot(*image_view.shape[2:]))
    image_view = image_view.reshape(straighten)
    image_noise_view = image_noise_view.reshape(straighten)

    H = np.eye(window_size**n_dims, dtype=np.float) * lamb
    H = np.lib.stride_tricks.as_strided(H, shape=(image_view.shape[0], window_size**n_dims, window_size**n_dims),
                                        strides=(0, 8 * window_size**n_dims, 8))

    sim_local = _local_similarity(a=image_view, b=image_noise_view, H=H, **kwargs)
    return sim_local.reshape(img_shape)

def _local_similarity(a, b, H, *args, **kwargs):
    """ Local Similarity between an image and estimated noise. """
    A = np.array([np.diag(a[i]) for i in range(len(a))])
    B = np.array([np.diag(b[i]) for i in range(len(b))])
    c1 = _shaping_conjugate_gradient(L=A, H=H, d=b, *args, **kwargs)
    c2 = _shaping_conjugate_gradient(L=B, H=H, d=a, *args, **kwargs)
    return np.sum(c1 * c2, axis=1)

def _shaping_conjugate_gradient(L, H, d, tol=1e-5, N=20):
    """ Vectorized Shaping Conjugate gradient Algorithm for a system with smoothing operator.
    Fomel, Sergey. "`Shaping regularization in geophysical-estimation problems
    <https://library.seg.org/doi/10.1190/1.2433716>`_".
    Variables and parameters are preserved as in the paper.
    """
    p = np.zeros_like(d)
    m = np.zeros_like(d)
    r = -d
    sp = np.zeros_like(d)
    sm = np.zeros_like(d)
    sr = np.zeros_like(d)
    EPS = 1e-5
    for i in range(N):
        gm = (np.transpose(L, axes=[0, 2, 1]) @ r[..., np.newaxis]).squeeze() - m
        gp = (np.transpose(H, axes=[0, 2, 1]) @ gm[..., np.newaxis]).squeeze() + p
        gm = H @ gp[..., np.newaxis]
        gr = L @ gm

        rho = np.sum(gp ** 2, axis=1)
        if i == 0:
            beta = np.zeros((L.shape[0], 1))
            rho0 = rho
        else:
            beta = (rho / (rho_hat + EPS))[..., np.newaxis]
            if np.all(beta < tol) or np.all(rho / (rho0 + EPS) < tol):
                return m

        sp = gp + beta * sp
        sm = gm.squeeze() + beta * sm
        sr = gr.squeeze() + beta * sr

        alpha = rho / (np.sum(sr ** 2, axis=1) + np.sum(sp ** 2, axis=1) - np.sum(sm ** 2, axis=1) + EPS)
        alpha = alpha[..., np.newaxis]

        p -= alpha * sp
        m -= alpha * sm
        r -= alpha * sr
        rho_hat = rho
    return m



def fourier_power_spectrum(image, prediction, fourier_map='pred', map_to=None, **kwargs):
    """ Fourier Power Spectrum for an image.

    Parameters
    ----------
    fourier_map : str
        If 'image', computes power spectrum for `image`.
        If 'pred', computes power spectrum for `prediction`.

    Returns
    -------
    np.ndarray
        Array of the same shape.
    """
    image = image if fourier_map == 'image' else prediction
    image = image.squeeze()
    img_fft = fftpack.fft2(image, **kwargs)
    shift_fft = fftpack.fftshift(img_fft)
    spectrum = np.abs(shift_fft)**2
    return np.log10(spectrum).squeeze()
