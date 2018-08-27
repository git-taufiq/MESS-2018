from scipy import fftpack
from numpy import (asarray, iscomplexobj, newaxis, zeros)
import math

def hilbert(x, N=None, axis=0):
    """
    Compute the analytic signal, using the Hilbert transform.

    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.

    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`

    Notes
    -----
    The analytic signal ``x_a(t)`` of signal ``x(t)`` is:

    .. math:: x_a = F^{-1}(F(x) 2U) = x + i y

    where `F` is the Fourier transform, `U` the unit step function,
    and `y` the Hilbert transform of `x`.

    In other words, the negative half of the frequency spectrum is zeroed
    out, turning the real-valued signal into a complex signal.  The Hilbert
    transformed signal can be obtained from ``np.imag(hilbert(x))``, and the
    original signal from ``np.real(hilbert(x))``.

    """
    x = asarray(x)
    if iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = fftpack.fft(x, N, axis=axis)
    h = zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[ind]
    x = fftpack.ifft(Xf * h, axis=axis)
    return x