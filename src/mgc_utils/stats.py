"""
A set of functions performing various statistical tasks. Note that some of these are redefinitions from numpy or scipy,
which have been edited to add in some features (most commonly, adding an `axes` parameter).
"""
__all__ = [
    "autocorrelate",
    "convolve",
    "correlate",
    "cov",
    "nancov",
]

import numpy as np
from scipy.signal import fftconvolve, _sigtools, choose_conv_method
import warnings

from ._numpy_redefinitions import _inputs_swap_needed, _np_conv_ok, _reverse_and_conj


def autocorrelate(x, mode="same", method="auto", axes=None):
    """
    Autocorrelates by passing in `x` as both input arguments to `correlate`,
    and normalises the result by 1/N (not included in scipy's implementation).
    """
    # Do autocorrelation: c_{av}[k] = sum_n x[n+k] * conj(x[n])
    acf = correlate(x, x, mode, method, axes)

    # Normalise: divide each term by n
    if axes is None:
        axes = tuple(range(x.ndim))
    else:
        axes = tuple(axes)

    N = np.ones(x.ndim, dtype=int)
    for ax in axes:
        N[ax] = acf.shape[ax]

    div = np.ones(N)
    if mode == "valid":
        for n in N:
            div *= n
    elif mode in {"same", "full"}:
        for ax, dim in enumerate(N):
            shape = np.ones(len(N), dtype=int)
            shape[ax] = dim
            d = (dim - np.abs(np.asarray(range(dim)) - int(dim / 2))).reshape(shape)
            div *= d

    return acf / div


def convolve(in1, in2, mode="full", method="auto", axes=None):
    """
    Copy of scipy's `convolve` to work with the additional `axes` parameter.
    """

    volume = np.asarray(in1)
    kernel = np.asarray(in2)

    if volume.ndim == kernel.ndim == 0:
        return volume * kernel
    elif volume.ndim != kernel.ndim:
        raise ValueError("volume and kernel should have the same " "dimensionality")

    if _inputs_swap_needed(mode, volume.shape, kernel.shape):
        # Convolution is commutative; order doesn't have any effect on output
        volume, kernel = kernel, volume

    if method == "auto":
        method = choose_conv_method(volume, kernel, mode=mode)

    if method == "fft":
        out = fftconvolve(volume, kernel, mode=mode, axes=axes)
        result_type = np.result_type(volume, kernel)
        if result_type.kind in {"u", "i"}:
            out = np.around(out)

        if np.isnan(out.flat[0]) or np.isinf(out.flat[0]):
            warnings.warn(
                "Use of fft convolution on input with NAN or inf"
                " results in NAN or inf output. Consider using"
                " method='direct' instead.",
                category=RuntimeWarning,
                stacklevel=2,
            )

        return out.astype(result_type)
    elif method == "direct":
        # fastpath to faster numpy.convolve for 1d inputs when possible
        if _np_conv_ok(volume, kernel, mode):
            return np.convolve(volume, kernel, mode)

        return correlate(volume, _reverse_and_conj(kernel), mode, "direct")
    else:
        raise ValueError("Acceptable method flags are 'auto'," " 'direct', or 'fft'.")


_modedict = {"same": 1, "full": 2}


def correlate(in1, in2, mode="same", method="auto", axes=None):
    """
    Copy of scipy.signal's fftconvolve, with an additional axes parameter.
    """
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if in1.ndim == in2.ndim == 0:
        return in1 * in2.conj()
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")

    # Don't use _valfrommode, since correlate should not accept numeric modes
    try:
        val = _modedict[mode]
    except KeyError as e:
        raise ValueError("Acceptable mode flags are" " 'same', or 'full'.") from e

    # this either calls fftconvolve or this function with method=='direct'
    if method in ("fft", "auto"):
        return convolve(in1, _reverse_and_conj(in2, axes=axes), mode, method, axes)

    elif method == "direct":
        # fastpath to faster numpy.correlate for 1d inputs when possible
        if _np_conv_ok(in1, in2, mode):
            return np.correlate(in1, in2, mode)

        # _correlateND is far slower when in2.size > in1.size, so swap them
        # and then undo the effect afterward if mode == 'full'.  Also, it fails
        # with 'valid' mode if in2 is larger than in1, so swap those, too.
        # Don't swap inputs for 'same' mode, since shape of in1 matters.
        swapped_inputs = (
            (mode == "full")
            and (in2.size > in1.size)
            or _inputs_swap_needed(mode, in1.shape, in2.shape, axes=axes)
        )

        if swapped_inputs:
            in1, in2 = in2, in1

        if mode == "valid":
            ps = [i - j + 1 for i, j in zip(in1.shape, in2.shape)]
            out = np.empty(ps, in1.dtype)

            z = _sigtools._correlateND(in1, in2, out, val)

        else:
            ps = [i + j - 1 for i, j in zip(in1.shape, in2.shape)]

            # zero pad input
            in1zpadded = np.zeros(ps, in1.dtype)
            sc = tuple(slice(0, i) for i in in1.shape)
            in1zpadded[sc] = in1.copy()

            if mode == "full":
                out = np.empty(ps, in1.dtype)
            elif mode == "same":
                out = np.empty(in1.shape, in1.dtype)

            z = _sigtools._correlateND(in1zpadded, in2, out, val)

        if swapped_inputs:
            # Reverse and conjugate to undo the effect of swapping inputs
            z = _reverse_and_conj(z, axes=axes)

        return z

    else:
        raise ValueError("Acceptable method flags are 'auto'," " 'direct', or 'fft'.")


def cov(m, varaxis=0, obsaxis=1, dtype=None, pseudo=False):
    """
    Alternative implementation of numpy's cov function, which may compute the
    pseudo-covariance matrix (in which Cov[X, X*] is computed instead of
    Cov[X, X]). As I don't really care about weights and things either I removed
    those - would need adding back if I ever want this in numpy.
    """
    # Handles complex arrays too
    X = np.array(m, ndmin=2, dtype=dtype)

    if X.shape[0] == 0:
        return np.array([]).reshape(0, 0)

    if varaxis != int(varaxis):
        raise ValueError("varaxis must be integer")
    if varaxis > m.ndim or varaxis < -X.ndim:
        raise ValueError("varaxis must correspond to an axis in m")
    varaxis = varaxis % X.ndim

    if obsaxis != int(obsaxis):
        raise ValueError("obsaxis must be integer")
    if obsaxis > m.ndim or obsaxis < -X.ndim:
        raise ValueError("obsaxis must correspond to an axis in m")
    obsaxis = obsaxis % X.ndim

    if obsaxis == varaxis:
        raise ValueError("obsaxis and varaxis cannot be equal")

    avg, w_sum = np.average(X, axis=obsaxis, returned=True)
    w_sum = w_sum[0]

    # Determine the normalization
    fact = X.shape[obsaxis]

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
        fact = 0.0

    X -= np.expand_dims(avg, obsaxis)
    if not pseudo:
        X_c = X.conj()
    else:
        X_c = X

    # Construct einsum signatures. As I want to preserve multiple outside axes,
    #   prefer this formulation to numpy's default `dot`.
    # Say we have a fairly typical example where X has 2 features and 10000
    #   observations (i.e. shape (2, 10000)). Within `np.cov(X)` this boils
    #   down to `np.dot(X, X.conj().T)`. This is equivalent to
    #   `np.einsum('ik,jk->ij', X, X.conj())`. The following code constructs
    #   this signature for X of arbitrary shape, when variables and
    #   observations are in the defined axes.
    input_signature_1 = ""
    input_signature_2 = ""
    output_signature = ""
    t = 0
    for i in range(X.ndim):
        if i not in [varaxis, obsaxis]:
            # Start the repeat axes from "l". Unlikely that we will have so many
            # axes that we will go beyond "z" - could change this if needed.
            input_signature_1 += chr(108 + t)
            input_signature_2 += chr(108 + t)
            output_signature += chr(108 + t)
            t += 1
        elif i == varaxis:
            # Reserve "i" and "j" for variable axes
            input_signature_1 += "i"
            input_signature_2 += "j"
            output_signature += "ij"
        else:
            # Reserve "k" for observation axis.
            input_signature_1 += "k"
            input_signature_2 += "k"

    c = np.einsum(
        f"{input_signature_1},{input_signature_2}->{output_signature}", X, X_c
    )
    c *= np.true_divide(1, fact)
    return c.squeeze()


def nancov(x, y=None):
    """
    A method to compute covariance which does not take nan values into account.
    This is not guaranteed to be any better or worse than the regular cov
    function - it will be about as good as numpy's nanmean function.

    Parameters
    ----------
    x : ndarray (M, N)
        M-feature array of N observations. If 1D, it is assumed that the array
        contains multiple observations of a single feature. May contain nan
        values.
    y : ndarray (O, P), optional
        Optional additional O-feature array of P observations. May not be the
        same size as `x`, but if not included then autocovariance of `x` is
        computed. The default is None.

    Returns
    -------
    cov : ndarray (M, O)
        Covariance matrix between `x` and `y`.

    """
    if y is None:
        y = x
    if x.ndim <= 1:
        x = x.reshape(1, -1)
    if y.ndim <= 1:
        y = y.reshape(1, -1)

    return np.nanmean(
        (
            (x - np.nanmean(x, axis=1).reshape(-1, 1)).reshape(
                x.shape[0], 1, x.shape[1]
            )
            * (y - np.nanmean(y, axis=1).reshape(-1, 1))
            .reshape(1, y.shape[0], y.shape[1])
            .conj()
        ),
        axis=-1,
    ).squeeze()
