"""
A set of functions performing various statistical tasks. Note that some of these are redefinitions from numpy or scipy,
which have been edited to add in some features (most commonly, adding an `axes` parameter).
"""
__all__ = [
    'auc', 'autocorrelate', 'convolve', 'correlate', 'cov', 'nancov', 'roc',
]

import numpy as np
from scipy.signal import fftconvolve, _sigtools, choose_conv_method
import warnings

from ._numpy_redefinitions import _inputs_swap_needed, _np_conv_ok, _reverse_and_conj
from .vis import opheim_simpl


def auc(x, y):
    """
    Compute the area under the curve of a given set of `x` and `y` values. This
    is designed to work with the result of the `roc` function, thus it is
    expected that the function is monotonic (increasing or decreasing).

    Parameters
    ----------
    x : ndarray (N,)
        Input x-coordinate values.
    y : ndarray (N,)
        Input y-coordinate values.

    Returns
    -------
    area : float
        Area under the curve described by `x` and `y`.

    """
    dx = np.diff(x)
    direction = 1
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError('x is neither increasing nor decreasing : {}.'.format(x))
    return direction * np.trapz(y, x)


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


def cov(
        m,
        varaxis=0,
        obsaxis=1,
        dtype=None,
        pseudo=False
):
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
    if varaxis > m.ndim or varaxis < - X.ndim:
        raise ValueError("varaxis must correspond to an axis in m")
    varaxis = varaxis % X.ndim

    if obsaxis != int(obsaxis):
        raise ValueError("obsaxis must be integer")
    if obsaxis > m.ndim or obsaxis < - X.ndim:
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
    input_signature_1 = ''
    input_signature_2 = ''
    output_signature = ''
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
            input_signature_1 += 'i'
            input_signature_2 += 'j'
            output_signature += 'ij'
        else:
            # Reserve "k" for observation axis.
            input_signature_1 += 'k'
            input_signature_2 += 'k'

    c = np.einsum(
        f'{input_signature_1},{input_signature_2}->{output_signature}',
        X, X_c
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


def roc(pos, neg, reduce_size=False):
    """
    Compute receiver-operating characteristic curve from known positive and
    negative distributions.

    This works by sorting `pos` and `neg` into ascending order and take the
    union:
    ```
    x, idxs = np.unique(np.vstack([pos, neg]), return_index=True)
    ```
    where `idxs` points to the location in the composite array that each
    element in `x` originally came from. As such, the part of `idxs` which
    comes from `pos` will be monotonically increasing as `pos` is ascending;
    and the part from `neg` will be also, with these two increasing arrays
    spliced together in some way. Flip the order so that `x` and `idxs` are
    decreasing.

    Considering a position `loc` in both `x` and `idxs` which originally comes
    from `pos`, then all of the values in the slice `pos[:idxs[loc]]` must be
    less than `x[loc]` because `pos` is ascending. Therefore `idxs[loc]` is
    equal to the number of elements in `pos` which are less than `x[loc]`.
    This is exactly the false -ve rate when divided by the total number of
    elements in `pos`, so subtract from 1 to get true +ve.

    Suppose that `loc+1` in `x` originally comes from `neg`. As `x` is
    descending, then `neg[idxs[loc+1]] < pos[idxs[loc]]`, thus the number of
    values in `pos` less than `x[loc+1]` is the same as the number of values
    less than `x[loc]`.

    Parameters
    ----------
    pos : ndarray (N1,)
        Observations made when the null hypothesis is true. Multiple features
        is not currently supported.
    neg : ndarray (N2,)
        Observations made when the null hypothesis is false. Multiple features
        is not currently supported.
    reduce_size : bool
        Arrays of thresholds, true positives and false positives can be huge.
        Set this to `True` to reduce the overall size of the array using Opheim
        simplification, with a tolerance of `1e-5`, i.e. steps should now be
        ~1e-5 between points.

    Returns
    -------
    x : ndarray (M,)
        The list of thresholds which are used, equivalent to the ascending
        values in the union of `pos` and `neg`.
    tps : ndarray (M,)
        True positive rate, or the percentage of weighted values in `pos` which
        exceed the value in `x`.
    fps : ndarray (M,)
        False positive rate, or the percentage of weighted values in `neg`
        which exceed `x`.

    """
    # Check data
    pos, neg = np.squeeze(pos), np.squeeze(neg)
    if pos.ndim < 2:
        pos = pos.reshape(-1, 1)
    elif pos.ndim > 2:
        raise ValueError(
            'pos expected to have 2 dimensions, found {}.'.format(pos.ndim)
        )
    elif pos.shape[1] != 1:
        warnings.warn(
            'pos should only have one feature, currently has {}.'.format(pos.shape[0])
        )
    if neg.ndim < 2:
        neg = neg.reshape(-1, 1)
    elif neg.ndim > 2:
        raise ValueError(
            'neg expected to have 2 dimensions, found {}.'.format(neg.ndim)
        )
    elif neg.shape[1] != 1:
        warnings.warn(
            'neg should only have one feature, currently has {}.'.format(neg.shape[0])
        )

    # # Check weights - leave out for now as working with weights will require
    # # some reformulation.
    # if pos_wts is None:
    #     pos_wts = np.ones(pos.shape[1])
    # else:
    #     pos_wts = np.squeeze(pos_wts)
    #     if pos_wts.ndim > 1 or pos_wts.shape[0] != pos.shape[1]:
    #         raise ValueError(
    #             'pos_wts expected to have shape equal to number of features in pos, found shape {}'
    #             .format(pos_wts.shape)
    #         )
    # if neg_wts is None:
    #     neg_wts = np.ones(neg.shape[1])
    # else:
    #     neg_wts = np.squeeze(neg_wts)
    #     if neg_wts.ndim > 1 or neg_wts.shape[0] != neg.shape[1]:
    #         raise ValueError(
    #             'neg_wts expected to have shape equal to number of features in neg, found shape {}'
    #             .format(neg_wts.shape)
    #         )

    # Sort into ascending order.
    pos, neg = np.sort(pos, axis=0)[::-1, :], np.sort(neg, axis=0)[::-1, :]
    # Sort unique values into descending order.
    x, idxs = np.unique(np.vstack([pos, neg]), return_index=True)
    # x, idxs = x[::-1], idxs[::-1]
    tps, fps = idxs.copy(), idxs.copy()

    for loc, idx in enumerate(idxs):
        # Find the parts of `tps` which come from `neg` and set them to the
        # most recently seen value in `pos`.
        if idx >= pos.size:
            # At the start, all values in the array must be < x[0].
            if loc == 0:
                tps[loc] = pos.size
            # At any other time, we must have either just seen a value from
            # `pos` on the previous iteration, or set the value on the previous
            # iteration. Either way, `loc-1` must contain a value from `pos`.
            else:
                tps[loc] = tps[loc - 1]
        # Find the parts of `fps` from `pos` and set them to the most recently
        # seen value in `neg`.
        else:
            if loc == 0:
                fps[loc] = pos.size + neg.size
            else:
                fps[loc] = fps[loc - 1]
    # `fps` must now point to the part of `np.vstack([pos, neg])` which comes
    # from `neg`, so correct for the presence of `pos`.
    fps -= pos.size

    # `tps` and `fps` contain the number of elements in `pos` and `neg` which
    # are less than a given threshold in `x`. To get true +ve and -ve rates,
    # want the number of elements larger than the threshold in `x`.
    tps, fps = tps / pos.size, fps / neg.size

    # Ensure that `tps` and `fps` both start at 1 and end at 0.
    tps, fps = np.hstack([1, tps, 0]), np.hstack([1, fps, 0])
    x = np.hstack([x[0], x, x[-1]])

    if reduce_size and x.shape[0] > 10000:
        try:
            mask = opheim_simpl(fps, tps, 1e-5)
        except Exception:
            try:
                mask = opheim_simpl(fps, tps, 2e-5)
            except Exception:
                mask = np.full(x.shape, True)
        x, tps, fps = x[mask], tps[mask], fps[mask]

    return x, tps, fps