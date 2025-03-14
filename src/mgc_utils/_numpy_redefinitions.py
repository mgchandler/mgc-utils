# -*- coding: utf-8 -*-
"""
A set of functions which make slight adaptations to some of those found in
`numpy`, to make life a bit easier. The useful functions are listed below, the
helper functions are not. 

nancov :        Compute the covariance, ignoring NaN values. Takes inputs and
                returns outputs in the same way as `np.cov()`.
cov :           Entirely identical to `np.cov`, apart from the fact that it
                takes `pseudo` as a keyword argument. Boolean value expected,
                if `True` then compute Cov[X, X*], if `False` then compute 
                Cov[X, X].
correlate :     Identical to `np.correlate` but also takes `axes` as a keyword
                argument, the axes to perform correlation over.
convolve :      Identical to `np.convolve`, but also takes `axes` as a keyword
                argument.
autocorrelate : Take a single input and correlate, but also do the 1/N
                normalisation which numpy neglects.
                
"""
__all__ = [
    '_inputs_swap_needed', '_np_conv_ok', '_reverse_and_conj',
]

import numpy as np


def _inputs_swap_needed(mode, shape1, shape2, axes=None):
    """
    Helper function from scipy.signal._signaltools
    """
    if mode != "valid":
        return False

    if not shape1:
        return False

    if axes is None:
        axes = range(len(shape1))

    ok1 = all(shape1[i] >= shape2[i] for i in axes)
    ok2 = all(shape2[i] >= shape1[i] for i in axes)

    if not (ok1 or ok2):
        raise ValueError(
            "For 'valid' mode, one must be at least "
            "as large as the other in every dimension"
        )

    return not ok1


def _np_conv_ok(volume, kernel, mode):
    """
    Helper function from scipy.signal._signaltools
    """
    if volume.ndim == kernel.ndim == 1:
        if mode in ("full", "valid"):
            return True
        elif mode == "same":
            return volume.size >= kernel.size
    else:
        return False


def _reverse_and_conj(x, axes=None):
    """
    Helper function from scipy.signal._signaltools
    """
    if axes is None:
        reverse = (slice(None, None, -1),) * x.ndim
    else:
        slices = [
            [
                slice(None),
            ]
            * x.ndim,
            [
                slice(None, None, -1),
            ]
            * x.ndim,
        ]
        reverse = tuple(slices[ax in np.mod(axes, x.ndim)][ax] for ax in range(x.ndim))
    return x[reverse].conj()