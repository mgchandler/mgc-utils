# -*- coding: utf-8 -*-
"""
A set of functions to assist with hypothesis testing described in [1].

A set of additional functions to assist with the auto-correlation and
hypothesis testing method described in [1]. Functions from `np_redef` are
re-exported here to try and avoid any confusion with `numpy` in scripts which
use both this module and `np_redef`.

References
----------
[1] - new paper citation.
[2] - R. DE MAESSCHALCK, et al., The Mahalanobis distance, Chemometrics and
        Intelligent Laboratory Systems, Volume 50, Issue 1, p. 1-18, Jan 2000,
        doi:10.1016/S0169-7439(99)00047-7
[3] - K. V. MARDIA, Measures of multivariate skewness and kurtosis with
        applications, Biometrika, Volume 57, Issue 3, p. 519–530, Dec 1970,
        doi:10.1093/biomet/57.3.519
[4] - L. J. NELSON, et al., Ply-orientation measurements in composites using
        structure-tensor analysis of volumetric ultrasonic data, Composites A,
        Volume 104, p. 108-119, Jan 2018, doi:10.1016/j.compositesa.2017.10.027
        
"""
__all__ = [
    'bounding_ellipse', 'mahalanobis',  'mardia_test', 'transform_observations',
]

import numpy as np
from numpy.linalg import cholesky, inv, solve
from scipy.stats import chi2, norm

from .stats import cov


def bounding_ellipse(mean, covariance, sigma=1, n=100):
    """
    Returns the ellipse for which all values are `sigma`-far away from the mean
    of the normal distribution with provided mean and covariance. All points on
    this ellipse will have Mahalanobis distance of `sigma`:
    ```
    >>> ellipse = bounding_ellipse(mean, covariance)
    >>> dist = mahalanobis(ellipse, mean, covariance)
    >>> print(np.sqrt(dist))
    [1. 1. 1. ... 1. 1. 1.]
    ```

    Parameters
    ----------
    mean : ndarray[float], (2,)
        Mean vector of the normal distribution.
    covariance : ndarray[float], (2, 2)
        Covariance matrix of the normal distribution.
    sigma : float, optional
        Required distance from the normal distribution. The default is 1.
    n : int, optional
        Number of points on the ellipse. The default is 100.

    Returns
    -------
    ndarray[float], (N, 2)
        Set of points on the ellipse.
        
    Notes
    -----
    This is a function which is intended to aid plotting, therefore it is
    limited to only accept distributions with two features.

    """
    mean, covariance = np.squeeze(mean), np.squeeze(covariance)
    if mean.shape != (2,):
        raise ValueError('`mean` should have length 2; found shape {}'.format(mean.shape))
    mean = mean.reshape(1, 2)
    if covariance.shape != (2, 2):
        raise ValueError('`covariance` should have shape (2, 2); found shape {}'.format(covariance.shape))
    
    unit_circle = np.vstack([
        np.cos(np.linspace(0, 2*np.pi, n)),
        np.sin(np.linspace(0, 2*np.pi, n)),
    ])
    lower = cholesky(covariance)
    return sigma * np.dot(lower, unit_circle).transpose() + mean


def mahalanobis(x, mean, covariance):
    """
    Compute the square of Mahalanobis distance [3] of all observations in `x`:
        s² = (x - μ) Σ⁻¹ (x - μ)*

    Parameters
    ----------
    x : ndarray (M, N)
        N-feature array of M observations of test points, for which the distance
        from the multivariate normal distribution described by `mu` and `sig` will
        be computed.
    mean : ndarray (N,)
        N-feature means.
    covariance : ndarray (N, N)
        N-feature covariance.

    Returns
    -------
    s2 : ndarray (M,)
        Square of distance from feature means to test points, normalised by
        covariance.
    
    References
    ----------
    [2] - R. DE MAESSCHALCK, et al., The Mahalanobis distance, Chemometrics and
            Intelligent Laboratory Systems, Volume 50, Issue 1, p. 1-18, Jan
            2000, doi:10.1016/S0169-7439(99)00047-7

    """
    x, mean, covariance = np.asarray(x), np.asarray(mean), np.asarray(covariance)
    return np.einsum(
        '...ij,...ji->...i',
        x - mean, solve(covariance, (x - mean).swapaxes(-2, -1).conj()),
    ).real


def mardia_test(x, alpha=0.05):
    """
    Perform the Mardia test [2] for normality with multi-dimensional data.
    Alternatively, check that Mahalanobis distance of `x` is distributed with χ².

    Parameters
    ----------
    x : ndarray (d, n)
        Data to be tested for normality, with features in axis 0 and
        observations in axis 1. If 1D, this may be relaxed to shape (n,).
    alpha : float, optional
        Significance level on which normality will be tested. The default is
        0.05.

    Returns
    -------
    result : bool
        True if both of the tests performed do not reject the null hypothesis
        (i.e. both p1 and p2 > alpha), False otherwise. Treat it as True/False
        of whether the provided data is normal.

    References
    ----------
    [3] - K. V. MARDIA, Measures of multivariate skewness and kurtosis with
            applications, Biometrika, Volume 57, Issue 3, p. 519–530, Dec 1970,
            doi:10.1093/biomet/57.3.519

    """
    x = np.asarray(x)
    if x.ndim < 2:
        x = x.reshape(1, -1)
    elif x.ndim > 2:
        raise ValueError(
            'x must be 2D containing rows of features and columns of observations.'
        )

    (d, n) = x.shape

    if x.dtype == complex:
        x = np.vstack([x.real, x.imag])

    mean = np.mean(x, axis=1).reshape(-1, 1)
    covr = np.atleast_2d(cov(x))

    a_to_sum = np.einsum('ji,jk', (x - mean).conj(), solve(covr, x - mean))
    a = 1 / (6 * n) * np.einsum('ij,ij,ij', a_to_sum, a_to_sum, a_to_sum).real
    b = np.sqrt(n / (8 * d * (d + 2))) * (
        1 / n * np.einsum('ii', a_to_sum**2) - d * (d + 2)
    )
    p1 = 1 - chi2.cdf(a, d * (d + 1) * (d + 2) / 6)
    p2 = 2 * (1 - norm.cdf(abs(b)))

    return p1 >= alpha and p2 >= alpha


def transform_observations(x, mean, covariance):
    """
    Transform the observations in `x` to the basis defined by the distribution:
        s = (x - μ) A
    where Σ⁻¹ = A Aᵀ, found using the Cholesky decomposition of Σ⁻¹.

    Parameters
    ----------
    x : ndarray (M, 1, N)
        N-feature array of M observations of test points, for which the 
        transformation will be computed
    mean : ndarray (M, 1, N)
        N-feature means.
    covariance : ndarray (M, N, N)
        N-feature covariance.

    Returns
    -------
    s : ndarray (M, N)
        Vectors from the origin in the basis defined by the multivariate normal
        distribution.
    
    Examples
    --------
    Say we have a test point `x`, which is being compared to a distribution
    with mean `mu` and covariance matrix `cov`. Then
    ```
    >>> test_points = np.reshape([1., 2.], (1, 2))
    >>> eg_mean     = np.reshape([1.5, 0.], (1, 2))
    >>> eg_cov      = np.asarray([[1.0, 0.3], [0.3, 2.0]])
    >>> s = transform_observations(test_points, eg_mean, eg_cov)
    >>> assert np.allclose(
    ...     mahalanobis(test_points, eg_mean, eg_cov),
    ...     np.dot(s, s.T)
    ... )
    ```
    
    We can test multiple points against the same distribution
    ```
    >>> test_points = 2 * np.random.rand(3, 1, 2) - 1
    >>> s = transform_observations(test_points, eg_mean, eg_cov)
    >>> for i in range(3):
    ...     assert np.allclose(
    ...         transform_observations(test_points[i, :, :], eg_mean, eg_cov),
    ...         s[i, :]
    ...     )
    ```
    Note that the same behaviour will be found if `mu` has shape `(1, 1, N)`
    and `cov` has shape `(1, N, N)`.
    
    We can also test multiple points against multiple distributions
    ```
    >>> eg_mean = 2 * np.random.rand(3, 1, 2) - 1
    >>> eg_cov = 2 * np.random.rand(3, 2, 2) - 1
    >>> eg_cov = eg_cov * eg_cov.transpose(0, 2, 1) + 2 * np.eye(2)
    >>> s = transform_observations(test_points, eg_mean, eg_cov)
    >>> for i in range(3):
    ...     assert np.allclose(
    ...         transform_observations(test_points[i, :, :], eg_mean[i, :, :], eg_cov[i, :, :]),
    ...         s[i, :]
    ...     )
    ```
    """
    x, mean, covariance = np.asarray(x), np.asarray(mean), np.asarray(covariance)
    return np.einsum(
        '...ij,...ji->...i',
        x - mean, cholesky(inv(covariance))
    ).real