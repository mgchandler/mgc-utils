"""
Functions relating to NDT, which fall outside the scope of being put into Arim (for whatever reason).
"""
__all__ = [
    'ply_orientation',
]

import numpy as np
from .hypothesis_testing import mahalanobis
from .stats import convolve


def ply_orientation(grid, tfm, p=0.4e-3):
    """
    Computes the measured ply orientation from a TFM image of a composite
    material using the structure tensor. Note if `arim` is used to compute this
    `tfm`, then `tfm.res` should be passed to this function.

    Smoothing functions are hard-coded to follow the recommendations of Nelson
    et al. [4].

    Parameters
    ----------
    grid : arim.core.Grid
        The grid over which the TFM is computed. Only spacing is used, so
        technically it can be any object with attributes `dx`, `dy`, `dz`.
    tfm : ndarray[complex]
        TFM image produced over a composite material. Can be 2D or 3D (N.B. 3D
        implementation currently not working).
    p : float, optional
        Approximate ply thickness. Used in smoothing functions to produce the
        recommended kernel. The default is .4e-3.

    Returns
    -------
    ply_angle : ndarray[float]
        Measured ply angle relative to the last axis of the TFM image. Will
        have the same shape as `tfm`.

    References
    ----------
    [4] - L. J. NELSON, et al., Ply-orientation measurements in composites
            using structure-tensor analysis of volumetric ultrasonic data,
            Composites A, Volume 126, p. 105581, November 2019,
            doi:10.1016/j.compositesa.2017.10.027

    """
    # Compute sine and cosine of phase angle.
    phase = np.angle(tfm)
    sin, cos = np.sin(phase), np.cos(phase)

    # Smooth sine and cosine w/ Gaussian kernel:
    #   ```G1(r) = A * exp(-(r.T * C * r)/2)```
    # Standard deviation model: recommendation.
    sigxx, sigyy, sigzz = p, p, p / 15
    if tfm.ndim == 2:
        # Go up to 6σ, very little going on outside due to exponential decay.
        x = np.arange(0, 12 * sigxx, grid.dx)
        z = np.arange(0, 12 * sigzz, grid.dz)
        [x, z] = np.meshgrid(x - x.mean(), z - z.mean(), indexing="ij")
        r = np.vstack([x.ravel(), z.ravel()])
        # Covariance: `C = cov**(-1)`. Will invert inside `mahalanobis`.
        cov1 = [sigxx**2, sigzz**2] * np.eye(2)
        cov2 = [(2 * sigxx) ** 2, (2 * sigzz) ** 2] * np.eye(2)
        # Get nominal normal direction, used in ply angle calc later. Use z_hat.
        normal = np.asarray([0, 1]).reshape(-1, 1)
    elif tfm.ndim == 3:
        x = np.arange(0, 12 * sigxx, grid.dx)
        y = np.arange(0, 12 * sigyy, grid.dy)
        z = np.arange(0, 12 * sigzz, grid.dz)
        [x, y, z] = np.meshgrid(x - x.mean(), y - y.mean(), z - z.mean(), indexing="ij")
        r = np.vstack([x.ravel(), y.ravel(), z.ravel()])
        cov1 = [sigxx**2, sigyy**2, sigzz**2] * np.eye(3)
        cov2 = [(2 * sigxx) ** 2, (2 * sigyy) ** 2, (2 * sigzz) ** 2] * np.eye(3)
        normal = np.asarray([0, 0, 1]).reshape(-1, 1)
    else:
        raise ValueError(
            'Too many dimensions in `tfm` - should contain a single 2D or 3D TFM image.'
        )
    g1 = (
        # `A` defined s.t. ∫ G1(r) dr = 1
        (2 * np.pi) ** (-len(x.shape) / 2)
        * np.sqrt(np.linalg.det(cov1))
        * np.exp(-mahalanobis(r.transpose(), 0, cov1) / 2).reshape(x.shape)
    )
    sin_smoothed = convolve(sin, g1, mode='same')
    cos_smoothed = convolve(cos, g1, mode='same')

    # Compute derivatives:
    if tfm.ndim == 2:
        d = np.asarray([-1, 0, 1])
        s = np.asarray([3, 10, 3])
        # Could probably automate this better.
        scharr_kern = np.zeros((2, 3, 3))
        scharr_kern[0, :, :] = np.dot(d.reshape(-1, 1), s.reshape(1, -1)) / 32
        scharr_kern[1, :, :] = np.dot(s.reshape(-1, 1), d.reshape(1, -1)) / 32
    elif tfm.ndim == 3:
        raise NotImplementedError(
            '3D implementation not yet working - need to work out the Scharr kernel in 3D.'
        )
        # d = np.asarray([0])
        # s = np.asarray([0])
        # scharr_kern = np.zeros((3, 3, 3, 3))
    grad = np.zeros((tfm.ndim, *tfm.shape))
    for dim in range(tfm.ndim):
        grad[dim] = cos_smoothed * convolve(
            sin_smoothed, scharr_kern[dim], mode='same'
        ) - sin_smoothed * convolve(cos_smoothed, scharr_kern[dim], mode='same')

    # Create structure tensor
    structure = grad.reshape(tfm.ndim, 1, *tfm.shape) * grad.reshape(
        1, tfm.ndim, *tfm.shape
    )
    # Smoothing (`g2`) goes here.
    g2 = (
        (2 * np.pi) ** (-len(x.shape) / 2)
        * np.sqrt(np.linalg.det(cov2))
        * np.exp(-mahalanobis(r.transpose(), 0, cov2) / 2).reshape(x.shape)
    ).reshape(1, 1, *x.shape)
    # g2 = np.asarray([1]).reshape([1 for _ in structure.shape])
    structure_smoothed = convolve(structure, g2, mode='same').reshape(
        tfm.ndim, tfm.ndim, -1
    )

    # Compute eigenvectors and ply angle - 2D.
    evecs = np.vstack([
        2 * structure_smoothed[0, 1],
        (
            structure_smoothed[1, 1]
            - structure_smoothed[0, 0]
            + np.sqrt(
                (structure_smoothed[0, 0] - structure_smoothed[1, 1]) ** 2
                + 4 * structure_smoothed[0, 1] ** 2
            )
        ),
    ]).transpose()
    ply_angle = (
        np.arccos(
            np.dot(evecs, normal).squeeze()
            / (np.linalg.norm(evecs, axis=1) * np.linalg.norm(normal))
        )
        * np.sign([
            np.linalg.det(np.hstack([normal, evecs[i : i + 1, :].transpose()]))
            for i in range(evecs.shape[0])
        ])
    ).reshape(tfm.shape)

    return ply_angle