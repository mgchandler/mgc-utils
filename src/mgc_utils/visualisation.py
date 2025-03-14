"""
A set of functions which solve different tasks in visualisation. This may be matplotlib figures which are plotted really
often, so it's useful to have them all in one place, or other helper functions.
"""
__all__ = [
    'plot_complex_tfm', 'opheim_simpl',
]

from arim.ut import decibel
import matplotlib.pyplot as plt
import numpy as np
import os
from warnings import warn


def plot_complex_tfm(
        tfm,
        grid,
        fig_title="",
        dB=False,
        show=True,
        save=False,
        savename="untitled",
        savedir=".",
):
    # Point is to make a figure. If neither interested in showing or saving, don't bother.
    if not show and not save:
        return

    fig, axs = plt.subplots(4, 1, figsize=(8, 8), dpi=200)
    imr = axs[0].imshow(
        np.real(tfm.transpose()),
        extent=[
            grid.xmin * 1e3,
            grid.xmax * 1e3,
            grid.zmax * 1e3,
            grid.zmin * 1e3,
        ],
        vmin=-np.abs(tfm).max(), vmax=np.abs(tfm).max(),
        cmap='RdBu',
    )
    axs[1].imshow(
        np.imag(tfm.transpose()),
        extent=[
            grid.xmin * 1e3,
            grid.xmax * 1e3,
            grid.zmax * 1e3,
            grid.zmin * 1e3,
        ],
        vmin=-np.abs(tfm).max(), vmax=np.abs(tfm).max(),
        cmap='RdBu',
    )
    if dB:
        ima = axs[2].imshow(
            decibel(tfm.transpose()),
            extent=[
                grid.xmin * 1e3,
                grid.xmax * 1e3,
                grid.zmax * 1e3,
                grid.zmin * 1e3,
            ],
            vmin=-40, vmax=0,
            cmap='viridis',
        )
    else:
        ima = axs[2].imshow(
            np.abs(tfm.transpose()),
            extent=[
                grid.xmin * 1e3,
                grid.xmax * 1e3,
                grid.zmax * 1e3,
                grid.zmin * 1e3,
            ],
            vmin=0, vmax=np.abs(tfm).max(),
            cmap='viridis',
        )
    imp = axs[3].imshow(
        np.angle(tfm.transpose()),
        extent=[
            grid.xmin * 1e3,
            grid.xmax * 1e3,
            grid.zmax * 1e3,
            grid.zmin * 1e3,
        ],
        vmin=-np.pi, vmax=np.pi,
        cmap='hsv',
    )
    axs[0].set_title(fig_title)
    [ax.set_xticklabels([]) for ax in axs[:3]]

    fig.colorbar(
        imr, ax=axs[:2], label='(arb)', shrink=.99, aspect=20,
    )
    if dB:
        fig.colorbar(
            ima, ax=axs[2], label='(dB)', shrink=.99, aspect=10,
        )
    else:
        fig.colorbar(
            ima, ax=axs[2], label='(arb)', shrink=.99, aspect=10,
        )
    rad = fig.colorbar(
        imp, ax=axs[3], label='(rad)', shrink=.99, aspect=10,
        ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    )
    rad.set_ticklabels(['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])

    axs[-1].set_xlabel('$y$ (mm)')
    [ax.set_ylabel('$z$ (mm)') for ax in axs]
    if save:
        if savename == "":
            warn('Warning - no title provided. Saving as "untitled.png".')
            savename = "untitled"
        if savename[-4:] != ".png":
            savename = f"{savename}.png"
        plt.savefig(os.path.join(savedir, savename), bbox_inches='tight')
    if show:
        plt.show()
    plt.close('all')

    return


def opheim_simpl(x, y, tol):
    """
    Performs Opheim path simplification algorithm on (x, y) data. A path
    contained in advancing `x`, `y` will be simplified by tolerance `tol` with
    the algorithm:
    1) Select the first vertex as a `key`.
    2) Find the first vertex after `key` which is more than distance `tol`
       away (call it `next`) and links these two vertices with `line`.
    3) Find the vertices which exist between `key` and `next`. Find the last
       one which sits within `tol` of `line` found in (2). Alternatively, find
       the vertex which (when connected to its previous point) forms a segment
       which has an angle with `line` greater than 90° (to deal with spikes
       which should typically be preserved). Call it `last`. Remove all points
       between `key` and `last`.
    4) Set `key = last` and repeat from (2), until all points are exhausted.

    Parameters
    ----------
    x : ndarray[float] (N,)
        x-coordinates.
    y : ndarray[float] (N,)
        y-coordinates.
    tol : float
        Magnitude tolerance used to iterate through (x, y).

    Returns
    -------
    mask : ndarray[bool] (N,)
        Mask to be applied to `x` and `y` to remove the relevant data points.

    Notes
    -----
    This function is mostly useful for plotting curves which may have a large
    number of points but are relatively smooth, e.g. the results of the `roc`
    function. Not all of the data is needed to visually represent it. Consider
    using this if `matplotlib.pyplot` is really slow at plotting curves with a
    lot of points.

    """
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('`x` and `y` must be 1-dimensional ndarrays.')
    if x.shape[0] != y.shape[0]:
        raise ValueError('`x` and `y` must have the same shape.')
    mask = np.full(x.shape, False)
    mask[0], mask[-1] = True, True

    i, N = 0, x.shape[0]
    while i < N - 2:
        # Find the first vertex beyond `tol`
        j = i + 1
        v = np.asarray([x[j] - x[i], y[j] - y[i]])
        while j < N and np.linalg.norm(v) <= tol:
            j = j + 1
            v = np.asarray([x[j] - x[i], y[j] - y[i]])
        v = v / np.linalg.norm(v)

        # Unit normal between `i`, `j`.
        norm = [v[1], -v[0]]

        # Find the last point which is within `tol` of the line connecting
        # point `i` to point `j`. Alternatively, the last point within a
        # direction change of pi/2.
        while j < N - 1:
            # Perpendicular distance from `i -> j` line
            v1 = [x[j + 1] - x[i], y[j + 1] - y[i]]
            d = np.abs(np.dot(norm, v1))
            if d > tol:
                break

            # Angle between line and current segment.
            v2 = [x[j + 1] - x[j], y[j + 1] - y[j]]
            cos = np.dot(v, v2)
            if cos <= 0:
                break

            j += 1
        i = j
        mask[i] = True

    return mask