"""Plotting of emissivity profiles on the TCV poloidal cross-section."""

from __future__ import annotations

from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

__all__ = ["plot_profile", "show_samples"]


def _tcv_patch(image_shape) -> PathPatch:
    """TCV vessel outline as a matplotlib patch in image coordinates."""
    with resources.files("plasma_diffusion.resources").joinpath(
            "tcv_shape_coords.npy").open("rb") as fh:
        coords = np.load(fh)
    coords = coords.copy()
    coords[0, 0], coords[0, 1] = 0.675, 1.0015
    coords[-1, 0] = 0.684
    lr, lz = 0.5, 1.5
    h = lz / image_shape[0]
    n_z = round(lz / h)
    n_r = round(lr / h)
    coords[:, 0] = coords[:, 0] * (n_r - 1) - 0.5 * h
    coords[:, 1] = coords[:, 1] * (n_z - 1) - 0.5 * h
    return PathPatch(MplPath(coords.tolist()), facecolor="none")


def plot_profile(image, ax=None, figsize=(2, 3), contour_image=None,
                 levels: int = 15, lcfs_width: float = 0.75,
                 contour_width: float = 0.2, contour_color: str = "w",
                 cmap: str = "viridis", vmin=None, vmax=None,
                 colorbar: bool = False, clip_to_vessel: bool = True):
    """Plot one emissivity profile, optionally with flux-surface contours.

    Parameters
    ----------
    image:
        Profile of shape ``(H, W)`` in the image convention (row 0 = top).
    contour_image:
        Optional poloidal-flux map of the same shape; its isolines are drawn
        on top, with the ``psi = 0`` contour (last closed flux surface)
        highlighted.
    clip_to_vessel:
        Clip the plot to the TCV vessel outline.

    Returns
    -------
    The matplotlib axis.
    """
    image = np.asarray(image)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    contours = None
    if contour_image is not None:
        contours = ax.contour(np.flip(np.asarray(contour_image), 0),
                              origin="lower", levels=levels, antialiased=True,
                              colors=contour_color,
                              negative_linestyles="solid", linewidths=0.1)
        widths = contour_width * np.ones(contours.levels.size)
        lcfs = np.where(contours.levels == 0)[0]
        if lcfs.size:
            widths[lcfs[0]] = lcfs_width
        contours.set_linewidth(widths)

    im = ax.imshow(np.flip(image, 0), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlim([-0.75, image.shape[1] + 0.75])
    ax.set_ylim([-0.75, image.shape[0] + 0.75])

    if clip_to_vessel:
        patch = _tcv_patch(image.shape)
        ax.add_patch(patch)
        im.set_clip_path(patch)
        if contours is not None:
            contours.set_clip_path(patch)

    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.08)
    return ax


def show_samples(samples, titles=None, title=None, contour_images=None,
                 figsize=None, cmap: str = "viridis", vmin=None, vmax=None,
                 colorbar: bool = False):
    """Show a row of profiles side by side.

    ``samples`` can be a numpy array or torch tensor of shape
    ``(N, H, W)`` (or ``(N, 1, H, W)``); ``contour_images`` is an optional
    list of flux maps, one per panel.
    """
    samples = np.asarray([np.asarray(getattr(s, "cpu", lambda: s)()).squeeze()
                          for s in samples])
    n = samples.shape[0]
    if figsize is None:
        figsize = (1.8 * n, 4)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    axes = np.atleast_1d(axes)
    for i in range(n):
        plot_profile(samples[i], ax=axes[i], cmap=cmap, vmin=vmin, vmax=vmax,
                     colorbar=colorbar,
                     contour_image=None if contour_images is None
                     else contour_images[i])
        if titles is not None:
            axes[i].set_title(titles[i], fontsize=9)
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    return fig, axes
