"""EPS panel / colourbar helpers for the phantom-study result figures (Figs. 6-9).

Panels are drawn with ``tomo_fusion.tools.plotting_fcts.plot_profile`` (TCV-clipped profile with
flux-surface contours), exactly as in the paper.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import common  # noqa: E402,F401  (puts tomo_fusion on sys.path)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

from tomo_fusion.tools import plotting_fcts as tomo_plots  # noqa: E402

H, W = cfg.H, cfg.W


def ground_truth(idx: int):
    """Ground-truth phantom and its equilibrium psi (float64, as in the paper)."""
    gt = np.load(cfg.TEST_SET, mmap_mode="r")[idx]
    psi = np.load(cfg.TEST_PSIS, mmap_mode="r")[idx]
    return np.array(gt), np.array(psi)


def study_image(method, diag, sigma, array, idx):
    """One phantom's (H, W) image from a phantom-study array (post_means/post_stds/maps)."""
    d = common.study_dir(method, diag, sigma)
    if not d.exists():
        raise FileNotFoundError(
            f"{d} not found: run the corresponding script in reproduce/02_phantom_studies first")
    return np.load(d / f"{array}.npy", mmap_mode="r")[idx].reshape(H, W)


def fmt_tick(t):
    """Tick label: 2 decimals, trailing zeros stripped, in math mode."""
    return f"${f'{t:.2f}'.rstrip('0').rstrip('.')}$"


def save_panel(img, psi, cmap, path, vmin=None, vmax=None, lwidth=0.2, lcfs_width=1, levels=15,
               distill=False):
    """Save one TCV-clipped profile with flux-surface contours as EPS.

    ``distill=True`` post-processes with Ghostscript ps2write (needs ``gs``). Most paper
    panels went through this step (matplotlib distills EPS whenever ``text.usetex`` is
    on); it only re-encodes the file and shifts strokes/images by < 1 pt.
    """
    plt.rcParams["ps.usedistiller"] = "ghostscript" if distill else None
    tomo_plots.plot_profile(img, tcv_plot_clip=True, contour_image=psi,
                            cmap=cmap, ax=None, contour_color="w",
                            vmin=vmin, vmax=vmax, levels=levels,
                            aspect=None, lcfs_width=lcfs_width, lwidth=lwidth)
    plt.savefig(path, format="eps", bbox_inches="tight")
    plt.close("all")


def save_colorbar(cmap, vmax, path, ticks=None, ticklabels=None, fontsize=8):
    """Standalone vertical colourbar [0, vmax] (ticks default to 0, vmax/2, vmax)."""
    if ticks is None:
        ticks = [0.0, vmax / 2.0, vmax]
    if ticklabels is None:
        ticklabels = [fmt_tick(t) for t in ticks]
    fig, ax = plt.subplots(figsize=(0.6, 3))
    cbar = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=vmax), cmap=cmap),
                        cax=ax, orientation="vertical")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels, fontsize=fontsize)
    fig.savefig(path, format="eps", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_latex_colorbar(cmap, vmax, path, ticks, ticklabels, horizontal=False, fontsize=12, aspect=5):
    """LaTeX-typeset colourbar as used in the paper figures.

    Vertical: ticks at ``ticks`` (figs. 6/7). Horizontal: min/max ticks only (fig. 8).
    Needs a LaTeX install with type1cm and dvips (matplotlib ``text.usetex``); falls
    back to matplotlib mathtext otherwise (same layout, slightly different font).
    """
    try:
        _latex_colorbar(cmap, vmax, path, ticks, ticklabels, horizontal, fontsize, aspect, usetex=True)
    except RuntimeError as e:
        global _WARNED
        if not _WARNED:
            print(f"  [warn] LaTeX unavailable for colourbars ({str(e).splitlines()[0][:60]}...);"
                  " using mathtext instead")
            _WARNED = True
        _latex_colorbar(cmap, vmax, path, ticks, ticklabels, horizontal, fontsize, aspect, usetex=False)
    finally:
        plt.close("all")
        plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "dejavusans"})


_WARNED = False


def _latex_colorbar(cmap, vmax, path, ticks, ticklabels, horizontal, fontsize, aspect, usetex):
    plt.close("all")
    plt.rcParams.update({"text.usetex": usetex, "font.family": "sans-serif",
                         "mathtext.fontset": "cm"})
    tbp = np.zeros((H, W))
    tbp[0, 0] = vmax
    plt.figure(figsize=(2, 3))
    r = plt.imshow(tbp, cmap)
    plt.gca().set_visible(False)
    if horizontal:
        cbar = plt.colorbar(r, orientation="horizontal", aspect=aspect, shrink=0.3)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.set_xticks(ticks)
        cbar.ax.set_xticklabels(ticklabels)
    else:
        cbar = plt.colorbar(r, aspect=20)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.set_yticks(ticks)
        cbar.ax.set_yticklabels(ticklabels)
    plt.axis("off")
    plt.savefig(path, format="eps", bbox_inches="tight")
