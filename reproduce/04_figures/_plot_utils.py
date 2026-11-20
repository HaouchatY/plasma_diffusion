"""Plotting helpers shared by the figure scripts in ``reproduce/04_figures``.

All emissivity panels in the paper are drawn with
``tomo_fusion.tools.plotting_fcts.plot_profile`` (TCV vessel clipping, flux-surface
contours) and saved as individual EPS files that the LaTeX figures assemble.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402,F401  (puts tomo_fusion on sys.path)
from tomo_fusion.tools import plotting_fcts as tomo_plots  # noqa: E402

plot_profile = tomo_plots.plot_profile


def _latex_available() -> bool:
    import shutil
    import subprocess
    if not (shutil.which("latex") and shutil.which("kpsewhich") and shutil.which("dvipng")):
        return False
    try:  # matplotlib's usetex preamble needs type1cm
        return bool(subprocess.run(["kpsewhich", "type1cm.sty"], capture_output=True,
                                   text=True).stdout.strip())
    except OSError:
        return False


def use_latex(enable: bool = True) -> None:
    """Computer-Modern text, as in the paper's colourbars and data plots.

    Uses a real LaTeX installation when available (as for the paper), otherwise falls
    back to matplotlib's built-in Computer Modern mathtext (visually equivalent).
    """
    if not enable:
        plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "dejavusans",
                             "font.family": "sans-serif"})
    elif _latex_available():
        plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    else:
        plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm",
                             "font.family": "serif", "font.serif": ["cmr10"],
                             "axes.formatter.use_mathtext": True})


def save_profile(img, path, psi=None, cmap="viridis", vmin=None, vmax=None,
                 levels=15, lwidth=0.2, lcfs_width=1):
    """Save one emissivity panel (TCV-clipped, optional flux-surface contours) as EPS.

    Mirrors the notebooks' ``plt.figure(figsize=(3,3)); plot_profile(..., ax=None)``
    followed by ``savefig(bbox_inches="tight")``.
    """
    plot_profile(np.asarray(img), tcv_plot_clip=True, contour_image=psi, cmap=cmap,
                 ax=None, colorbar=False, contour_color="w", vmin=vmin, vmax=vmax,
                 levels=levels, aspect=None, lcfs_width=lcfs_width, lwidth=lwidth)
    plt.axis("off")
    plt.savefig(path, format="eps", bbox_inches="tight")
    plt.close("all")


def fmt_tick(t: float) -> str:
    """2 decimals, trailing zeros stripped, LaTeX math (e.g. ``$0.5$``)."""
    return f"${f'{t:.2f}'.rstrip('0').rstrip('.')}$"


def save_colorbar(cmap, vmax, path, orientation="vertical", ticks=None, ticklabels=None,
                  fontsize=8, vmin=0.0, figsize=None, pad_inches=0.05, tick_length=None,
                  linewidth=None):
    """Standalone colourbar [vmin, vmax] saved as EPS (ticks default to [0, vmax/2, vmax])."""
    if ticks is None:
        ticks = [vmin, (vmin + vmax) / 2.0, vmax]
    if ticklabels is None:
        ticklabels = [fmt_tick(t) for t in ticks]
    if figsize is None:
        figsize = (0.6, 3) if orientation == "vertical" else (3, 0.6)
    fig, ax = plt.subplots(figsize=figsize)
    cbar = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                        cax=ax, orientation=orientation)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels, fontsize=fontsize)
    if tick_length is not None or linewidth is not None:
        cbar.ax.tick_params(length=tick_length, width=linewidth)
    if linewidth is not None:
        cbar.outline.set_linewidth(linewidth)
    fig.savefig(path, format="eps", bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)


def draw_los(params, groups, ax=None, widen=True):
    """Overlay lines of sight on a profile drawn by ``plot_profile`` (pixel coordinates).

    ``params`` is an (M, 2) array of (p, theta) line parameters. ``groups`` is a list of
    ``(i_stop, r_max, (y_lo, y_hi), opened_side)`` tuples: line ``i`` uses the first group
    with ``i < i_stop``; its abscissae run over ``linspace(0, r_max)``, points with
    ``y_lo < y < y_hi`` are kept (``None`` keeps all) and the line is clipped to the vessel
    outline opened on ``opened_side`` ("top", "right", "bottom" or None) so that fans
    entering from outside the vessel stay visible up to the detector.
    The vessel patch is the one returned by the last ``plot_profile`` call (``ax.patches``).
    ``widen`` enlarges the view so that the detectors outside the vessel are visible.
    """
    ax = ax or plt.gca()
    patch = [p for p in ax.patches if isinstance(p, PathPatch)][0]
    v0, codes = patch.get_path().vertices, patch.get_path().codes
    x_max, y_max, y_min = v0[:, 0].max(), v0[:, 1].max(), v0[:, 1].min()

    def opened(side):
        if side is None:
            return patch
        v = v0.copy()
        if side == "top":
            v[v[:, 1] > y_max - 10, 1] = 300
        elif side == "right":
            v[v[:, 0] > x_max - 10, 0] = 300
        elif side == "bottom":
            v[v[:, 1] < y_min + 10, 1] = -300
        p = PathPatch(MplPath(v, codes), facecolor="none", edgecolor="none")
        ax.add_patch(p)
        return p

    clips = {}
    center = np.array([120, 41], dtype=int) / 2
    Lz = 1.5
    h = Lz / 120
    for i in range(params.shape[0]):
        p, theta = params[i]
        if np.isclose(np.tan(theta), 0, atol=1e-6):
            lc = ax.vlines(center[1] + p, 0, Lz, "r", linewidth=0.3)
            lc.set_clip_path(patch)
            continue
        i_stop, r_max, yr, side = next((g for g in groups if i < g[0]), (None, 40.4, None, None))
        r = np.linspace(0.0, r_max, 4000)
        y = center[0] + p / h / np.sin(theta) - (r - center[1]) / np.tan(theta)
        keep = np.ones_like(y, bool) if yr is None else (y > yr[0]) & (y < yr[1])
        lw = 0.3 if i_stop is not None else 0.4
        if side not in clips:
            clips[side] = opened(side)
        for ln in ax.plot(r[keep], y[keep], "r", linewidth=lw):
            ln.set_clip_path(clips[side])
    if widen:
        ax.set_xlim([-3, 50])
        ax.set_ylim([-10.5, 130])


# Line-of-sight groups per diagnostic (ordering of the geometry rows), as in the paper
LOS_GROUPS = {
    # 20 top vertical, 40 + 20 horizontal (outer midplane), 20 divertor
    "sxr": [(20, 40.4, (0.0, 130), "top"),
            (60, 52.4, (0.4, 119.6), "right"),
            (80, 52.4, None, "right"),
            (100, 47.4, (-10.0, 119.6), "bottom")],
    # every other line of sight: 64 top + 64 bottom
    "dmpx": [(64, 40.4, (0.0, 127), "top"),
             (128, 40.4, (-7.0, 119.6), "bottom")],
    # every other line of sight, all from the top
    "pilatus": [(10 ** 6, 40.4, (0.0, 129), "top")],
}
