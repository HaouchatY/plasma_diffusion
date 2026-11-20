"""Fig. `fig:los_configurations` - lines of sight of the SXR, DMPX and Pilatus diagnostics.

Writes Figures/diagnostics/{sxr,dmpx,pilatus}_LoS_configuration.eps.
CPU only, a few seconds.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402
import common  # noqa: E402,F401
import _plot_utils as pu  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def draw(diag: str, path: Path) -> None:
    params = np.load(cfg.LOS_PARAMS[diag])
    if diag != "sxr":
        params = params[::2]  # every other line of sight, for readability
    pu.plot_profile(np.ones((120, 41)), levels=12, figsize=(3.5, 5), tcv_plot_clip=True,
                    ax=None, cmap="gray", vmax=1, vmin=0)
    pu.draw_los(params, pu.LOS_GROUPS[diag])
    plt.savefig(path, format="eps", bbox_inches="tight")
    plt.close("all")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diags", nargs="+", default=["sxr", "dmpx", "pilatus"])
    args = ap.parse_args()
    out = cfg.figure_dir("diagnostics")
    for d in args.diags:
        draw(d, out / f"{d}_LoS_configuration.eps")
        print("wrote", out / f"{d}_LoS_configuration.eps")


if __name__ == "__main__":
    main()
