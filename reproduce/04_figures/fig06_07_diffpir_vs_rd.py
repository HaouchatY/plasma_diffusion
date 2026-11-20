"""Figs. 6 and 7 (fig:diffpir_vs_rd_sxr, fig:diffpir_vs_rd_dmpx): DiffPIR vs RD.

Ground truth, posterior means and pixel-wise posterior std of DiffPIR and RD for one
test phantom: SXR at eta_medium (sigma=0.25, phantom 661) and DMPX at eta_low
(sigma=0.1, phantom 9). Mean panels are named ``*_mean_psnr{..}_ssim{..}.eps`` with
the metrics of that phantom, as referenced by the paper's LaTeX.

Inputs:  results/phantom_studies/{diffpir,rd}_{diag}_{sigma}/
Outputs: figures/diffpir_vs_rd/{diag}_sigma{sigma}_idx{idx}/*.eps
Runtime: a few seconds (CPU). The std colourbars are LaTeX-typeset (needs LaTeX).

    python fig06_07_diffpir_vs_rd.py                      # both paper panels
    python fig06_07_diffpir_vs_rd.py --diag sxr --sigma 0.5 --idx 4 --auto-cbar
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _phantom_panels as pp  # noqa: E402

# (diag, sigma, idx) -> hand-chosen std-colourbar (vmax, ticks, labels) used in the paper:
# {method: (vmax, ticks, labels)}.
PAPER_PANELS = {
    ("sxr", 0.25, 661): {
        "rd": (0.22, [0, 0.11, 0.22], [r"$0$", r"$0.11$", r"$0.22$"]),
        "diffpir": (0.06, [0, 0.03, 0.06], [r"$0$", r"$0.03$", r"$0.06$"]),
    },
    ("dmpx", 0.1, 9): {
        "rd": (0.27, [0, 0.13, 0.27], [r"$0$", r"$0.13$", r"$0.26$"]),
        "diffpir": (0.12, [0, 0.06, 0.12], [r"$0$", r"$0.06$", r"$0.12$"]),
    },
}


def make(diag, sigma, idx, auto_cbar=False):
    gt, psi = pp.ground_truth(idx)
    m = {}
    for method in ("diffpir", "rd"):
        m[method] = {
            "mean": pp.study_image(method, diag, sigma, "post_means", idx),
            "std": pp.study_image(method, diag, sigma, "post_stds", idx),
            "metrics": pd.read_csv(common.study_dir(method, diag, sigma) / "metrics.csv"),
        }
    dpsnr, dssim = m["diffpir"]["metrics"].loc[idx, ["psnr", "ssim"]]
    rpsnr, rssim = m["rd"]["metrics"].loc[idx, ["psnr_mean", "ssim_mean"]]

    out = cfg.figure_dir(f"diffpir_vs_rd/{diag}_sigma{sigma}_idx{idx}")
    vmax_mean = float(gt.max())
    err_d = np.abs(gt - m["diffpir"]["mean"])
    err_r = np.abs(gt - m["rd"]["mean"])

    pp.save_panel(gt, psi, "viridis", out / "phantom.eps", vmin=0, vmax=vmax_mean)
    pp.save_panel(m["diffpir"]["mean"], psi, "viridis",
                  out / f"diffpir_mean_psnr{dpsnr:.2f}_ssim{dssim:.3f}.eps", vmin=0, vmax=vmax_mean)
    pp.save_panel(m["rd"]["mean"], psi, "viridis",
                  out / f"rd_mean_psnr{rpsnr:.2f}_ssim{rssim:.3f}.eps", vmin=0, vmax=vmax_mean)
    pp.save_panel(m["diffpir"]["std"], psi, "inferno", out / "diffpir_std.eps", vmin=0)
    pp.save_panel(m["rd"]["std"], psi, "inferno", out / "rd_std.eps", vmin=0)
    pp.save_panel(err_d, psi, "gray", out / "diffpir_err.eps", vmin=0)
    pp.save_panel(err_r, psi, "gray", out / "rd_err.eps", vmin=0)

    pp.save_colorbar("viridis", 1.0, out / "cbar_mean.eps", ticks=[0.0, 0.5, 1.0])
    pp.save_colorbar("gray", float(err_d.max()), out / "cbar_diffpir_err.eps")
    pp.save_colorbar("gray", float(err_r.max()), out / "cbar_rd_err.eps")

    # std colourbars: the paper uses hand-rounded LaTeX colourbars; --auto-cbar derives
    # them from the data (std panels themselves are always auto-scaled to [0, max]).
    paper = PAPER_PANELS.get((diag, sigma, idx))
    for method in ("diffpir", "rd"):
        smax = float(m[method]["std"].max())
        if paper is None or auto_cbar:
            ticks = [0, smax / 2, smax]
            spec = (smax, ticks, [pp.fmt_tick(t) for t in ticks])
        else:
            spec = paper[method]
        pp.save_latex_colorbar("inferno", spec[0], out / f"cbar_{method}_std.eps", spec[1], spec[2])
        print(f"  {method} std max = {smax:.4f}, colourbar top = {spec[0]}")
    print(f"{diag} sigma={sigma} idx={idx}: DiffPIR psnr {dpsnr:.2f} ssim {dssim:.3f} | "
          f"RD psnr {rpsnr:.2f} ssim {rssim:.3f} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", choices=list(cfg.GEOMETRY))
    ap.add_argument("--sigma", type=float)
    ap.add_argument("--idx", type=int)
    ap.add_argument("--auto-cbar", action="store_true", help="std colourbars from data max")
    a = ap.parse_args()
    if a.diag is None:
        for key in PAPER_PANELS:
            make(*key, auto_cbar=a.auto_cbar)
    else:
        make(a.diag, a.sigma, a.idx, auto_cbar=a.auto_cbar)


if __name__ == "__main__":
    main()
