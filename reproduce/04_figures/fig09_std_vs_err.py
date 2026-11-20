"""Fig. 9 (fig:std_vs_err): posterior std vs absolute error, RD and DiffRD (Pilatus).

For three test phantoms (666, 608, 969) at eta_medium (sigma=0.35): pixel-wise
posterior std and |ground truth - posterior mean| of RD and DiffRD. Contours are
drawn with 5 levels and thin lines (as in the paper). The posterior means are written
as well (not shown in the paper).

Inputs:  results/phantom_studies/{rd,diffrd}_pilatus_0.35/
Outputs: figures/std_vs_err/pilatus_all_sigma{sigma}_idx{idx}/{rd,diffrd}_{std,..._err}.eps
Runtime: a few seconds (CPU).

    python fig09_std_vs_err.py                  # paper figure
    python fig09_std_vs_err.py --idx 317 --sigma 0.7
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

PAPER_IDXS = (666, 608, 969)


def make(sigma, idx):
    diag = "pilatus"
    gt, psi = pp.ground_truth(idx)
    out = cfg.figure_dir(f"std_vs_err/pilatus_all_sigma{sigma}_idx{idx}")

    rd_df = pd.read_csv(common.study_dir("rd", diag, sigma) / "metrics.csv")
    drd_df = pd.read_csv(common.study_dir("diffrd", diag, sigma) / "metrics.csv")
    rd_mean = pp.study_image("rd", diag, sigma, "post_means", idx)
    drd_mean = pp.study_image("diffrd", diag, sigma, "post_means", idx)
    rd_psnr, rd_ssim = rd_df.loc[idx, ["psnr_mean", "ssim_mean"]]
    drd_psnr, drd_ssim = drd_df.loc[idx, ["psnr", "ssim"]]

    vmax_mean = float(gt.max())
    for img, name in [(gt, "phantom"),
                      (rd_mean, f"rd_mean_psnr{rd_psnr:.2f}_ssim{rd_ssim:.3f}"),
                      (drd_mean, f"diffrd_mean_psnr{drd_psnr:.2f}_ssim{drd_ssim:.3f}")]:
        pp.save_panel(img, psi, "viridis", out / f"{name}.eps", vmin=0, vmax=vmax_mean)

    thin = dict(vmin=0, lcfs_width=1, lwidth=0.04, levels=5)
    pp.save_panel(pp.study_image("rd", diag, sigma, "post_stds", idx), psi, "inferno", out / "rd_std.eps", **thin)
    pp.save_panel(pp.study_image("diffrd", diag, sigma, "post_stds", idx), psi, "inferno", out / "diffrd_std.eps", **thin)
    pp.save_panel(np.abs(gt - rd_mean), psi, "gray", out / "rd_mean_err.eps", **thin)
    pp.save_panel(np.abs(gt - drd_mean), psi, "gray", out / "diffrd_err.eps", **thin)
    print(f"pilatus sigma={sigma} idx={idx}: RD psnr {rd_psnr:.2f} | DiffRD psnr {drd_psnr:.2f} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--idx", type=int, nargs="+", default=list(PAPER_IDXS))
    a = ap.parse_args()
    for idx in a.idx:
        make(a.sigma, idx)


if __name__ == "__main__":
    main()
