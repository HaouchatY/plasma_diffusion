"""Fig. 8 (fig:pilatus): Pilatus reconstructions of all methods for one phantom.

Ground truth, RD MAP, RD mean, DiffPIR, DiffRD_clip and DiffRD posterior means
(top row) and the posterior std of RD, DiffPIR, DiffRD_clip, DiffRD with their
colourbars (bottom row), Pilatus at eta_high (sigma=0.7), test phantom 52. Absolute
error panels are written too (not shown in the paper).

Inputs:  results/phantom_studies/{rd,diffpir,diffrd_clip,diffrd}_pilatus_{sigma}/
Outputs: figures/pilatus/pilatus_all_sigma{sigma}_idx{idx}/*.eps
Runtime: a few seconds (CPU). Colourbars are LaTeX-typeset (needs LaTeX).

    python fig08_pilatus.py                       # paper figure (sigma 0.7, idx 52)
    python fig08_pilatus.py --sigma 0.35 --idx 612
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

# method key -> (file stem used in the paper, results method)
METHODS = {
    "rd": ("rd", "rd"),
    "diffpir": ("diffpir_noclip", "diffpir"),
    "diffrd_clip": ("diffpir_clip", "diffrd_clip"),
    "diffrd": ("diffrd", "diffrd"),
}


def make(sigma, idx, cbar_fontsize=11):
    diag = "pilatus"
    gt, psi = pp.ground_truth(idx)
    vmax_mean = float(gt.max())
    out = cfg.figure_dir(f"pilatus/pilatus_all_sigma{sigma}_idx{idx}")

    res = {}
    for key, (stem, method) in METHODS.items():
        df = pd.read_csv(common.study_dir(method, diag, sigma) / "metrics.csv")
        res[key] = {
            "stem": stem,
            "mean": pp.study_image(method, diag, sigma, "post_means", idx),
            "std": pp.study_image(method, diag, sigma, "post_stds", idx),
            "psnr": float(df.loc[idx, "psnr_mean" if key == "rd" else "psnr"]),
            "ssim": float(df.loc[idx, "ssim_mean" if key == "rd" else "ssim"]),
        }
        if key == "rd":
            rd_map = pp.study_image(method, diag, sigma, "maps", idx)
            map_psnr, map_ssim = float(df.loc[idx, "psnr_map"]), float(df.loc[idx, "ssim_map"])

    mean_panels = [(gt, "phantom"), (rd_map, f"rd_map_psnr{map_psnr:.2f}_ssim{map_ssim:.3f}")]
    mean_panels += [(r["mean"], f"{r['stem']}_mean_psnr{r['psnr']:.2f}_ssim{r['ssim']:.3f}")
                    for r in res.values()]
    std_panels = [(r["std"], f"{r['stem']}_std") for r in res.values()]
    err_panels = [(np.abs(gt - res["rd"]["mean"]), "rd_mean_err"), (np.abs(gt - rd_map), "rd_map_err")]
    err_panels += [(np.abs(gt - r["mean"]), f"{r['stem']}_err") for k, r in res.items() if k != "rd"]

    for img, name in mean_panels:
        pp.save_panel(img, psi, "viridis", out / f"{name}.eps", vmin=0, vmax=vmax_mean)
    for img, name in std_panels + err_panels:
        pp.save_panel(img, psi, "inferno" if name.endswith("_std") else "gray",
                      out / f"{name}.eps", vmin=0)

    # horizontal colourbars with min/max ticks only
    pp.save_latex_colorbar("viridis", vmax_mean, out / "cbar_mean.eps", [0, vmax_mean],
                           [r"$0$", pp.fmt_tick(vmax_mean)], horizontal=True, fontsize=cbar_fontsize, aspect=4)
    for img, name in std_panels + err_panels:
        v = float(img.max())
        pp.save_latex_colorbar("inferno" if name.endswith("_std") else "gray", v,
                               out / f"cbar_{name}.eps", [0, v], [r"$0$", pp.fmt_tick(v)],
                               horizontal=True, fontsize=cbar_fontsize)

    print(f"pilatus sigma={sigma} idx={idx}: RD MAP psnr {map_psnr:.2f} | " +
          " | ".join(f"{k} psnr {r['psnr']:.2f}" for k, r in res.items()) + f" -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma", type=float, default=0.7)
    ap.add_argument("--idx", type=int, default=52)
    a = ap.parse_args()
    make(a.sigma, a.idx)


if __name__ == "__main__":
    main()
