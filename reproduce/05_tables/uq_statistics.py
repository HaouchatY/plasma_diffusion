"""Uncertainty-quantification statistics quoted in Sec. "Results" (no table/figure).

For every test phantom, restricted to its core pixels, with err = |mean - x_gt| and
z = err / std:

    r_std_err    Pearson corr(std, err)       -> r_{sigma,err} in the paper
    r_mean_err   Pearson corr(mean, err)      -> r_{mu,err}
    cov_1sigma   fraction of pixels with z <= 1  (Gaussian ideal 0.683)
    cov_2sigma   fraction of pixels with z <= 2  (Gaussian ideal 0.954)

plus Spearman variants, a mismatched-phantom control (std of phantom i vs error of
phantom i+500), a partial correlation removing the amplitude dependence, the Winkler
interval score (alpha = 0.01) and an over-confidence score. Values are then averaged
over the 1000 phantoms (mean, std, median).

Pilatus is evaluated for RD, DiffPIR, DiffRD_clip and DiffRD. NOTE: the original
analysis (and the sentence "r^DiffPIR is only 10-15% smaller than r^DiffRD" in the
paper) used the *clipped* DiffPIR run, i.e. DiffRD_clip, under the label "DiffPIR".
Both are reported here under their table names; see the README.

Inputs:  results/phantom_studies/{rd,diffpir,diffrd_clip,diffrd}_{diag}_{sigma}/post_{means,stds}.npy
Outputs: figures/uq_statistics/uq_metrics_{per_phantom,summary,summary_formatted}.csv
Runtime: ~5 min (CPU).

    python uq_statistics.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, pearsonr, rankdata, spearmanr  # noqa: E402

from tomo_fusion.tools import helpers as tomo_helps  # noqa: E402

H, W = cfg.H, cfg.W
ALPHA = 0.01              # interval-score level (central 99% interval, 200x violation penalty)
MISMATCH_OFFSET = 500     # phantom i's std is paired with phantom (i + 500) % N's error
MISMATCH_MIN_PX = 400     # minimum core-mask overlap for the mismatch control

# (diag) -> [(label, results method)]
METHODS = {
    "sxr": [("RD", "rd"), ("DiffPIR", "diffpir")],
    "dmpx": [("RD", "rd"), ("DiffPIR", "diffpir")],
    "pilatus": [("RD", "rd"), ("DiffPIR", "diffpir"), ("DiffRD_clip", "diffrd_clip"), ("DiffRD", "diffrd")],
}
METRICS = ["rho_std_err", "rho_mean_err", "rho_partial", "rho_mismatch", "rho_specific",
           "r_std_err", "r_mean_err", "r_mismatch", "r_specific", "cv_std",
           "cov_1sigma", "cov_2sigma", "is_range", "is_err", "overconf"]


def _partial_spearman(s, e, m):
    """Spearman(std, err) with the amplitude (mean) dependence projected out."""
    rs, re, rm = (rankdata(x) for x in (s, e, m))
    c = np.corrcoef([rs, re, rm])
    rse, rsm, rem = c[0, 1], c[0, 2], c[1, 2]
    return float((rse - rsm * rem) / np.sqrt(max((1 - rsm ** 2) * (1 - rem ** 2), 1e-12)))


def uq_metrics(gt, mean, std, mask, alpha=ALPHA):
    """UQ metrics for a single phantom, evaluated on core pixels only."""
    g, m, s = gt[mask], mean[mask], std[mask]
    err = np.abs(m - g)
    z = err / s
    k = norm.ppf(1 - alpha / 2)
    iscore = 2 * k * s + (2 / alpha) * np.maximum(err - k * s, 0.0)  # Winkler interval score
    return {
        "rho_std_err": spearmanr(s, err).statistic,
        "rho_mean_err": spearmanr(m, err).statistic,
        "rho_partial": _partial_spearman(s, err, m),
        "r_std_err": pearsonr(s, err)[0],
        "r_mean_err": pearsonr(m, err)[0],
        "cv_std": float(s.std() / s.mean()),
        "cov_1sigma": float(np.mean(z <= 1.0)),
        "cov_2sigma": float(np.mean(z <= 2.0)),
        "is_range": float(iscore.mean() / np.ptp(g)),
        "is_err": float(iscore.mean() / err.mean()),
        "overconf": float(np.mean(np.maximum(z - 1.0, 0.0))),
        "n_core": int(mask.sum()),
        "std_mean": float(s.mean()),
        "gt_range": float(np.ptp(g)),
    }


def per_phantom_table(gt, masks, diag, label, method, sigma):
    st = common.load_study(method, diag, sigma)
    mean = st["post_means"].reshape(-1, H, W)
    std = st["post_stds"].reshape(-1, H, W)
    n = len(gt)
    df = pd.DataFrame([uq_metrics(gt[i], mean[i], std[i], masks[i]) for i in range(n)])
    df = df.assign(diag=diag, method=label, noise=sigma, sample_idx=np.arange(n))
    mm = []
    for i in range(n):
        j = (i + MISMATCH_OFFSET) % n
        mk = masks[i] & masks[j]
        if mk.sum() < MISMATCH_MIN_PX:
            mm.append((np.nan, np.nan))
            continue
        s_i, e_j = std[i][mk], np.abs(mean[j][mk] - gt[j][mk])
        mm.append((float(spearmanr(s_i, e_j).statistic), float(pearsonr(s_i, e_j)[0])))
    df["rho_mismatch"], df["r_mismatch"] = np.array(mm).T
    df["rho_specific"] = df["rho_std_err"] - df["rho_mismatch"]
    df["r_specific"] = df["r_std_err"] - df["r_mismatch"]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", nargs="+", default=list(METHODS), choices=list(METHODS))
    a = ap.parse_args()

    gt = np.load(cfg.TEST_SET)
    psi = np.load(cfg.TEST_PSIS)
    trim = np.load(cfg.TEST_TRIM)
    masks = np.array([tomo_helps.define_core_mask(psi=psi[i], dim_shape=(H, W), trim_values_x=trim[i])
                      for i in range(len(gt))])

    tables = []
    for diag in a.diag:
        for sigma in cfg.NOISE_LEVELS[diag]:
            for label, method in METHODS[diag]:
                print(f"{diag} sigma={sigma} {label}")
                tables.append(per_phantom_table(gt, masks, diag, label, method, sigma))
    per_phantom = pd.concat(tables, ignore_index=True)[
        ["diag", "noise", "method", "sample_idx", "n_core", "std_mean", "gt_range"] + METRICS]

    out = cfg.figure_dir("uq_statistics")
    per_phantom.to_csv(out / "uq_metrics_per_phantom.csv", index=False)
    summary = (per_phantom.groupby(["diag", "noise", "method"], sort=False)[METRICS]
               .agg(["mean", "std", "median"]).round(3))
    summary.to_csv(out / "uq_metrics_summary.csv")
    formatted = summary.stack(level=0, future_stack=True).apply(
        lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1).unstack(level=-1)[METRICS]
    formatted.to_csv(out / "uq_metrics_summary_formatted.csv")

    # the quantities discussed in the paper text
    key = summary.xs("mean", level=1, axis=1)[["r_std_err", "r_mean_err", "cov_1sigma", "cov_2sigma"]]
    pd.set_option("display.width", 200)
    print(key.to_string())
    degenerate = per_phantom[per_phantom["std_mean"] > per_phantom["gt_range"]]
    if len(degenerate):
        print("\nruns with a degenerate posterior (core std > signal range); use medians for is_*:")
        print(degenerate[["diag", "noise", "method", "sample_idx", "std_mean"]].to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
