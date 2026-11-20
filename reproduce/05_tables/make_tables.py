"""Tables 1-3 (tab:sxr, tab:dmpx, tab:pilatus): PSNR / SSIM over the 1000 test phantoms.

Each cell is mean +- std over the test set (pandas std, ddof=1). The best mean per
metric and noise level is set in bold, and starred when the best method is better
than *each* competitor according to a one-sided paired Wilcoxon signed-rank test at
p < 0.05 on the per-phantom values (scipy.stats.wilcoxon, alternative="greater").

PSNR is printed with one decimal (rounded). The paper prints SSIM with two decimals
by *truncation* (e.g. 0.996 -> 0.99), which is the default here (--ssim-format paper);
--ssim-format round uses ordinary rounding.

Inputs:  results/phantom_studies/{diffpir,rd,diffrd_clip,diffrd}_{diag}_{sigma}/metrics.csv
Outputs: figures/Tables/{sxr,dmpx,pilatus}.tex  and  figures/Tables/wilcoxon_pvalues.csv
Runtime: seconds (CPU).

    python make_tables.py
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import common  # noqa: E402

import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

# table rows: (label, results method, column prefix in metrics.csv -> {psnr,ssim}{suffix})
ROWS = {
    "sxr": [("RD (mean)", "rd", "_mean"), ("DiffPIR", "diffpir", "")],
    "dmpx": [("RD (mean)", "rd", "_mean"), ("DiffPIR", "diffpir", "")],
    "pilatus": [("RD (mean)", "rd", "_mean"), ("RD (MAP)", "rd", "_map"),
                ("DiffPIR", "diffpir", ""), (r"DiffRD$_{\text{clip}}$", "diffrd_clip", ""),
                ("DiffRD", "diffrd", "")],
}
TITLES = {"sxr": "SXR", "dmpx": "DMPX", "pilatus": "Pilatus"}
ALPHA = 0.05


def per_phantom(diag, sigma, metric):
    """{row label: per-phantom metric values (Series indexed by sample_idx)}."""
    out = {}
    for label, method, suffix in ROWS[diag]:
        df = common.load_study(method, diag, sigma, arrays=())["metrics"]
        out[label] = df.set_index("sample_idx")[metric + suffix]
    return out


def best_and_star(values):
    """Label of the best mean, whether it beats every competitor (Wilcoxon), and p-values."""
    means = {k: v.mean() for k, v in values.items()}
    best = max(means, key=means.get)
    pvals = {}
    for k, v in values.items():
        if k == best:
            continue
        a, b = values[best].align(v, join="inner")
        pvals[k] = wilcoxon(a.values, b.values, alternative="greater").pvalue
    return best, all(p < ALPHA for p in pvals.values()), pvals


def fmt(metric, mean, std, ssim_format):
    if metric == "psnr":
        return f"{mean:.1f} \\pm {std:.1f}"
    if ssim_format == "paper":
        trunc = lambda x: math.floor(x * 100 + 1e-9) / 100  # noqa: E731
        return f"{trunc(mean):.2f} \\pm {max(trunc(std), 0.01):.2f}"
    return f"{mean:.2f} \\pm {std:.2f}"


def make_table(diag, ssim_format):
    sigmas = cfg.NOISE_LEVELS[diag]
    labels = [r[0] for r in ROWS[diag]]
    width = max(len(lbl) for lbl in labels)
    cells, stats = {}, []
    for metric in ("psnr", "ssim"):
        for sigma in sigmas:
            vals = per_phantom(diag, sigma, metric)
            best, star, pvals = best_and_star(vals)
            for lbl, v in vals.items():
                s = fmt(metric, v.mean(), v.std(), ssim_format)
                if lbl == best:
                    s = f"\\mathbf{{{s}}}" + ("^*" if star else "")
                cells[(metric, sigma, lbl)] = f"${s}$"
            stats += [{"diag": diag, "metric": metric, "sigma": sigma, "best": best,
                       "vs": k, "p_value": p, "significant": p < ALPHA} for k, p in pvals.items()]

    n = len(labels)
    lines = [r"\begin{table}[t!]", r"\centering", r"\begin{tabular}{llccc}", r"\toprule",
             r"Metric & Method & $\eta_{\text{low}}$ & $\eta_{\text{medium}}$ & $\eta_{\text{high}}$ \\"]
    for metric in ("psnr", "ssim"):
        lines += [r"\midrule", f"\\multirow{{{n}}}{{*}}{{{metric.upper()}}}"]
        for lbl in labels:
            row = " & ".join(f"{cells[(metric, s, lbl)]:<26}" for s in sigmas)
            lines.append(f" & {lbl:<{width}} & {row.rstrip()} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\captionsetup{justification=justified,singlelinecheck=false}",
              f"\\caption{{{TITLES[diag]} reconstruction metrics by method and noise level, reporting "
              "average and standard deviation over the test set. Best value in bold; $^*$ denotes that "
              "the best-performing method is, statistically, significantly better than the other.}",
              f"\\label{{tab:{diag}}}", r"\end{table}"]
    return "\n".join(lines) + "\n", stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ssim-format", choices=["paper", "round"], default="paper")
    a = ap.parse_args()
    out = cfg.figure_dir("Tables")
    all_stats = []
    for diag in ROWS:
        tex, stats = make_table(diag, a.ssim_format)
        (out / f"{diag}.tex").write_text(tex)
        all_stats += stats
        print(f"wrote {out / f'{diag}.tex'}")
    df = pd.DataFrame(all_stats)
    df.to_csv(out / "wilcoxon_pvalues.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
