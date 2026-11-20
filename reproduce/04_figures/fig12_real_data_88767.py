"""Fig. "real_data_88767_fluct": background (t=1.000 s) and fluctuation reconstructions
(t=1.037-1.041 s) of TCV #88767, RD (MAP) vs DiffPIR posterior mean.

Reads results/real_data/{88767_baseline,88767_fluct} (reproduce/03_real_data/run_shot_88767.py)
and writes figures/real_data/shot88767_{baseline_t1000ms,fluct_t10{37..41}ms}_{rd,diffpir}.eps.
Fluctuation panels use viridis clipped at 0; RD panels are scaled individually, DiffPIR panels
share one vmax (max over the 10 fluctuation slices). CPU, seconds.

--as-published reproduces the panels exactly as they appear in the submitted paper. The
original plotting notebook selected the *times* [1:2] / [2:7] but took the *images* from
index 0, so each panel shows the reconstruction one slot earlier than its label
(background 0.990 s instead of 1.000 s; fluctuations 1.035-1.039 s instead of 1.037-1.041 s),
drawn with the flux surfaces of the labelled time. Default: images match their labels.

    python reproduce/04_figures/fig12_real_data_88767.py [--as-published]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_real_data"))

import numpy as np  # noqa: E402

import _real_data as rd  # noqa: E402
import config as cfg  # noqa: E402

BASELINE_SEL = slice(1, 2)    # t = 1.000 s
FLUCT_SEL = slice(2, 7)       # t = 1.037 ... 1.041 s


def physical_means(run):
    return np.asarray(run["posterior_tomo"]).mean(axis=1).squeeze(1) * run["params"][:, 0, None, None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-published", action="store_true")
    ap.add_argument("--overview", action="store_true")
    args = ap.parse_args()

    _, psi = rd.load_shot(88767)
    out = cfg.figure_dir("real_data")

    for name, sel, tag in [("88767_baseline", BASELINE_SEL, "baseline"), ("88767_fluct", FLUCT_SEL, "fluct")]:
        run = rd.load_run(name)
        maps = run["im_maps"].squeeze((1, 2))
        means = physical_means(run)
        times, frames = run["inv_times"][sel], run["frames"][sel]
        img_idx = np.arange(len(run["frames"]))[sel]
        if args.as_published:
            img_idx = img_idx - sel.start
        if tag == "fluct":
            # as in the paper: RD panels scaled individually, DiffPIR panels share the
            # maximum over all 10 fluctuation slices of the run
            kw_rd = dict(vmin=0)
            kw_dp = dict(vmin=0, vmax=np.abs(means).max())
        else:
            kw_rd = kw_dp = {}
        for t, fr, i in zip(times, frames, img_idx):
            t_ms = int(round(t * 1000))
            rd.save_profile_eps(maps[i], psi[fr], out / f"shot88767_{tag}_t{t_ms}ms_rd.eps", **kw_rd)
            rd.save_profile_eps(means[i], psi[fr], out / f"shot88767_{tag}_t{t_ms}ms_diffpir.eps", **kw_dp)
        if args.overview:
            titles = [f"t={t:.3f}s" for t in times]
            psis = [psi[fr] for fr in frames]
            rd.save_overview([maps[i] for i in img_idx], psis, titles, f"#88767 {tag} RD (MAP)",
                             out / f"shot88767_{tag}_rd_overview.png", **kw_rd)
            rd.save_overview([means[i] for i in img_idx], psis, titles, f"#88767 {tag} DiffPIR mean",
                             out / f"shot88767_{tag}_diffpir_overview.png", **kw_dp)
    print(f"saved #88767 panels to {out}")


if __name__ == "__main__":
    main()
