"""Fig. "real_data_85270": RD (MAP) vs DiffPIR posterior mean, TCV #85270, 7 time slices.

Reads results/real_data/85270 (from reproduce/03_real_data/run_shot_85270.py) and writes
figures/real_data/shot85270_core_mask_t{240,...,1020}ms_{rd,diffpir}.eps. CPU, seconds.

    python reproduce/04_figures/fig10_real_data_85270.py [--overview]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_real_data"))

import numpy as np  # noqa: E402

import _real_data as rd  # noqa: E402
import config as cfg  # noqa: E402

SLICES = [3, 6, 10, 11, 12, 14, 16]   # t = 0.24, 0.42, 0.66, 0.72, 0.78, 0.90, 1.02 s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="85270", help="results/real_data sub-directory")
    ap.add_argument("--overview", action="store_true", help="also save a one-row PNG overview")
    args = ap.parse_args()

    run = rd.load_run(args.run, mmap=True)
    _, psi = rd.load_shot(85270)
    out = cfg.figure_dir("real_data")
    alpha = run["params"][:, 0]
    pos = {int(f): k for k, f in enumerate(run["frames"])}

    maps, means, psis, titles = [], [], [], []
    for s in SLICES:
        k = pos[s]
        t_ms = int(round(run["inv_times"][k] * 1000))
        im_map = run["im_maps"][k].squeeze()
        mean = np.asarray(run["posterior_tomo"][k]).mean(axis=0).squeeze() * alpha[k]   # back to physical units
        rd.save_profile_eps(im_map, psi[s], out / f"shot85270_core_mask_t{t_ms}ms_rd.eps")
        rd.save_profile_eps(mean, psi[s], out / f"shot85270_core_mask_t{t_ms}ms_diffpir.eps")
        maps.append(im_map), means.append(mean), psis.append(psi[s]), titles.append(f"t={t_ms / 1000:.2f}s")
    if args.overview:
        rd.save_overview(maps, psis, titles, "#85270 RD (MAP)", out / "shot85270_core_mask_rd_overview.png")
        rd.save_overview(means, psis, titles, "#85270 DiffPIR mean", out / "shot85270_core_mask_diffpir_overview.png")
    print(f"saved {2 * len(SLICES)} panels to {out}")


if __name__ == "__main__":
    main()
