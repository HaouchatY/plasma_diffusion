"""Fig. "real_data_posterior_samples": DiffPIR std and six posterior samples, #85270 at t=0.66 s.

Reads results/real_data/85270 and writes figures/real_data_posterior_samples/
shot85270_std.eps and shot85270_core_mask_t660ms_diffpir_sample{2,3,4,5,7,8}.eps.
Samples and std are shown in normalized units (common colour scale starting at 0). CPU, seconds.

    python reproduce/04_figures/fig11_real_data_posterior_samples.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_real_data"))

import numpy as np  # noqa: E402

import _real_data as rd  # noqa: E402
import config as cfg  # noqa: E402

SLICE = 10                                   # t = 0.66 s
# panel number n in the file name -> posterior sample index 2*(n-1)+6 (samples 8, 10, ..., 20)
PANELS = [2, 3, 4, 5, 7, 8]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="85270")
    args = ap.parse_args()

    run = rd.load_run(args.run, mmap=True)
    _, psi = rd.load_shot(85270)
    out = cfg.figure_dir("real_data_posterior_samples")
    k = {int(f): i for i, f in enumerate(run["frames"])}[SLICE]
    post = np.asarray(run["posterior_tomo"][k])[:, 0]            # (100, H, W), normalized units
    t_ms = int(round(run["inv_times"][k] * 1000))

    rd.save_profile_eps(post.reshape(len(post), -1).std(axis=0).reshape(post.shape[1:]), psi[SLICE],
                        out / "shot85270_std.eps", vmin=0, cmap="inferno")
    for n in PANELS:
        rd.save_profile_eps(post[2 * (n - 1) + 6], psi[SLICE],
                            out / f"shot85270_core_mask_t{t_ms}ms_diffpir_sample{n}.eps", vmin=0)
    print(f"saved {1 + len(PANELS)} panels to {out}")


if __name__ == "__main__":
    main()
