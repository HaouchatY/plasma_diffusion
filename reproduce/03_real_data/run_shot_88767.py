"""TCV discharge 88767 (N2 injection): background and fluctuation reconstructions.

Produces the data behind Fig. "real_data_88767_fluct". Uses the 0.1 ms time grid
(shot_data_full_88767_finetimegrid.mat, 2000 frames from t~0.9 s):

* background: raw measurements at frames 900, 1000, ..., 1900 (t = 0.99 ... 1.09 s)
  -> results/real_data/88767_baseline/
* fluctuations: measurements minus the pre-injection reference (mean of frames
  1000-1049, t = 1.000-1.005 s) at frames 1350, 1360, ..., 1440 (t = 1.035 ... 1.044 s)
  -> results/real_data/88767_fluct/

Both are reconstructed with RD (MAP, CV-tuned anisotropy) and DiffPIR exactly as
for shot 85270 (sigma = 0.25, lambda = 100, zeta = 0.8, core mask).

Runtime: 21 slices, ~1 min each (CPU for RD, GPU for DiffPIR).

    python reproduce/03_real_data/run_shot_88767.py [--part baseline fluct]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _real_data as rd  # noqa: E402
import config as cfg  # noqa: E402

BASELINE_FRAMES = np.arange(900, 2000)[::100]            # 11 slices, 10 ms apart
FLUCT_FRAMES = np.arange(1000, 2000)[::10][35:45]        # 10 slices, 1 ms apart
REFERENCE_FRAMES = slice(1000, 1050)                      # pre-injection background


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", nargs="+", choices=["baseline", "fluct"], default=["baseline", "fluct"])
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="restrict to these positions within the selected frame list (for testing)")
    ap.add_argument("--num-samples", type=int, default=cfg.NUM_POSTERIOR_SAMPLES)
    ap.add_argument("--suffix", default="", help="appended to the output directory names")
    args = ap.parse_args()

    data, psi = rd.load_shot(88767)
    meas = data["measurements"]
    sel = (lambda fr: fr) if args.frames is None else (lambda fr: fr[np.asarray(args.frames)])

    if "baseline" in args.part:
        rd.run_slices("88767_baseline" + args.suffix, data, psi, meas, sel(BASELINE_FRAMES),
                      num_samples=args.num_samples)
    if "fluct" in args.part:
        fluct = meas - meas[REFERENCE_FRAMES].mean(axis=0, keepdims=True)
        rd.run_slices("88767_fluct" + args.suffix, data, psi, fluct, sel(FLUCT_FRAMES),
                      num_samples=args.num_samples)


if __name__ == "__main__":
    main()
