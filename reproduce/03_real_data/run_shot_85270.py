"""TCV discharge 85270: RD (MAP) and DiffPIR reconstructions of all 28 SXR time slices.

Produces the data behind Fig. "real_data_85270" and Fig. "real_data_posterior_samples".
Output: results/real_data/85270/ (layout documented in _real_data.py).

Runtime: per slice ~1 min CV tuning + MAP (CPU) and ~10 s DiffPIR (100 samples, GPU);
the full shot takes ~30-40 min on one GPU + CPU (much longer for DiffPIR on CPU).

    python reproduce/03_real_data/run_shot_85270.py [--frames 10] [--num-samples 100]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _real_data as rd  # noqa: E402
import config as cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="subset of time-slice indices (default: all 28)")
    ap.add_argument("--num-samples", type=int, default=cfg.NUM_POSTERIOR_SAMPLES)
    ap.add_argument("--name", default="85270", help="output sub-directory of results/real_data")
    args = ap.parse_args()

    data, psi = rd.load_shot(85270)
    meas = data["measurements"]
    frames = np.arange(meas.shape[0]) if args.frames is None else np.asarray(args.frames)
    rd.run_slices(args.name, data, psi, meas, frames, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
