"""Projections of the first 1e4 training profiles through each geometry.

``y_train_{d}.npy`` (10000, M_d) = X_train[:1e4] @ A_d^T (A_d normalised by its max).
Used only by the amplitude-normalisation procedure (coefficient c* in the paper,
``--procedure`` runs). The arrays are shipped in data/, so this script is only needed
if you regenerate the training set. Requires PD_TRAIN_DIR. Runtime: ~1 min.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import common  # noqa: E402

import numpy as np  # noqa: E402


def load_training_profiles(n, train_dir=None):
    train_dir = Path(train_dir or cfg.TRAIN_DIR or "")
    if not (train_dir / "sxr_sample_0.npy").exists():
        sys.exit("Set PD_TRAIN_DIR to the directory containing sxr_sample_{i}.npy")
    return np.stack([np.load(train_dir / f"sxr_sample_{i}.npy") for i in range(n)]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=cfg.DATA)
    args = ap.parse_args()
    x = load_training_profiles(args.num).reshape(args.num, -1)
    for diag in cfg.GEOMETRY:
        y = x @ common.load_geometry(diag).T
        np.save(args.out / f"y_train_{diag}.npy", y)
        print(f"{diag}: {y.shape}")


if __name__ == "__main__":
    main()
