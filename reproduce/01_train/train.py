"""Retrain the diffusion prior (paper Sec. "Training"; checkpoint data/prior_sxr.pth).

Same setup as the paper run (original ``run_diffusion.py`` + ``utils.py``):
TinyUNet noise predictor (318,764 parameters), DDPM with T=1000 linear betas in
[1e-4, 2e-2], epsilon-prediction MSE loss, Adam lr 1e-3, batch size 512, shuffled
1e5 training profiles (120x40, float32). The paper checkpoint is the one after
epoch index 2193 (i.e. 2194 epochs; the original run was stopped manually),
saved here as ``prior_epoch_2193.pth``.

The 1e5 training profiles (3.9 GB) are not shipped with the repository; point
PD_TRAIN_DIR at a directory of ``sxr_sample_{i}.npy`` files.

Runtime: ~16-19 h on one V100 (~30 s/epoch). GPU training is not bit-reproducible.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import common  # noqa: E402

import numpy as np  # noqa: E402
from plasma_diffusion.training import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", type=Path, default=cfg.TRAIN_DIR,
                    help="directory with sxr_sample_{i}.npy (default: $PD_TRAIN_DIR)")
    ap.add_argument("--num-profiles", type=int, default=100000)
    ap.add_argument("--epochs", type=int, default=2194)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--out", type=Path, default=cfg.RESULTS / "training")
    args = ap.parse_args()
    if args.train_dir is None or not (args.train_dir / "sxr_sample_0.npy").exists():
        sys.exit("Training data not found: set PD_TRAIN_DIR or pass --train-dir")

    print(f"loading {args.num_profiles} profiles from {args.train_dir}")
    profiles = np.stack([np.load(args.train_dir / f"sxr_sample_{i}.npy")
                         for i in range(args.num_profiles)]).astype(np.float32)
    common.seed_everything()
    train(profiles, args.out, num_epochs=args.epochs, batch_size=512, lr=1e-3,
          num_train_steps=cfg.NUM_TRAIN_STEPS, checkpoint_every=args.checkpoint_every)
    print(f"checkpoints in {args.out}; use prior_epoch_{args.epochs - 1:04d}.pth "
          f"in place of {cfg.CHECKPOINT.name}")


if __name__ == "__main__":
    main()
