"""Figs. `fig:prior_sampling_process` and `fig:prior_samples` - unconditional samples of the prior.

Draws 99 DDIM samples (250 steps, eta=0) from the trained prior with seed 42 and writes
  Figures/prior_sampling_process/diffusion_process_step_{step}_{i}.eps
      (reverse-process snapshots of the first sample at steps 1, 151, 211, 235, 244, 246, 249)
  Figures/prior_samples/prior_sample_{i}.eps
      (samples 6, 8, 14, 2, 21, 43, 10)
The paper figures were produced on a CUDA GPU: the initial noise is drawn with the CUDA
generator, so the exact same samples are obtained only on a GPU (`--device cuda`, default
when available); on CPU the script runs (~1 min) but draws different samples.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402
import common  # noqa: E402
import _plot_utils as pu  # noqa: E402

import torch  # noqa: E402

PROCESS_STEPS = [1, 151, 211, 235, 244, 246, 249]
SAMPLE_IDX = [6, 8, 14, 2, 21, 43, 10]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--num-samples", type=int, default=99)
    args = ap.parse_args()

    dev = torch.device(args.device) if args.device else common.device()
    # the paper ran on a V100 (no TF32); TF32 on Ampere+ GPUs makes samples drift
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    common.seed_everything(args.seed)
    model, schedule = common.load_model(dev)
    samples, process = common.pdiff.sample_prior(
        model, schedule, image_shape=(1, cfg.H, cfg.W), num_samples=args.num_samples,
        num_inference_steps=cfg.NUM_INFERENCE_STEPS, eta=0.0, clip_range=None)

    out = cfg.figure_dir("prior_sampling_process")
    for i, step in enumerate(PROCESS_STEPS):
        pu.save_profile(process[step][0].numpy(), out / f"diffusion_process_step_{step}_{i}.eps")
    out = cfg.figure_dir("prior_samples")
    for i, idx in enumerate(SAMPLE_IDX):
        pu.save_profile(samples[idx, 0].cpu().numpy(), out / f"prior_sample_{i}.eps")
    print("wrote", out.parent / "prior_sampling_process", "and", out)


if __name__ == "__main__":
    main()
