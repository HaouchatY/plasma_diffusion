"""Fig. `fig:posterior_sampling_process` - DiffPIR and DiffRD reverse processes (SXR).

Test phantom 9, SXR geometry, eta_high noise (sigma_y = 0.5), seed 42. Writes to
Figures/posterior_sampling_process/:
  ground_truth.eps                          phantom with flux surfaces
  sxr_LoS_configuration_clipped.eps         phantom + SXR lines of sight (clipped to vessel)
  sxr_LoS_configuration.eps                 same, fans drawn up to the detectors (not in paper)
  sxr_tomo_data.csv                         y_true / y_noisy, plotted by pgfplots in the .tex
  diffusion_process_sxr_step_{s}_{i}.eps    DiffPIR x0-estimates of the first sample
  diffrd_process_sxr_step_{s}_{i}.eps       DiffRD x0-estimates of the first sample
DiffPIR: lambda=100, zeta=0.8. DiffRD: lambda=5, zeta=0.9, lambda_rd=1e-4, anisotropy 1e-3,
clipped to the core mask. 100 posterior samples each (only the first is shown).

The paper figure was produced on a CUDA GPU; the noise realisation and the samples come
from the CUDA generator, so they are reproduced exactly only with `--device cuda`
(default when available). On CPU the script runs (~2 min) with a different noise draw.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402
import common  # noqa: E402
import _plot_utils as pu  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tomo_fusion.tools import helpers as tomo_helps  # noqa: E402

SAMPLE_IDX = 9
SIGMA = 0.5
DIFFPIR_STEPS = [0, 35, 71, 106, 142, 177, 249]
DIFFRD_STEPS = [0, 35, 71, 106, 177, 213, 249]


def save_los_over_profile(gt, psi, path, clipped):
    """SXR lines of sight over the phantom (drawn at 75% opacity)."""
    pu.plot_profile(gt, levels=12, tcv_plot_clip=True, ax=None, contour_image=psi, alpha_imshow=0.75)
    groups = pu.LOS_GROUPS["sxr"]
    if clipped:  # every line clipped to the vessel, default view
        groups = [(g[0], g[1], g[2], None) for g in groups]
    pu.draw_los(np.load(cfg.LOS_PARAMS["sxr"]), groups, widen=not clipped)
    plt.savefig(path, format="eps", bbox_inches="tight")
    plt.close("all")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--num-samples", type=int, default=cfg.NUM_POSTERIOR_SAMPLES)
    args = ap.parse_args()
    dev = torch.device(args.device) if args.device else common.device()
    out = cfg.figure_dir("posterior_sampling_process")

    torch.backends.cudnn.deterministic = True
    # the paper ran on a V100 (no TF32); TF32 on Ampere+ GPUs makes samples drift
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model, schedule = common.load_model(dev)
    A = common.load_geometry("sxr")
    x_np, psi_np = common.load_test_set([SAMPLE_IDX])
    gt, psi = x_np[0], psi_np[0]
    trim = np.load(cfg.TEST_TRIM)[SAMPLE_IDX, :]

    # noisy data (first draw of the generator after seeding, as in the paper)
    common.seed_everything(args.seed)
    A_t = torch.from_numpy(A).to(dev)
    y_true = A_t @ torch.from_numpy(gt).to(dev).flatten()
    y = y_true + SIGMA * torch.randn(A.shape[0], device=dev)

    # ---- top row: phantom, lines of sight, data ----
    pu.save_profile(gt, out / "ground_truth.eps", psi=psi)
    save_los_over_profile(gt, psi, out / "sxr_LoS_configuration_clipped.eps", clipped=True)
    save_los_over_profile(gt, psi, out / "sxr_LoS_configuration.eps", clipped=False)
    y_np = y.cpu().numpy()
    pd.DataFrame({"y_true": y_true.cpu().numpy(), "y_noisy": y_np}).to_csv(
        out / "sxr_tomo_data.csv", index=False)

    # ---- DiffPIR ----
    _, _, process = common.pdiff.sample_posterior_diffpir(
        model, schedule, A, y_np, image_shape=(1, cfg.H, cfg.W), num_samples=args.num_samples,
        num_inference_steps=cfg.NUM_INFERENCE_STEPS, zeta=0.8, sigma_y=SIGMA, lambda_=100)
    for i, s in enumerate(DIFFPIR_STEPS):
        pu.save_profile(process[s][0].numpy(), out / f"diffusion_process_sxr_step_{s}_{i}.eps")

    # ---- DiffRD ----
    L, _ = common.pdiff.build_rd_operator(psi, anis_param=1e-3, sampling=0.0125)
    mask_core = tomo_helps.define_core_mask(psi=psi, dim_shape=(cfg.H, cfg.W), trim_values_x=trim)
    _, _, process_rd = common.pdiff.sample_posterior_diffpir(
        model, schedule, A, y_np, image_shape=(1, cfg.H, cfg.W), num_samples=args.num_samples,
        num_inference_steps=cfg.NUM_INFERENCE_STEPS, zeta=0.9, sigma_y=SIGMA, lambda_=5,
        clipping_mask=mask_core.astype(np.float32), rd_matrix=L, lambda_rd=1e-4)
    for i, s in enumerate(DIFFRD_STEPS):
        pu.save_profile(process_rd[s][0].numpy(), out / f"diffrd_process_sxr_step_{s}_{i}.eps")
    print("wrote", out)


if __name__ == "__main__":
    main()
