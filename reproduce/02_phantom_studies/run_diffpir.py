"""DiffPIR / DiffRD_clip / DiffRD phantom studies (Tables 1-3, Figs. 6-9, UQ statistics).

For each of the 1000 test phantoms: y = A x + sigma n (fixed noise draws), draw 100
posterior samples (250 DDIM steps), store the posterior mean/std and PSNR/SSIM of
the mean. Port of the original diffusion_{sxr,dmpx,pilatus}_phantom_analysis.py and
diffrd_pilatus_phantom_analysis_lownoise.py.

Methods (hyperparameters in config.py):
  diffpir      DiffPIR, all diagnostics                         (lambda_, zeta) = cfg.DIFFPIR
  diffrd_clip  DiffPIR + clipping to the plasma core (Pilatus)  cfg.DIFFRD_CLIP
  diffrd       DiffRD_clip + anisotropic RD term lambda_rd ||L x||^2 (Pilatus), cfg.DIFFRD.
               L is built from each phantom's equilibrium with the anisotropy alpha
               tuned by the RD baseline: needs the RD alpha-tuning runs of run_rd.py
               (rd_cv_pilatus_{0.35,0.7}, rd_cvcam_pilatus_0.07).

Output: results/phantom_studies/{method}_{diag}_{sigma}/{metrics.csv,post_means.npy,post_stds.npy}
(chunks with --start/--end, merge with merge_chunks.py).

Runtime per phantom (100 samples): ~10 s DiffPIR, ~14 s DiffRD on a V100; i.e. ~3 GPU-h
per (method, diag, sigma). On CPU expect ~1-2 min per phantom.

Reproducibility: the original runs seeded torch once (seed 42) and processed phantoms
0..999 sequentially in one process on a CUDA GPU; the same is done here when the full
range is run in one process. Chunked runs reseed at the chunk start, so individual
posterior draws differ (statistics are unaffected).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import common  # noqa: E402
import _study  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import plasma_diffusion as pdiff  # noqa: E402
from tomo_fusion.tools import helpers as tomo_helps  # noqa: E402

RD_SAMPLING = 0.0125


def rd_alphas(sigma: float, procedure: bool) -> np.ndarray:
    """Per-phantom anisotropy alpha tuned by the RD baseline (Pilatus).

    eta_low (0.07) uses the 'CV by camera' tuning, eta_med/high the random-CV tuning,
    exactly as in the paper runs. If those RD tuning runs are not available, the values
    used in the paper (data/rd_anis_params_pilatus.csv) are used instead."""
    name = "rd_cvcam" if sigma == 0.07 else "rd_cv"
    path = _study.study_path(name, "pilatus", sigma, procedure) / "metrics.csv"
    if path.exists():
        return pd.read_csv(path).sort_values("sample_idx")["best_anis_param"].to_numpy()
    shipped = cfg.DATA / "rd_anis_params_pilatus.csv"
    if procedure or not shipped.exists():
        sys.exit(f"{path} missing: run run_rd.py --diag pilatus --sigma {sigma} "
                 f"--cv-strategy {'by_camera' if sigma == 0.07 else 'random'} first")
    print(f"using paper anisotropy parameters from {shipped} "
          f"(rerun run_rd.py to recompute them, ~80 CPU-h per noise level)")
    df = pd.read_csv(shipped)
    return df[df.sigma == sigma].sort_values("sample_idx")["best_anis_param"].to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["diffpir", "diffrd_clip", "diffrd"], required=True)
    ap.add_argument("--diag", choices=list(cfg.GEOMETRY), required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=cfg.NUM_TEST)
    ap.add_argument("--num-samples", type=int, default=cfg.NUM_POSTERIOR_SAMPLES)
    ap.add_argument("--procedure", action="store_true",
                    help="amplitude-normalisation check: rescale data by random factors and "
                         "estimate sigma and the normalisation coefficient c* from the data")
    ap.add_argument("--out", type=Path, default=None, help="override output directory")
    args = ap.parse_args()
    if args.method != "diffpir" and args.diag != "pilatus":
        ap.error("diffrd/diffrd_clip were only run on Pilatus")

    dev = common.device()
    common.seed_everything(cfg.SEED)
    torch.backends.cudnn.deterministic = True     # as in the original runs
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model, schedule = common.load_model(dev)

    if args.method == "diffpir":
        lam, zeta = cfg.DIFFPIR[args.diag]
        lam_rd = None
    elif args.method == "diffrd_clip":
        (lam, zeta), lam_rd = cfg.DIFFRD_CLIP, None
    else:
        lam, zeta, lam_rd = cfg.DIFFRD[args.sigma]
        alphas = rd_alphas(args.sigma, args.procedure)

    idx = np.arange(args.start, args.end)
    A = torch.from_numpy(common.load_geometry(args.diag)).to(dev)
    x_all, psis = common.load_test_set(idx)
    x_all = torch.from_numpy(x_all).to(dev)
    trims = torch.from_numpy(np.load(cfg.TEST_TRIM)[idx]).to(dev).to(torch.float32)
    noise = torch.from_numpy(np.load(cfg.NOISE[args.diag])[idx]).to(dev).to(torch.float32)
    y_clean = x_all.reshape(len(idx), -1) @ A.T
    if args.procedure:
        scaling = np.load(cfg.SCALING)[idx].astype(np.float32)
        y_train = np.load(cfg.Y_TRAIN[args.diag])

    out = args.out or _study.study_path(args.method, args.diag, args.sigma, args.procedure)
    print(f"{args.method} {args.diag} sigma={args.sigma} lambda={lam} zeta={zeta} "
          f"lambda_rd={lam_rd} phantoms {args.start}-{args.end} -> {out}")

    rows, means, stds = [], [], []
    for k, i in enumerate(idx):
        y = y_clean[k] + args.sigma * noise[k]
        x_gt = x_all[k]
        if args.procedure:
            scale = float(scaling[k])
            y, x_gt = y * scale, x_gt * scale
            y_np = y.cpu().numpy()
            c = pdiff.normalization_coefficient(y_np, y_train)
            sigma_used = pdiff.estimate_noise_std(y_np) / c
            y = y / c
        else:
            scale, c, sigma_used = 1.0, 1.0, args.sigma

        mask = rd_mat = None
        if args.method != "diffpir":
            mask = tomo_helps.define_core_mask(psi=psis[k], dim_shape=(cfg.H, cfg.W),
                                               trim_values_x=trims[k])
        if args.method == "diffrd":
            rd_mat, _ = pdiff.build_rd_operator(psis[k], anis_param=alphas[i], sampling=RD_SAMPLING)

        post, _, _ = pdiff.sample_posterior_diffpir(
            model, schedule, A, y.cpu().numpy(), image_shape=(1, cfg.H, cfg.W),
            num_samples=args.num_samples, num_inference_steps=cfg.NUM_INFERENCE_STEPS,
            zeta=zeta, sigma_y=sigma_used, lambda_=lam, clipping_mask=mask,
            rd_matrix=rd_mat, lambda_rd=lam_rd if lam_rd is not None else 0.0, progress=False)

        mean = (post.mean(dim=0).squeeze() * c).cpu().numpy()
        std = (post.std(dim=0).squeeze() * c).cpu().numpy()
        gt = x_gt.cpu().numpy()
        row = {"sample_idx": int(i), **common.metrics(gt, mean), "post_std_mean": float(std.mean()),
               "sigma_used": float(sigma_used), "alpha": float(c), "scaling": scale,
               "lambda_": lam, "zeta": zeta}
        if args.method != "diffpir":
            row["clip"] = True
        if lam_rd is not None:
            row.update(lambda_rd=lam_rd, anis_param=float(alphas[i]))
        rows.append(row)
        means.append(mean.ravel())
        stds.append(std.ravel())
        print(f"  phantom {i}: psnr={row['psnr']:.2f} ssim={row['ssim']:.3f}", flush=True)

    _study.save(out, args.start, args.end, rows, {"post_means": means, "post_stds": stds})


if __name__ == "__main__":
    main()
