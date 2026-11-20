"""RD baseline phantom studies (Tables 1-3, Figs. 6-9, UQ statistics).

Model-based reconstruction with an anisotropic reaction-diffusion prior aligned with
the flux surfaces of each phantom's equilibrium (Hamm et al., PPCF 2025):
  1. anisotropy alpha tuned by 5-fold cross-validation on the data (13 values in
     logspace(-4, 0)), with regularisation weight lambda = 0.1 and the core mask;
  2. MAP (positivity-constrained) + posterior mean/std by ULA sampling.
Port of the original rd_phantom_analysis.py / rd_phantom_analysis_ula_more_samples.py.
Fully CPU / numpy and seeded per phantom: reruns are deterministic.

Paper runs (see run_all.sh):
  SXR, DMPX   --cv-strategy random, 1e5 ULA samples (+1e3 burn-in)       -> rd_{diag}_{sigma}
  Pilatus     two steps:
    a) alpha tuning: --cv-strategy random (eta_med/high) -> rd_cv_pilatus_{sigma}
                     --cv-strategy by_camera (eta_low)   -> rd_cvcam_pilatus_0.07
       (by_camera = folds made of contiguous blocks of lines of sight)
    b) --alpha-from-tuning: fixed alpha from a), 5e5 ULA samples + 1e4 burn-in
                                                          -> rd_pilatus_{sigma}
  The step-a alphas are also used by DiffRD (run_diffpir.py --method diffrd).

ULA sample count for Pilatus: the committed original script says 1e6 samples, but the
CSVs behind Table 3 record 5e5 (and ~380 s/phantom, 5x the 1e5 runs); the paper text
says 1e5. The default here (5e5 + 1e4 burn-in) is what produced the paper numbers.

Output: results/phantom_studies/<name>_{diag}_{sigma}/{metrics.csv, maps.npy,
post_means.npy, post_stds.npy} (arrays (1000, H*W) float32); with --start/--end the
results go to chunks/, merge with merge_chunks.py. Runtime per phantom on one core:
~80-100 s (1e5 ULA incl. CV), ~380 s (5e5 ULA); the original ran 20 processes x 50
phantoms (see run_all.sh), i.e. ~4 h wall per (diag, sigma) for 1e5 samples.
"""
import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import common  # noqa: E402
import _study  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import plasma_diffusion as pdiff  # noqa: E402
import tomo_fusion.bayesian_computations as bcomp  # noqa: E402
import tomo_fusion.functionals_definition as fct_def  # noqa: E402
import tomo_fusion.hyperparameter_tuning as hyper_tune  # noqa: E402
from tomo_fusion.tools import helpers as tomo_helps  # noqa: E402

ALPHA_INIT = 1e-2
SAMPLING = 0.0125
ANIS_GRID = np.logspace(-4, 0, 13)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", choices=list(cfg.GEOMETRY), required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=cfg.NUM_TEST)
    ap.add_argument("--cv-strategy", choices=["random", "by_camera"], default="random")
    ap.add_argument("--alpha-from-tuning", action="store_true",
                    help="skip CV; use the alphas of the Pilatus tuning run (rd_cv / rd_cvcam)")
    ap.add_argument("--ula-samples", type=int, default=None,
                    help="default 1e5 (5e5 with --alpha-from-tuning)")
    ap.add_argument("--burn-in", type=int, default=None,
                    help="default 1e3 (1e4 with --alpha-from-tuning)")
    ap.add_argument("--name", default=None,
                    help="study name (default: rd, rd_cv / rd_cvcam for Pilatus tuning runs)")
    ap.add_argument("--procedure", action="store_true",
                    help="amplitude-normalisation check (random rescaling, estimated sigma and c*)")
    args = ap.parse_args()

    fixed = args.alpha_from_tuning
    ula_samples = args.ula_samples or (int(5e5) if fixed else int(1e5))
    burn_in = args.burn_in or (int(1e4) if fixed else int(1e3))
    if args.name:
        name = args.name
    elif args.diag == "pilatus" and not fixed:
        name = "rd_cvcam" if args.cv_strategy == "by_camera" else "rd_cv"
    else:
        name = "rd"
    out = _study.study_path(name, args.diag, args.sigma, args.procedure)

    idx = np.arange(args.start, args.end)
    if fixed:
        tune = "rd_cvcam" if args.sigma == 0.07 else "rd_cv"
        tdir = _study.study_path(tune, args.diag, args.sigma, args.procedure)
        if not (tdir / "metrics.csv").exists():
            sys.exit(f"{tdir}/metrics.csv missing: run the alpha-tuning step first")
        alphas = pd.read_csv(tdir / "metrics.csv").set_index("sample_idx").loc[idx, "best_anis_param"].to_numpy()

    # original numerics: float64 geometry and phantoms, float32 noise
    A = np.load(cfg.GEOMETRY[args.diag])
    A /= A.max()
    A_csr = sp.csr_matrix(A)
    x_all = np.load(cfg.TEST_SET)[idx]
    psis = np.load(cfg.TEST_PSIS)[idx]
    trims = np.load(cfg.TEST_TRIM)[idx]
    noise = np.load(cfg.NOISE[args.diag])[idx]
    y_clean = x_all.reshape(len(idx), -1) @ A.T
    if args.procedure:
        scaling = np.load(cfg.SCALING)[idx]
        y_train = np.load(cfg.Y_TRAIN[args.diag]).astype(np.float64)

    print(f"RD {args.diag} sigma={args.sigma} cv={'fixed' if fixed else args.cv_strategy} "
          f"ULA {ula_samples}+{burn_in} phantoms {args.start}-{args.end} -> {out}")
    rows, maps, means, stds = [], [], [], []
    for k, i in enumerate(idx):
        np.random.seed(i)
        y = y_clean[k] + args.sigma * noise[k]
        x_gt = copy.deepcopy(x_all[k])
        if args.procedure:
            scale = float(scaling[k])
            y, x_gt = y * scale, x_gt * scale
            c = pdiff.normalization_coefficient(y, y_train)
            sigma_used = pdiff.estimate_noise_std(y) / c
            y = y / c
        else:
            scale, c, sigma_used = 1.0, 1.0, args.sigma

        mask_core = tomo_helps.define_core_mask(psi=psis[k], dim_shape=(cfg.H, cfg.W), trim_values_x=trims[k])
        f, g = fct_def.define_loglikelihoodfromdata_and_logprior(
            y, psis[k], fwd_matrix=A_csr, reconstruction_shape=(1, cfg.H, cfg.W),
            sigma_err=sigma_used, reg_fct_type="anisotropic",
            alpha=alphas[k] if fixed else ALPHA_INIT, sampling=SAMPLING, seed=k)

        if fixed:
            best_alpha, best_idx = float(alphas[k]), -1
        else:
            key = f"CV_full_{args.cv_strategy}"
            cv = hyper_tune.anis_param_tuning(
                f, g, reg_param=cfg.RD_REG_PARAM, tuning_techniques=["CV_full"],
                with_pos_constraint=True, clipping_mask=mask_core,
                cv_strategy=[args.cv_strategy], anis_params=ANIS_GRID, plot=False)[key]
            best_idx = int(np.argmin(cv[1, :]))
            best_alpha = float(cv[0, best_idx])
            g = hyper_tune._redefine_anis_param_logprior(g, best_alpha)

        t0 = time.time()
        uq = bcomp.run_ula(f, g, cfg.RD_REG_PARAM, psis[k], trims[k], with_pos_constraint=True,
                           clip_iterations="core", compute_stats_wrt_MAP=True,
                           estimate_quantiles=False, estimate_tomo_data_stats=False,
                           estimate_peak_location=False, samples=ula_samples, burn_in=burn_in)
        t_ula = time.time() - t0

        im_map, im_mean, im_std = uq["im_MAP"] * c, uq["mean"] * c, np.sqrt(uq["var"]) * c
        m_map, m_mean = common.metrics(x_gt, im_map), common.metrics(x_gt, im_mean)
        rows.append({
            "sample_idx": int(i), "sigma_y_tomo": args.sigma, "procedure": args.procedure,
            "scale": scale, "alpha_norm": float(c), "sigma_used": float(sigma_used),
            "reg_param": cfg.RD_REG_PARAM, "reg_fct_type": "anisotropic", "sampling_param": SAMPLING,
            "cv_strategy": "fixed" if fixed else args.cv_strategy,
            "best_anis_param": best_alpha, "best_anis_idx": best_idx,
            "ula_samples": ula_samples, "ula_burn_in": burn_in, "time_ula": t_ula,
            **{f"{key}_map": v for key, v in m_map.items()},
            **{f"{key}_mean": v for key, v in m_mean.items()},
            "post_std_mean": float(im_std.mean()),
        })
        maps.append(im_map.ravel())
        means.append(im_mean.ravel())
        stds.append(im_std.ravel())
        print(f"  phantom {i}: alpha={best_alpha:.4g} psnr_map={m_map['psnr']:.2f} "
              f"psnr_mean={m_mean['psnr']:.2f} ({t_ula:.0f} s)", flush=True)

    _study.save(out, args.start, args.end, rows,
                {"maps": maps, "post_means": means, "post_stds": stds})


if __name__ == "__main__":
    main()
