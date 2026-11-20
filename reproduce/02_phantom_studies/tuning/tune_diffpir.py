"""Hyperparameter grid search for DiffPIR / DiffRD (paper: "grid search on a few
representative samples"; no figure or table).

For each noise level and grid point, reconstructs the first 10 test phantoms
(100 posterior samples each) and stores per-phantom PSNR/SSIM. Port of the original
diffusion_hyperparam_tuning.py (DiffPIR) and rdd_hyperparam_tuning.py (DiffRD, --rd).

Grids (as in the paper runs):
  SXR      lambda in {100..500}, zeta in {0.2..1.0}
  DMPX     lambda in {50..500},  zeta in {0.2..1.4}
  Pilatus  lambda in {50..500},  zeta in {0.2..1.4}, without and with core clipping
  DiffRD   lambda in {1,5,10,25,50,100,200}, zeta=0.9, lambda_rd in logspace(-9,-2,8),
           core clipping, alpha from the RD "by camera" tuning run (rd_cvcam_pilatus_S,
           obtained with run_rd.py --cv-strategy by_camera --sigma S --start 0 --end 10)

Noise: torch.manual_seed(i) / randn for phantom i (not the fixed noise files), as in
the original. Output: results/tuning/<diag>[_rd]/metrics_<sigma>_zeta<z>_lambda<l>[...].csv
Runtime: 10 phantoms x ~10 s per grid point on a GPU, i.e. ~10-30 GPU-h per diagnostic.
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg  # noqa: E402
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import plasma_diffusion as pdiff  # noqa: E402
from tomo_fusion.tools import helpers as tomo_helps  # noqa: E402

GRIDS = {
    "sxr": ([100, 200, 300, 400, 500], [0.2, 0.4, 0.6, 0.8, 1.0], [False]),
    "dmpx": ([50, 100, 200, 300, 400, 500], [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4], [False]),
    "pilatus": ([50, 100, 200, 300, 400, 500], [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4], [False, True]),
}
RD_GRID = ([1, 5, 10, 25, 50, 100, 200], [0.9], [True], list(np.logspace(-9, -2, 8)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", choices=list(cfg.GEOMETRY), required=True)
    ap.add_argument("--rd", action="store_true", help="DiffRD grid (Pilatus only)")
    ap.add_argument("--sigma", type=float, nargs="*", default=None)
    ap.add_argument("--num-phantoms", type=int, default=10)
    args = ap.parse_args()
    if args.rd and args.diag != "pilatus":
        ap.error("--rd only for pilatus")

    dev = common.device()
    model, schedule = common.load_model(dev)
    n = args.num_phantoms
    A = torch.from_numpy(common.load_geometry(args.diag)).to(dev)
    x_all, psis = common.load_test_set(np.arange(n))
    x_all = torch.from_numpy(x_all).to(dev)
    trims = torch.from_numpy(np.load(cfg.TEST_TRIM)[:n]).to(dev).to(torch.float32)
    y_clean = x_all.reshape(n, -1) @ A.T
    lambdas, zetas, clips, lam_rds = RD_GRID if args.rd else (*GRIDS[args.diag], [None])
    out = cfg.RESULTS / "tuning" / (f"{args.diag}_rd" if args.rd else args.diag)
    out.mkdir(parents=True, exist_ok=True)

    for sigma in args.sigma or cfg.NOISE_LEVELS[args.diag]:
        if args.rd:
            tdir = cfg.RESULTS / "phantom_studies" / f"rd_cvcam_pilatus_{sigma}"
            alphas = pd.read_csv(tdir / "metrics.csv").set_index("sample_idx").loc[range(n), "best_anis_param"].to_numpy()
        for clip, zeta, lam, lam_rd in itertools.product(clips, zetas, lambdas, lam_rds):
            name = f"metrics_{sigma}_zeta{zeta}_lambda{lam}"
            name += (f"_lambda_rd{lam_rd}" if lam_rd is not None else "") + (f"_clip{clip}" if args.diag == "pilatus" else "")
            if (out / f"{name}.csv").exists():
                continue
            rows = []
            for i in range(n):
                np.random.seed(i)
                torch.manual_seed(i)
                y = y_clean[i] + sigma * torch.randn_like(y_clean[i])
                mask = (tomo_helps.define_core_mask(psi=psis[i], dim_shape=(cfg.H, cfg.W), trim_values_x=trims[i])
                        if clip else None)
                L = pdiff.build_rd_operator(psis[i], anis_param=alphas[i])[0] if args.rd else None
                post, _, _ = pdiff.sample_posterior_diffpir(
                    model, schedule, A, y.cpu().numpy(), num_samples=cfg.NUM_POSTERIOR_SAMPLES,
                    num_inference_steps=cfg.NUM_INFERENCE_STEPS, zeta=zeta, sigma_y=sigma, lambda_=lam,
                    clipping_mask=mask, rd_matrix=L, lambda_rd=lam_rd or 0.0, progress=False)
                mean = post.mean(0).squeeze().cpu().numpy()
                rows.append({"sample_idx": i, **common.metrics(x_all[i].cpu().numpy(), mean),
                             "post_std_mean": float(post.std(0).mean()), "sigma_used": sigma,
                             "zeta": zeta, "lambda_": lam, "lambda_rd": lam_rd, "clip": clip})
            df = pd.DataFrame(rows)
            df.to_csv(out / f"{name}.csv", index=False)
            print(f"{name}: psnr {df.psnr.mean():.2f} ssim {df.ssim.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
