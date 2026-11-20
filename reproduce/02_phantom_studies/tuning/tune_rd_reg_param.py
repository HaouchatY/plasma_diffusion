"""Choice of the RD regularisation weight lambda = 0.1 (paper Sec. "RD baseline").

On the first 100 test phantoms (sigma = 0.25, true anisotropy alpha of each phantom),
finds the lambda in logspace(-3, 1, 13) minimising the MAP error, then compares the
MSE obtained with the mean / median best lambda and with the fixed lambda = 0.1.
Port of the original reaction_diffusion/phantom_studies/reg_param_tuning.py; its
outputs ship in data/rd_reg_param_tuning/reg_param_tuning_{sxr,dmpx,pilatus}/.

Needs data/test_set/alpha_random_values.npy (written by 00_prepare/make_test_set.py).
Output: results/tuning/rd_reg_param_{diag}/*.npy. Runtime: ~1-2 h CPU per diagnostic.
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg  # noqa: E402
import common  # noqa: E402,F401

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import tomo_fusion.bayesian_computations as bcomp  # noqa: E402
import tomo_fusion.functionals_definition as fct_def  # noqa: E402
import tomo_fusion.hyperparameter_tuning as hyper_tune  # noqa: E402
from tomo_fusion.tools import helpers as tomo_helps  # noqa: E402

SIGMA = 0.25
REG_PARAMS = np.logspace(-3, 1, 13)


def setup(diag, A, x, psi, alpha, trim, idx):
    mask = tomo_helps.define_core_mask(psi=psi, dim_shape=(cfg.H, cfg.W), trim_values_x=trim)
    # noise drawn inside with np.random.seed(idx)
    f, g = fct_def.define_loglikelihood_and_logprior(
        ground_truth=copy.deepcopy(x), psi=psi, fwd_matrix=A, sigma_err=SIGMA,
        reg_fct_type="anisotropic", alpha=alpha, plot=False, seed=idx)
    return f, g, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", choices=list(cfg.GEOMETRY), required=True)
    ap.add_argument("--num-phantoms", type=int, default=100)
    ap.add_argument("--alphas", type=Path, default=cfg.DATA / "test_set" / "alpha_random_values.npy")
    args = ap.parse_args()
    alpha_file = args.alphas
    if not alpha_file.exists():
        sys.exit(f"{alpha_file} missing: run 00_prepare/make_test_set.py --out <dir> and copy it")

    idx = np.arange(args.num_phantoms)
    x_all, psis = np.load(cfg.TEST_SET)[idx], np.load(cfg.TEST_PSIS)[idx]
    alphas, trims = np.load(alpha_file)[idx], np.load(cfg.TEST_TRIM)[idx]
    A = np.load(cfg.GEOMETRY[args.diag])
    A = sp.csr_matrix(A) / A.max()
    out = cfg.RESULTS / "tuning" / f"rd_reg_param_{args.diag}"
    out.mkdir(parents=True, exist_ok=True)

    data, best = [], []
    for i in idx:
        f, g, mask = setup(args.diag, A, x_all[i], psis[i], alphas[i], trims[i], i)
        d = hyper_tune.reg_param_tuning(f, g, tuning_techniques=["GT"], ground_truth=x_all[i],
                                        with_pos_constraint=True, clipping_mask=mask,
                                        cv_strategy="random", map_scaling_factor=1,
                                        reg_params=REG_PARAMS, plot=False)
        data.append(d)
        best.append(d["GT"][0, np.argmin(d["GT"][2, :])])
        print(f"phantom {i}: best lambda = {best[-1]:.3g}", flush=True)
    best = np.array(best)
    nb_occ = np.bincount([np.argmin(d["GT"][2, :]) for d in data], minlength=REG_PARAMS.size)
    lam_mean, lam_median = best.mean(), np.median(best)

    factors = {"avg": [], "median": [], "001": []}
    for k, i in enumerate(idx):
        f, g, mask = setup(args.diag, A, x_all[i], psis[i], alphas[i], trims[i], i)
        best_mse = data[k]["GT"][2].min()
        for key, lam in (("avg", lam_mean), ("median", lam_median), ("001", 1e-1)):
            im = bcomp.compute_MAP(f, g, lam, with_pos_constraint=True, clipping_mask=mask)
            factors[key].append(np.mean((im - x_all[i]) ** 2) / best_mse)

    np.save(out / "best_hyperparams.npy", best)
    np.save(out / "nb_occurrences.npy", nb_occ)
    np.save(out / "reg_param_mean.npy", lam_mean)
    np.save(out / "reg_param_median.npy", lam_median)
    for key, v in factors.items():
        np.save(out / f"factors_{key}_wrt_best.npy", np.array(v))
    print(f"lambda: mean {lam_mean:.3g}, median {lam_median:.3g}; MSE ratio vs best with "
          f"lambda=0.1: median {np.median(factors['001']):.3f}")


if __name__ == "__main__":
    main()
