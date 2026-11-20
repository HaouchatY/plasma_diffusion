"""Generate the 1000-phantom synthetic SXR test set (paper Sec. "Synthetic data", Fig. 2).

Each phantom is a point source diffused anisotropically along the flux surfaces of a
randomly stretched TCV equilibrium (data/test_set/magnetic_equilibrium.npy) on a
240x80 grid, clipped to the plasma core, plus a Gaussian background, then resized
to 120x40. Port of the original ``test_set_generation/generate_sxr_phantoms.py``
(seeds 5032026 for the background amplitudes, 1200603 for the phantoms).

Outputs (in --out, default data/test_set, i.e. overwrites the shipped arrays):
    sxr_samples_with_background_coarse.npy  (N, 120, 40)  ground-truth phantoms
    psis_coarse.npy                         (N, 120, 40)  equilibria
    trimming_values.npy                     (N, 4)
    alpha_random_values.npy                 (N,)          true anisotropy of each phantom
                                                          (used by tuning/tune_rd_reg_param.py)

Runtime: ~20-30 min on CPU for 1000 phantoms (up to 1e4 sparse diffusion steps each);
``--gpu`` uses cupy. Phantoms are generated sequentially from one RNG stream, so
``--num k`` reproduces exactly the first k phantoms.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import common  # noqa: E402,F401

import numpy as np  # noqa: E402
import skimage.transform as skimt  # noqa: E402
from pyxu_diffops.operator import AnisDiffusionOp  # noqa: E402


def define_core_mask(psi, dim_shape, xpoint_idx_base_psi=90, trim_values_x=(0, 120)):
    if psi.shape != dim_shape:
        psi = skimt.resize(psi, dim_shape, anti_aliasing=False, mode="edge")
    mask = np.where(psi < 0)
    xpoint_loc = int(dim_shape[0] * (xpoint_idx_base_psi - trim_values_x[0])
                     / (trim_values_x[1] - trim_values_x[0]))
    keep = mask[0] < xpoint_loc
    mask_core = np.zeros(dim_shape, dtype=bool)
    mask_core[mask[0][keep], mask[1][keep]] = True
    return mask_core


def add_gaussian_background(phantom, psi, mask, peak_value, sigma_rel_radius=0.5):
    g = peak_value * np.exp(-((psi - psi.min()) / psi.min()) ** 2 / (2 * sigma_rel_radius ** 2))
    out = phantom + g * mask
    return out / out.max()


def generate(psi_fct, nsamples, peak_values, seed=1200603, dim_shape=(1, 240, 80),
             sampling=0.00625, steps_nb_factor=4, psi_upper_rel=0.4, psi_lower_rel=1.0,
             xpoint_idx_base_psi=90, gpu=False, verbose=True):
    if gpu:
        import cupy as cp
    psi_up = psi_upper_rel * psi_fct.min()
    psi_lo = psi_lower_rel * psi_fct.min()
    np.random.seed(seed)
    upper_t, lower_t, hfs_t, lfs_t = 37, 27, 5, 3
    trimming = np.zeros((nsamples, 4))
    alphas = np.zeros(nsamples)
    alpha_bounds = np.array([np.log(5e-4), np.log(5e-1)])
    steps_bounds = steps_nb_factor * np.array([int(2.5e2), int(2.5e3)])
    phantoms, psis = [], []
    for i in range(nsamples):
        # random stretch of the base equilibrium
        up = int(upper_t * np.random.rand())
        lo = int(120 - lower_t * np.random.rand())
        hfs = int(hfs_t * np.random.rand())
        lfs = int(40 - lfs_t * np.random.rand())
        trimming[i] = [up, lo, hfs, lfs]
        psi = skimt.resize(psi_fct[up:lo, hfs:lfs], dim_shape[1:], anti_aliasing=True, mode="edge")

        # random source location between two flux surfaces, above the X-point
        locs = np.where(np.logical_and(psi <= psi_up, psi >= psi_lo))
        xpoint_loc = int(dim_shape[1] * (xpoint_idx_base_psi - trimming[i, 0])
                         / (trimming[i, 1] - trimming[i, 0]))
        keep = locs[0] < xpoint_loc
        locs = (locs[0][keep], locs[1][keep])
        src = np.random.randint(0, len(locs[0]) - 1, size=(1,))[0]
        x0 = np.zeros(dim_shape[1:], dtype=psi_fct.dtype)
        x0[locs[0][src]:locs[0][src] + 2, locs[1][src]:locs[1][src] + 2] = 1

        alpha = alphas[i] = np.exp(np.random.uniform(alpha_bounds[0], alpha_bounds[1]))
        nsteps = np.random.randint(steps_bounds[0], steps_bounds[1])
        mask_core = define_core_mask(psi, dim_shape[1:], trim_values_x=[trimming[i, 0], trimming[i, 1]])
        if verbose:
            print(f"phantom {i}: psi_src={psi[locs[0][src], locs[1][src]]:.4f} "
                  f"alpha={alpha:.3e} steps={nsteps}", flush=True)

        psi_op = cp.array(psi) if gpu else psi
        op = AnisDiffusionOp(dim_shape=dim_shape, alpha=alpha, diff_method_struct_tens="fd",
                             freezing_arr=psi_op, sampling=sampling, matrix_based_impl=True,
                             gpu=gpu, dtype=psi_fct.dtype)
        G = op._grad_matrix_based.mat
        x = x0.flatten()
        m = mask_core.flatten()
        if gpu:
            x, m = cp.array(x), cp.array(m)
        step = 0.95 * 0.25 * sampling ** 2
        for _ in range(nsteps):
            x -= step * G.dot(x)
            x[x < 0] = 0
            x *= m
        x[x < 0] = 0
        x *= m
        x /= x.max()
        if gpu:
            x = x.get()
        x = x.reshape(dim_shape[1:])
        phantoms.append(add_gaussian_background(x, psi, mask_core, peak_values[i]))
        psis.append(psi)

    coarse = lambda a: skimt.resize(a, (cfg.H, cfg.W), anti_aliasing=False, mode="edge")  # noqa: E731
    return (np.array([coarse(p) for p in phantoms]), np.array([coarse(p) for p in psis]), trimming, alphas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", type=int, default=cfg.NUM_TEST)
    ap.add_argument("--out", type=Path, default=cfg.DATA / "test_set")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    psi_eq = np.load(cfg.DATA / "test_set" / "magnetic_equilibrium.npy")
    # background amplitudes are drawn for all 1000 phantoms first (separate seed)
    np.random.seed(5032026)
    peak_vals = 0.2 + 0.8 * np.random.rand(cfg.NUM_TEST)

    x, psi, trim, alphas = generate(psi_eq, args.num, peak_vals, gpu=args.gpu)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "sxr_samples_with_background_coarse.npy", x)
    np.save(args.out / "psis_coarse.npy", psi)
    np.save(args.out / "trimming_values.npy", trim)
    np.save(args.out / "alpha_random_values.npy", alphas)
    print(f"saved {len(x)} phantoms to {args.out}")


if __name__ == "__main__":
    main()
