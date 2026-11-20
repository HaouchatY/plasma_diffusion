"""Shared code for the TCV experimental-data reconstructions (Sec. "Experimental data").

Each time slice is reconstructed twice from the same SXR measurement vector:

* RD (MAP): anisotropic reaction-diffusion prior aligned with the LIUQE
  equilibrium, anisotropy parameter tuned by cross-validation (blind to any
  ground truth), MAP with positivity constraint, restricted to the core mask;
* DiffPIR: 100 posterior samples from the diffusion prior (lambda=100, zeta=0.8).

The measurement is brought to the training amplitude scale by the normalization
coefficient alpha (nearest training measurement), and a fixed noise level
sigma=0.25 (in normalized units) is assumed.

Results layout (one directory per run, e.g. results/real_data/85270/):
    posterior_tomo.npy  (T, 100, 1, H, W) float32  DiffPIR samples, *normalized* units (multiply by alpha)
    im_maps.npy         (T, 1, 1, H, W)   float32  RD MAP, physical units (already multiplied by alpha)
    params.npy          (T, 3)            float64  [alpha, CV-tuned anisotropy, sigma]
    inv_times.npy       (T,)              float64  time of each slice (s)
    frames.npy          (T,)              int      frame index in the shot file
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import scipy.io as scio  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import skimage.transform as skimt  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402,F401  (sets sys.path for plasma_diffusion / tomo_fusion)
import config as cfg  # noqa: E402
import plasma_diffusion as pdiff  # noqa: E402

H, W = cfg.H, cfg.W
SIGMA = 0.25                         # assumed noise level (normalized units)
RD_REG_PARAM = cfg.RD_REG_PARAM      # 0.1
RD_ANIS_INIT = 1e-2
ANIS_GRID = np.logspace(-4, 0, 13)   # CV grid for the anisotropy parameter
DIFFPIR_LAMBDA, DIFFPIR_ZETA = 100, 0.8


def load_shot(key):
    """Shot record (``data`` struct of the .mat file) and its equilibria on the grid."""
    data = scio.loadmat(cfg.SHOTS[key])["data"][0, 0]
    psi = pdiff.interpolate_equilibria(data, nz=H, nr=W)
    return data, psi


def run_dir(name: str) -> Path:
    return cfg.result_dir(f"real_data/{name}")


def load_run(name: str, mmap: bool = False):
    d = cfg.RESULTS / "real_data" / name
    if not d.exists():
        raise FileNotFoundError(f"{d} not found: run reproduce/03_real_data first")
    mode = "r" if mmap else None
    return {k: np.load(d / f"{k}.npy", mmap_mode=mode if k == "posterior_tomo" else None)
            for k in ("posterior_tomo", "im_maps", "params", "inv_times", "frames")}


def core_mask(psi_slice: np.ndarray) -> np.ndarray:
    """Confined region (psi < 5e-4) above the X-point."""
    mask = psi_slice < 0.0005
    row_sums = mask.sum(axis=1)
    xpt = 60 + int(np.argmin(row_sums[60:]))
    mask[xpt:, :] = False
    return mask


def tcv_mask() -> np.ndarray:
    m = np.load(cfg.TCV_MASK)
    m = skimt.resize(m, (H, W), anti_aliasing=False, mode="edge")
    m[m > 0] = 1
    return m


class Reconstructor:
    """RD-MAP + DiffPIR reconstruction of single SXR time slices."""

    def __init__(self, device=None, num_samples=cfg.NUM_POSTERIOR_SAMPLES):
        self.device = device or common.device()
        self.model, self.schedule = common.load_model(self.device)
        A = np.load(cfg.GEOMETRY["sxr"]).astype(np.float32)
        self.A_max = float(A.max())
        self.A = A / self.A_max
        self.y_train = np.load(cfg.Y_TRAIN["sxr"])
        self.num_samples = num_samples

    def measurement(self, data, signal_row):
        """Raw detector row -> line integrals in normalized-geometry units, good channels only."""
        good = data["good_channels"].flatten() - 1
        y = np.asarray(signal_row, dtype=np.float64) * data["etendues"].flatten() * 4 * np.pi / self.A_max
        return y[good].astype(np.float32), good

    def __call__(self, y, good, psi_slice, use_core_mask=True):
        # deferred: pyxu is only needed for the RD part
        from pyxu_diffops.operator import AnisDiffusionOp
        import tomo_fusion.bayesian_computations as bcomp
        import tomo_fusion.functionals_definition as fct_def
        import tomo_fusion.hyperparameter_tuning as hyper_tune

        alpha = pdiff.normalization_coefficient(y, self.y_train[:, good])
        y_n = y / alpha
        mask = core_mask(psi_slice) if use_core_mask else tcv_mask()

        # ---- RD: CV-tuned anisotropy, then MAP --------------------------------
        f = fct_def._DataFidelityFunctional(
            dim_shape=(1, H, W), noisy_tomo_data=y_n, sigma_err=SIGMA,
            geometry_matrix=sp.csr_matrix(self.A)[good, :])
        g = AnisDiffusionOp(dim_shape=(1, H, W), alpha=RD_ANIS_INIT, diff_method_struct_tens="fd",
                            freezing_arr=psi_slice, sampling=0.0125, matrix_based_impl=True)
        cv = hyper_tune.anis_param_tuning(
            f, g, reg_param=RD_REG_PARAM, tuning_techniques=["CV_full"], with_pos_constraint=True,
            clipping_mask=mask, cv_strategy=["random"], anis_params=ANIS_GRID, plot=False)
        best_anis = cv["CV_full_random"][0, np.argmin(cv["CV_full_random"][1, :])]
        g = hyper_tune._redefine_anis_param_logprior(g, best_anis)
        im_map = bcomp.compute_MAP(f, g, RD_REG_PARAM, with_pos_constraint=True,
                                   clipping_mask=mask, show_progress=False) * alpha

        # ---- DiffPIR ---------------------------------------------------------
        post, _, _ = pdiff.sample_posterior_diffpir(
            self.model, self.schedule, self.A[good, :], y_n, image_shape=(1, H, W),
            num_samples=self.num_samples, num_inference_steps=cfg.NUM_INFERENCE_STEPS,
            zeta=DIFFPIR_ZETA, sigma_y=SIGMA, lambda_=DIFFPIR_LAMBDA, progress=False)

        return (post.detach().cpu().numpy().astype(np.float32),
                np.asarray(im_map, dtype=np.float32).reshape(1, 1, H, W),
                np.array([alpha, best_anis, SIGMA]))


def run_slices(name, data, psi, signal, frames, device=None, num_samples=cfg.NUM_POSTERIOR_SAMPLES):
    """Reconstruct ``signal[frames]`` and save the run under results/real_data/<name>/."""
    common.seed_everything()
    rec = Reconstructor(device, num_samples)
    times = data["measurements_times"].flatten()
    posts, maps, params = [], [], []
    for k, fr in enumerate(frames):
        y, good = rec.measurement(data, signal[fr])
        post, im_map, par = rec(y, good, psi[fr])
        posts.append(post), maps.append(im_map), params.append(par)
        print(f"[{name}] {k + 1}/{len(frames)} t={times[fr]:.4f}s alpha={par[0]:.4e} anis={par[1]:.2e}",
              flush=True)
    out = run_dir(name)
    np.save(out / "posterior_tomo.npy", np.stack(posts))
    np.save(out / "im_maps.npy", np.stack(maps))
    np.save(out / "params.npy", np.stack(params))
    np.save(out / "inv_times.npy", times[np.asarray(frames)])
    np.save(out / "frames.npy", np.asarray(frames))
    print(f"saved {out}")
    return out


# ---------------------------------------------------------------------------
# Plotting (shared by reproduce/04_figures/fig1[0-2]_real_data_*.py)
# ---------------------------------------------------------------------------
def save_profile_eps(img, psi_slice, path, cmap="viridis", vmin=None, vmax=None, contour_color="w"):
    """Single profile panel on the TCV cross-section with flux-surface contours (paper style)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tomo_fusion.tools import plotting_fcts as tomo_plots

    plt.figure(figsize=(3, 3))
    tomo_plots.plot_profile(np.asarray(img), tcv_plot_clip=True, contour_image=psi_slice, cmap=cmap,
                            ax=None, colorbar=False, contour_color=contour_color, vmin=vmin, vmax=vmax,
                            aspect=None, lcfs_width=1, lwidth=0.2)
    plt.axis("off")
    plt.savefig(path, format=Path(path).suffix[1:], bbox_inches="tight")
    plt.close("all")


def save_overview(images, psis, titles, suptitle, path, cmap="viridis", vmin=None, vmax=None,
                  contour_color="w"):
    """Row of panels (quick-look, not used in the paper)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tomo_fusion.tools import plotting_fcts as tomo_plots

    n = len(images)
    fig, ax = plt.subplots(1, n, figsize=(2 * n, 5), squeeze=False)
    for i in range(n):
        tomo_plots.plot_profile(np.asarray(images[i]), tcv_plot_clip=True, contour_image=psis[i], cmap=cmap,
                                ax=ax[0, i], colorbar=False, contour_color=contour_color, vmin=vmin,
                                vmax=vmax, aspect=None, lcfs_width=1, lwidth=0.2)
        ax[0, i].set_title(titles[i], fontsize=9)
        ax[0, i].axis("off")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close("all")
