"""Fig. "limitations" (fig:limitations): out-of-distribution SXR reconstructions.

Two synthetic phantoms, sigma_y = 0.25 (SXR eta_medium):

* doublet (top row): a doublet profile built by anisotropic diffusion of two
  seed pixels along the flux surfaces of TCV shot #87262 (time index 26);
  data are used as is (the phantom is normalised to 1, no amplitude scaling).
* strike point (bottom row): test phantom #7 (smoothed) plus a Gaussian
  radiation blob at the outer divertor strike point; data are brought to the
  training amplitude by the normalisation coefficient alpha (Sec. 3) and the
  RD reconstruction is restricted to the TCV vessel only (not to the core).

For each case: RD MAP + RD posterior mean/std (anisotropic RD prior, 1e5 ULA
samples) and DiffPIR posterior mean/std (100 samples, lambda=100, zeta=0.8).
Also prints the doublet metrics quoted in the text ("metrics not shown").

Outputs (same file names as in the paper sources, Figures/limitations/):
    gt, rd_map, rd_mean, rd_std, diffpir_mean, diffpir_std  (.eps, strike point)
    the same with suffix _doublet, and the horizontal colour bars
    cbar_horiz, cbar_rd_std_horiz, cbar_diffpir_std_horiz (+ _doublet).
Raw reconstructions are cached in results/limitations/{strikepoint,doublet}.npz.

Runtime (CPU): ~2 min DiffPIR + ~3 min ULA per case, plus ~1 min
for the CV of the anisotropy parameter of the doublet. Pass --plot-only to
re-plot from the cached reconstructions.

The measurement noise of the paper figure was drawn with an unseeded torch.randn, so the reconstructions here
match the paper qualitatively, not bit-for-bit.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import common  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as scio  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import skimage.transform as skimt  # noqa: E402
import torch  # noqa: E402
from matplotlib import cm, colors as mcolors  # noqa: E402
from scipy import ndimage  # noqa: E402
from skimage.filters import gaussian  # noqa: E402

import plasma_diffusion as pdiff  # noqa: E402
import tomo_fusion.bayesian_computations as bcomp  # noqa: E402
import tomo_fusion.functionals_definition as fct_def  # noqa: E402
import tomo_fusion.hyperparameter_tuning as hyper_tune  # noqa: E402
from tomo_fusion.tools import plotting_fcts as tomo_plots  # noqa: E402
from pyxu_diffops.operator import AnisDiffusionOp  # noqa: E402

H, W = cfg.H, cfg.W
SIGMA = 0.25                      # SXR eta_medium
LAMBDA, ZETA = cfg.DIFFPIR["sxr"]  # (100, 0.8)
RD_SAMPLING = 0.0125              # pixel size used by the RD operators
ANIS_GRID = np.logspace(-4, 0, 13)

# strike-point phantom
SP_IDX = 7
SP_BLOB = dict(z=H - 2, r=W // 2 - 1, std=3.0, amp=0.8 * 1.2)
SP_ANIS = ANIS_GRID[7]            # 0.0215..., CV-tuned value hard-coded in the original
# doublet phantom
DB_SHOT, DB_TIME = 87262, 26
DB_SEEDS = [(31, 21), (87, 21)]
DB_DIFF_STEPS = 2000
DB_CORE_PSI, DB_ROI_PSI = 0.41, 0.395
DB_ANIS = ANIS_GRID[6]            # 0.01, value used for the paper panels (CV result of the original run)


# ---------------------------------------------------------------------------
# phantoms
# ---------------------------------------------------------------------------
def tcv_mask() -> np.ndarray:
    m = skimt.resize(np.load(cfg.TCV_MASK), (H, W), anti_aliasing=False, mode="edge")
    m[m > 0] = 1
    return m


def strikepoint_phantom():
    x, psi = common.load_test_set([SP_IDX])
    x = gaussian(x[0], sigma=1).astype(np.float32)
    zz, rr = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    b = SP_BLOB
    x += (b["amp"] * np.exp(-((zz - b["z"]) ** 2 + (rr - b["r"]) ** 2) / (2 * b["std"] ** 2))).astype(np.float32)
    return x, psi[0]


def doublet_psi() -> np.ndarray:
    """LIUQE flux of the doublet shot on the reconstruction grid.

    Same as plasma_diffusion.interpolate_equilibria but without the transpose:
    the flux map of this (doublet) shot file is stored as (z, r).
    """
    d = scio.loadmat(cfg.SHOTS[DB_SHOT])["data"][0, 0]
    rs, zs = d["liuqe_rs"][:, 0], d["liuqe_zs"][:, 0]
    dr_l, dz_l = np.mean(np.diff(rs)), np.mean(np.diff(zs))
    zmin, zmax, rmin, rmax = -0.75, 0.75, 0.624, 1.1376
    dz, dr = (zmax - zmin) / H, (rmax - rmin) / W
    grid_z = np.linspace(zmin, zmax, H + 1)[:-1] + dz / 2
    grid_r = np.linspace(rmin, rmax, W + 1)[:-1] + dr / 2
    coords = np.zeros((2, H * W))
    coords[0] = np.repeat(-(np.flip(grid_z) - zs.max()) / dz_l, W)
    coords[1] = np.tile((grid_r - rs.min()) / dr_l, H)
    return ndimage.map_coordinates(np.flip(d["psi"][:, :, DB_TIME], axis=0), coords,
                                   order=2, mode="nearest", prefilter=True).reshape(H, W)


def doublet_phantom(psi: np.ndarray) -> np.ndarray:
    """Diffuse two seed pixels along the flux surfaces, inside psi > 0.41."""
    x = np.zeros((H, W))
    for z, r in DB_SEEDS:
        x[z, r] = 1.0
    x = x.ravel()
    op = AnisDiffusionOp(dim_shape=(1, H, W), alpha=5e-2, diff_method_struct_tens="fd",
                         freezing_arr=psi, sampling=RD_SAMPLING, matrix_based_impl=True)
    L = op._grad_matrix_based.mat
    core = (psi > DB_CORE_PSI).ravel()
    for _ in range(DB_DIFF_STEPS):
        x -= 0.95 * (0.25 * RD_SAMPLING ** 2) * L.dot(x)
        x[x < 0] = 0
        x *= core
    return (x / x.max()).reshape(H, W).astype(np.float32)


# ---------------------------------------------------------------------------
# reconstructions
# ---------------------------------------------------------------------------
def diffpir(model, schedule, A, y, sigma, seed):
    common.seed_everything(seed)
    post, _, _ = pdiff.sample_posterior_diffpir(
        model, schedule, A, y, image_shape=(1, H, W),
        num_samples=cfg.NUM_POSTERIOR_SAMPLES, num_inference_steps=cfg.NUM_INFERENCE_STEPS,
        zeta=ZETA, sigma_y=sigma, lambda_=LAMBDA)
    return post.mean(0)[0].cpu().numpy(), post.std(0)[0].cpu().numpy()


def rd(A, y, psi, sigma, roi_mask, anis=None, ula_samples=int(1e5)):
    """RD MAP and ULA posterior mean/std; anis=None -> 5-fold CV over ANIS_GRID."""
    f, g = fct_def.define_loglikelihoodfromdata_and_logprior(
        y.ravel(), psi, fwd_matrix=sp.csr_matrix(A), reconstruction_shape=(1, H, W),
        sigma_err=sigma, reg_fct_type="anisotropic", alpha=1e-2, sampling=RD_SAMPLING)
    if anis is None:
        cv = hyper_tune.anis_param_tuning(
            f, g, reg_param=cfg.RD_REG_PARAM, tuning_techniques=["CV_full"],
            with_pos_constraint=True, clipping_mask=roi_mask, cv_strategy=["random"],
            anis_params=ANIS_GRID, plot=False)["CV_full_random"]
        anis = float(cv[0, np.argmin(cv[1, :])])
    g = hyper_tune._redefine_anis_param_logprior(g, anis)
    x_map = bcomp.compute_MAP(f, g, cfg.RD_REG_PARAM, with_pos_constraint=True,
                              clipping_mask=roi_mask, show_progress=False)
    uq = bcomp.run_ula(f, g, cfg.RD_REG_PARAM, psi, np.load(cfg.TEST_TRIM)[SP_IDX],
                       with_pos_constraint=True, clipping_mask=roi_mask,
                       compute_stats_wrt_MAP=True, samples=ula_samples)
    return (np.asarray(x_map).reshape(H, W), np.asarray(uq["mean"]).reshape(H, W),
            np.sqrt(np.asarray(uq["var"])).reshape(H, W), anis)


def run_case(case, model, schedule, args):
    rng = np.random.default_rng(args.seed)
    A = common.load_geometry("sxr")
    if case == "strikepoint":
        x, psi = strikepoint_phantom()
        y = A @ x.ravel() + SIGMA * rng.standard_normal(A.shape[0]).astype(np.float32)
        # amplitude normalisation against the training projections (Sec. 3)
        alpha = pdiff.normalization_coefficient(y, np.load(cfg.Y_TRAIN["sxr"]))
        roi, anis = tcv_mask(), SP_ANIS
    else:
        psi = doublet_psi()
        x = doublet_phantom(psi)
        y = A @ x.ravel() + SIGMA * rng.standard_normal(A.shape[0]).astype(np.float32)
        alpha = 1.0      # phantom already at unit amplitude
        roi, anis = (psi > DB_ROI_PSI).astype(bool), (None if args.tune_anis else DB_ANIS)
    print(f"[{case}] alpha = {alpha:.4f}")
    y_n, s_n = y / alpha, SIGMA / alpha
    dp_mean, dp_std = diffpir(model, schedule, A, y_n, s_n, args.seed)
    rd_map, rd_mean, rd_std, anis = rd(A, y_n, psi, s_n, roi, anis, args.ula_samples)
    print(f"[{case}] RD anisotropy parameter = {anis:.4g}")
    out = dict(x=x, psi=psi, y=y, alpha=alpha, anis=anis,
               diffpir_mean=dp_mean * alpha, diffpir_std=dp_std * alpha,
               rd_map=rd_map * alpha, rd_mean=rd_mean * alpha, rd_std=rd_std * alpha)
    np.savez(cfg.result_dir("limitations") / f"{case}.npz", **out)
    return out


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def nice_vmax(v):
    """Round v up to 1, 1.5, 2, 2.5, 3, 4, 5, 7.5 or 10 x 10^k."""
    decade = 10.0 ** math.floor(math.log10(v))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10):
        if m * decade >= v * (1 - 1e-12):
            return float(f"{m * decade:.6g}")


def save_panel(img, psi, path, cmap, vmax, levels):
    plt.figure(figsize=(3, 3))
    tomo_plots.plot_profile(np.asarray(img), tcv_plot_clip=True, contour_image=psi, cmap=cmap,
                            colorbar=False, contour_color="w", vmin=0, vmax=vmax,
                            aspect=None, lcfs_width=1, lwidth=0.2, levels=levels)
    plt.axis("off")
    plt.savefig(path, format="eps", bbox_inches="tight")
    plt.close("all")


def _fmt_tick(t):
    return f"${f'{t:.3f}'.rstrip('0').rstrip('.')}$"


# Horizontal colour bars: every bar drawn on an identical canvas (bar 0.465 x
# 0.093 in, 11 pt labels) and saved without a tight bbox, so that LaTeX scales
# all of them identically. The canvas is sized for the widest tick label.
CBAR_BAR_W, CBAR_BAR_H, CBAR_FS, CBAR_TOP = 0.465, 0.093, 11, 0.03


def _cbar_fig(vmax, cmap, figsize, rect):
    fig = plt.figure(figsize=figsize)
    cb = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(0, vmax), cmap=cmap),
                      cax=fig.add_axes(rect), orientation="horizontal")
    cb.ax.tick_params(labelsize=CBAR_FS)
    cb.set_ticks([0, vmax])
    cb.set_ticklabels([_fmt_tick(0), _fmt_tick(vmax)])
    return fig


def save_cbars(specs, ref_vmaxes, out):
    probe = (3.0, 1.5)
    rect0 = [(1 - CBAR_BAR_W / probe[0]) / 2, 0.5, CBAR_BAR_W / probe[0], CBAR_BAR_H / probe[1]]
    ox = oy = 0.0
    for vmax in ref_vmaxes:
        fig = _cbar_fig(vmax, "viridis", probe, rect0)
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        bl, bb = rect0[0] * probe[0], rect0[1] * probe[1]
        ox, oy = max(ox, bl - tb.x0, tb.x1 - (bl + CBAR_BAR_W)), max(oy, bb - tb.y0)
        plt.close(fig)
    canvas = (CBAR_BAR_W + 2 * (ox + 0.01), CBAR_TOP + CBAR_BAR_H + oy + 0.01)
    rect = [(canvas[0] - CBAR_BAR_W) / 2 / canvas[0], (canvas[1] - CBAR_TOP - CBAR_BAR_H) / canvas[1],
            CBAR_BAR_W / canvas[0], CBAR_BAR_H / canvas[1]]
    for vmax, cmap, name in specs:
        fig = _cbar_fig(vmax, cmap, canvas, rect)
        fig.savefig(out / name, format="eps")
        plt.close(fig)


def plot_case(case, r, out):
    if case == "strikepoint":
        sfx, levels = "", 15
        rd_vmax, dp_vmax = 0.1, 0.035   # fixed by hand in the paper (see README)
        dp_panel_vmax = 0.032           # paper: panel drawn with 0.032, colour bar labelled 0.035
    else:
        sfx, levels = "_doublet", 10
        m = tcv_mask().astype(bool)     # 99.5th percentile inside the vessel, rounded up
        rd_vmax = nice_vmax(np.percentile(r["rd_std"][m], 99.5))
        dp_vmax = nice_vmax(np.percentile(r["diffpir_std"][m], 99.5))
        dp_panel_vmax = dp_vmax
    mx = float(r["x"].max())
    psi = r["psi"]
    for name, img in [("gt", r["x"]), ("rd_map", r["rd_map"]), ("rd_mean", r["rd_mean"]),
                      ("diffpir_mean", r["diffpir_mean"])]:
        save_panel(img / mx, psi, out / f"{name}{sfx}.eps", "viridis", 1, levels)
    save_panel(r["rd_std"], psi, out / f"rd_std{sfx}.eps", "inferno", rd_vmax, levels)
    save_panel(r["diffpir_std"], psi, out / f"diffpir_std{sfx}.eps", "inferno", dp_panel_vmax, levels)
    specs = [(1.0, "viridis", f"cbar_horiz{sfx}.eps"),
             (rd_vmax, "inferno", f"cbar_rd_std_horiz{sfx}.eps"),
             (dp_vmax, "inferno", f"cbar_diffpir_std_horiz{sfx}.eps")]
    try:   # the paper used LaTeX tick labels; fall back to mathtext if LaTeX is incomplete
        with plt.rc_context({"text.usetex": True, "font.family": "sans-serif"}):
            save_cbars(specs, [1.0, 0.1, 0.035, rd_vmax, dp_vmax], out)
    except RuntimeError:
        print("LaTeX not usable by matplotlib, colour bars drawn with mathtext")
        plt.close("all")
        save_cbars(specs, [1.0, 0.1, 0.035, rd_vmax, dp_vmax], out)
    print(f"[{case}] std colour limits: RD {rd_vmax}, DiffPIR {dp_vmax}")


def print_metrics(case, r):
    print(f"[{case}] metrics w.r.t. ground truth:")
    for name in ("rd_map", "rd_mean", "diffpir_mean"):
        m = common.metrics(r["x"], r[name].astype(np.float32))
        print(f"    {name:13s} PSNR {m['psnr']:5.2f} dB   SSIM {m['ssim']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", choices=["doublet", "strikepoint", "all"], default="all")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--ula-samples", type=int, default=int(1e5))
    ap.add_argument("--tune-anis", action="store_true",
                    help="doublet: tune the RD anisotropy parameter by 5-fold CV instead of using 0.01")
    ap.add_argument("--plot-only", action="store_true",
                    help="re-plot from results/limitations/*.npz")
    args = ap.parse_args()

    cases = ["doublet", "strikepoint"] if args.case == "all" else [args.case]
    out = cfg.figure_dir("limitations")
    model = schedule = None
    for case in cases:
        if args.plot_only:
            r = dict(np.load(cfg.RESULTS / "limitations" / f"{case}.npz"))
        else:
            if model is None:
                model, schedule = common.load_model()
            r = run_case(case, model, schedule, args)
        print_metrics(case, r)
        plot_case(case, r, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
