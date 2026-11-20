"""Shared loaders, measurement model and result I/O for the reproduction scripts."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import config as cfg

# make `plasma_diffusion` (repo root) and `tomo_fusion` (vendored RD code) importable
for p in (cfg.REPO, cfg.ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import plasma_diffusion as pdiff  # noqa: E402

METHODS = ("diffpir", "diffrd_clip", "diffrd", "rd")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = cfg.SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # TF32 matmuls change the sampled trajectories; the paper runs used full fp32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_model(dev=None):
    dev = dev or device()
    model = pdiff.load_pretrained(cfg.CHECKPOINT, device=dev)
    schedule = pdiff.build_schedule(num_train_steps=cfg.NUM_TRAIN_STEPS, device=dev)
    return model, schedule


def load_geometry(diag: str) -> np.ndarray:
    """Geometry matrix (M, H*W), normalised by its maximum as in the paper."""
    A = np.load(cfg.GEOMETRY[diag]).astype(np.float32)
    return A / A.max()


def load_test_set(indices=None):
    """Ground-truth phantoms (N, H, W) and their equilibria psi (N, H, W)."""
    idx = slice(None) if indices is None else indices
    x = np.load(cfg.TEST_SET)[idx].astype(np.float32)
    psi = np.load(cfg.TEST_PSIS)[idx]
    return x, psi


def measurements(diag: str, sigma: float, indices=None):
    """Noisy synthetic data y_i = A x_i + sigma * n_i with the fixed noise draws n_i.

    Returns (A, x_gt, y_noisy) with shapes (M, HW), (N, H, W), (N, M).
    """
    idx = np.arange(cfg.NUM_TEST) if indices is None else np.asarray(indices)
    A = load_geometry(diag)
    x, _ = load_test_set(idx)
    noise = np.load(cfg.NOISE[diag])[idx].astype(np.float32)
    y = x.reshape(len(idx), -1) @ A.T + sigma * noise
    return A, x, y


def metrics(gt: np.ndarray, est: np.ndarray) -> dict:
    """PSNR / SSIM with data range of the ground truth (as in all paper tables)."""
    rng = float(gt.max() - gt.min())
    return {
        "psnr": peak_signal_noise_ratio(gt, est, data_range=rng),
        "ssim": structural_similarity(gt, est, data_range=rng),
        "mse": float(((gt - est) ** 2).mean()),
    }


# ---------------------------------------------------------------------------
# Phantom-study results layout:
#   results/phantom_studies/{method}_{diag}_{sigma}/
#       metrics.csv      one row per test phantom (sample_idx, psnr, ssim, ...)
#       post_means.npy   (1000, H*W) float32 posterior means
#       post_stds.npy    (1000, H*W) float32 posterior pixel-wise std
#       maps.npy         (1000, H*W) float32 MAP estimates (RD only)
# ---------------------------------------------------------------------------
def study_dir(method: str, diag: str, sigma: float, create: bool = False) -> Path:
    assert method in METHODS, method
    p = cfg.RESULTS / "phantom_studies" / f"{method}_{diag}_{sigma}"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def load_study(method: str, diag: str, sigma: float, arrays=("post_means", "post_stds")):
    d = study_dir(method, diag, sigma)
    if not d.exists():
        raise FileNotFoundError(
            f"{d} not found: run the corresponding script in reproduce/02_phantom_studies first")
    out = {"metrics": pd.read_csv(d / "metrics.csv")}
    for a in arrays:
        out[a] = np.load(d / f"{a}.npy")
    return out
