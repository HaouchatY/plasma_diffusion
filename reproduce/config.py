"""Paths and experiment constants shared by all reproduction scripts.

Every script in ``reproduce/`` imports this module. Paths can be overridden
with environment variables so that large outputs can live outside the repo:

    PD_RESULTS_DIR   where experiment outputs (posteriors, metrics) are written
                     (default: reproduce/results)
    PD_FIGURES_DIR   where regenerated figures/tables are written
                     (default: reproduce/figures)
    PD_TRAIN_DIR     directory of the 1e5 training profiles sxr_sample_{i}.npy
                     (only needed to retrain the prior)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = ROOT / "data"

RESULTS = Path(os.environ.get("PD_RESULTS_DIR", ROOT / "results"))
FIGURES = Path(os.environ.get("PD_FIGURES_DIR", ROOT / "figures"))
TRAIN_DIR = Path(os.environ["PD_TRAIN_DIR"]) if "PD_TRAIN_DIR" in os.environ else None

CHECKPOINT = REPO / "data" / "prior_sxr.pth"   # = fusion_epoch_2193.pth used in the paper

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
TEST_SET = DATA / "test_set" / "sxr_samples_with_background_coarse.npy"   # (1000, 120, 40)
TEST_PSIS = DATA / "test_set" / "psis_coarse.npy"                         # (1000, 120, 40)
TEST_TRIM = DATA / "test_set" / "trimming_values.npy"                     # (1000,)
GEOMETRY = {
    "sxr": DATA / "geometry" / "forward_model_sxr_full_geometry.npy",      # (100, 4800)
    "dmpx": DATA / "geometry" / "dmpx_geometry_matrix.npy",                # (256, 4800)
    "pilatus": DATA / "geometry" / "pilatus_geometry_matrix.npy",          # (80, 4800)
}
NOISE = {d: DATA / "noise" / f"noise_realizations_{d}.npy" for d in GEOMETRY}  # (1000, M)
SCALING = DATA / "noise" / "scaling_params.npy"                             # (1000,)
Y_TRAIN = {d: DATA / f"y_train_{d}.npy" for d in GEOMETRY}  # projections of the first 1e4 training profiles
TCV_MASK = DATA / "masks" / "tcv_mask_1_subpixels_NINO.npy"
LOS_PARAMS = {d: DATA / "diagnostics_los" / f"{d}_LoS_params.npy" for d in GEOMETRY}
SHOTS = {
    85270: DATA / "experimental" / "shot_data_85270.mat",
    87262: DATA / "experimental" / "shot_data_87262.mat",
    88767: DATA / "experimental" / "shot_data_full_88767_finetimegrid.mat",
}

# ---------------------------------------------------------------------------
# Experiment constants (Sec. "Results" of the paper)
# ---------------------------------------------------------------------------
H, W = 120, 40
NUM_TEST = 1000
NUM_TRAIN_STEPS = 1000
NUM_INFERENCE_STEPS = 250
NUM_POSTERIOR_SAMPLES = 100
SEED = 42

# eta_low / eta_medium / eta_high per diagnostic
NOISE_LEVELS = {
    "sxr": [0.05, 0.25, 0.5],
    "dmpx": [0.1, 0.5, 1.0],
    "pilatus": [0.07, 0.35, 0.7],
}

# DiffPIR hyperparameters (lambda_, zeta), from the grid search
DIFFPIR = {"sxr": (100, 0.8), "dmpx": (300, 0.8), "pilatus": (200, 0.8)}
# DiffRD_clip = DiffPIR restricted to the reconstruction region (Pilatus)
DIFFRD_CLIP = (100, 0.8)
# DiffRD (lambda_, zeta, lambda_rd) per Pilatus noise level
DIFFRD = {0.07: (100, 0.9, 1e-5), 0.35: (10, 0.9, 1e-4), 0.7: (10, 0.9, 1e-4)}

# RD baseline: anisotropic reaction-diffusion prior, alpha tuned by CV, ULA sampling
RD_REG_PARAM = 1e-1


def result_dir(name: str) -> Path:
    p = RESULTS / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def figure_dir(name: str) -> Path:
    p = FIGURES / name
    p.mkdir(parents=True, exist_ok=True)
    return p
