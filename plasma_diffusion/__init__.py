"""Diffusion priors for plasma tomography.

Bayesian data-driven reconstruction of plasma emissivity profiles from
tomographic measurements, using diffusion models as learned priors
(DiffPIR posterior sampling and its equilibrium-informed DiffRD variant).

Reference: D. Hamm*, Y. Haouchat*, C. Theiler, M. Unser,
"Diffusion Priors for Plasma Tomography: Bayesian Data-Driven Methods
Applied to SXR Reconstruction".
"""

from .data import (closest_training_measurement, estimate_noise_std,
                   fit_affine_noise_model, normalization_coefficient)
from .diffusion import build_schedule, make_timesteps
from .experimental import build_measurement, interpolate_equilibria
from .model import TinyUNet, load_pretrained
from .plotting import plot_profile, show_samples
from .rd import build_rd_operator, load_rd_operator, save_rd_operator
from .sampling import (sample_posterior_diffpir, sample_posterior_dps,
                       sample_prior)

__version__ = "1.0.0"

__all__ = [
    "TinyUNet",
    "load_pretrained",
    "build_schedule",
    "make_timesteps",
    "sample_prior",
    "sample_posterior_diffpir",
    "sample_posterior_dps",
    "normalization_coefficient",
    "closest_training_measurement",
    "estimate_noise_std",
    "fit_affine_noise_model",
    "interpolate_equilibria",
    "build_measurement",
    "build_rd_operator",
    "save_rd_operator",
    "load_rd_operator",
    "plot_profile",
    "show_samples",
]
