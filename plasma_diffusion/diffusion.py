"""DDPM diffusion schedule and elementary reverse-process updates.

We adopt the variance-preserving DDPM formulation: noisy profiles are
constructed as ``x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) z`` with a linear
beta schedule, and reverse sampling uses DDIM updates whose stochasticity is
controlled by ``eta`` (0 = deterministic).
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["build_schedule", "make_timesteps", "predict_x0_from_eps", "ddim_update"]


def build_schedule(num_train_steps: int = 1000, beta_start: float = 1e-4,
                   beta_end: float = 2e-2,
                   device: str | torch.device = "cpu") -> dict:
    """Linear-beta DDPM schedule.

    Returns a dict with keys ``num_train_steps``, ``betas``, ``alphas`` and
    ``alpha_bar`` (cumulative product of alphas).
    """
    betas = torch.linspace(beta_start, beta_end, num_train_steps,
                           device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        "num_train_steps": num_train_steps,
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
    }


def make_timesteps(num_train_steps: int, num_inference_steps: int) -> list[int]:
    """Uniformly spaced decreasing timesteps for accelerated sampling."""
    return np.linspace(num_train_steps - 1, 0, num_inference_steps,
                       dtype=np.int64).tolist()


def predict_x0_from_eps(x_t: torch.Tensor, eps_t: torch.Tensor,
                        alpha_bar_t: torch.Tensor) -> torch.Tensor:
    """Tweedie estimate of the clean profile from the predicted noise."""
    return (x_t - torch.sqrt(1 - alpha_bar_t) * eps_t) / torch.sqrt(alpha_bar_t)


def ddim_update(x0_hat: torch.Tensor, eps_t: torch.Tensor, t: int, t_prev: int,
                alpha_bar: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
    """One reverse DDIM step from level ``t`` to ``t_prev``.

    ``eta = 0`` gives the deterministic DDIM update; ``eta = 1`` recovers
    DDPM-like stochasticity.
    """
    if t_prev < 0:
        return x0_hat

    a_t = alpha_bar[t]
    a_prev = alpha_bar[t_prev]

    sigma_t = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
    c_t = torch.sqrt(torch.clamp(1 - a_prev - sigma_t**2, min=0.0))
    noise = torch.randn_like(x0_hat) if eta > 0 else torch.zeros_like(x0_hat)

    return torch.sqrt(a_prev) * x0_hat + c_t * eps_t + sigma_t * noise
