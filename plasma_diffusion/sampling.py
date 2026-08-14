"""Prior and posterior sampling with the learned diffusion prior.

Implements the three samplers used in the paper:

- :func:`sample_prior` -- unconditional DDIM sampling from the learned prior.
- :func:`sample_posterior_diffpir` -- DiffPIR posterior sampling for the
  linear inverse problem ``y = T x + noise`` (Zhu et al., CVPRW 2023), plus
  the equilibrium-informed **DiffRD** variant obtained by passing the
  anisotropic reaction-diffusion gradient operator ``rd_matrix`` (Algorithm 1
  of the paper).
- :func:`sample_posterior_dps` -- DPS-style guidance (Chung et al., 2023),
  included for comparison.

All samplers run on the device of ``model`` and are ``torch.no_grad``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from tqdm.auto import tqdm

from .diffusion import ddim_update, make_timesteps, predict_x0_from_eps

__all__ = ["sample_prior", "sample_posterior_diffpir", "sample_posterior_dps"]


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _init_x(x_init, num_samples: int, image_shape, device) -> torch.Tensor:
    if x_init is None:
        return torch.randn(num_samples, *image_shape, device=device)
    x = torch.as_tensor(x_init, dtype=torch.float32, device=device)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.size(0) == 1 and num_samples > 1:
        x = x.repeat(num_samples, 1, 1, 1)
    return x


def _as_dense_tensor(mat, device) -> torch.Tensor:
    if sp.issparse(mat):
        mat = mat.toarray()
    return torch.as_tensor(np.asarray(mat), dtype=torch.float32, device=device)


@torch.no_grad()
def sample_prior(model, schedule, image_shape=(1, 120, 40), num_samples: int = 8,
                 num_inference_steps: int = 250, eta: float = 0.0,
                 clip_range=None, progress: bool = True):
    """Draw unconditional samples from the learned prior.

    Parameters
    ----------
    model:
        Trained noise predictor ``eps_theta(x_t, t)``.
    schedule:
        Output of :func:`plasma_diffusion.diffusion.build_schedule`.
    image_shape:
        Shape ``(C, H, W)`` of one sample.
    num_samples:
        Number of independent samples drawn in parallel.
    num_inference_steps:
        Number of reverse steps (<= the training steps; uniformly spaced).
    eta:
        DDIM stochasticity (0 = deterministic).
    clip_range:
        Optional ``(low, high)`` clamp applied to the intermediate ``x0``
        predictions.

    Returns
    -------
    samples:
        Tensor of shape ``(num_samples, C, H, W)``.
    process:
        List of intermediate ``x0`` predictions of the first sample (CPU),
        useful to visualize the sampling trajectory.
    """
    device = _model_device(model)
    model.eval()
    x = torch.randn(num_samples, *image_shape, device=device)
    process = []

    alpha_bar = schedule["alpha_bar"]
    timesteps = make_timesteps(schedule["num_train_steps"], num_inference_steps)
    iterator = tqdm(timesteps, desc="Prior sampling") if progress else timesteps

    for i, t in enumerate(iterator):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        eps_t = model(x, t_batch)
        x0_hat = predict_x0_from_eps(x, eps_t, alpha_bar[t])
        if clip_range is not None:
            x0_hat = x0_hat.clamp(*clip_range)

        x = ddim_update(x0_hat, eps_t, t, t_prev, alpha_bar, eta=eta)
        process.append(x[0].detach().cpu())

    return x, process


@torch.no_grad()
def sample_posterior_diffpir(model, schedule, A, y, image_shape=(1, 120, 40),
                             num_samples: int = 8, num_inference_steps: int = 250,
                             zeta: float = 0.8, sigma_y: float = 0.05,
                             lambda_: float = 100.0, clip_range=None,
                             x_init=None, clipping_mask=None, rd_matrix=None,
                             lambda_rd: float = 1e-4, progress: bool = True):
    """DiffPIR / DiffRD posterior sampling for ``y = A x + noise``.

    At each reverse step the network denoises the iterate into a prediction
    ``x0_hat`` (prior step), then the data proximal subproblem

    ``min_x ||y - A x||^2 + rho_t ||x - x0_hat||^2 (+ lambda_rd sbar_t ||L x||^2)``

    is solved in closed form (likelihood step), with the coupling
    ``rho_t = lambda_ * sigma_y^2 / sbar_t^2`` strengthening as the effective
    noise level ``sbar_t`` decreases. Passing the anisotropic gradient
    operator ``L`` as ``rd_matrix`` enables the equilibrium-informed DiffRD
    variant; passing ``clipping_mask`` restricts the estimate to the
    reconstruction region.

    Parameters
    ----------
    A:
        Forward model (geometry matrix), shape ``(M, N)`` with ``N = H*W``.
        Numpy array or torch tensor.
    y:
        Measurement vector of shape ``(M,)`` (or a batch ``(B, M)``).
    sigma_y:
        Noise standard deviation of the measurements (same units as ``y``).
    zeta:
        Stochasticity of the reverse update (``eta`` of the DDIM step).
    lambda_:
        DiffPIR regularization strength (paper value: 100 for SXR).
    rd_matrix:
        Optional sparse/dense anisotropic gradient matrix ``L`` (with ``N``
        columns) built from the magnetic equilibrium; enables DiffRD.
    lambda_rd:
        Strength of the RD term (its effective weight decays as
        ``lambda_rd * sbar_t`` along the reverse process). Paper values:
        1e-5 to 1e-4 depending on the noise level.
    clipping_mask:
        Optional ``(H, W)`` binary mask multiplied onto the estimate at every
        step (geometry clipping).

    Returns
    -------
    samples:
        Tensor of shape ``(num_samples, C, H, W)``.
    residual_history:
        Mean measurement-residual norm at each step.
    process:
        Intermediate ``x0`` estimates of the first sample (CPU).
    """
    device = _model_device(model)
    model.eval()
    alpha_bar = schedule["alpha_bar"]

    A = _as_dense_tensor(A, device)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    x = _init_x(x_init, num_samples, image_shape, device)

    if clipping_mask is not None:
        clipping_mask = torch.as_tensor(np.asarray(clipping_mask),
                                        dtype=torch.float32, device=device)

    # constant pieces of the proximal system
    AtA = A.T @ A
    identity = torch.eye(A.shape[1], device=device)
    LtL = None
    if rd_matrix is not None:
        L = _as_dense_tensor(rd_matrix, device)
        LtL = L.T @ L

    timesteps = make_timesteps(schedule["num_train_steps"], num_inference_steps)
    iterator = tqdm(timesteps, desc="Posterior sampling") if progress else timesteps
    residual_history, process = [], []

    for i, t in enumerate(iterator):
        sbar_t = torch.sqrt((1 - alpha_bar[t]) / alpha_bar[t])
        rho_t = lambda_ * (sigma_y**2 / sbar_t**2)

        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)

        eps_model = model(x, t_batch)
        x0_hat = predict_x0_from_eps(x, eps_model, alpha_bar[t])

        # likelihood (proximal) step: (AtA + rho_t I [+ c_t LtL]) x = At y + rho_t x0_hat
        H = AtA + rho_t * identity
        if LtL is not None:
            H = H + (lambda_rd * sbar_t) * LtL
        b = y @ A + rho_t * x0_hat.flatten(1)
        x0_guided = torch.linalg.solve(H, b.T).T.view_as(x0_hat)

        if clip_range is not None:
            x0_guided = x0_guided.clamp(*clip_range)
        if clipping_mask is not None:
            x0_guided = x0_guided * clipping_mask

        x = ddim_update(x0_guided, eps_model, t, t_prev, alpha_bar, eta=zeta)

        residual = x0_hat.flatten(1) @ A.T - y
        residual_history.append(residual.norm(dim=1).mean().item())
        process.append(x0_guided[0].detach().cpu())

    return x, residual_history, process


@torch.no_grad()
def sample_posterior_dps(model, schedule, A, y, image_shape=(1, 120, 40),
                         num_samples: int = 8, num_inference_steps: int = 250,
                         eta: float = 0.0, sigma_y: float = 0.05,
                         guidance_scale: float = 0.2, clip_range=None,
                         x_init=None, progress: bool = True):
    """DPS-style posterior sampling (norm-scaled likelihood guidance on x0).

    Included as a lightweight alternative to DiffPIR; see
    :func:`sample_posterior_diffpir` for the parameters shared by the two
    samplers.
    """
    device = _model_device(model)
    model.eval()
    alpha_bar = schedule["alpha_bar"]

    A = _as_dense_tensor(A, device)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    x = _init_x(x_init, num_samples, image_shape, device)

    timesteps = make_timesteps(schedule["num_train_steps"], num_inference_steps)
    iterator = tqdm(timesteps, desc="Posterior sampling") if progress else timesteps
    residual_history, process = [], []

    for i, t in enumerate(iterator):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)

        eps_t = model(x, t_batch)
        x0_hat = predict_x0_from_eps(x, eps_t, alpha_bar[t])

        residual = x0_hat.flatten(1) @ A.T - y
        grad_x0 = (residual @ A) / (sigma_y**2 * A.shape[0])
        grad_x0 = grad_x0.view_as(x0_hat)

        step_t = guidance_scale * torch.sqrt(1 - alpha_bar[t])
        grad_norm = grad_x0.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-8)
        x0_guided = x0_hat - step_t * grad_x0 / grad_norm

        if clip_range is not None:
            x0_guided = x0_guided.clamp(*clip_range)

        x = ddim_update(x0_guided, eps_t, t, t_prev, alpha_bar, eta=eta)
        residual_history.append(residual.norm(dim=1).mean().item())
        process.append(x0_guided[0].detach().cpu())

    return x, residual_history, process
