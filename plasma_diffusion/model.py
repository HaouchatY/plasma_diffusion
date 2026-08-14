"""Noise-prediction network for the diffusion prior.

A small time-conditioned U-Net (~300k parameters) that predicts the noise
component of a corrupted emissivity profile. The architecture is intentionally
lightweight: TCV soft X-ray profiles are 120 x 40 images, and a compact model
is sufficient to learn an expressive prior while keeping posterior sampling
fast.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

__all__ = ["TinyUNet", "load_pretrained"]


def _sinusoidal_embedding(n_steps: int, dim: int) -> torch.Tensor:
    """Fixed sinusoidal embedding table of shape (n_steps, dim)."""
    positions = torch.arange(n_steps, dtype=torch.float32).unsqueeze(1)
    div_term = torch.pow(
        10000.0, (2 * torch.arange(dim, dtype=torch.float32)) / dim
    ).unsqueeze(0)
    emb = positions / div_term
    emb[0::2] = torch.sin(emb[0::2])
    emb[1::2] = torch.cos(emb[1::2])
    return emb


class _Conv(nn.Module):
    """Conv2d + GroupNorm + SiLU."""

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, normalize: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(1, out_c) if normalize else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


def _block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(_Conv(in_c, out_c), _Conv(out_c, out_c), _Conv(out_c, out_c))


def _up_block(in_c: int) -> nn.Sequential:
    return nn.Sequential(_Conv(in_c, in_c // 2), _Conv(in_c // 2, in_c // 4),
                         _Conv(in_c // 4, in_c // 4))


class TinyUNet(nn.Module):
    """Time-conditioned U-Net noise predictor ``eps_theta(x_t, t)``.

    Parameters
    ----------
    in_c, out_c:
        Number of input/output channels (1 for emissivity profiles).
    n_steps:
        Number of diffusion training steps (size of the time-embedding table).
    time_emb_dim:
        Dimension of the sinusoidal time embedding.
    """

    def __init__(self, in_c: int = 1, out_c: int = 1, n_steps: int = 1000,
                 time_emb_dim: int = 100):
        super().__init__()

        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = _sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.weight.requires_grad_(False)

        self.te1 = self._make_te(time_emb_dim, in_c)
        self.b1 = _block(in_c, 10)
        self.down1 = nn.Conv2d(10, 10, kernel_size=4, stride=2, padding=1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = _block(10, 20)
        self.down2 = nn.Conv2d(20, 20, kernel_size=4, stride=2, padding=1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = _block(20, 40)
        self.down3 = nn.Conv2d(40, 40, kernel_size=4, stride=2, padding=1)

        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(_Conv(40, 20), _Conv(20, 20), _Conv(20, 40))

        self.up1 = nn.ConvTranspose2d(40, 40, kernel_size=4, stride=2, padding=1)
        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = _up_block(80)

        self.up2 = nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = _up_block(40)

        self.up3 = nn.ConvTranspose2d(10, 10, kernel_size=4, stride=2, padding=1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = _block(20, 10)

        self.conv_out = nn.Conv2d(10, out_c, kernel_size=3, stride=1, padding=1)

    def _make_te(self, dim_in: int, dim_out: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(),
                             nn.Linear(dim_out, dim_out))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        te = self.time_embed(t)
        b = x.size(0)

        h1 = self.b1(x + self.te1(te).view(b, -1, 1, 1))
        h2 = self.b2(self.down1(h1) + self.te2(te).view(b, -1, 1, 1))
        h3 = self.b3(self.down2(h2) + self.te3(te).view(b, -1, 1, 1))

        h_mid = self.b_mid(self.down3(h3) + self.te_mid(te).view(b, -1, 1, 1))

        u1 = torch.cat([h3, self.up1(h_mid)], dim=1)
        u1 = self.b4(u1 + self.te4(te).view(b, -1, 1, 1))

        u2 = torch.cat([h2, self.up2(u1)], dim=1)
        u2 = self.b5(u2 + self.te5(te).view(b, -1, 1, 1))

        u3 = torch.cat([h1, self.up3(u2)], dim=1)
        u3 = self.b_out(u3 + self.te_out(te).view(b, -1, 1, 1))

        return self.conv_out(u3)


def load_pretrained(path: str | Path, device: str | torch.device = "cpu",
                    n_steps: int = 1000) -> TinyUNet:
    """Load a trained :class:`TinyUNet` checkpoint.

    Accepts either a plain ``state_dict`` of the network or a checkpoint of the
    full training wrapper (keys prefixed with ``network.``).
    """
    device = torch.device(device)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("network.") for k in state):
        state = {k[len("network."):]: v for k, v in state.items()
                 if k.startswith("network.")}
    model = TinyUNet(n_steps=n_steps).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
