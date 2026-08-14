"""DDPM training of the diffusion prior on synthetic emissivity profiles.

The network learns to predict the noise added to clean profiles across all
diffusion levels (standard DDPM objective). Run as a module for command-line
training::

    python -m plasma_diffusion.training --data-glob "path/to/sxr_sample_*.npy" \\
        --epochs 2000 --batch-size 512 --out checkpoints/

The training set of the paper consists of physically realistic synthetic
emissivity phantoms built from magnetic equilibria of TCV discharges (see the
paper and its references for the generation procedure).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .diffusion import build_schedule
from .model import TinyUNet

__all__ = ["add_noise", "train"]


def add_noise(x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor,
              schedule: dict) -> torch.Tensor:
    """Forward (noising) process: ``x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) z``."""
    abar = schedule["alpha_bar"][timesteps].view(-1, 1, 1, 1)
    return torch.sqrt(abar) * x0 + torch.sqrt(1 - abar) * noise


def train(profiles: np.ndarray, out_dir: str | Path, num_epochs: int = 2000,
          batch_size: int = 512, lr: float = 1e-3, num_train_steps: int = 1000,
          device: str | torch.device | None = None,
          checkpoint_every: int = 100) -> TinyUNet:
    """Train a :class:`TinyUNet` prior on a stack of clean profiles.

    Parameters
    ----------
    profiles:
        Training profiles of shape ``(K, H, W)``.
    out_dir:
        Directory where checkpoints (``prior_epoch_XXXX.pth``) are written.

    Returns
    -------
    The trained network (last epoch).
    """
    device = torch.device(device if device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = torch.as_tensor(np.asarray(profiles, dtype=np.float32)).unsqueeze(1)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    schedule = build_schedule(num_train_steps=num_train_steps, device=device)
    model = TinyUNet(n_steps=num_train_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            batch = batch.to(device, non_blocking=True)
            noise = torch.randn_like(batch)
            timesteps = torch.randint(0, num_train_steps, (batch.size(0),),
                                      device=device)
            noisy = add_noise(batch, noise, timesteps, schedule)
            loss = F.mse_loss(model(noisy, timesteps), noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1

        print(f"epoch {epoch}: loss = {epoch_loss / len(loader):.5f}")
        if (epoch + 1) % checkpoint_every == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), out_dir / f"prior_epoch_{epoch:04d}.pth")

    return model


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-glob", required=True,
                        help="glob matching the training profile .npy files")
    parser.add_argument("--out", default="checkpoints",
                        help="output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-train-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.data_glob))
    if not paths:
        raise SystemExit(f"no files match {args.data_glob!r}")
    print(f"loading {len(paths)} training profiles...")
    profiles = np.stack([np.load(p) for p in paths])

    train(profiles, args.out, num_epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr,
          num_train_steps=args.num_train_steps,
          checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    _main()
