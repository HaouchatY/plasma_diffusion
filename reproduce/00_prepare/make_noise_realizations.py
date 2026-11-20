"""Fixed noise draws and amplitude scalings for the phantom studies (Tables 1-3).

For each diagnostic d, ``noise_realizations_{d}.npy`` (1000, M_d) holds one standard
normal vector per test phantom; the noisy data are y_i = A x_i + sigma * n_i. The same
draws are reused for every noise level and method, so all methods see identical data.
``scaling_params.npy`` (1000,) holds random amplitudes in [1, 1000] used only by the
amplitude-normalisation check (``--procedure``). Seed 42 for both.

NOTE: the shipped arrays were drawn with ``torch.randn`` on a CUDA device, whose RNG
differs from the CPU one. Running this script on CPU produces statistically
equivalent but different noise vectors (scaling_params is identical). Keep the
shipped arrays to reproduce the paper's numbers exactly. Runtime: seconds.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import common  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=cfg.DATA / "noise")
    args = ap.parse_args()
    dev = common.device()
    if dev.type != "cuda":
        print("WARNING: CPU RNG -> noise vectors will differ from the shipped (CUDA-drawn) ones")
    args.out.mkdir(parents=True, exist_ok=True)
    for diag in cfg.GEOMETRY:
        M = common.load_geometry(diag).shape[0]
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        noise = torch.randn(cfg.NUM_TEST, M, device=dev).cpu().numpy()
        scaling = 1 + 999 * np.random.rand(cfg.NUM_TEST)
        np.save(args.out / f"noise_realizations_{diag}.npy", noise)
        print(f"{diag}: noise {noise.shape}")
    np.save(args.out / "scaling_params.npy", scaling)


if __name__ == "__main__":
    main()
