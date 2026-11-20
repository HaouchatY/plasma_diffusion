"""Fig. `fig:training_data` - example synthetic emissivity phantoms, plus the shared colourbar.

Writes Figures/dataset/test_sample_{0..6}.eps (test-set phantoms 0, 1, 12, 16, 22, 25, 55
with their flux surfaces) and Figures/dataset/phantoms_cbar.eps (viridis, [0, 1]), the
colourbar re-used by the prior-sample, posterior-sampling and result figures.
CPU only, a few seconds. Requires a LaTeX installation for the colourbar labels.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402
import common  # noqa: E402
import _plot_utils as pu  # noqa: E402

INDICES = [0, 1, 12, 16, 22, 25, 55]


def save_phantoms_cbar(path: Path) -> None:
    """Viridis colourbar for profiles normalised to [0, 1] (ticks 0, 0.5, 1)."""
    pu.use_latex(True)
    pu.save_colorbar("viridis", 1.0, path, ticks=[0, 0.5, 1], ticklabels=[r"$0$", r"$0.5$", r"$1$"],
                     fontsize=13, figsize=(0.15, 3.1), pad_inches=0.02,
                     tick_length=7, linewidth=0.8)
    pu.use_latex(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indices", nargs="+", type=int, default=INDICES)
    args = ap.parse_args()
    out = cfg.figure_dir("dataset")
    x, psi = common.load_test_set(args.indices)
    for i in range(len(args.indices)):
        pu.save_profile(x[i], out / f"test_sample_{i}.eps", psi=psi[i])
    save_phantoms_cbar(out / "phantoms_cbar.eps")
    print("wrote", out)


if __name__ == "__main__":
    main()
