"""Merge chunked phantom-study outputs (run_diffpir.py / run_rd.py with --start/--end).

    python merge_chunks.py results/phantom_studies/rd_sxr_0.25 [more dirs ...]
    python merge_chunks.py --all          # every study directory that has chunks/

Writes metrics.csv (sorted by sample_idx) and the concatenated (1000, H*W) arrays.
Runtime: seconds.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import _study  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    dirs = args.dirs
    if args.all:
        dirs = sorted(p.parent for p in (cfg.RESULTS / "phantom_studies").glob("*/chunks"))
    if not dirs:
        ap.error("no study directories given")
    for d in dirs:
        _study.merge(d)


if __name__ == "__main__":
    main()
