"""Chunked result I/O shared by run_diffpir.py and run_rd.py.

A study directory (see common.study_dir) is filled either directly (full range
0-1000 in one process) or through chunks written to ``<dir>/chunks/`` and merged
by merge_chunks.py into the final ``metrics.csv`` and ``(1000, H*W)`` arrays.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def study_path(name: str, diag: str, sigma: float, procedure: bool = False) -> Path:
    """Directory of a study; ``procedure=True`` runs (amplitude-normalisation check)
    get a ``_procedure`` suffix so they never overwrite the paper runs."""
    suffix = "_procedure" if procedure else ""
    return cfg.RESULTS / "phantom_studies" / f"{name}_{diag}_{sigma}{suffix}"


def save(out_dir: Path, start: int, end: int, rows: list, arrays: dict) -> Path:
    """Save rows (list of dicts) and arrays {name: list of (H*W,) vectors}.

    Full range -> final files in out_dir; partial range -> out_dir/chunks/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    full = start == 0 and end == cfg.NUM_TEST
    target = out_dir if full else out_dir / "chunks"
    target.mkdir(exist_ok=True)
    tag = "" if full else f"_{start:04d}_{end:04d}"
    pd.DataFrame(rows).to_csv(target / f"metrics{tag}.csv", index=False)
    for name, vecs in arrays.items():
        np.save(target / f"{name}{tag}.npy", np.asarray(vecs, dtype=np.float32))
    print(f"saved {len(rows)} phantoms to {target}")
    return target


def merge(out_dir: Path) -> None:
    chunk_dir = out_dir / "chunks"
    csvs = sorted(chunk_dir.glob("metrics_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no chunks in {chunk_dir}")
    tags = [c.stem[len("metrics"):] for c in csvs]
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df = df.sort_values("sample_idx").reset_index(drop=True)
    if len(df) != cfg.NUM_TEST or df.sample_idx.duplicated().any():
        print(f"WARNING: {len(df)} rows / {df.sample_idx.nunique()} unique phantoms in {out_dir}")
    df.to_csv(out_dir / "metrics.csv", index=False)
    names = sorted({p.name[: -len(t) - 4] for t in tags for p in chunk_dir.glob(f"*{t}.npy")})
    for name in names:
        arr = np.concatenate([np.load(chunk_dir / f"{name}{t}.npy") for t in tags])
        np.save(out_dir / f"{name}.npy", arr)
    print(f"merged {len(csvs)} chunks -> {out_dir} ({len(df)} phantoms; arrays: {names})")
