#!/usr/bin/env bash
# All phantom-study runs behind Tables 1-3, Figs. 6-9 and the UQ statistics.
#
# Compute (paper setup):
#   DiffPIR SXR / DMPX / Pilatus, DiffRD_clip  ~3 GPU-h per (diag, sigma)  -> ~36 GPU-h
#   DiffRD Pilatus                             ~4 GPU-h per sigma          -> ~12 GPU-h
#   RD SXR / DMPX (CV + 1e5 ULA)               ~90 CPU-h per (diag, sigma) -> ~550 CPU-h
#   RD Pilatus alpha tuning (CV + 1e5 ULA)     ~80 CPU-h per sigma         -> ~240 CPU-h
#   RD Pilatus final (5e5 ULA)                 ~110 CPU-h per sigma        -> ~320 CPU-h
# RD runs are split over NPROC processes (the original used 20 x 50 phantoms).
#
# Usage: bash run_all.sh [diffpir|rd|all]   (PD_RESULTS_DIR to change the output location)
# Add --procedure to any command to rerun the amplitude-normalisation check
# (paper: "phantom studies confirmed the procedure"); this doubles the cost.
set -euo pipefail
cd "$(dirname "$0")"
WHAT=${1:-all}
NPROC=${NPROC:-20}
CHUNK=$((1000 / NPROC))

rd_parallel() {  # rd_parallel <run_rd.py args...>
  for i in $(seq 0 $((NPROC - 1))); do
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u run_rd.py "$@" --start $((i * CHUNK)) --end $(((i + 1) * CHUNK)) &
  done
  wait
}

if [[ $WHAT == diffpir || $WHAT == all ]]; then
  for s in 0.05 0.25 0.5; do python -u run_diffpir.py --method diffpir --diag sxr --sigma $s; done
  for s in 0.1 0.5 1.0;   do python -u run_diffpir.py --method diffpir --diag dmpx --sigma $s; done
  for s in 0.07 0.35 0.7; do
    python -u run_diffpir.py --method diffpir     --diag pilatus --sigma $s
    python -u run_diffpir.py --method diffrd_clip --diag pilatus --sigma $s
  done
fi

if [[ $WHAT == rd || $WHAT == all ]]; then
  for s in 0.05 0.25 0.5; do rd_parallel --diag sxr --sigma $s;  done
  for s in 0.1 0.5 1.0;   do rd_parallel --diag dmpx --sigma $s; done
  # Pilatus: alpha tuning, then long ULA chains with the tuned alpha
  rd_parallel --diag pilatus --sigma 0.07 --cv-strategy by_camera
  for s in 0.35 0.7; do rd_parallel --diag pilatus --sigma $s --cv-strategy random; done
  python merge_chunks.py --all
  for s in 0.07 0.35 0.7; do rd_parallel --diag pilatus --sigma $s --alpha-from-tuning; done
  python merge_chunks.py --all
fi

if [[ $WHAT == all ]]; then
  # DiffRD uses the RD-tuned anisotropy alpha of each phantom
  for s in 0.07 0.35 0.7; do python -u run_diffpir.py --method diffrd --diag pilatus --sigma $s; done
fi
