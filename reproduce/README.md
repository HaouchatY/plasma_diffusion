# Reproducing the paper

This folder contains everything needed to reproduce the figures, tables and numbers of

> D. Hamm\*, Y. Haouchat\*, C. Theiler, M. Unser,
> *Diffusion Priors for Plasma Tomography: Bayesian Data-Driven Methods Applied to SXR Reconstruction*

All experiments use the public `plasma_diffusion` package of this repository and the
pretrained prior `data/prior_sxr.pth` (the checkpoint used for every result in the paper).
The reaction-diffusion (RD) baseline code is vendored in `tomo_fusion/`.

## Installation

```bash
git clone https://github.com/HaouchatY/plasma_diffusion.git
cd plasma_diffusion
pip install -e .
pip install -r reproduce/requirements.txt     # adds pyxu / pyxu-diffops for the RD baseline
```

A CUDA GPU is strongly recommended for the diffusion runs (DiffPIR/DiffRD); the RD baseline
runs on CPU. LaTeX (`latex`, `dvips`, `type1cm`) is used for colourbar labels when available,
otherwise matplotlib's built-in Computer Modern fonts are used.

Outputs are written to `reproduce/results/` (experiment outputs) and `reproduce/figures/`
(figures and tables). Override with `PD_RESULTS_DIR` / `PD_FIGURES_DIR`. All figure files
use the file names of the paper's LaTeX sources.

## Paper item → script

| Paper item | Script (in `reproduce/`) | Needs | Cost |
|---|---|---|---|
| Fig. 1 — lines of sight (SXR, DMPX, Pilatus) | `04_figures/fig01_diagnostics.py` | inputs | seconds |
| Fig. 2 — test phantoms (+ shared colourbar) | `04_figures/fig02_dataset.py` | inputs | seconds |
| Figs. 3, 4 — prior sampling process, prior samples | `04_figures/fig03_04_prior_sampling.py` | prior | ~10 s GPU |
| Fig. 5 — posterior sampling process (DiffPIR, DiffRD) | `04_figures/fig05_posterior_sampling_process.py` | prior | ~1 min GPU |
| Fig. 6, 7 — DiffPIR vs RD, SXR / DMPX | `04_figures/fig06_07_diffpir_vs_rd.py` | phantom studies | seconds |
| Fig. 8 — Pilatus, all methods | `04_figures/fig08_pilatus.py` | phantom studies | seconds |
| Fig. 9 — posterior std vs error (Pilatus) | `04_figures/fig09_std_vs_err.py` | phantom studies | seconds |
| Tables 1–3 (incl. Wilcoxon significance) | `05_tables/make_tables.py` | phantom studies | seconds |
| UQ statistics in the text (r<sub>σ,err</sub>, r<sub>μ,err</sub>, coverage) | `05_tables/uq_statistics.py` | phantom studies | ~4 min |
| Fig. 10 — TCV #85270, RD vs DiffPIR | `03_real_data/run_shot_85270.py` → `04_figures/fig10_real_data_85270.py` | prior | see below |
| Fig. 11 — #85270 posterior samples | `04_figures/fig11_real_data_posterior_samples.py` | run_shot_85270 | seconds |
| Fig. 12 — #88767 background and fluctuations | `03_real_data/run_shot_88767.py` → `04_figures/fig12_real_data_88767.py` | prior | see below |
| Fig. 13 — limitations (doublet, strike point) | `04_figures/fig13_limitations.py` | prior | ~25 min CPU |

Pipeline:

```
00_prepare/          (optional) regenerate the shipped inputs in data/
01_train/            (optional) retrain the prior            -> data/prior_sxr.pth
02_phantom_studies/  DiffPIR / DiffRD / RD on 1000 test phantoms -> results/phantom_studies/
03_real_data/        TCV shots #85270, #88767                 -> results/real_data/
04_figures/, 05_tables/   figures and tables                  -> figures/
```

## Inputs (`reproduce/data/`)

| File | Content |
|---|---|
| `test_set/sxr_samples_with_background_coarse.npy`, `psis_coarse.npy`, `trimming_values.npy`, `alpha_random_values.npy` | the 1000 test phantoms (120×40) and their equilibria |
| `test_set/magnetic_equilibrium.npy` | equilibria used to generate the test set |
| `geometry/*.npy` | geometry matrices: SXR (100 LoS), DMPX (256), Pilatus (80) |
| `noise/noise_realizations_{sxr,dmpx,pilatus}.npy`, `noise/scaling_params.npy` | fixed noise draws used for all phantom studies |
| `y_train_{sxr,dmpx,pilatus}.npy` | projections of the first 10⁴ training profiles (amplitude normalisation) |
| `rd_anis_params_pilatus.csv` | per-phantom anisotropy tuned by the RD baseline, used by DiffRD (see below) |
| `rd_reg_param_tuning/` | tuning data behind the RD regularisation weight λ = 0.1 |
| `experimental/shot_data_{85270,87262}.mat`, `shot_data_full_88767_finetimegrid.mat` | TCV shots (SXR measurements + LIUQE equilibria) |
| `masks/`, `diagnostics_los/` | TCV vessel mask, line-of-sight parameters for Fig. 1 |

The 10⁵ training profiles (3.9 GB) are not included; they are only needed to retrain the prior
(`01_train/train.py`, `PD_TRAIN_DIR=...`).

---

## 1. Preparation and training (optional)

| Script | Produces | Runtime |
|---|---|---|
| `00_prepare/make_test_set.py` | `data/test_set/*` — regenerates the shipped arrays bit-exactly | ~20–30 min CPU |
| `00_prepare/make_noise_realizations.py` | `data/noise/*` (seed 42) | seconds |
| `00_prepare/make_y_train.py` | `data/y_train_*.npy` (needs `PD_TRAIN_DIR`) | ~1 min |
| `01_train/train.py` | prior checkpoint (needs `PD_TRAIN_DIR`) | ~16–19 h on a V100 |

The shipped noise was drawn with `torch.randn` on a CUDA GPU; on CPU, `make_noise_realizations.py`
gives different (statistically equivalent) draws, so keep the shipped files to reproduce the paper
numbers. Training: TinyUNet (318,764 parameters), DDPM with T = 1000 and linear β ∈ [10⁻⁴, 2·10⁻²],
ε-MSE loss, Adam (lr 10⁻³), batch 512; the paper checkpoint is the one after 2194 epochs.

## 2. Phantom studies — Tables 1–3, Figs. 6–9, UQ statistics

```bash
bash reproduce/02_phantom_studies/run_all.sh [diffpir|rd|all]
```

runs every configuration of the paper in dependency order. Individual runs:

| Command (`02_phantom_studies/`) | Method | Output (`results/phantom_studies/`) |
|---|---|---|
| `run_diffpir.py --method diffpir --diag D --sigma S` | DiffPIR (λ = 100 SXR, 300 DMPX, 200 Pilatus; ζ = 0.8) | `diffpir_D_S/` |
| `run_diffpir.py --method diffrd_clip --diag pilatus --sigma S` | DiffPIR restricted to the reconstruction region (λ = 100, ζ = 0.8) | `diffrd_clip_pilatus_S/` |
| `run_diffpir.py --method diffrd --diag pilatus --sigma S` | DiffRD, (λ, ζ, λ<sub>RD</sub>) = (100, 0.9, 10⁻⁵) at η<sub>low</sub>, (10, 0.9, 10⁻⁴) at η<sub>med/high</sub> | `diffrd_pilatus_S/` |
| `run_rd.py --diag D --sigma S` | RD: anisotropy tuned by cross-validation, λ = 0.1, MAP + 10⁵ ULA samples | `rd_D_S/` |
| `run_rd.py --diag pilatus --sigma 0.07 --cv-strategy by_camera`, `--sigma {0.35,0.7} --cv-strategy random` | Pilatus anisotropy tuning | `rd_cvcam_pilatus_0.07/`, `rd_cv_pilatus_S/` |
| `run_rd.py --diag pilatus --sigma S --alpha-from-tuning` | Pilatus RD of Table 3: tuned anisotropy, 5·10⁵ ULA samples + 10⁴ burn-in | `rd_pilatus_S/` |
| `merge_chunks.py --all` | merges runs split with `--start/--end` | |

Noise levels (η<sub>low/med/high</sub>): SXR 0.05/0.25/0.5, DMPX 0.1/0.5/1.0, Pilatus 0.07/0.35/0.7.
Each study directory holds `metrics.csv` (one row per phantom) and `post_means.npy`,
`post_stds.npy` (+ `maps.npy` for RD), each of shape (1000, 4800). Add `--procedure` to rerun with
the amplitude-normalisation procedure.

DiffRD uses the anisotropy parameter tuned by the RD baseline for each phantom. If the RD tuning
runs are not available, `run_diffpir.py --method diffrd` uses the values from the paper runs
(`data/rd_anis_params_pilatus.csv`), so DiffRD can be rerun without ~240 CPU-h of RD tuning.

Hyperparameter searches: `tuning/tune_diffpir.py --diag D [--rd]` (DiffPIR / DiffRD grid search on
10 phantoms) and `tuning/tune_rd_reg_param.py --diag D` (RD weight λ).

Then:

```bash
python reproduce/04_figures/fig06_07_diffpir_vs_rd.py
python reproduce/04_figures/fig08_pilatus.py
python reproduce/04_figures/fig09_std_vs_err.py
python reproduce/05_tables/make_tables.py      # Tables/{sxr,dmpx,pilatus}.tex + wilcoxon_pvalues.csv
python reproduce/05_tables/uq_statistics.py
```

Table entries are mean ± std over the 1000 phantoms; bold/stars follow a one-sided paired Wilcoxon
signed-rank test (best method vs. each competitor, p < 0.05). SSIM values are truncated to two
decimals as in the paper (`--ssim-format round` for rounding). The figure scripts accept
`--diag/--sigma/--idx` to show any other phantom.

**Compute.** DiffPIR ≈ 10 s and DiffRD ≈ 14 s per phantom on a V100 (100 samples, 250 steps),
≈ 50 GPU-h in total. RD ≈ 80–100 s per phantom and core (≈ 380 s with 5·10⁵ ULA samples),
≈ 1100 CPU-h in total; `run_all.sh` splits each RD run over `NPROC` processes (default 20).

## 3. Experimental data — Figs. 10–12

```bash
python reproduce/03_real_data/run_shot_85270.py     # 28 time slices  -> results/real_data/85270/
python reproduce/03_real_data/run_shot_88767.py     # background (11) + fluctuation (10) slices
python reproduce/04_figures/fig10_real_data_85270.py
python reproduce/04_figures/fig11_real_data_posterior_samples.py
python reproduce/04_figures/fig12_real_data_88767.py
```

Per time slice: amplitude normalisation against `y_train_sxr.npy` with σ = 0.25 (normalised units);
RD with cross-validated anisotropy over `logspace(-4, 0, 13)`, λ = 0.1, positivity and core mask
(MAP); DiffPIR with λ = 100, ζ = 0.8, 100 samples. The cross-validation dominates the cost
(~30 min per slice on one CPU process; ~10 s per slice for DiffPIR on a GPU).
For #88767, the background uses frames 900:2000:100 and the fluctuations frames 1350:1450:10 after
subtracting the mean of frames 1000–1049.

Result layout per run: `posterior_tomo.npy` (T, 100, 1, 120, 40) DiffPIR samples in normalised
units (multiply by α), `im_maps.npy` RD MAP in physical units, `params.npy` (T, 3) = [α, anisotropy,
σ], `inv_times.npy`, `frames.npy`.

`fig12_real_data_88767.py --as-published` reproduces the panel/time pairing of the submitted figure.

## 4. Limitations — Fig. 13

```bash
python reproduce/04_figures/fig13_limitations.py              # both rows
python reproduce/04_figures/fig13_limitations.py --plot-only  # re-plot from results/limitations/*.npz
```

Top row: a doublet built by anisotropic diffusion (2000 steps, α = 5·10⁻²) of two seed pixels along
the flux surfaces of #87262 (time index 26). Bottom row: test phantom #7 with a Gaussian blob at the
outer strike point. σ<sub>y</sub> = 0.25; RD with MAP + 10⁵ ULA samples; DiffPIR λ = 100, ζ = 0.8.
The script also prints the doublet PSNR/SSIM of RD (MAP), μ<sub>RD</sub> and μ<sub>DiffPIR</sub>
mentioned in the text.

---

## Reproducibility notes

- `plasma_diffusion.sample_posterior_diffpir` is bit-identical to the research code used for the
  paper for a fixed seed (with and without the RD term and clipping mask).
- RD runs are deterministic: reruns reproduce the paper's per-phantom anisotropy, PSNR/SSIM and
  posterior means to all digits.
- DiffPIR/DiffRD were run on a CUDA GPU, seeded once (seed 42) and processing phantoms
  sequentially. Individual posterior draws depend on hardware and chunking; the reported statistics
  do not (reruns differ by the Monte-Carlo error of a 100-sample average). The scripts disable TF32
  so that recent GPUs follow the same trajectories as the original V100 runs.
- Figs. 3–5 are stochastic: run on a GPU to obtain the panels of the paper.
- The measurement noise of Fig. 13 was not seeded in the original run, so the reconstructions and
  printed metrics differ slightly from the paper; the phantoms are identical.
