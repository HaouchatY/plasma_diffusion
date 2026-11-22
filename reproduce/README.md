# Reproducing the paper

Scripts and input data to reproduce the results of *Diffusion Priors for Plasma Tomography*.

## Installation

```bash
pip install -e .
pip install -r reproduce/requirements.txt
```

A CUDA GPU is recommended for the diffusion runs; the RD baseline runs on CPU.

## Usage

```bash
# 1. phantom studies (DiffPIR, DiffRD, RD on SXR, DMPX, Pilatus)
bash reproduce/02_phantom_studies/run_all.sh

# 2. experimental TCV shots
python reproduce/03_real_data/run_shot_85270.py
python reproduce/03_real_data/run_shot_88767.py

# 3. figures and tables
python reproduce/04_figures/<script>.py
python reproduce/05_tables/make_tables.py
python reproduce/05_tables/uq_statistics.py
```

Results are written to `reproduce/results/`, figures and tables to `reproduce/figures/`.

## Structure

```
00_prepare/          regenerate the inputs in data/ (optional)
01_train/            train the prior (optional, training set not included)
02_phantom_studies/  experiments on the 1000 test phantoms
03_real_data/        experimental TCV shots
04_figures/          one script per figure
05_tables/           tables and uncertainty statistics
data/                test phantoms, geometries, noise realizations, TCV shots
tomo_fusion/         RD baseline
config.py            paths and hyperparameters
```

## Notes

- The phantom studies are expensive: about 50 GPU-hours for DiffPIR/DiffRD and about 1100 CPU-hours for RD.
- DiffRD uses the anisotropy tuned by RD. If the RD runs are missing, the values from the paper (`data/rd_anis_params_pilatus.csv`) are used.
- RD is deterministic. DiffPIR samples depend on the hardware, but the reported statistics do not.
