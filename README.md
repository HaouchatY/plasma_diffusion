# Diffusion Priors for Plasma Tomography

Official code for *Diffusion Priors for Plasma Tomography: Bayesian Data-Driven Methods Applied to SXR Reconstruction*,
D. Hamm\*, Y. Haouchat\*, C. Theiler, M. Unser.

A diffusion model is trained as a prior on plasma emissivity profiles and used for posterior sampling
with DiffPIR, giving a reconstruction and an uncertainty map for any diagnostic geometry.

## Installation

```bash
pip install -e .
```

## Usage

```python
import numpy as np
import plasma_diffusion as pdiff

model = pdiff.load_pretrained("data/prior_sxr.pth", device="cuda")
schedule = pdiff.build_schedule(device="cuda")

A = np.load("data/geometry_sxr.npy")
A = A / A.max()
x = np.load("data/phantoms_example.npy")[0]
y = A @ x.ravel() + 0.05 * np.random.randn(A.shape[0])

posterior, _, _ = pdiff.sample_posterior_diffpir(
    model, schedule, A, y, num_samples=50, sigma_y=0.05, lambda_=100, zeta=0.8,
)
mean, std = posterior.mean(0)[0], posterior.std(0)[0]
```

See [`tutorial.ipynb`](tutorial.ipynb) for a complete example, and [`reproduce/`](reproduce/) to reproduce the paper.

## Citation

```bibtex
@article{hamm2026diffusion,
  title   = {Diffusion Priors for Plasma Tomography: Bayesian Data-Driven
             Methods Applied to SXR Reconstruction},
  author  = {Hamm, Daniele and Haouchat, Youssef and Theiler, Christian
             and Unser, Michael},
  year    = {2026},
}
```

## License

[MIT](LICENSE)
