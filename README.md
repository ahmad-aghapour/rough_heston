
# Rough Heston (Markov Approximation) on GPU + Diffusion Model for Conditional Surface Sampling

This project uses a **Markov approximation of the rough Heston model** to **parallelize option pricing on GPU**. With this GPU acceleration we generate a large dataset of option price surfaces, train a **diffusion model** on those surfaces, and then **sample full surfaces conditionally on partial observations** (e.g., a handful of points on the grid).



## Method & References

**Foundation (rough Heston simulation).**  
The GPU pricer follows the fast simulation approach for rough volatility introduced by **Ma & Wu (2022)** in *Quantitative Finance*: “A fast algorithm for simulation of rough volatility models.” We adopt the **Markov (multi-factor) approximation** of the fractional kernel to enable batched, vectorized simulation suitable for GPUs.  
Link: https://doi.org/10.1080/14697688.2021.1970213

For early times (near \(t=0\)), we improve accuracy using the **Hybrid Scheme** of **Bennedsen, Lunde & Pakkanen (2017)** for Brownian semistationary processes: approximate the kernel by a power law near zero and by a step function elsewhere.  

To efficiently apply the rough kernel history term, we approximate the fractional kernel with a **sum of exponentials (SOE)** following **Li (2010)**’s fast time-stepping method for fractional integrals. This yields simple recursive updates (no full history storage) and maps naturally to GPU.  


---

## What This Project Does
1. **GPU Monte Carlo pricing** under (approximate) rough Heston:
   - Markovian reformulation via exponential-sum kernel.
   - Fully vectorized PyTorch ops; prices & variance simulated in batches.
2. **Dataset generation**:
   - Sample model parameters across ranges; for each sample, compute a **(T, K)** grid (e.g., **32×32**) of option prices.
3. **Diffusion model training**:
   - Treat each price grid as a single-channel image and train a **variance-preserving diffusion model (UNet)** to learn the distribution of realistic surfaces.
4. **Conditional sampling from partial observations**:
   - Given **sparse observations** (subset of grid points), perform **guided reverse diffusion** to reconstruct a plausible **full surface** consistent with those observations.


## Quick Start
> Adjust paths and hyperparameters to match your setup. Start small (smoke test), then scale up on GPU.

```bash
# 1) Install
pip install -r requirements.txt
````

```python
# 2) Simulate a small surface (example)
import torch, numpy as np
from rough_vol import (
  simulate_rough_vol_batch_torch,
  compute_vol_surface,
  f_heston_like_torch, g_heston_like_torch
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = dict(alpha=0.30, theta=0.13, rho=-0.5, V0=0.04, kappa=0.10, nu=0.13)

# batched path simulation (tiny sizes for a quick check)
t,S,V = simulate_rough_vol_batch_torch(
  alpha=params["alpha"],
  f=lambda V: f_heston_like_torch(V, kappa=params["kappa"], theta=params["theta"]),
  g=lambda V: g_heston_like_torch(V, nu=params["nu"]),
  V0=params["V0"], S0=100.0, r=0.0, rho=params["rho"],
  T=0.2, N=50, N_exp=2, M=500, device=device
)

# build a (T,K) price grid (default 32×32)
surface = compute_vol_surface(tuple(params.values()), device=device, show=False)
np.save("data/surfaces/example_surface.npy", surface)
```


here’s a drop-in replacement with **4 images** (the fourth is the reconstruction **error**). just paste this into your README:


## Results (placeholders — add your figures)

| Observed points                                | Reconstruction (ours)                            | Ground truth                              | Error (ours vs. ground truth)               |
| ---------------------------------------------- | ------------------------------------------------ | ----------------------------------------- | ------------------------------------------- |
| <img src="figs\Option Price partial observation.png" width="220" alt="Observed 100 pts"/> | <img src="figs/Postorior sampling option price rHeston.png" width="220" alt="Reconstruction"/> | <img src="figs/Actual Option Prices rHeston.png" width="220" alt="Ground truth"/> | <img src="figs/Actual Option Prices rHeston.png" width="220" alt="Error heatmap"/> |




* Grid: **32×32** maturities $T$ and strikes $K$
* Inference: **50** guided reverse-diffusion steps
* Metric: relative MSE on the full grid (report your numbers here)


## Acknowledgements

* Ma & Wu (2022), *A fast algorithm for simulation of rough volatility models*, Quantitative Finance. [https://doi.org/10.1080/14697688.2021.1970213](https://doi.org/10.1080/14697688.2021.1970213)
* Bennedsen, Lunde & Pakkanen (2017), *Hybrid scheme for Brownian semistationary processes*, Finance and Stochastics. [https://link.springer.com/article/10.1007/s00780-017-0335-5](https://link.springer.com/article/10.1007/s00780-017-0335-5)
* Li (2010), *A fast time stepping method for evaluating fractional integrals*, SIAM J. Sci. Comput. [https://epubs.siam.org/doi/10.1137/080736533](https://epubs.siam.org/doi/10.1137/080736533)


