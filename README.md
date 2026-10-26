# LN-MMSBM: Continuous-Weight Multilayer Stochastic Block Model in Python

A research prototype implementing **regularized tensor power iterations** for directed, continuous-weight economic flow networks. 

This project extends the Mixture Multilayer Stochastic Block Model (MMSBM) and the TWIST algorithm from [Jing et al. (*Annals of Statistics*, 2021)](https://doi.org/10.1214/21-AOS2079) from binary, symmetric graphs to **asymmetric, rectangular, lognormal-weighted tensors**.

---

## Model Formulation

Let $\mathcal{X} \in \mathbb{R}_{>0}^{N_P \times N_C \times L}$ be a 3-way continuous flow tensor observed across $N_P$ source nodes (producers), $N_C$ target nodes (consumers), and $L$ layers (e.g., commodity sectors, transaction types, or time periods). 

Taking log-transformed flows $\mathcal{Y}_{ijl} = \log \mathcal{X}_{ijl}$, the conditional generative model is:

$$\mathcal{Y}_{ijl} \mid c_l=c, \, z^{P,c}_i=a, \, z^{C,c}_j=b \;\sim\; \mathcal{N}\big(\mu_{c,a,b} + \alpha_l, \, \sigma^2\big)$$

where:
- $c_l \in \{1, \dots, m\}$ is the latent class of layer $l$.
- $z^{P,c}_i \in \{1, \dots, K_P\}$ and $z^{C,c}_j \in \{1, \dots, K_C\}$ denote role-specific community memberships under layer class $c$.
- $\mu_{c,a,b}$ represents the block log-mean flow from source community $a$ to target community $b$ under network type $c$.
- $\alpha_l$ is an optional layer-level intercept (estimated or nuisance-centered).
- $\sigma^2$ is the common conditional residual variance on the log scale.

### Low-Rank Tucker Structure & Rectangular Power Iteration

Conditioned on group assignments, the population mean tensor $\mathbf{\Theta} = \mathbb{E}[\mathcal{Y} \mid \text{labels}]$ admits a low-rank Tucker representation:

$$\mathbf{\Theta} \approx \mathcal{C} \times_1 \mathbf{U}_P \times_2 \mathbf{U}_C \times_3 \mathbf{W}$$

where $\mathbf{U}_P \in \mathbb{R}^{N_P \times r_P}$, $\mathbf{U}_C \in \mathbb{R}^{N_C \times r_C}$, and $\mathbf{W} \in \mathbb{R}^{L \times m}$ span the leading left singular subspaces across the three tensor modes. 

To prevent highly connected hub nodes from skewing the singular subspaces, factors are updated via alternating projections with row-norm clipping (Jing et al., Eq. 3.1):

$$\mathcal{P}(\mathbf{U}) = \operatorname{SVD}_r\Big(\operatorname{clip}_\delta(\mathbf{U})\Big), \quad \text{where } \operatorname{clip}_\delta(\mathbf{U})_{i,:} = \mathbf{U}_{i,:} \cdot \min\left(1, \, \frac{\delta}{\|\mathbf{U}_{i,:}\|_2}\right)$$

The mode-wise factor updates proceed via tensor contractions:
- $\mathbf{U}_P^{(t+1)} = \operatorname{SVD}_{r_P} \Big( \mathcal{M}_1\big(\mathcal{Y} \times_2 \mathcal{P}(\mathbf{U}_C^{(t)})^\top \times_3 \mathcal{P}(\mathbf{W}^{(t)})^\top \big) \Big)$
- $\mathbf{U}_C^{(t+1)} = \operatorname{SVD}_{r_C} \Big( \mathcal{M}_2\big(\mathcal{Y} \times_1 \mathcal{P}(\mathbf{U}_P^{(t)})^\top \times_3 \mathcal{P}(\mathbf{W}^{(t)})^\top \big) \Big)$
- $\mathbf{W}^{(t+1)} = \operatorname{SVD}_{m} \Big( \mathcal{M}_3\big(\mathcal{Y} \times_1 \mathcal{P}(\mathbf{U}_P^{(t)})^\top \times_2 \mathcal{P}(\mathbf{U}_C^{(t)})^\top \big) \Big)$

Following convergence, $K$-means clustering on $\mathbf{W}$ identifies latent layer types, while spectral clustering on the mode contractions recovers global and layer-specific source/target communities.

---

## Installation & Quick Start

Environment requirements: Python $\ge$ 3.10 (tested on Python 3.13 via `uv`).

```bash
# Clone and install in editable mode with development dependencies
git clone https://github.com/MaEconStat/ln-mmsbm-python.git
cd ln-mmsbm-python
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"

# Run test suite
pytest
```

### Minimal Usage Example

```python
from ln_mmsbm import simulate_lognormal, fit_lognormal

# 1. Generate synthetic continuous flow tensor: (n_source, n_target, n_layers)
sim = simulate_lognormal(
    n_source=60, n_target=50, n_layers=12,
    n_types=2, k_source=2, k_target=2,
    seed=42
)

# 2. Fit regularized rectangular tensor decomposition
model = fit_lognormal(
    sim.weights,
    n_types=2,
    k_source=2,
    k_target=2,
    center_layers=True,
    seed=42
)

# 3. Inspect recovered latent structures
print("Recovered layer types:", model.layer_types)
print("Source community partition shape:", model.local_source.shape)
print("Estimated log-scale noise variance:", round(model.log_variance, 4))
```

---

## Empirical Benchmark & Signal-to-Noise Regimes

To evaluate recovery fidelity across varying signal-to-noise ratios (SNR), synthetic benchmarks were conducted across 6 contrast-noise configurations (42 sources, 39 targets, 10 layers, 2 types):

| Block Contrast ($\Delta\mu$) | Noise ($\sigma$) | Layer ARI | Raw-Weight Baseline ARI | Source Global ARI | Target Global ARI |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.2 | 0.8 | **1.000** | 0.667 | 0.826 | 0.680 |
| 0.2 | 1.5 | **0.640** | 0.000 | 0.242 | 0.143 |
| 0.6 | 0.8 | **1.000** | 1.000 | 1.000 | 1.000 |
| 0.6 | 1.5 | **1.000** | 0.000 | 1.000 | 0.953 |
| 1.6 | 0.8 | **1.000** | 1.000 | 1.000 | 1.000 |
| 1.6 | 1.5 | **1.000** | 0.000 | 1.000 | 1.000 |

*Metrics report mean Adjusted Rand Index (ARI) over paired random seeds. Full script: `experiments/simulation_benchmark.py`.*

### Boundary Observations:
1. **SNR Phase Transition**: Under moderate-to-high signal contrast ($\Delta\mu/\sigma \ge 0.75$), regularized log-tensor power iteration perfectly recovers layer classes and community partitions ($\text{ARI} = 1.000$).
2. **Weak-Signal Degradation**: In weak-signal regimes ($\Delta\mu=0.2, \sigma=1.5$), perturbation of the leading singular subspace causes clustering degradation ($\text{ARI} \approx 0.14 - 0.24$), consistent with theoretical information-theoretic limits.
3. **Log-Domain Superiority**: Decomposing raw, unscaled positive weights exhibits catastrophic failure whenever noise variance is elevated ($\text{ARI} = 0.000$).

---

## Methodological Boundaries & Assumptions

- **Strict Positivity**: The current likelihood assumes strictly positive weights ($\mathcal{X}_{ijl} > 0$). Structural zeros in empirical networks represent a qualitative absence of interaction and should be handled via a two-part / Hurdle process ($D_{ijl} \sim \text{Bernoulli}, \, \mathcal{X}_{ijl} \mid D_{ijl}=1 \sim \text{Lognormal}$) rather than arbitrary offsets like $\log(1+x)$.
- **Homoskedastic Quadratic Loss**: The unweighted Frobenius tensor decomposition is equivalent to conditional MLE under equal residual variance $\sigma^2$. Complex heteroskedasticity across sectors requires weighted tensor contractions.
- **Theoretical Guarantees**: Non-asymptotic concentration bounds established for Bernoulli graphs in Jing et al. (2021) do not automatically transfer to continuous Gaussian tensors without separate sub-Gaussian spectral perturbation proofs.

---

## Literature & Attribution

- **Methodological Reference**: Jing, B.-Y., Li, T., Lyu, Z., & Xia, D. (2021). *Community detection on mixture multilayer networks via regularized tensor decomposition*. **The Annals of Statistics**, 49(6), 3181–3205. [DOI: 10.1214/21-AOS2079](https://doi.org/10.1214/21-AOS2079).
- **R Package Reference**: Li, T., Lyu, Z., Ren, C., & Xia, D. (2023). *rMultiNet: An R Package For Multilayer Networks Analysis*. [arXiv:2302.04437](https://arxiv.org/abs/2302.04437).

*License: MIT. This repository contains an independent Python implementation and continuous-weight extension.*
