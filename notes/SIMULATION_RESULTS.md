# Exploratory simulation, 2026-10-26

This is an **engineering smoke-test comparison**, not a formal Monte Carlo study or a claim of statistical superiority. Run from the repository root with `python experiments/simulation_benchmark.py`.

- Model: 42 source nodes, 39 target nodes, 10 layers, 2 network types, 2 source groups per type, 2 target groups per type.
- Block log means drawn independently from `Normal(0, contrast²)` per seed, balanced memberships, common Gaussian log-scale noise with standard deviation `sigma`.
- Three seeds each for six settings (18 fits total; seeds are reused across regimes as a paired comparison).
- Baseline comparison: run identical regularized rectangular Tucker decomposition on **raw positive weights**, then K-means on layer-factor rows. This is a heuristic raw-weight baseline, not the original Bernoulli paper's statistical model.
- ARI = adjusted Rand index (1 perfect; 0 chance-level expectation; negative possible). Global labels encode each node's cross-type membership tuple. Local-type metrics align estimated layer-type labels with ground truth using optimal assignment before calculating ARIs.

| Block contrast | Sigma | Log-TWIST layer ARI | Raw-weight layer ARI | Log-TWIST source global ARI | Log-TWIST target global ARI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.8 | 1.000 | 0.667 | 0.826 | 0.680 |
| 0.2 | 1.5 | 0.640 | 0.000 | 0.242 | 0.143 |
| 0.6 | 0.8 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6 | 1.5 | 1.000 | 0.000 | 1.000 | 0.953 |
| 1.6 | 0.8 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1.6 | 1.5 | 1.000 | 0.000 | 1.000 | 1.000 |

These are means of **only three seeds** for each setting. In the weak-block/high-noise regime (`contrast=0.2`, `sigma=1.5`), layer and node communities deteriorate substantially. Results depend on the particular random seed, simulation design, tuning, and available layer count; do **not** infer universal advantages from this table. Runtime varies by computer environment, and no confidence interval or held-out likelihood comparison has been completed.

Before homepage publication of quantitative claims: ≥100 repetitions per setting, balanced and imbalanced layer frequencies, rank/cluster-count sensitivity, heteroskedastic log-noise, structural-zero scenarios, R baseline numerical replication, and uncertainty intervals.
