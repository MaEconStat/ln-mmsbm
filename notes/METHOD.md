# Method specification / 方法说明 (v0.1)

## 1. What the original paper does (attributed)

Jing, Li, Lyu, and Xia (2021) study the **mixture multilayer stochastic block model** (MMSBM). Each layer `ell` belongs to a latent class `c_ell ∈ {1,...,m}`. Each class has a node-community partition `Z_c` and within-/between-community **Bernoulli edge probabilities** `B_c`. The paper derives a low-rank Tucker representation of the population *symmetric* adjacency tensor and proposes TWIST: spectral warm start, capped-row regularization with restoration of an orthogonal rank-r subspace, updates of common node and layer singular spaces, then clustering global/local nodes and layers.

The main paper's Algorithm 1 (arXiv 2002.04457, pp. 11–13 of manuscript) and equations (2.1)–(2.6), (3.1) are the methodological baseline. Unlike rMultiNet, this implementation re-orthogonalizes after row clipping, as equation (3.1) explicitly prescribes; performs parallel mode updates; and projects a final least-squares core from all current orthonormal factors. R `PowerIteration` uses row clipping without explicit reorthogonalization, sequential updates of modes 1–3 and a core step based on a previously projected tensor. Accordingly neither output nor objective trajectory is promised to match numerically.

## 2. Proposed lognormal model (our adaptation, not in Jing et al.)

Let `X ∈ (0,∞)^(n_P × n_C × L)` be a positive producer-to-consumer weight tensor (the roles can correspond to distinct node sets, or the same entities in distinct roles). Let `c_ell` be the latent layer class, and let `z^{P,c}_i`, `z^{C,c}_j` be class-specific source and target community labels. Model:

```text
Y_{ijell} = log X_{ijell}
Y_{ijell} | c_ell=c, z^{P,c}_i=a, z^{C,c}_j=b
             ~ Normal(mu_{c,a,b} + alpha_ell, sigma^2)
```

`alpha_ell` is an OPTIONAL fixed layer intercept (no simultaneous identification of arbitrary intercept and unrestricted block means without a centering constraint). The first version either sets `alpha=0` (`center_layers=False`) or removes the empirical per-layer log mean (`center_layers=True`) for spectral estimation. The latter is a nuisance-correction heuristic, not the result of a jointly identified likelihood model.

For x > 0:

```text
f(x | mu, sigma^2) = exp[-(log(x)-mu)^2 / (2*sigma^2)] / (x*sigma*sqrt(2*pi))
```

At constant `sigma^2`, maximizing conditional lognormal likelihood over the block means is equivalent to minimizing the sum of squares on `Y=log(X)`. Allowing both `mu` and `sigma^2` free, the lognormal family has sufficient statistics `(log x, (log x)^2)`, natural parameters `(mu/sigma^2, -1/(2 sigma^2))`, and support `x>0`. Hence this is one explicit exponential-family extension, **not a solver for arbitrary exponential-family observations**.

## 3. Rectangular low-rank mean tensor

Let `Z^P_c ∈ {0,1}^{n_P × K_P}`, `Z^C_c ∈ {0,1}^{n_C × K_C}`, and `M_c ∈ R^{K_P × K_C}`. Conditional mean matrix at layer `ell`:

```text
E[Y_ell | labels] = Z^P_{c_ell} M_{c_ell} (Z^C_{c_ell})^T + alpha_ell 11^T.
```

With alpha=0 or removed, form tensor `Theta = E[Y | labels]`. Its mode-1, mode-2 and mode-3 ranks are respectively bounded by `r_P ≤ m(K_P-1)+1`, `r_C ≤ m(K_C-1)+1`, and `r_L ≤ m` under common cluster counts and common constant membership columns. No equality is guaranteed: some patterns can be linearly dependent, making component/rank selection non-identifiable.

A Tucker decomposition is:

```text
Theta ≈ C ×1 U_P ×2 U_C ×3 W,
U_P^T U_P = I, U_C^T U_C = I, W^T W = I.
```

The rectangular analogue of TWIST uses alternating regularized projections:

```text
U_P(new) = top-rP left singular vectors of M1(Y ×2 P(U_C)^T ×3 P(W)^T)
U_C(new) = top-rC left singular vectors of M2(Y ×1 P(U_P)^T ×3 P(W)^T)
W(new)   = top-m  left singular vectors of M3(Y ×1 P(U_P)^T ×2 P(U_C)^T)
```

Here `P(U) = SVD_r(row_clip(U, delta))`. Factors are all updated using the *previous* iteration (simultaneous update). The code projects the core `C=Y ×1 U_P^T ×2 U_C^T ×3 W^T`. This minimizes unweighted squared reconstruction error for the chosen orthonormal factor spaces but is not guaranteed to monotonically decrease under clipping or a global optimum.

## 4. Estimation and recovery

1. Initialize source and target spaces by SVD of the sum of log-adjacency layers; initialize layer space from log-tensor contracted by these node spaces.
2. Apply capped-row updates until the relative reconstruction error stabilizes or `max_iter` is reached.
3. Cluster rows of layer embedding using K-means to recover `c_ell` (labels only identifiable up to permutation).
4. Independently cluster rows of source and target embeddings to find joint/global partitions across layer types.
5. For each estimated layer type, average its **log-weight matrices** and perform SVD/K-means on source and target singular embeddings for local partitions. This is a prototype spectral post-processing rule, NOT identical to the paper's symmetric local clustering.
6. Given hard labels, estimate group means by within-block averages of log weights; calculate pooled MLE residual log variance. This second stage is not a joint full likelihood optimum.

## 5. Structural zeros, asymmetric roles, and identifiability

- If X=0 is an actual non-link: `log(0)` is undefined; **do not** replace with `log(1+X)` while retaining lognormal claims. A possible hurdle extension is:
  `D_ijell ~ Bernoulli(rho_{c,a,b})` and `log X | D=1 ~ Normal(mu_{c,a,b},sigma^2_{c,a,b})`, with the proper masked likelihood. This is planned, not implemented.
- If zero means "missing" rather than "no flow", use separate missingness indicators and appropriately masked estimation. Sparse weighted panels also require careful algorithm and theory.
- If n_P=n_C and users want one visualized symmetric matrix, the block dilation `[[0,Y],[Y.T,0]]` is a **computational construction**, not observed same-role edges. The rectangular 3-mode representation is the primary scientific model; the block-dilation idea is left for future compatibility testing.
- Lognormal assumptions are falsifiable. In economic transaction/carbon-flow datasets, heavy tails, dependence, censoring, cross-layer correlation and structural identities may violate independent homogeneous Gaussian log errors.
- With `center_layers=True`, data on the original log scale has a nuisance layer mean added back for parameter calculations; empirical layer normalization can remove meaningful between-layer signal, so report both centered and uncentered sensitivity results.

## 6. Theory that cannot simply be inherited

Bernoulli-specific sparsity, p_max, and concentration statements in Jing et al. **do not transfer** to lognormal-weight observations. On the log scale, possible proofs would require explicit independence or dependence structure, bounded or suitably controlled Gaussian heteroskedastic variance, minimum singular-value separation, incoherence / community balance, enough layers per type, and a warm start. A new perturbation and misclustering theorem must be derived; no formal theorem is claimed here.
