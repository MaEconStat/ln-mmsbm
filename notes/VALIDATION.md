# Validation protocol — separate coding correctness from scientific replication

## A. Current automated tests

Run `python -m pip install -e ".[test]" && pytest`.

Tests check input validation, symmetry and orientation, output ranks/dimensions, orthogonality and tensor reconstruction, random-seed repeatability, analytic conditional Gaussian block MLE, and layer-centering invariance to multiplicative layer scaling. They **do not** constitute a published benchmark or a formal comparison against the original R package.

## B. Required before describing as a faithful R/Python reproduction

1. Pin the exact rMultiNet Git commit `0d922beb9abdff7255c65d0378c3097fea2cdf69` (main as inspected on 2026-10-26). Create small deterministic tensor fixtures in neutral numerical format; note R uses column-major array order and NumPy C-order unless specified.
2. Run R `InitializationMMSBM` / `PowerIteration` and Python symmetric baseline on the same fixtures. Compare column spaces by projection matrix distances `||U U^T - V V^T||_F` (not vector entries or signs); compare layer partitions by ARI; compare reconstruction residuals.
3. Record differences: R row clipping vs paper's clipped+reorthogonalized equation; R sequential updates vs paper's two-factor update; R final core construction. If exact replication is important, create a compatibility backend for each behavior rather than modifying the main math-correct approach silently.
4. Report versions of R, rTensor, Python, NumPy, scikit-learn, BLAS, and random seeds. Preserve R raw results / script as a regression fixture (with proper attribution).

## C. Scientific benchmarks before public performance claims

- Binary symmetric MMSBM: compare paper-style TWIST vs R, global-node/layer/local ARI, averaged over ≥100 seeds with varying n, L, sparsity p, separation and m.
- Positive continuous: compare log-TWIST (this package) to raw-weight TWIST, mode-3 spectral flattening, a pooled-SVD reference, and a direct Gaussian block likelihood procedure if feasible.
- Evaluate layer ARI, type-aligned source/target local ARI, global ARI, held-out log-likelihood and elapsed seconds, with confidence intervals.
- Vary sigma, block separation, K, layer balance, heteroskedasticity, influential nodes and layer intercepts; include log-scale QQ and zero-rate diagnostics.
- Test real positive flow networks only after verifying provenance, missingness, zeros, monetary deflators and producer/consumer node semantics.
- If adding zeros, compare a hurdle likelihood fitted with positivity indicators; `log1p` can be included as a predictive heuristic baseline but never labeled an exact lognormal fit.
- Establish train/test splits by held-out edges/layers without data leakage; check label alignment before interpreting type-specific ARI.

## D. Release gates

- [x] Code can be unit tested without original or private datasets.
- [x] Attribution, scope, mathematical assumptions documented.
- [ ] Verified against original R results.
- [ ] Multi-seed quantitative benchmark and uncertainty intervals.
- [ ] Parameter/rank selection sensitivity and robustness checks.
- [ ] Real-data validation with network semantics audit.
- [ ] Zero/missing-value generative extension.
- [ ] Continuous-model error-bound proof.

No unsupported checkboxes should be presented as completed.
