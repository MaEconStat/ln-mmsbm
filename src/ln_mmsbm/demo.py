"""Run with `python -m ln_mmsbm.demo`. All data are synthetic."""
from __future__ import annotations
from sklearn.metrics import adjusted_rand_score
from .simulate import simulate_lognormal, simulate_bernoulli
from .fit import fit_lognormal
from .binary import fit_bernoulli


def main():
    binary=simulate_bernoulli(n_nodes=80,n_layers=16,seed=21)
    binary_fit=fit_bernoulli(binary.weights,2,2,seed=21)
    weighted=simulate_lognormal(n_source=90,n_target=75,n_layers=18,seed=21)
    fitted=fit_lognormal(weighted.weights,2,2,2,seed=21)
    print("Synthetic Bernoulli MMSBM baseline")
    print(f"  TWIST relative residual: {binary_fit.decomposition.relative_residuals[-1]:.4f}")
    print(f"  Layer-type adjusted Rand index: {adjusted_rand_score(binary.layer_types,binary_fit.layer_types):.4f}")
    print("Synthetic Lognormal LN-MMSBM rectangular extension")
    print(f"  TWIST relative residual (on log scale): {fitted.decomposition.relative_residuals[-1]:.4f}")
    print(f"  Layer-type adjusted Rand index: {adjusted_rand_score(weighted.layer_types,fitted.layer_types):.4f}")
    print(f"  Estimated conditional log variance: {fitted.log_variance:.4f}")
    print("These are single-seed toy outputs, NOT validated performance claims.")


if __name__ == "__main__":
    main()
