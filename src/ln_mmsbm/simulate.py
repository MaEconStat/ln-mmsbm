"""Independent generators for Bernoulli MMSBM and Lognormal LN-MMSBM."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SyntheticNetwork:
    weights: NDArray[np.float64]                # source x target x layer
    layer_types: NDArray[np.int64]              # layer type label
    source_communities: NDArray[np.int64]       # network type x source
    target_communities: NDArray[np.int64]       # network type x target
    block_parameters: NDArray[np.float64]      # network type x source group x target group
    sigma: float | None = None


def _balanced_labels(n: int, groups: int, rng: np.random.Generator) -> NDArray[np.int64]:
    if n < groups or groups < 1:
        raise ValueError("Each group requires at least one observation")
    return rng.permutation(np.arange(n, dtype=int) % groups)


def simulate_lognormal(
    n_source: int = 90,
    n_target: int = 80,
    n_layers: int = 18,
    n_types: int = 2,
    k_source: int = 2,
    k_target: int = 2,
    *,
    block_means: NDArray[np.float64] | None = None,
    sigma: float = 0.25,
    seed: int = 0,
    layer_offsets: NDArray[np.float64] | None = None,
) -> SyntheticNetwork:
    """Generate strictly positive, directed/bipartite weighted networks.

    Conditional on latent network type and the two endpoint memberships:
    log(X[i,j,l]) ~ Normal(block_means[type_l, g_src, g_dst] + offset_l, sigma^2).
    No structural zeros or censoring are generated in this first version.
    """
    rng = np.random.default_rng(seed)
    if min(n_source,n_target,n_layers,n_types,k_source,k_target) < 1 or n_layers < n_types:
        raise ValueError("Invalid sizes, or fewer layers than mixture types")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive")
    if block_means is None:
        block_means = rng.normal(0.5, 1.6, (n_types,k_source,k_target))
    block_means = np.asarray(block_means, dtype=float)
    if block_means.shape != (n_types,k_source,k_target) or not np.isfinite(block_means).all():
        raise ValueError("Invalid block means")
    offsets = np.zeros(n_layers) if layer_offsets is None else np.asarray(layer_offsets,dtype=float)
    if offsets.shape != (n_layers,) or not np.isfinite(offsets).all():
        raise ValueError("Invalid layer offsets")
    layer_types = _balanced_labels(n_layers,n_types,rng)
    src = np.stack([_balanced_labels(n_source,k_source,rng) for _ in range(n_types)])
    dst = np.stack([_balanced_labels(n_target,k_target,rng) for _ in range(n_types)])
    y = np.empty((n_source,n_target,n_layers))
    for l,c in enumerate(layer_types):
        means = block_means[c][np.ix_(src[c], dst[c])]
        y[:,:,l] = rng.normal(loc=means+offsets[l],scale=sigma)
    return SyntheticNetwork(np.exp(y),layer_types,src,dst,block_means,sigma)


def simulate_bernoulli(
    n_nodes: int = 100, n_layers: int = 20, n_types: int = 2,
    k: int = 2, *, p_in: float = 0.85, p_out: float = 0.08, seed: int = 0,
) -> SyntheticNetwork:
    """Generate binary, undirected MMSBM with zero diagonal as comparison data."""
    if min(n_nodes,n_layers,n_types,k) < 1 or n_layers < n_types:
        raise ValueError("Invalid dimensions")
    if not (0 <= p_out <= 1 and 0 <= p_in <= 1):
        raise ValueError("Probabilities must be in [0,1]")
    rng = np.random.default_rng(seed)
    types = _balanced_labels(n_layers,n_types,rng)
    src = np.stack([_balanced_labels(n_nodes,k,rng) for _ in range(n_types)])
    B = np.full((n_types,k,k),p_out)
    for c in range(n_types): np.fill_diagonal(B[c],p_in)
    out = np.zeros((n_nodes,n_nodes,n_layers))
    ii,jj = np.triu_indices(n_nodes,1)
    for l,c in enumerate(types):
        prob=B[c,src[c,ii],src[c,jj]]
        draw=rng.binomial(1,prob).astype(float)
        out[ii,jj,l]=draw
        out[jj,ii,l]=draw
    return SyntheticNetwork(out,types,src,src.copy(),B,None)
