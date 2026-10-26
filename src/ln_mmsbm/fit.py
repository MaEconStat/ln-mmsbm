"""Spectral clustering and conditional Gaussian block parameter estimation.

No claim is made that spectral clustering globally maximizes the latent likelihood.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from .twist import TwistResult, twist


@dataclass(frozen=True)
class FittedLN:
    decomposition: TwistResult
    layer_types: NDArray[np.int64]              # n_layers
    global_source: NDArray[np.int64]            # n_source
    global_target: NDArray[np.int64]            # n_target
    local_source: NDArray[np.int64]             # n_types x n_source
    local_target: NDArray[np.int64]             # n_types x n_target
    block_log_means: NDArray[np.float64]        # n_types x k_source x k_target
    log_variance: float                         # common conditional variance MLE
    layer_offsets: NDArray[np.float64]         # optional centering offsets


def _kmeans(z: NDArray[np.float64], k: int, seed: int) -> NDArray[np.int64]:
    if k > z.shape[0] or k < 1: raise ValueError("Invalid cluster count")
    if k == 1: return np.zeros(z.shape[0],dtype=int)
    return KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(z)


def _spectral_communities(matrix: NDArray[np.float64], k_s: int, k_t: int, seed:int):
    u,s,vt=np.linalg.svd(matrix,full_matrices=False)
    if k_s>u.shape[1] or k_t>vt.shape[0]:
        raise ValueError("Local rank exceeds supported matrix rank")
    # US and VS are more stable across nearly tied singular directions.
    return _kmeans(u[:,:k_s]*s[:k_s],k_s,seed),_kmeans(vt[:k_t,:].T*s[:k_t],k_t,seed)


def estimate_blocks(
    weights: NDArray[np.float64],
    layer_types: NDArray[np.int64],
    source_groups: NDArray[np.int64],
    target_groups: NDArray[np.int64],
    *, layer_offsets: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64],float]:
    """Conditional MLE means and a pooled sigma^2 for known/estimated hard labels.

    Empty blocks are returned as NaN and excluded from pooled residual variance.
    This is *not* a missing-data/hurdle likelihood.
    """
    x=np.asarray(weights,dtype=float)
    if x.ndim!=3 or not np.isfinite(x).all() or np.any(x<=0):
        raise ValueError("Lognormal observations must all be finite and strictly positive")
    n1,n2,L=x.shape
    layers=np.asarray(layer_types,dtype=int)
    sg=np.asarray(source_groups,dtype=int)
    tg=np.asarray(target_groups,dtype=int)
    m,k_shape=sg.shape[0],tg.shape[0]
    if layers.shape!=(L,) or sg.shape[1]!=n1 or tg.shape[1]!=n2 or k_shape!=m:
        raise ValueError("Incompatible memberships")
    if min(layers.min(),sg.min(),tg.min())<0 or layers.max()>=m:
        raise ValueError("Invalid cluster indices")
    offsets=np.zeros(L) if layer_offsets is None else np.asarray(layer_offsets,dtype=float)
    if offsets.shape!=(L,) or not np.isfinite(offsets).all():
        raise ValueError("Invalid layer offsets")
    y=np.log(x)-offsets[None,None,:]
    k1=int(sg.max()+1)
    k2=int(tg.max()+1)
    mus=np.full((m,k1,k2),np.nan)
    squared_error=0.0
    n_obs=0
    for c in range(m):
        sl=np.flatnonzero(layers==c)
        for a in range(k1):
            i=np.flatnonzero(sg[c]==a)
            for b in range(k2):
                j=np.flatnonzero(tg[c]==b)
                if not len(i) or not len(j) or not len(sl): continue
                block=y[np.ix_(i,j,sl)]
                mu=float(block.mean())
                mus[c,a,b]=mu
                squared_error+=float(np.sum((block-mu)**2))
                n_obs+=block.size
    if n_obs==0: raise ValueError("No block observations")
    return mus,squared_error/n_obs


def fit_lognormal(
    weights: NDArray[np.float64],
    n_types: int,
    k_source: int,
    k_target: int,
    *,
    rank_source: int | None = None,
    rank_target: int | None = None,
    center_layers: bool = False,
    seed: int = 0,
    max_iter: int = 30,
) -> FittedLN:
    """End-to-end rectangular lognormal mixture multilayer spectral pipeline.

    center_layers subtracts log-layer means before decomposition to remove a
    nuisance layer intercept. Offsets are added back in fitted block means.
    Inference requires sufficiently separated mean block matrices and layers;
    K, m and Tucker ranks are supplied, NOT automatically selected.
    """
    x=np.asarray(weights,dtype=float)
    if x.ndim!=3 or min(x.shape)<2 or not np.isfinite(x).all() or np.any(x<=0):
        raise ValueError("weights must be a 3D, finite, strictly positive tensor")
    n1,n2,L=x.shape
    if not 1<=n_types<=L or not 1<=k_source<=n1 or not 1<=k_target<=n2:
        raise ValueError("Invalid community and type counts")
    if n_types>min(n1,n2):
        raise ValueError("For this low-rank implementation n_types must <= both node counts")
    y=np.log(x)
    offsets=y.mean(axis=(0,1)) if center_layers else np.zeros(L)
    yc=y-offsets[None,None,:]
    # Upper bound from the union of type-specific indicator subspaces:
    # dim(sum_j span(Z_j)) <= m*(K-1)+1 (assuming each partition has a 1 vector).
    r1=rank_source or min(n1, n_types*(k_source-1)+1)
    r2=rank_target or min(n2, n_types*(k_target-1)+1)
    decomp=twist(yc,(r1,r2,n_types),symmetric=False,max_iter=max_iter)
    types=_kmeans(decomp.layer_factors,n_types,seed)
    gs=_kmeans(decomp.source_factors,min(n1,k_source**n_types),seed)
    gt=_kmeans(decomp.target_factors,min(n2,k_target**n_types),seed)
    ls=np.empty((n_types,n1),dtype=int)
    lt=np.empty((n_types,n2),dtype=int)
    for c in range(n_types):
        idx=np.flatnonzero(types==c)
        if not len(idx): raise RuntimeError("Empty inferred layer group")
        mat=yc[:,:,idx].mean(axis=2)
        ls[c],lt[c]=_spectral_communities(mat,k_source,k_target,seed)
    mu,sigma2=estimate_blocks(x,types,ls,lt,layer_offsets=offsets)
    return FittedLN(decomp,types,gs,gt,ls,lt,mu,sigma2,offsets)


def fit_binary_baseline(
    adjacency: NDArray[np.float64], n_types: int, k: int,
    *, rank_nodes: int | None = None, seed: int = 0, max_iter: int = 30,
) -> tuple[TwistResult,NDArray[np.int64],NDArray[np.int64]]:
    """Original-model-inspired symmetric TWIST baseline (not exact R port)."""
    a=np.asarray(adjacency,dtype=float)
    if a.ndim!=3 or not np.isin(a,[0.,1.]).all():
        raise ValueError("Baseline accepts Bernoulli (0/1) adjacency only")
    r=rank_nodes or min(a.shape[0],n_types*(k-1)+1)
    res=twist(a,(r,r,n_types),symmetric=True,max_iter=max_iter)
    return res,_kmeans(res.layer_factors,n_types,seed),_kmeans(res.source_factors,min(a.shape[0],k**n_types),seed)
