"""Binary symmetric MMSBM: a full TWIST-style clustering pipeline.

Reference: Jing et al. (2021), methodology steps 1–4, not a bitwise R port.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .twist import TwistResult
from .fit import fit_binary_baseline, _kmeans


@dataclass(frozen=True)
class FittedBernoulli:
    decomposition: TwistResult
    layer_types: NDArray[np.int64]
    global_nodes: NDArray[np.int64]
    local_nodes: NDArray[np.int64]      # n_types x n_nodes
    block_probabilities: NDArray[np.float64]  # n_types x k x k


def fit_bernoulli(
    adjacency: NDArray[np.float64], n_types: int, k: int,
    *, rank_nodes: int | None = None, seed: int = 0, max_iter: int = 30,
) -> FittedBernoulli:
    """Cluster original-model symmetric binary multilayer networks.

    Stage 1: TWIST-style common node and layer embeddings.
    Stage 2: layer types and global node groups via K-means.
    Stage 3: pooled class-wise adjacency eigenspace and local communities.
    Stage 4: Bernoulli block probabilities conditional on hard labels.

    Diagonal self-edges are ignored in block-probability estimation.
    """
    a=np.asarray(adjacency,dtype=float)
    decomp,layers,global_labels=fit_binary_baseline(a,n_types,k,rank_nodes=rank_nodes,
                                                    seed=seed,max_iter=max_iter)
    n=a.shape[0]
    local=np.empty((n_types,n),dtype=int)
    block_probs=np.full((n_types,k,k),np.nan)
    for c in range(n_types):
        sel=np.flatnonzero(layers==c)
        summed=a[:,:,sel].sum(axis=2)
        eigenvalues,eigenvectors=np.linalg.eigh(summed)
        order=np.argsort(np.abs(eigenvalues))[::-1]
        embeds=eigenvectors[:,order[:k]]
        local[c]=_kmeans(embeds,k,seed)
        for g1 in range(k):
            i=np.flatnonzero(local[c]==g1)
            for g2 in range(k):
                j=np.flatnonzero(local[c]==g2)
                if not len(i) or not len(j): continue
                if g1==g2:
                    if len(i)<2: continue
                    upper=np.triu_indices(len(i),k=1)
                    vals=a[np.ix_(i,i,sel)][upper[0],upper[1],:]
                else:
                    vals=a[np.ix_(i,j,sel)]
                block_probs[c,g1,g2]=float(vals.mean())
    return FittedBernoulli(decomp,layers,global_labels,local,block_probs)
