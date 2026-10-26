"""Tensor spectral decomposition inspired by Algorithm 1 (TWIST), Jing et al. (2021).

The symmetric variant retains one common node factor; the rectangular variant is
an explicit extension for directed producer-consumer weighted networks.
This is not a claim of bit-for-bit equivalence to the rMultiNet R implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class TwistResult:
    source_factors: Array
    target_factors: Array
    layer_factors: Array
    core: Array
    relative_residuals: tuple[float, ...]
    converged: bool
    n_iterations: int


def _left_singular_vectors(matrix: Array, rank: int) -> Array:
    """Return leading left singular space with deterministic orientation."""
    u = np.linalg.svd(matrix, full_matrices=False)[0][:, :rank]
    if u.shape[1] != rank:
        raise ValueError("Requested rank exceeds the singular-space dimension")
    # Sign convention only; rotations within repeated singular spaces are inherent.
    for j in range(rank):
        pivot = np.argmax(np.abs(u[:, j]))
        if u[pivot, j] < 0:
            u[:, j] *= -1
    return u


def _row_regularize(u: Array, delta: float) -> Array:
    """Equation (3.1): clip row norms then restore a rank-r orthonormal basis."""
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    clipped = u * np.minimum(1.0, delta / np.maximum(norms, 1e-15))
    return _left_singular_vectors(clipped, u.shape[1])


def _mode1(x: Array, v: Array, w: Array) -> Array:
    return np.einsum("ijl,jb,lc->ibc", x, v, w, optimize=True).reshape(x.shape[0], -1)


def _mode2(x: Array, u: Array, w: Array) -> Array:
    return np.einsum("ijl,ia,lc->jac", x, u, w, optimize=True).reshape(x.shape[1], -1)


def _mode3(x: Array, u: Array, v: Array) -> Array:
    return np.einsum("ijl,ia,jb->lab", x, u, v, optimize=True).reshape(x.shape[2], -1)


def _core(x: Array, u: Array, v: Array, w: Array) -> Array:
    return np.einsum("ijl,ia,jb,lc->abc", x, u, v, w, optimize=True)


def _check_tensor(x: Array, ranks: tuple[int, int, int], symmetric: bool) -> Array:
    a = np.asarray(x, dtype=float)
    if a.ndim != 3 or min(a.shape) < 2:
        raise ValueError("Expected a 3D tensor with each dimension >= 2")
    if not np.isfinite(a).all():
        raise ValueError("Tensor must contain only finite values")
    if len(ranks) != 3 or any(int(v) != v or v < 1 for v in ranks):
        raise ValueError("Ranks must be a tuple of three positive integers")
    n1, n2, l = a.shape
    r1, r2, r3 = ranks
    if r1 > min(n1, r2*r3) or r2 > min(n2, r1*r3) or r3 > min(l, r1*r2):
        raise ValueError("Incompatible Tucker ranks for tensor dimensions")
    if symmetric:
        if n1 != n2 or r1 != r2 or not np.allclose(a, a.transpose(1, 0, 2), atol=1e-10):
            raise ValueError("Symmetric TWIST requires square symmetric slices and equal node ranks")
    return a


def twist(
    tensor: Array,
    ranks: tuple[int, int, int],
    *,
    symmetric: bool = False,
    delta_source: float | None = None,
    delta_target: float | None = None,
    delta_layers: float | None = None,
    max_iter: int = 30,
    tol: float = 1e-6,
) -> TwistResult:
    """Fit a regularized spectral Tucker model to any finite numeric tensor.

    For symmetric=True: common node subspace U=V (TWIST-style).
    For symmetric=False: independent U and V (rectangular extension).

    Regularization follows row-clipping + SVD as in the PAPER (3.1), not merely
    the un-reorthogonalized row-clipping in the released R package. Updates use
    the previous iteration for all modes (simultaneous/Jacobi-style).

    Parameters delta_* are L2 row-norm caps; default 2*sqrt(rank/dimension).
    The objective is Frobenius approximation, not full heteroskedastic ML.
    """
    x = _check_tensor(tensor, ranks, symmetric)
    if not isinstance(max_iter, int) or max_iter < 1 or tol <= 0:
        raise ValueError("max_iter must be positive integer; tol must be positive")
    n1, n2, n_l = x.shape
    r1, r2, r3 = ranks
    ds = float(delta_source) if delta_source is not None else 2.0 * np.sqrt(r1/n1)
    dt = float(delta_target) if delta_target is not None else 2.0 * np.sqrt(r2/n2)
    dl = float(delta_layers) if delta_layers is not None else 2.0 * np.sqrt(r3/n_l)

    # Warm initialization: SVD of pooled adjacency / log-weight matrix.
    pooled = x.sum(axis=2)
    if symmetric:
        u = _left_singular_vectors(pooled, r1)
        v = u.copy()
    else:
        left, _, right = np.linalg.svd(pooled, full_matrices=False)
        u, v = left[:, :r1], right.T[:, :r2]
    w = _left_singular_vectors(_mode3(x, u, v), r3)
    norm_sq = float(np.sum(x*x))
    if norm_sq <= 1e-25:
        raise ValueError("The tensor is numerically all zero; memberships are not identified")
    norm = np.sqrt(norm_sq)
    residuals: list[float] = []
    converged = False

    for _ in range(max_iter):
        ur = _row_regularize(u, ds)
        vr = ur if symmetric else _row_regularize(v, dt)
        wr = _row_regularize(w, dl)
        u_new = _left_singular_vectors(_mode1(x, vr, wr), r1)
        v_new = u_new.copy() if symmetric else _left_singular_vectors(_mode2(x, ur, wr), r2)
        w_new = _left_singular_vectors(_mode3(x, ur, vr), r3)
        u, v, w = u_new, v_new, w_new
        c = _core(x, u, v, w)
        # Exact projection residual for orthonormal factor matrices.
        residual = np.sqrt(max(0.0, norm_sq - float(np.sum(c*c)))) / norm
        residuals.append(float(residual))
        if len(residuals) > 1 and abs(residuals[-1] - residuals[-2]) < tol:
            converged = True
            break

    return TwistResult(u, v, w, _core(x, u, v, w), tuple(residuals), converged, len(residuals))


def reconstruct(result: TwistResult) -> Array:
    """Reconstruct the fitted tensor on the same scale as twist's input."""
    return np.einsum("abc,ia,jb,lc->ijl", result.core, result.source_factors,
                     result.target_factors, result.layer_factors, optimize=True)
