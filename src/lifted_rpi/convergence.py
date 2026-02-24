"""
Convergence diagnostics for the fixed-point iteration.

Provides several metrics to monitor the progress of the outside-in
iteration Z_{k+1} = F(Z_k):

- ``hausdorff_pointcloud`` : symmetric Hausdorff distance between
  two finite point clouds, computed via KD-trees.  Optionally
  normalised by the AABB diagonal.
- ``support_gap`` / ``support_gap_with_dirs`` : approximate
  symmetric support-function gap using random projection directions.
- ``aabb_volume`` : axis-aligned bounding box (hyper)volume.
- ``count_facets`` : number of convex-hull facets (for diagnostics).

"""
from __future__ import annotations

import math
import numpy as np
import scipy.spatial
from typing import Optional
from scipy.spatial import cKDTree, ConvexHull

__all__ = [
    "aabb_volume",
    "support_gap",
    "make_dirs",
    "support_gap_with_dirs",
    "hausdorff_pointcloud",
    "count_facets",
    "support_outer_poly",
    "rel_change",
]


def aabb_volume(points: np.ndarray) -> float:
    """Axis-aligned bounding box volume (0 if empty)."""
    if points.size == 0:
        return 0.0
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 0.0)
    return float(np.prod(np.maximum(span, 1e-12)))


def support_gap(
    VA: np.ndarray,
    VB: np.ndarray,
    num_dirs: Optional[int] = None,
    seed: int = 0,
) -> float:
    """
    Approximate symmetric support-function gap between convex hulls of two
    point clouds.  Returns +inf if exactly one is empty; 0 if both empty.
    """
    if VA.size == 0 and VB.size == 0:
        return 0.0
    if VA.size == 0 or VB.size == 0:
        return float("inf")

    d = VA.shape[1]
    n = max(VA.shape[0], VB.shape[0])
    if num_dirs is None:
        num_dirs = int(math.ceil(2 * d * math.log(n + 1)))

    rng = np.random.default_rng(seed)
    D = rng.standard_normal((num_dirs, d)).astype(VA.dtype, copy=False)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-12

    sA = (VA @ D.T).max(axis=0)
    sB = (VB @ D.T).max(axis=0)
    return float(np.max(np.abs(sA - sB)))


def make_dirs(dim: int, q: int = 128, seed: int = 0) -> np.ndarray:
    """Reusable bank of directions for support-function evaluation."""
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((q, dim)).astype(np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-12
    return D


def support_gap_with_dirs(
    VA: np.ndarray, VB: np.ndarray, D: np.ndarray
) -> float:
    """Support-gap using a pre-made direction bank D (shape q×d)."""
    if VA.size == 0 and VB.size == 0:
        return 0.0
    if VA.size == 0 or VB.size == 0:
        return float("inf")
    sA = (VA @ D.T).max(axis=0)
    sB = (VB @ D.T).max(axis=0)
    return float(np.max(np.abs(sA - sB)))


def hausdorff_pointcloud(
    VA: np.ndarray,
    VB: np.ndarray,
    *,
    normalize: bool = True,
    scale: Optional[float] = None,
) -> float:
    """Symmetric Hausdorff distance between two finite point clouds.

    Uses KD-trees for nearest-neighbour queries.  The symmetric distance
    is  max( max_{a in A} d(a, B),  max_{b in B} d(b, A) ).

    Parameters
    ----------
    VA, VB : ndarray
        Point clouds of shape (N_a, d) and (N_b, d).
    normalize : bool
        If True, divide by `scale` (or the AABB diagonal of A union B).
    scale : float, optional
        Custom normalisation constant (overrides AABB diagonal).

    Returns
    -------
    float
        The (optionally normalised) Hausdorff distance.
        Returns +inf if exactly one set is empty, 0 if both empty.
    """
    if VA.size == 0 and VB.size == 0:
        return 0.0
    if VA.size == 0 or VB.size == 0:
        return float("inf")

    treeA = cKDTree(VA)
    treeB = cKDTree(VB)
    d_AB, _ = treeB.query(VA, k=1, workers=-1)
    d_BA, _ = treeA.query(VB, k=1, workers=-1)
    hd = float(max(np.max(d_AB), np.max(d_BA)))

    if not normalize:
        return hd

    if scale is None:
        both = np.vstack([VA, VB])
        if both.shape[0] == 0:
            scale = 1.0
        else:
            lo = both.min(axis=0)
            hi = both.max(axis=0)
            diag = float(np.linalg.norm(hi - lo))
            scale = diag if diag > 1e-12 else 1.0
    return hd / scale


def count_facets(V: np.ndarray) -> int:
    """
    Number of convex-hull facets of a point cloud.
    Returns 0 for degenerate / small clouds.
    """
    if V.size == 0 or V.shape[0] < V.shape[1] + 1:
        return 0
    try:
        hull = ConvexHull(V, qhull_options="QJ")
        return hull.simplices.shape[0]
    except (ValueError, scipy.spatial.QhullError):
        return 0


def support_outer_poly(
    V: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """
    Compute support-function values h_V(d) = max_{v ∈ V} d·v for all d in D.
    """
    if V.size == 0:
        return np.full(D.shape[0], -np.inf)
    return (V @ D.T).max(axis=0)


def rel_change(prev: float, curr: float) -> float:
    """Relative change |curr - prev| / max(|prev|, ε)."""
    return abs(curr - prev) / max(abs(prev), 1e-12)
