"""
Lightweight vertex-cloud (V-set) wrapper and helpers.

Sets are represented as raw point clouds (no convex hull stored).
Hull computation is deferred to plotting utilities when needed.

Key types
---------
VSet       -- Immutable vertex cloud with deduplication and optional
              float32 storage (saves GPU/CPU memory for large clouds).
GraphSet   -- Callable-based disturbance set: bounds_fn(x, u) -> (lb, ub).
              Used by the engine for graph-set clipping.

Helper functions: box_corners, apply_linear_to_cloud,
clip_cloud_by_polytope(_gpu), downsample_cloud.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Tuple

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


# ----------------------------- V-set wrapper -----------------------------

@dataclass
class VSet:
    """Lightweight vertex cloud (no hull)."""
    V: np.ndarray               # shape (N, d)
    name: Optional[str] = None
    dtype: str = "float32"      # store as float32 to save memory by default

    def __post_init__(self):
        V = np.asarray(self.V)
        if V.size == 0:
            object.__setattr__(self, 'V', np.zeros((0, V.shape[1] if V.ndim > 1 else 0), dtype=self.dtype))
            return

        if self.dtype == "float32":
            V_typed = V.astype(np.float32, copy=False)
        elif self.dtype == "float64":
            V_typed = V.astype(np.float64, copy=False)
        else:
            V_typed = V

        V_processed = np.unique(V_typed, axis=0)

        if V_processed.size == 0:
            object.__setattr__(self, 'V', np.zeros((0, V.shape[1] if V.ndim > 1 else 0), dtype=self.dtype))
        else:
            object.__setattr__(self, 'V', V_processed)

    @property
    def dim(self) -> int:
        return 0 if self.V.ndim == 1 else (self.V.shape[1] if self.V.size else 0)

    @property
    def nverts(self) -> int:
        return 0 if self.V.ndim == 1 else (self.V.shape[0] if self.V.size else 0)


# ----------------------------- helpers ---------------------------------

def box_corners(lb: np.ndarray, ub: np.ndarray, dtype="float32") -> np.ndarray:
    """Generate 2^d corners of an axis-aligned box [lb, ub]."""
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    assert lb.shape == ub.shape
    d = lb.size
    corners = np.array(np.meshgrid(*[[lb[i], ub[i]] for i in range(d)], indexing="ij"))
    V = corners.reshape(d, -1).T
    return V.astype(np.float32 if dtype == "float32" else np.float64, copy=False)


def apply_linear_to_cloud(V: np.ndarray, M: np.ndarray, out_dtype="float32") -> np.ndarray:
    """Pointwise linear map y = M x for a vertex cloud."""
    Y = V @ np.asarray(M, dtype=V.dtype).T
    if out_dtype == "float32":
        return Y.astype(np.float32, copy=False)
    elif out_dtype == "float64":
        return Y.astype(np.float64, copy=False)
    return Y


def clip_cloud_by_polytope(V: np.ndarray, G, tol: float = 1e-9) -> np.ndarray:
    """
    UNDER-approximate intersection with H-rep polytope G by keeping only points inside:
        { x : A_G x <= b_G }.
    NOTE: This does NOT add boundary intersection points; no hulling/refinement.
    """
    if V.size == 0:
        return V
    A, b = G.A, G.b
    mask = np.all(A @ V.T - b[:, None] <= tol, axis=0)
    return V[mask]


def clip_cloud_by_polytope_gpu(V: np.ndarray, G, tol: float = 1e-9) -> np.ndarray:
    """
    GPU-accelerated halfspace clipping.
    Returns points satisfying A @ x <= b.
    """
    if cp is None:
        return clip_cloud_by_polytope(V, G, tol)
    if V.size == 0:
        return V

    V_gpu = cp.asarray(V, dtype=cp.float32)
    A_gpu = cp.asarray(G.A, dtype=cp.float32)
    b_gpu = cp.asarray(G.b, dtype=cp.float32)

    residuals = A_gpu @ V_gpu.T - b_gpu[:, None]
    mask = cp.all(residuals <= tol, axis=0)

    V_clipped = V_gpu[mask]
    return cp.asnumpy(V_clipped)


def downsample_cloud(V: np.ndarray,
                     max_points: int = 50_000,
                     method: str = "grid",
                     grid_eps: float = None,
                     seed: int = 0) -> np.ndarray:
    """Reduce a point cloud V (N, d) to at most ``max_points`` rows.

    Three methods are available:

    - ``grid`` : snap coordinates to a uniform grid, then deduplicate.
      Preserves the shape of the cloud better than random sampling.
      The grid spacing is computed as the d-th root of
      (bounding-box volume / max_points).
    - ``random`` : uniform random subset (fast but may miss extremes).
    - ``stride`` : deterministic stride (stable across runs).

    Returns the reduced array with unique rows.
    """
    V = np.asarray(V)
    if V.size == 0 or V.shape[0] <= max_points:
        return np.unique(V, axis=0)

    if method == "grid":
        if grid_eps is None:
            spread = np.maximum(1e-12, V.max(axis=0, keepdims=True) - V.min(axis=0, keepdims=True))
            k = max_points
            d = V.shape[1]
            grid_eps = float((np.prod(spread)**(1.0/d)) / (k**(1.0/d) + 1e-9))
        Q = np.floor(V / grid_eps + 0.5).astype(np.int64)
        Vu = np.unique(Q, axis=0)
        Vc = Vu.astype(float) * grid_eps
        if Vc.shape[0] > max_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(Vc.shape[0], size=max_points, replace=False)
            Vc = Vc[idx]
        return Vc

    elif method == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(V.shape[0], size=max_points, replace=False)
        return V[idx]

    elif method == "stride":
        step = max(1, V.shape[0] // max_points)
        return V[::step][:max_points]

    else:
        return downsample_cloud(V, max_points, "random", seed=seed)


# ----------------------------- GraphSet ---------------------------------

@dataclass(frozen=True)
class GraphSet:
    """
    Encodes G = { (x,v,w): w in W_hat(x, Kx+v) }.
    bounds_fn(x,u) must return (lb, ub) in R^{N x w} with
      W_hat(x,u) <= { w : lb(x,u) <= w <= ub(x,u) }.
    """
    bounds_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    name: str = "G"
    tol: float = 1e-9
