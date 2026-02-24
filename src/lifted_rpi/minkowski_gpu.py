"""
GPU Minkowski-sum + extreme-point filtering (JAX + CuPy).

Pipeline
--------
1. Build all pairwise Minkowski-sum candidates on GPU via JAX.
2. Transfer to CuPy using DLPack (zero-copy when backends match).
3. Multi-pass extreme-point filtering entirely on GPU (CuPy).
4. Return NumPy arrays (and optionally a SciPy ConvexHull on CPU).

The extreme-point filter projects all candidate points onto a bank of
random unit directions and retains any point that is maximal in at
least one direction.  Multiple passes with independent direction sets
are run, and the union of retained points is returned.

Notes
-----
- For zero-copy DLPack, JAX and CuPy must share the same CUDA backend.
  If a backend mismatch is detected, falls back to safe host copy.
- To force JAX to use CUDA in Colab:
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"

Dependencies: jax, cupy, numpy, scipy.

"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
except ImportError:
    jax = None
    jnp = None
    jit = None

try:
    import cupy as cp
except ImportError:
    cp = None

from scipy.spatial import ConvexHull

__all__ = ["MinkowskiGPU"]


# ---------------------------- JAX kernel ------------------------------------

if jit is not None:
    @jit
    def _minkowski_sum_candidates_jax(V1: jnp.ndarray, V2: jnp.ndarray) -> jnp.ndarray:
        """Return all m*n pairwise sums (m*n, d) computed on device via JAX."""
        return (V1[:, None, :] + V2[None, :, :]).reshape(-1, V1.shape[1])
else:
    def _minkowski_sum_candidates_jax(V1, V2):
        """CPU fallback: pairwise sums via NumPy."""
        return (V1[:, None, :] + V2[None, :, :]).reshape(-1, V1.shape[1])


# ---------------------------- class engine ----------------------------------

@dataclass
class MinkowskiGPU:
    """GPU pipeline for Minkowski-sum + extreme-point filtering.

    Parameters
    ----------
    num_dirs : int or None
        Number of random directions for the extreme-point filter.
        If None, uses ceil(2 * d * log(n+1)).
    tol : float
        Tolerance for considering a point maximal along a direction.
    seed : int or None
        RNG seed for reproducible random directions.
    force_host_copy : bool
        If True, skip DLPack and copy via host NumPy.
    n_filter_passes : int
        Number of filtering passes (union of maxima across passes).
    """
    num_dirs: Optional[int] = None
    tol: float = 1e-9
    seed: Optional[int] = None
    force_host_copy: bool = False
    n_filter_passes: int = 3

    def _jax_to_cupy(self, x) -> "cp.ndarray":
        """Best-effort JAX→CuPy. Try zero-copy DLPack; fall back to host copy."""
        if cp is None:
            return np.asarray(x)
        if self.force_host_copy:
            return cp.asarray(np.asarray(x))
        try:
            return cp.from_dlpack(x)
        except (TypeError, RuntimeError, BufferError):
            try:
                dlpack_capsule = jax.dlpack.to_dlpack(x)
                return cp.from_dlpack(dlpack_capsule)
            except (TypeError, RuntimeError, BufferError):
                return cp.asarray(np.asarray(x))

    @staticmethod
    def _filter_extreme_points_cupy_multipass(
        pts: "cp.ndarray",
        num_dirs: Optional[int],
        tol: float,
        seed: Optional[int],
        n_passes: int = 3
    ) -> "cp.ndarray":
        """
        Multi-pass extreme point filtering.
        Runs n_passes with different random directions and takes union of all
        points that are maximal in at least one direction across any pass.

        Returns a CuPy boolean mask.
        """
        if cp is None:
            raise RuntimeError("CuPy is not available")
        if pts.ndim != 2:
            raise ValueError("points must be a 2D array")
        n, d = pts.shape
        if n == 0:
            return cp.zeros((0,), dtype=cp.bool_)

        if num_dirs is None:
            num_dirs = int(math.ceil(2 * d * math.log(n + 1) / math.sqrt(n_passes)))

        union_mask = cp.zeros(n, dtype=cp.bool_)

        for pass_idx in range(n_passes):
            pass_seed = seed + pass_idx if seed is not None else pass_idx
            rs = cp.random.RandomState(pass_seed)

            dirs = rs.randn(num_dirs, d, dtype=pts.dtype)
            norms = cp.linalg.norm(dirs, axis=1, keepdims=True)
            norms = cp.where(norms == 0, 1.0, norms)
            dirs = dirs / norms

            proj = pts @ dirs.T                             # (n, num_dirs)
            max_vals = proj.max(axis=0, keepdims=True)      # (1, num_dirs)
            pass_mask = (proj >= (max_vals - tol)).any(axis=1)
            union_mask = union_mask | pass_mask

        return union_mask

    def filter_on_gpu(self, cands_jax) -> Tuple[np.ndarray, np.ndarray]:
        """Filter extreme points on GPU using multi-pass approach."""
        cands_cp = self._jax_to_cupy(cands_jax)

        mask_cp = self._filter_extreme_points_cupy_multipass(
            cands_cp, self.num_dirs, self.tol, self.seed, self.n_filter_passes
        )

        filtered_cp = cands_cp[mask_cp]
        return cp.asnumpy(cands_cp), cp.asnumpy(filtered_cp)

    def candidates(self, V1: np.ndarray, V2: np.ndarray):
        """Build pairwise sums with JAX (returns JAX array on device)."""
        if jnp is not None:
            V1_j = jnp.asarray(V1)
            V2_j = jnp.asarray(V2)
            C = _minkowski_sum_candidates_jax(V1_j, V2_j)
            C.block_until_ready()
            return C
        else:
            return _minkowski_sum_candidates_jax(V1, V2)

    def minkowski_sum_hull(self, V1: np.ndarray, V2: np.ndarray):
        """End-to-end: candidates (JAX) → filter (CuPy) → hull (SciPy)."""
        cands_j = self.candidates(V1, V2)
        all_pts_np, filt_pts_np = self.filter_on_gpu(cands_j)
        if filt_pts_np.shape[0] < 4 and filt_pts_np.shape[1] >= 3:
            hull = None
        else:
            hull = ConvexHull(filt_pts_np)
        return all_pts_np, filt_pts_np, hull
