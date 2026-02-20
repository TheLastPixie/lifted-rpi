"""
GPU-accelerated operations for the lifted-RPI pipeline.

Uses PyTorch CUDA for operations where GPU is significantly faster
than CPU NumPy/SciPy:

- ``unique_rows``: lexicographic row deduplication for large point
  clouds.  Replaces ``np.unique(V, axis=0)`` with
  ``torch.unique(dim=0)``, giving 50--100x speedup on 50k+ rows.

- ``knn_distances``: brute-force 1-nearest-neighbour distances for
  OOD detection.  Replaces ``sklearn.KDTree.query`` with batched
  squared-distance on GPU, giving 5--10x speedup for 50k queries
  against 2 500 training points.

- ``nystroem_predict``: full Nystroem RBF transform + multi-target
  Ridge prediction in a single GPU round-trip.  Replaces separate
  sklearn ``transform`` + ``predict`` calls.

All functions accept and return NumPy arrays.  GPU memory transfers
are handled internally.  Functions gracefully fall back to ``None``
when GPU is unavailable so callers can branch to CPU paths.

Typical usage::

    from lifted_rpi.speedup import gpu_ops

    if gpu_ops.init():
        V_deduped = gpu_ops.unique_rows(V)
    else:
        V_deduped = np.unique(V, axis=0)

"""
from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple

# --------------- module state ---------------
_torch = None        # lazy-loaded torch module
_device = None       # torch.device('cuda')
_ready: bool = False

# Minimum rows to bother shipping to GPU (below this, CPU is faster)
_MIN_ROWS_UNIQUE = 2000
_MIN_ROWS_KNN = 1000


# --------------- initialization ---------------

def init(verbose: bool = True) -> bool:
    """Try to initialise PyTorch CUDA.  Returns True if GPU is ready.

    Safe to call multiple times; subsequent calls are no-ops.
    Performs a warm-up of the key kernels (unique, cdist, matmul) so
    that the first real call is not penalised by JIT compilation.
    """
    global _torch, _device, _ready
    if _ready:
        return True

    try:
        import torch
        if not torch.cuda.is_available():
            if verbose:
                print("[gpu_ops] CUDA not available")
            return False

        _torch = torch
        _device = torch.device("cuda")

        # Warm up the three hot kernels with realistic shapes
        if verbose:
            print(f"[gpu_ops] warming up on {torch.cuda.get_device_name(0)} ...")

        # 1. unique(dim=0)
        _w = torch.randn(10000, 8, device=_device, dtype=torch.float32)
        _ = torch.unique(_w, dim=0)
        del _w

        # 2. cdist
        _a = torch.randn(500, 6, device=_device, dtype=torch.float32)
        _b = torch.randn(200, 6, device=_device, dtype=torch.float32)
        _ = torch.cdist(_a, _b)
        del _a, _b

        # 3. matmul
        _c = torch.randn(500, 200, device=_device, dtype=torch.float32)
        _d = torch.randn(200, 4, device=_device, dtype=torch.float32)
        _ = _c @ _d
        del _c, _d

        torch.cuda.synchronize()
        _ready = True

        mem_free = torch.cuda.mem_get_info()[0] / 1e9
        if verbose:
            print(f"[gpu_ops] ready  ({mem_free:.1f} GB free)")
        return True

    except (ImportError, RuntimeError) as exc:
        if verbose:
            print(f"[gpu_ops] init failed: {exc}")
        return False


def is_available() -> bool:
    """True if GPU ops are ready (init() succeeded)."""
    return _ready


# --------------- unique_rows ---------------

def unique_rows(V: np.ndarray) -> np.ndarray:
    """Drop-in replacement for ``np.unique(V, axis=0)``.

    Uses ``torch.unique(dim=0)`` on GPU for large arrays, falling back
    to NumPy for small arrays where transfer overhead dominates.

    Parameters
    ----------
    V : ndarray of shape (N, d)

    Returns
    -------
    ndarray of shape (M, d), M <= N, rows sorted lexicographically.
    """
    if not _ready or V.shape[0] < _MIN_ROWS_UNIQUE:
        return np.unique(V, axis=0)

    t = _torch.from_numpy(np.ascontiguousarray(V)).to(_device)
    u = _torch.unique(t, dim=0)
    return u.cpu().numpy()


# --------------- knn_distances ---------------

def knn_distances(
    queries: np.ndarray,
    train: np.ndarray,
    *,
    chunk: int = 10_000,
) -> np.ndarray:
    """Brute-force 1-NN Euclidean distances on GPU.

    Returns ``d[i] = min_j ||queries[i] - train[j]||`` using batched
    squared-distance computation (avoids building the full N_q x N_t
    matrix at once when N_q is large).

    Parameters
    ----------
    queries : (N_q, d)
    train   : (N_t, d)
    chunk   : batch size for query rows (memory / speed trade-off)

    Returns
    -------
    ndarray of shape (N_q,) -- Euclidean distances.
    """
    if not _ready or queries.shape[0] < _MIN_ROWS_KNN:
        return None  # signal caller to use CPU fallback

    Q = _torch.from_numpy(queries.astype(np.float32, copy=False)).to(_device)
    T = _torch.from_numpy(train.astype(np.float32, copy=False)).to(_device)

    T_sq = (T * T).sum(dim=1)  # (N_t,)
    N = Q.shape[0]
    dists = _torch.empty(N, device=_device, dtype=_torch.float32)

    for i in range(0, N, chunk):
        qi = Q[i : i + chunk]
        qi_sq = (qi * qi).sum(dim=1, keepdim=True)  # (chunk, 1)
        d2 = qi_sq - 2.0 * (qi @ T.t()) + T_sq.unsqueeze(0)
        d2.clamp_(min=0.0)
        dists[i : i + chunk] = d2.min(dim=1).values.sqrt()

    return dists.cpu().numpy().astype(np.float64)


# --------------- nystroem_predict ---------------

def nystroem_predict(
    F: np.ndarray,
    *,
    components: np.ndarray,
    normalization: np.ndarray,
    gamma: float,
    coefs: List[np.ndarray],
    intercepts: List[float],
) -> Optional[List[np.ndarray]]:
    """Full Nystroem RBF transform + Ridge predictions on GPU.

    Computes::

        D2  = pairwise_sq_dist(F, components)    # (N, M)
        K   = exp(-gamma * D2)                    # RBF kernel
        Phi = K @ normalization                   # Nystroem features
        y_i = Phi @ coef_i + intercept_i          # Ridge predictions

    Parameters
    ----------
    F            : (N, d_feat) query features
    components   : (M, d_feat) Nystroem component centres
    normalization: (M, M) normalization matrix
    gamma        : RBF gamma
    coefs        : list of K arrays each (M,)
    intercepts   : list of K floats

    Returns
    -------
    list of K arrays each (N,) or None if GPU unavailable.
    """
    if not _ready:
        return None

    Fg = _torch.from_numpy(F.astype(np.float32, copy=False)).to(_device)
    Cg = _torch.from_numpy(components.astype(np.float32, copy=False)).to(_device)
    Ng = _torch.from_numpy(normalization.astype(np.float32, copy=False)).to(_device)

    # Pairwise squared Euclidean distance
    D2 = _torch.cdist(Fg, Cg).pow_(2)        # (N, M)
    K = _torch.exp_(-gamma * D2)              # in-place exp
    Phi = K @ Ng                              # (N, M)

    results: List[np.ndarray] = []
    for coef, intercept in zip(coefs, intercepts):
        w = _torch.from_numpy(coef.astype(np.float32, copy=False)).to(_device)
        pred = Phi @ w + intercept
        results.append(pred.cpu().numpy().astype(np.float64))

    return results


# --------------- hausdorff ---------------

def hausdorff(
    VA: np.ndarray,
    VB: np.ndarray,
    *,
    normalize: bool = True,
    scale: Optional[float] = None,
    chunk: int = 2000,
) -> Optional[float]:
    """Bidirectional Hausdorff distance on GPU (float32 batched sq-dist).

    For N > ~30 000 points the GPU brute-force approach is slower than
    scipy cKDTree.  This function returns None in that case so the
    caller can fall back to the CPU path.

    Use when N <= 25 000 or when an approximate metric is acceptable
    (e.g. convergence monitoring where cKDTree overhead dominates for
    repeated calls).
    """
    # For 50k x 50k brute-force the GPU is actually slower than cKDTree,
    # so only use GPU for moderate sizes.
    N = max(VA.shape[0], VB.shape[0])
    if not _ready or N > 30_000:
        return None

    A = _torch.from_numpy(VA.astype(np.float32, copy=False)).to(_device)
    B = _torch.from_numpy(VB.astype(np.float32, copy=False)).to(_device)

    def _directed(src, dst):
        dst_sq = (dst * dst).sum(dim=1)  # (N_dst,)
        max_min = _torch.tensor(0.0, device=_device)
        for i in range(0, src.shape[0], chunk):
            si = src[i : i + chunk]
            si_sq = (si * si).sum(dim=1, keepdim=True)
            d2 = si_sq - 2.0 * (si @ dst.t()) + dst_sq.unsqueeze(0)
            d2.clamp_(min=0.0)
            max_min = _torch.maximum(max_min, d2.min(dim=1).values.max())
        return max_min

    hd_sq = _torch.maximum(_directed(A, B), _directed(B, A))
    hd = float(hd_sq.sqrt())

    if not normalize:
        return hd

    if scale is None:
        both = _torch.cat([A, B], dim=0)
        lo = both.min(dim=0).values
        hi = both.max(dim=0).values
        diag = float(_torch.norm(hi - lo))
        scale = max(diag, 1e-12)

    return hd / scale


# --------------- minkowski_candidates ---------------

def minkowski_candidates(
    V1: np.ndarray,
    V2: np.ndarray,
) -> Optional[np.ndarray]:
    """Pairwise sum ``V1[i] + V2[j]`` on GPU.

    Returns (N1*N2, d) array of candidates, or None if GPU unavailable.
    """
    if not _ready:
        return None

    t1 = _torch.from_numpy(np.ascontiguousarray(V1)).to(_device)
    t2 = _torch.from_numpy(np.ascontiguousarray(V2)).to(_device)
    C = (t1.unsqueeze(1) + t2.unsqueeze(0)).reshape(-1, V1.shape[1])
    return C.cpu().numpy()
