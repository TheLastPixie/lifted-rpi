"""
Disturbance-set constructors.

``make_drag_graphset``
    Analytical velocity-dependent drag GraphSet with Gaussian noise:
        w(x,u) = -(beta1/m)|v| v - (beta2/m) u + noise
    Bounds are centre +/- sqrt(chi2_tau) * diag(sigma_wx, sigma_wy).

``suggest_W_box_for_drag``
    Compute a conservative axis-aligned W superset by grid-searching
    the (v, u) domain for worst-case deterministic disturbance.

``build_static_G_vset``
    Sample the full (x, v) domain and expand each sample into the
    2^w disturbance box corners to build a point-cloud representation
    of G for visualisation.

"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

from .vset import VSet, GraphSet, box_corners, downsample_cloud

if TYPE_CHECKING:
    from .engine import LiftedSetOpsGPU_NoHull
    from .polytope import Polytope


# ─────────────────── exact G cloud from a GraphSet ───────────────────

def make_graph_G_from_cloud(
    Ggraph: GraphSet,
    eng: "LiftedSetOpsGPU_NoHull",
    S_aug,
    *,
    max_pairs: int = 200,
    seed: int = 0,
    name: str = "G_graph_exact",
) -> VSet:
    """
    Build the *exact* graph G used in clipping, over the provided (x,v) cloud.
    Accepts either a VSet-like with .V/.verts or a raw ndarray of shape (N, n+m).
    For each chosen (x,v), we add the 2^w box corners of W_hat(x, Kx+v).
    """
    if isinstance(S_aug, np.ndarray):
        XV = np.asarray(S_aug)
    elif hasattr(S_aug, "V"):
        XV = np.asarray(S_aug.V)
    elif hasattr(S_aug, "verts"):
        XV = np.asarray(S_aug.verts)
    else:
        raise TypeError("S_aug must be ndarray or have .V / .verts")

    if XV.size == 0:
        return VSet(np.zeros((0, eng.n + eng.m + eng.w)), name=name)
    if XV.ndim != 2 or XV.shape[1] < eng.n + eng.m:
        raise ValueError(
            f"Expected (N, {eng.n + eng.m}) array of [x|v]; got shape {XV.shape}"
        )

    X = XV[:, :eng.n]
    Vv = XV[:, eng.n : eng.n + eng.m]
    U = (eng.K @ X.T).T + Vv

    N = X.shape[0]
    take = min(max_pairs, N) if max_pairs is not None else N
    idx = np.linspace(0, N - 1, take).astype(int)

    if not callable(getattr(Ggraph, "bounds_fn", None)):
        raise AttributeError("GraphSet has no bounds_fn; can't build exact G cloud.")
    lbW, ubW = Ggraph.bounds_fn(X[idx], U[idx])

    w = eng.w
    corner_bits = (
        np.array(np.meshgrid(*[[0, 1]] * w, indexing="ij")).reshape(w, -1).T
    )
    out = []
    for i in range(take):
        Wi = lbW[i] + corner_bits * (ubW[i] - lbW[i])
        Xi = np.repeat(X[idx[i] : idx[i] + 1], Wi.shape[0], axis=0)
        Vi = np.repeat(Vv[idx[i] : idx[i] + 1], Wi.shape[0], axis=0)
        out.append(np.hstack([Xi, Vi, Wi]))
    G_cloud = np.vstack(out)

    G_cloud = downsample_cloud(G_cloud, max_points=50_000, method="grid")
    return VSet(G_cloud, name=name)


# ──────────────── analytical drag GraphSet ────────────────

def make_drag_graphset(
    *,
    mass: float,
    beta1: float,
    beta2: float,
    sigma_wx: float,
    sigma_wy: float,
    chi2_tau: float = 3.841,
    vx_index: int = 1,
    vy_index: int = 3,
    name: str = "G_drag",
    tol: float = 1e-9,
) -> GraphSet:
    """Build a GraphSet for the velocity-dependent drag disturbance model.

    The deterministic component is
        w_det(x, u) = -(beta1/mass)|v| v  -  (beta2/mass) u
    and the stochastic envelope is
        centre +/- sqrt(chi2_tau) * diag(sigma_wx, sigma_wy).

    Parameters
    ----------
    mass : float
        Body mass.
    beta1, beta2 : float
        Quadratic and linear drag/inefficiency coefficients.
    sigma_wx, sigma_wy : float
        Per-axis noise standard deviations.
    chi2_tau : float
        Chi-squared threshold for the confidence interval (e.g. 3.841
        for 95 %, 6.635 for 99.5 %).
    vx_index, vy_index : int
        Column indices of the velocity components in the state vector.

    Returns
    -------
    GraphSet
        Callable graph set whose bounds_fn(X, U) returns (lb, ub).
    """
    sqrt_chi = float(np.sqrt(chi2_tau))
    noise = np.array([sigma_wx, sigma_wy], dtype=float)[None, :] * sqrt_chi

    def _bounds_fn(x: np.ndarray, u: np.ndarray):
        vx = x[:, vx_index]
        vy = x[:, vy_index]
        vmag = np.sqrt(vx * vx + vy * vy)

        det_wx = -(beta1 / mass) * vmag * vx - (beta2 / mass) * u[:, 0]
        det_wy = -(beta1 / mass) * vmag * vy - (beta2 / mass) * u[:, 1]
        center = np.stack([det_wx, det_wy], axis=1)

        lb = center - noise
        ub = center + noise
        return lb.astype(x.dtype, copy=False), ub.astype(x.dtype, copy=False)

    return GraphSet(bounds_fn=_bounds_fn, name=name, tol=tol)


# ──────────── global W-box superset for Minkowski ────────────

def suggest_W_box_for_drag(
    *,
    mass: float,
    beta1: float,
    beta2: float,
    sigma_wx: float,
    sigma_wy: float,
    chi2_tau: float,
    vx_range: Tuple[float, float],
    vy_range: Tuple[float, float],
    ux_range: Tuple[float, float],
    uy_range: Tuple[float, float],
    grid: int = 41,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coarse grid the (v_x,v_y,u_x,u_y) ranges, find worst-case deterministic |w|,
    and return axis-aligned lb, ub that contain center ± √(χ²)·σ on top.
    """
    vx = np.linspace(vx_range[0], vx_range[1], grid)
    vy = np.linspace(vy_range[0], vy_range[1], grid)
    ux = np.linspace(ux_range[0], ux_range[1], grid)
    uy = np.linspace(uy_range[0], uy_range[1], grid)
    VX, VY, UX, UY = np.meshgrid(vx, vy, ux, uy, indexing="ij")
    VM = np.sqrt(VX * VX + VY * VY)

    cw_x = -(beta1 / mass) * VM * VX - (beta2 / mass) * UX
    cw_y = -(beta1 / mass) * VM * VY - (beta2 / mass) * UY

    half_x = np.sqrt(chi2_tau) * sigma_wx
    half_y = np.sqrt(chi2_tau) * sigma_wy

    min_x = float(cw_x.min() - half_x)
    max_x = float(cw_x.max() + half_x)
    min_y = float(cw_y.min() - half_y)
    max_y = float(cw_y.max() + half_y)

    lb = np.array([min_x, min_y], dtype=float)
    ub = np.array([max_x, max_y], dtype=float)
    return lb, ub


# ────── static G domain for plotting (full grid) ──────

def build_static_G_vset(
    Ggraph: GraphSet,
    eng: "LiftedSetOpsGPU_NoHull",
    x_box: Tuple[np.ndarray, np.ndarray],
    v_box: Tuple[np.ndarray, np.ndarray],
    *,
    per_axis_levels: Optional[int] = 9,
    n_samples: int = 200_000,
    max_pairs: int = 120_000,
    seed: int = 0,
    name: str = "G_plot_static",
) -> VSet:
    """
    Build a static visualization VSet for G over the full (x,v) domain.
    """
    if not callable(getattr(Ggraph, "bounds_fn", None)):
        raise AttributeError("GraphSet has no bounds_fn; cannot build exact static G.")

    rng = np.random.default_rng(seed)

    x_lb, x_ub = np.asarray(x_box[0], float), np.asarray(x_box[1], float)
    v_lb, v_ub = np.asarray(v_box[0], float), np.asarray(v_box[1], float)
    d_xv = eng.n + eng.m

    XV = None
    if per_axis_levels is not None:
        total = per_axis_levels ** d_xv
        if total <= n_samples:
            axes = [np.linspace(x_lb[i], x_ub[i], per_axis_levels) for i in range(eng.n)]
            axes += [np.linspace(v_lb[j], v_ub[j], per_axis_levels) for j in range(eng.m)]
            mesh = np.meshgrid(*axes, indexing="ij")
            XV = np.stack([m.reshape(-1) for m in mesh], axis=1)
        else:
            XV = rng.uniform(
                np.r_[x_lb, v_lb], np.r_[x_ub, v_ub], size=(n_samples, d_xv)
            )
    else:
        XV = rng.uniform(
            np.r_[x_lb, v_lb], np.r_[x_ub, v_ub], size=(n_samples, d_xv)
        )

    if XV.shape[0] > max_pairs:
        idx = np.linspace(0, XV.shape[0] - 1, max_pairs).astype(int)
        XV = XV[idx]

    X = XV[:, : eng.n]
    Vv = XV[:, eng.n : eng.n + eng.m]
    U = (eng.K @ X.T).T + Vv

    lbW, ubW = Ggraph.bounds_fn(X, U)

    wdim = eng.w
    corner_bits = (
        np.array(np.meshgrid(*[[0, 1]] * wdim, indexing="ij")).reshape(wdim, -1).T
    )
    Wi = lbW[:, None, :] + corner_bits[None, :, :] * (ubW - lbW)[:, None, :]
    W_flat = Wi.reshape(-1, wdim)

    XV_rep = np.repeat(XV, corner_bits.shape[0], axis=0)
    G_cloud = np.hstack([XV_rep, W_flat])

    try:
        G_cloud = downsample_cloud(G_cloud, max_points=80_000, method="grid")
    except Exception:
        pass

    return VSet(G_cloud, name=name)


# ────── generic G-to-VSet for visualization ──────

def make_G_vset_for_plot(G, eng, n_aug, *, xv_box=None, samples=400, seed=0):
    """
    Return a VSet to visualize 'G' regardless of its type.
    """
    from .polytope import Polytope

    if isinstance(G, Polytope):
        try:
            V = G.vertices()
            return VSet(V, name=G.name or "G")
        except Exception:
            lb, ub = G.axis_bounds()
            if np.isfinite(lb).all() and np.isfinite(ub).all():
                return VSet(box_corners(lb, ub), name=(G.name or "G_box"))
            raise

    if G.__class__.__name__ == "GraphSet":
        rng = np.random.default_rng(seed)
        n, m, w = eng.n, eng.m, eng.w

        if xv_box is None:
            lb_xv = -2.0 * np.ones(n + m)
            ub_xv = 2.0 * np.ones(n + m)
        else:
            lb_xv, ub_xv = xv_box

        X = rng.uniform(lb_xv[:n], ub_xv[:n], size=(samples, n))
        V = rng.uniform(lb_xv[n:], ub_xv[n:], size=(samples, m))
        U = (eng.K @ X.T).T + V

        lbW, ubW = G.bounds_fn(X, U)
        corner_bits = (
            np.array(np.meshgrid(*[[0, 1]] * w, indexing="ij")).reshape(w, -1).T
        )
        take = min(samples, 40)
        cloud = []
        for i in range(take):
            Wi = lbW[i] + corner_bits * (ubW[i] - lbW[i])
            Xi = np.repeat(X[i : i + 1], Wi.shape[0], axis=0)
            Vi = np.repeat(V[i : i + 1], Wi.shape[0], axis=0)
            cloud.append(np.hstack([Xi, Vi, Wi]))
        return VSet(np.vstack(cloud), name=getattr(G, "name", "G_graph"))

    if isinstance(G, (list, tuple)):
        for g in G:
            if hasattr(g, "A") and hasattr(g, "b"):
                return make_G_vset_for_plot(g, eng, n_aug)
        for g in G:
            if callable(getattr(g, "bounds_fn", None)):
                return make_G_vset_for_plot(
                    g, eng, n_aug, xv_box=xv_box, samples=samples, seed=seed
                )

    raise TypeError(f"Don't know how to visualize G of type {type(G)}")
