"""
Initialisation helpers for the fixed-point iteration.

``build_Z0_inside_G``
    Construct the initial set Z0 as a subset of G by sampling (x, v)
    on a grid, querying the graph-set bounds for valid w values, and
    assembling augmented points z = [x; v; w].

``make_W_from_learned_G_envelope``
    Build the axis-aligned disturbance outer-approximation W from the
    learned G bounds envelope over the operating domain.

``make_DV_from_u_box``
    Compute the input perturbation set DV such that u = Kx + v stays
    within the physical actuator limits [u_min, u_max].

"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

from .vset import VSet, box_corners

if TYPE_CHECKING:
    from .engine import LiftedSetOpsGPU_NoHull

__all__ = [
    "build_Z0_inside_G",
    "make_W_from_learned_G_envelope",
    "make_DV_from_u_box",
]


def build_Z0_inside_G(
    eng: "LiftedSetOpsGPU_NoHull",
    G,
    *,
    xv_box=None,
    x_levels: int = 3,
    v_levels: int = 3,
    w_mode: str = "center",
    name: str = "Z_0_in_G",
) -> VSet:
    """Construct the initial set Z0 as a subset of the graph constraint G.

    For each (x_pre, v) sample on a grid, query G.bounds_fn to obtain the
    valid disturbance range, choose w inside those bounds (centre or
    corners), then set x = x_pre + D_top @ w to account for the
    disturbance injection.  The result is a VSet of augmented points
    z = [x; v; w] that are guaranteed to satisfy G.

    Parameters
    ----------
    eng : LiftedSetOpsGPU_NoHull
        Engine instance (provides K, D_top, dimensions).
    G : GraphSet
        Constraint graph set with bounds_fn(X, U).
    xv_box : tuple of ndarray, optional
        (lower, upper) bounds on the (x, v) sampling domain.
        Defaults to +/-0.5 in each coordinate.
    x_levels, v_levels : int
        Number of grid points per axis for x and v respectively.
    w_mode : str
        'center' places w at the midpoint of the bounds;
        'corners' places w at all 2^w box corners.

    Returns
    -------
    VSet
        Initial set in the augmented space.
    """
    n, m, w = eng.n, eng.m, eng.w

    # Use the engine's pre-computed D_top (handles Edist, dist_rows, and default)
    D_top = eng.D_top

    if xv_box is None:
        lb_xv = -0.5 * np.ones(n + m)
        ub_xv = 0.5 * np.ones(n + m)
    else:
        lb_xv, ub_xv = xv_box

    def _grid(lb, ub, L):
        axes = [np.linspace(lb[i], ub[i], L) for i in range(len(lb))]
        return np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(lb), -1).T

    Xpre_grid = _grid(lb_xv[:n], ub_xv[:n], x_levels)
    V_grid = _grid(lb_xv[n:], ub_xv[n:], v_levels)

    Xpre = np.repeat(Xpre_grid, V_grid.shape[0], axis=0)
    Vv = np.tile(V_grid, (Xpre_grid.shape[0], 1))
    Upre = (eng.K @ Xpre.T).T + Vv

    lbW, ubW = G.bounds_fn(Xpre, Upre)

    if w_mode == "center":
        Wpts = 0.5 * (lbW + ubW)
        X = Xpre + (Wpts @ D_top.T)
        Z = np.hstack([X, Vv, Wpts])
    elif w_mode == "corners":
        bits = (
            np.array(np.meshgrid(*[[0, 1]] * w, indexing="ij")).reshape(w, -1).T
        )
        clouds = []
        for i in range(Xpre.shape[0]):
            Wi = lbW[i] + bits * (ubW[i] - lbW[i])
            Xi = Xpre[i] + Wi @ D_top.T
            Vi = np.repeat(Vv[i : i + 1], Wi.shape[0], axis=0)
            clouds.append(np.hstack([Xi, Vi, Wi]))
        Z = np.vstack(clouds)
    else:
        raise ValueError("w_mode must be 'center' or 'corners'")

    Z = np.unique(Z, axis=0)
    return VSet(Z, name=name)


def make_W_from_learned_G_envelope(
    eng: "LiftedSetOpsGPU_NoHull",
    G,
    *,
    xv_box=None,
    x_levels: int = 5,
    v_levels: int = 5,
    add_local_grid: bool = True,
    local_levels: int = 3,
    name: str = "W_from_G",
) -> VSet:
    """Axis-aligned W superset from learned G bounds envelope."""
    n, m, w = eng.n, eng.m, eng.w
    if w != 2:
        raise NotImplementedError(
            f"make_W_from_learned_G_envelope currently requires w=2, got w={w}. "
            "Extend the grid code for general w as needed."
        )

    if xv_box is None:
        lb_xv = -0.5 * np.ones(n + m)
        ub_xv = 0.5 * np.ones(n + m)
    else:
        lb_xv, ub_xv = xv_box

    def _grid(lb, ub, L):
        axes = [np.linspace(lb[i], ub[i], L) for i in range(len(lb))]
        return np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(lb), -1).T

    Xpre_grid = _grid(lb_xv[:n], ub_xv[:n], x_levels)
    V_grid = _grid(lb_xv[n:], ub_xv[n:], v_levels)

    Xpre = np.repeat(Xpre_grid, V_grid.shape[0], axis=0)
    Vv = np.tile(V_grid, (Xpre_grid.shape[0], 1))
    Upre = (eng.K @ Xpre.T).T + Vv

    lbW, ubW = G.bounds_fn(Xpre, Upre)
    lb = lbW.min(axis=0)
    ub = ubW.max(axis=0)

    if hasattr(G, "lb_prior") and hasattr(G, "ub_prior"):
        lb = np.maximum(lb, G.lb_prior)
        ub = np.minimum(ub, G.ub_prior)

    W_list = [box_corners(lb, ub)]

    if add_local_grid:
        X0 = np.zeros((1, n))
        V0 = np.zeros((1, m))
        U0 = (eng.K @ X0.T).T + V0
        lb0, ub0 = G.bounds_fn(X0, U0)
        h = 0.5 * (ub0 - lb0).ravel()
        axes = [np.linspace(-h[i], h[i], local_levels) for i in range(w)]
        Wg = np.array(np.meshgrid(*axes, indexing="ij")).reshape(w, -1).T
        W_list.append(Wg)

    W_arr = np.unique(np.vstack(W_list), axis=0)
    return VSet(W_arr, name=name)


def make_DV_from_u_box(
    K: np.ndarray,
    xv_box,
    u_min: np.ndarray,
    u_max: np.ndarray,
    alpha_v: float = 0.95,
    safety: float = 0.9,
) -> Tuple[VSet, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ΔV so that u = Kx + v stays in [u_min,u_max] for all x in xv_box.

    Returns (DV_vset, hv, d, R_abs) where
      hv = steady-state v half-widths,
      d  = per-step Δv half-widths,
      R_abs = worst-case |Kx|.
    """
    lb_xv, ub_xv = xv_box
    n = K.shape[1]
    m = K.shape[0]
    hx = 0.5 * (ub_xv[:n] - lb_xv[:n])
    R_abs = np.zeros(m)
    for j in range(m):
        R_abs[j] = np.sum(np.abs(K[j, :]) * hx)

    umin = np.asarray(u_min)
    umax = np.asarray(u_max)
    hv = safety * np.maximum(0.0, umax - R_abs)
    d = (1.0 - alpha_v) * hv
    DV = VSet(box_corners(-d, d), name="ΔV_from_u_box")
    return DV, hv, d, R_abs
