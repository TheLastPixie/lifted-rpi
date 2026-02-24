"""
Convergence triplet plots: 3-D lifted-space visualisation.

Improved version with automatic ``px``/``vx`` token naming.
Uses ``plot_hull_3d`` from :mod:`.publication` for consistent mesh
rendering.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from .publication import plot_hull_3d, parts_from_vset


# ──── token → data columns ────

def _token_columns(vset, eng):
    """Map human-readable tokens to data columns.

    Token scheme (automatic for double-integrator layout
    ``[p_x, v_x, p_y, v_y, ...]``):

    ===== ===================
    Token Meaning
    ===== ===================
    px    position x
    vx    velocity x
    py    position y
    vy    velocity y
    ux    control x  (= Kx + v)
    wx    disturbance x
    …     (extends to z, etc.)
    ===== ===================
    """
    X, Vc, U, W = parts_from_vset(vset, eng)

    cols = {}
    n_pairs = eng.n // 2
    axes = ["x", "y", "z"]

    for i in range(n_pairs):
        axis = axes[i] if i < len(axes) else str(i + 1)
        cols[f"p{axis}"] = X[:, 2 * i]
        cols[f"v{axis}"] = X[:, 2 * i + 1]

    for j in range(eng.m):
        axis = axes[j] if j < len(axes) else str(j + 1)
        cols[f"u{axis}"] = U[:, j]

    for j in range(eng.w):
        axis = axes[j] if j < len(axes) else str(j + 1)
        cols[f"w{axis}"] = W[:, j]

    return cols


def _axis_label(tok: str) -> str:
    """Convert a token like ``'px'`` to a LaTeX axis label."""
    if len(tok) < 2:
        return tok
    prefix = tok[0].lower()
    subscript = tok[1:]
    if prefix == "p":
        return rf"$p_{subscript}$ [m]"
    if prefix == "v":
        return rf"$v_{subscript}$ [m/s]"
    if prefix == "u":
        return rf"$u_{subscript}$ [m/s$^2$]"
    if prefix == "w":
        return rf"$w_{subscript}$ [m/s$^2$]"
    return tok


def triplet_points(vset, eng, triplet):
    """Return an (N, 3) array of the three token columns from *vset*."""
    cols = _token_columns(vset, eng)
    a, b, c = triplet
    return np.c_[cols[a], cols[b], cols[c]]


# ──── main convergence plotter ────

_COLOR_G = "#2E86AB"
_COLOR_Z0 = "#A23B72"
_COLOR_MID = "#3C9D6D"
_COLOR_ZSTAR = "#F18F01"


def plot_convergence_triplet(
    history,
    G_vset,
    eng,
    triplet,
    *,
    title_prefix: str = "Convergence in lifted space",
    max_layers: int = 7,
    save: bool = True,
    save_dir: str = ".",
):
    """
    Plot convergence layers ``[Z_0, Z_1, …, Z*]`` projected onto a
    3-component triplet such as ``('px', 'ux', 'wx')``.

    Parameters
    ----------
    history : list[VSet]
        Iteration history ``[Z_0, …, Z*]``.
    G_vset : VSet
        Constraint graph set (drawn transparently in the background).
    eng : LiftedSetOpsGPU_NoHull
        Engine instance (needed for ``n, m, w, K``).
    triplet : tuple[str, str, str]
        Token names, e.g. ``('px', 'ux', 'wx')``.
    max_layers : int
        Maximum number of intermediate layers to render.
    save : bool
        Whether to save ``conv_{a}_{b}_{c}.{png,pdf}``.
    """
    K = len(history)
    if K <= max_layers:
        layers = list(range(K))
    else:
        idx = np.linspace(0, K - 1, max_layers).round().astype(int)
        layers = sorted(set(idx.tolist() + [0, K - 1]))

    fig = plt.figure(figsize=(4.6, 3.8))
    ax = fig.add_subplot(111, projection="3d")

    # G background
    PG = triplet_points(G_vset, eng, triplet)
    plot_hull_3d(ax, PG, _COLOR_G, alpha=0.18, label=r"$\mathcal{G}$",
                 edge_color=_COLOR_G, edge_alpha=0.35)

    # intermediate layers
    alphas = np.linspace(0.05, 0.1, len(layers))
    for k, a in zip(layers[:-1], alphas[:-1]):
        Pk = triplet_points(history[k], eng, triplet)
        col = _COLOR_Z0 if k == 0 else _COLOR_MID
        plot_hull_3d(ax, Pk, col, alpha=a,
                     label=(r"$Z_0$" if k == 0 else None),
                     edge_color=col, edge_alpha=0.2)

    # final Z*
    Pstar = triplet_points(history[-1], eng, triplet)
    plot_hull_3d(ax, Pstar, _COLOR_ZSTAR, alpha=0.8, label=r"$Z^*$",
                 edge_color=_COLOR_ZSTAR, edge_alpha=0.9)

    a, b, c = triplet
    ax.set_xlabel(_axis_label(a), labelpad=7)
    ax.set_ylabel(_axis_label(b), labelpad=7)
    ax.set_zlabel(_axis_label(c), labelpad=7)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.view_init(elev=22, azim=42)
    plt.tight_layout()

    if save:
        import os
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"conv_{a}_{b}_{c}.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        fig.savefig(fname.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")

    return fig
