"""
Publication-quality matplotlib plots for the lifted RPI operator.

Includes:
- ``plot_hull_3d`` : trisurf mesh with QJ joggled convex hull
- ``plot_hull_2d`` : filled / outline polygon
- ``parts_from_vset`` : split augmented vertices into (X, Vc, U, W)
- Pre-built publication figure generators (Figs 1-5)

Fonts default to Computer Modern (TeX) at 300 dpi.
"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ─── publication-quality rcParams ───
PUBLICATION_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.3,
}


def apply_publication_style():
    """Apply ``PUBLICATION_RCPARAMS`` to the global matplotlib state."""
    mpl.rcParams.update(PUBLICATION_RCPARAMS)


# ─── vertex extraction ───

def parts_from_vset(vset, eng):
    """Split augmented V-set into state (X), control perturbation (Vc),
    actual control (U = K X + V), and disturbance (W) arrays."""
    V = vset.V
    n, m, w = eng.n, eng.m, eng.w
    X = V[:, :n]
    Vc = V[:, n : n + m]
    W = V[:, n + m : n + m + w]
    U = (eng.K @ X.T).T + Vc
    return X, Vc, U, W


# ─── 3-D hull plot ───

def plot_hull_3d(
    ax,
    points,
    color,
    alpha: float = 0.30,
    label=None,
    edge_color=None,
    edge_alpha: float = 0.65,
):
    """
    Draw a trisurf mesh on a 3-D Axes.

    Pads with an epsilon tetrahedron when fewer than 4 unique points
    so that ``ConvexHull`` always succeeds.
    """
    P = np.asarray(points, dtype=float)
    P = np.unique(P, axis=0)
    if P.shape[0] < 4:
        c = P.mean(axis=0) if P.size else np.zeros(3)
        span = (P.max(axis=0) - P.min(axis=0)) if P.size else np.ones(3)
        eps = 1e-6 + 1e-3 * np.linalg.norm(span)
        pad = np.array(
            [[eps, 0.0, 0.0], [-eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
        )
        P = np.vstack([P, c + pad])

    hull = ConvexHull(P, qhull_options="QJ Pp Qt")
    tris = hull.simplices

    ax.plot_trisurf(
        P[:, 0],
        P[:, 1],
        P[:, 2],
        triangles=tris,
        color=color,
        alpha=alpha,
        shade=True,
        edgecolor=(edge_color or color),
        linewidth=0.4,
    )

    if label:
        ax.plot([], [], [], color=color, alpha=alpha, label=label, linewidth=3)

    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("persp")
    ax.set_box_aspect([1, 1, 0.8])


# ─── 2-D hull plot ───

def plot_hull_2d(
    ax,
    points,
    color,
    alpha: float = 0.30,
    label=None,
    fill: bool = True,
    edge_style: str = "-",
):
    """
    Draw a filled / outline polygon from the 2-D convex hull.
    """
    P = np.asarray(points, dtype=float)
    P = np.unique(P, axis=0)
    if P.shape[0] < 3:
        c = P.mean(axis=0) if P.size else np.zeros(2)
        span = (P.max(axis=0) - P.min(axis=0)) if P.size else np.ones(2)
        eps = 1e-6 + 1e-3 * np.linalg.norm(span)
        pad = np.array([[eps, 0.0], [0.0, eps], [-eps, -eps]])
        P = np.vstack([P, c + pad])

    hull = ConvexHull(P, qhull_options="QJ Pp")
    verts = P[hull.vertices]
    verts = np.vstack([verts, verts[0]])

    if fill:
        ax.fill(
            verts[:, 0],
            verts[:, 1],
            color=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=1.3,
            linestyle=edge_style,
            label=label,
        )
    else:
        ax.plot(
            verts[:, 0],
            verts[:, 1],
            color=color,
            linewidth=1.8,
            linestyle=edge_style,
            alpha=min(1.0, alpha * 3),
            label=label,
        )


# ─── pre-built figure generators ───

# Default palette used across all five standard figures
_COLOR_G = "#2E86AB"
_COLOR_Z0 = "#A23B72"
_COLOR_ZSTAR = "#F18F01"


def make_publication_figures(
    G_vset, Z0, Z_star, eng, *, G_bounds_fn=None, save: bool = True,
    save_dir: str = ".",
):
    """
    Generate all five publication figures and optionally save to
    ``fig_*.{png,pdf}`` in *save_dir*.

    Parameters
    ----------
    G_vset, Z0, Z_star : VSet
    eng : LiftedSetOpsGPU_NoHull
    G_bounds_fn : callable, optional
        ``G.bounds_fn`` for the contour plot (Fig 4).  Omit to skip Fig 4.
    save_dir : str
        Directory for saved figures (default: current directory).
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    apply_publication_style()

    X_G, V_G, U_G, W_G = parts_from_vset(G_vset, eng)
    X_0, V_0, U_0, W_0 = parts_from_vset(Z0, eng)
    X_s, V_s, U_s, W_s = parts_from_vset(Z_star, eng)

    ix_x1, ix_v1, ix_x2, ix_v2 = 0, 1, 2, 3
    iw1 = 0

    figs = {}

    # FIG 1: 3-D (p_x, p_y, v_x)
    fig1 = plt.figure(figsize=(4.2, 3.6))
    ax = fig1.add_subplot(111, projection="3d")
    plot_hull_3d(ax, np.c_[X_G[:, ix_x1], X_G[:, ix_x2], X_G[:, ix_v1]],
                 _COLOR_G, alpha=0.20, label=r"$\mathcal{G}$")
    plot_hull_3d(ax, np.c_[X_0[:, ix_x1], X_0[:, ix_x2], X_0[:, ix_v1]],
                 _COLOR_Z0, alpha=0.25, label=r"$Z_0$")
    plot_hull_3d(ax, np.c_[X_s[:, ix_x1], X_s[:, ix_x2], X_s[:, ix_v1]],
                 _COLOR_ZSTAR, alpha=0.35, label=r"$Z^*$")
    ax.set_xlabel(r"$p_x$ [m]", labelpad=7)
    ax.set_ylabel(r"$p_y$ [m]", labelpad=7)
    ax.set_zlabel(r"$v_x$ [m/s]", labelpad=7)
    ax.view_init(elev=20, azim=45)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_box_aspect([1, 1, 0.8])
    plt.tight_layout()
    if save:
        fig1.savefig(os.path.join(save_dir, "fig_x1_x2_v1_3d.png"), dpi=300, bbox_inches="tight")
        fig1.savefig(os.path.join(save_dir, "fig_x1_x2_v1_3d.pdf"), dpi=300, bbox_inches="tight")
    figs["x1_x2_v1_3d"] = fig1

    # FIG 2: 3-D (v_x, v_y, w_x)
    fig2 = plt.figure(figsize=(4.2, 3.6))
    ax = fig2.add_subplot(111, projection="3d")
    plot_hull_3d(ax, np.c_[X_G[:, ix_v1], X_G[:, ix_v2], W_G[:, iw1]],
                 _COLOR_G, alpha=0.20, label=r"$\mathcal{G}$")
    plot_hull_3d(ax, np.c_[X_0[:, ix_v1], X_0[:, ix_v2], W_0[:, iw1]],
                 _COLOR_Z0, alpha=0.25, label=r"$Z_0$")
    plot_hull_3d(ax, np.c_[X_s[:, ix_v1], X_s[:, ix_v2], W_s[:, iw1]],
                 _COLOR_ZSTAR, alpha=0.35, label=r"$Z^*$")
    ax.set_xlabel(r"$v_x$ [m/s]", labelpad=7)
    ax.set_ylabel(r"$v_y$ [m/s]", labelpad=7)
    ax.set_zlabel(r"$w_x$ [m/s$^2$]", labelpad=7)
    ax.view_init(elev=25, azim=-60)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_box_aspect([1, 1, 0.65])
    plt.tight_layout()
    if save:
        fig2.savefig(os.path.join(save_dir, "fig_v1_v2_w1_3d.png"), dpi=300, bbox_inches="tight")
        fig2.savefig(os.path.join(save_dir, "fig_v1_v2_w1_3d.pdf"), dpi=300, bbox_inches="tight")
    figs["v1_v2_w1_3d"] = fig2

    # FIG 3: 2-D (p_x, v_x)
    fig3, ax = plt.subplots(figsize=(4.4, 3.6))
    plot_hull_2d(ax, np.c_[X_G[:, ix_x1], X_G[:, ix_v1]], _COLOR_G,
                 alpha=0.15, label=r"$\mathcal{G}$", fill=True)
    plot_hull_2d(ax, np.c_[X_0[:, ix_x1], X_0[:, ix_v1]], _COLOR_Z0,
                 alpha=0.20, label=r"$Z_0$", fill=True)
    plot_hull_2d(ax, np.c_[X_s[:, ix_x1], X_s[:, ix_v1]], _COLOR_ZSTAR,
                 alpha=0.30, label=r"$Z^*$", fill=True)
    plot_hull_2d(ax, np.c_[X_0[:, ix_x1], X_0[:, ix_v1]], _COLOR_Z0,
                 alpha=0.9, fill=False, edge_style="--")
    plot_hull_2d(ax, np.c_[X_s[:, ix_x1], X_s[:, ix_v1]], _COLOR_ZSTAR,
                 alpha=1.0, fill=False, edge_style="-")
    ax.set_xlabel(r"$p_x$ [m]")
    ax.set_ylabel(r"$v_x$ [m/s]")
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    if save:
        fig3.savefig(os.path.join(save_dir, "fig_x1_v1.png"), dpi=300, bbox_inches="tight")
        fig3.savefig(os.path.join(save_dir, "fig_x1_v1.pdf"), dpi=300, bbox_inches="tight")
    figs["x1_v1"] = fig3

    # FIG 4: 2-D contour (v_x, u_x) colored by w_x, only if bounds_fn given
    if G_bounds_fn is not None:
        v1_min = np.percentile(
            np.r_[X_G[:, ix_v1], X_0[:, ix_v1], X_s[:, ix_v1]], 1
        )
        v1_max = np.percentile(
            np.r_[X_G[:, ix_v1], X_0[:, ix_v1], X_s[:, ix_v1]], 99
        )
        u1_min = np.percentile(np.r_[U_G[:, 0], U_0[:, 0], U_s[:, 0]], 1)
        u1_max = np.percentile(np.r_[U_G[:, 0], U_0[:, 0], U_s[:, 0]], 99)

        v1_lin = np.linspace(v1_min, v1_max, 120)
        u1_lin = np.linspace(u1_min, u1_max, 120)
        V1, U1 = np.meshgrid(v1_lin, u1_lin)

        Xg = np.zeros((V1.size, eng.n))
        Ug = np.zeros((V1.size, eng.m))
        Xg[:, ix_v1] = V1.ravel()
        Ug[:, 0] = U1.ravel()

        lbW, ubW = G_bounds_fn(Xg, Ug)
        W1_mid = 0.5 * (lbW[:, iw1] + ubW[:, iw1])

        fig4, ax = plt.subplots(figsize=(4.4, 3.6))
        w_range = W1_mid.max() - W1_mid.min()
        if w_range > 1e-6:
            levels = np.linspace(W1_mid.min(), W1_mid.max(), 15)
            cs = ax.contourf(V1, U1, W1_mid.reshape(V1.shape),
                             levels=levels, cmap="RdBu_r", alpha=0.8)
            ax.contour(V1, U1, W1_mid.reshape(V1.shape),
                       levels=levels[::3], colors="k", linewidths=0.5, alpha=0.5)
        else:
            cs = ax.pcolormesh(V1, U1, W1_mid.reshape(V1.shape),
                               cmap="RdBu_r", alpha=0.8, shading="auto")
        cbar = plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$w_x$ [m/s$^2$]", rotation=270, labelpad=14)

        ax.scatter(X_0[:, ix_v1], U_0[:, 0], c=_COLOR_Z0, s=10,
                   alpha=0.35, label=r"$Z_0$ proj.")
        ax.scatter(X_s[:, ix_v1], U_s[:, 0], c=_COLOR_ZSTAR, s=14,
                   alpha=0.55, label=r"$Z^*$ proj.")
        ax.set_xlabel(r"$v_x$ [m/s]")
        ax.set_ylabel(r"$u_x$ [m/s$^2$]")
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.legend(loc="upper left", framealpha=0.9)
        plt.tight_layout()
        if save:
            fig4.savefig(os.path.join(save_dir, "fig_v1_u1_contour.png"), dpi=300, bbox_inches="tight")
            fig4.savefig(os.path.join(save_dir, "fig_v1_u1_contour.pdf"), dpi=300, bbox_inches="tight")
        figs["v1_u1_contour"] = fig4

    # FIG 5: 2-D (p_y, v_y)
    fig5, ax = plt.subplots(figsize=(4.4, 3.6))
    plot_hull_2d(ax, np.c_[X_G[:, ix_x2], X_G[:, ix_v2]], _COLOR_G,
                 alpha=0.15, label=r"$\mathcal{G}$", fill=True)
    plot_hull_2d(ax, np.c_[X_0[:, ix_x2], X_0[:, ix_v2]], _COLOR_Z0,
                 alpha=0.20, label=r"$Z_0$", fill=True)
    plot_hull_2d(ax, np.c_[X_s[:, ix_x2], X_s[:, ix_v2]], _COLOR_ZSTAR,
                 alpha=0.30, label=r"$Z^*$", fill=True)
    plot_hull_2d(ax, np.c_[X_0[:, ix_x2], X_0[:, ix_v2]], _COLOR_Z0,
                 alpha=0.9, fill=False, edge_style="--")
    plot_hull_2d(ax, np.c_[X_s[:, ix_x2], X_s[:, ix_v2]], _COLOR_ZSTAR,
                 alpha=1.0, fill=False, edge_style="-")
    ax.set_xlabel(r"$p_y$ [m]")
    ax.set_ylabel(r"$v_y$ [m/s]")
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    if save:
        fig5.savefig(os.path.join(save_dir, "fig_x2_v2.png"), dpi=300, bbox_inches="tight")
        fig5.savefig(os.path.join(save_dir, "fig_x2_v2.pdf"), dpi=300, bbox_inches="tight")
    figs["x2_v2"] = fig5

    return figs
