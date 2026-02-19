#!/usr/bin/env python
"""
Generate publication-quality figures from saved pipeline results.

Reads the NPZ output of ``run_pipeline.py`` and (optionally) the saved
GP model to produce the full suite of figures used in the paper.

Figure steps
------------
  [1] Hausdorff convergence + normalised AABB volume
  [2] Per-iteration timing breakdown (forward vs. clipping)
  [3] 2-D state-space projections  (Z0 vs Z*)
  [4] 2-D full set comparison      (G vs Z0 vs Z*)
  [5] 3-D set comparison           (G vs Z0 vs Z*)
  [6] 3-D convergence evolution     (G -> Z0 -> intermediate -> Z*)
  [7] 3-D acceleration-disturbance scatter + GP mean surfaces
  [8] 2-D disturbance heatmap projections (contourf + scatter)

Steps 7-8 use simulation trajectory data (state/control/disturbance
histories) with a fresh GP fit.

All figures are saved as both PDF (vector) and PNG (300 dpi raster).
Text is rendered with Computer Modern (CMU) font via LaTeX.

Usage
-----
    python scripts/generate_figures.py
    python scripts/generate_figures.py --no-tex        # disable LaTeX
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════
# Colour palette  (identical to paper)
# ═══════════════════════════════════════════════════════════════════════
_COLOR_G     = "#2E86AB"   # Graph G
_COLOR_Z0    = "#A23B72"   # Initial set Z0
_COLOR_ZSTAR = "#F18F01"   # Fixed point Z*
_COLOR_MID   = "#3C9D6D"   # Intermediate iterates
_COLOR_TOL   = "#D32F2F"   # Tolerance line


# ═══════════════════════════════════════════════════════════════════════
# IEEE / LCSS style with LaTeX Computer Modern font
# ═══════════════════════════════════════════════════════════════════════

def apply_ieee_style(use_tex: bool = True):
    """Apply IEEE-compatible matplotlib RC style (Computer Modern font)."""
    import matplotlib as mpl

    style = {
        # ── Font (CMU via LaTeX, or fallback serif) ──
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman",
                        "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # ── Figure ──
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        # ── Lines / axes ──
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.25,
        "axes.grid": True,
        # ── LaTeX rendering ──
        "text.usetex": use_tex,
    }
    if use_tex:
        style["text.latex.preamble"] = (
            r"\usepackage{amsmath}"
            r"\usepackage{amssymb}"
            r"\usepackage[T1]{fontenc}"
        )
    mpl.rcParams.update(style)


# ═══════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════

def _save(fig, save_dir, name):
    """Save a figure as both PDF and PNG."""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(save_dir, f"{name}.{ext}"),
                    dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"  {name}.{{pdf,png}}")


def _hull_3d(ax, pts, color, alpha=0.30, label=None, edge_color=None):
    """Trisurf mesh from ConvexHull (joggled, triangle-faceted)."""
    from scipy.spatial import ConvexHull
    P = np.unique(np.asarray(pts, dtype=float), axis=0)
    if P.shape[0] < 4:
        c = P.mean(axis=0) if P.size else np.zeros(3)
        span = (P.max(0) - P.min(0)) if P.size else np.ones(3)
        eps = 1e-6 + 1e-3 * np.linalg.norm(span)
        pad = np.array([[eps, 0, 0], [-eps, 0, 0], [0, eps, 0], [0, 0, eps]])
        P = np.vstack([P, c + pad])
    hull = ConvexHull(P, qhull_options="QJ Pp Qt")
    ax.plot_trisurf(P[:, 0], P[:, 1], P[:, 2], triangles=hull.simplices,
                    color=color, alpha=alpha, shade=True,
                    edgecolor=(edge_color or color), linewidth=0.4)
    if label:
        ax.plot([], [], [], color=color, alpha=min(1.0, alpha + 0.3),
                label=label, linewidth=4)
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("persp")
    ax.set_box_aspect([1, 1, 0.8])


def _hull_2d(ax, pts, color, alpha=0.30, label=None, fill=True, ls="-"):
    """Filled/outline polygon from 2-D ConvexHull."""
    from scipy.spatial import ConvexHull
    P = np.unique(np.asarray(pts, dtype=float), axis=0)
    if P.shape[0] < 3:
        c = P.mean(axis=0) if P.size else np.zeros(2)
        span = (P.max(0) - P.min(0)) if P.size else np.ones(2)
        eps = 1e-6 + 1e-3 * np.linalg.norm(span)
        P = np.vstack([P, c + np.array([[eps, 0], [0, eps], [-eps, -eps]])])
    hull = ConvexHull(P, qhull_options="QJ Pp")
    v = P[hull.vertices]; v = np.vstack([v, v[0]])
    if fill:
        ax.fill(v[:, 0], v[:, 1], color=color, alpha=alpha,
                edgecolor=color, linewidth=1.3, linestyle=ls, label=label)
    else:
        ax.plot(v[:, 0], v[:, 1], color=color, linewidth=1.8,
                linestyle=ls, alpha=min(1.0, alpha * 3), label=label)


def _parts(V, K, n, m, w):
    """Split augmented vertex matrix into (X, Vc, U, W)."""
    X  = V[:, :n]
    Vc = V[:, n:n + m]
    W  = V[:, n + m:n + m + w]
    U  = (K @ X.T).T + Vc
    return X, Vc, U, W


def _token_cols(V, K, n, m, w):
    """Map tokens like 'px', 'vx', 'ux', 'wx' to data columns."""
    X, Vc, U, W = _parts(V, K, n, m, w)
    axes = ["x", "y", "z"]
    cols = {}
    for i in range(n // 2):
        a = axes[i] if i < len(axes) else str(i + 1)
        cols[f"p{a}"] = X[:, 2 * i]
        cols[f"v{a}"] = X[:, 2 * i + 1]
    for j in range(m):
        a = axes[j] if j < len(axes) else str(j + 1)
        cols[f"u{a}"] = U[:, j]
    for j in range(w):
        a = axes[j] if j < len(axes) else str(j + 1)
        cols[f"w{a}"] = W[:, j]
    return cols


# Token → rich LaTeX label with proper unit (Computer Modern via usetex)
_LABEL_MAP = {
    "px": (r"$p_x$",  r"[m]"),
    "py": (r"$p_y$",  r"[m]"),
    "vx": (r"$v_x$",  r"[m\,s$^{-1}$]"),
    "vy": (r"$v_y$",  r"[m\,s$^{-1}$]"),
    "ux": (r"$u_x$",  r"[m\,s$^{-2}$]"),
    "uy": (r"$u_y$",  r"[m\,s$^{-2}$]"),
    "wx": (r"$w_x$",  r"[m\,s$^{-2}$]"),
    "wy": (r"$w_y$",  r"[m\,s$^{-2}$]"),
}


def _tok_label(tok: str) -> str:
    """Token → full LaTeX axis label with unit, e.g. '$p_x$ [m]'."""
    sym, unit = _LABEL_MAP.get(tok, (tok, ""))
    return f"{sym} {unit}".strip()


# ═══════════════════════════════════════════════════════════════════════
# [1] Convergence metrics
# ═══════════════════════════════════════════════════════════════════════

def plot_convergence_metrics(data, save_dir, pfx="fig"):
    """Hausdorff convergence and normalised volume curves."""
    import matplotlib.pyplot as plt

    haus = data.get("hausdorff", np.array([]))
    vols = data.get("volumes_state_norm", np.array([]))
    tol = 3e-2

    if haus.size > 0:
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        iters = np.arange(1, len(haus) + 1)
        ax.semilogy(iters, haus, "o-", color=_COLOR_G, markersize=2.5,
                     linewidth=0.8,
                     label=r"$d_H^{\mathrm{norm}}(Z_k, Z_{k-1})$")
        ax.axhline(tol, ls="--", color=_COLOR_TOL, lw=0.7, alpha=0.8,
                    label=rf"$\tau = {tol}$")
        ax.set_xlabel(r"Iteration $k$")
        ax.set_ylabel(r"Normalised Hausdorff distance")
        ax.legend(loc="upper right", framealpha=0.92)
        ax.set_xlim(1, len(haus))
        plt.tight_layout()
        _save(fig, save_dir, f"{pfx}_hausdorff")
        plt.close(fig)

    if vols.size > 0:
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        ax.plot(np.arange(len(vols)), vols, "s-", color=_COLOR_ZSTAR,
                markersize=2, linewidth=0.8)
        ax.set_xlabel(r"Iteration $k$")
        ax.set_ylabel(r"$\mathrm{vol}(Z_k^x)\,/\,\mathrm{vol}(Z_0^x)$")
        ax.set_xlim(0, len(vols) - 1)
        plt.tight_layout()
        _save(fig, save_dir, f"{pfx}_volume")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# [2] Per-iteration timing
# ═══════════════════════════════════════════════════════════════════════

def plot_timing_breakdown(data, save_dir, pfx="fig"):
    """Stacked bar chart: forward vs clipping time per iteration."""
    import matplotlib.pyplot as plt

    t_fwd  = data.get("time_forward", np.array([]))
    t_clip = data.get("time_clip", np.array([]))
    if t_fwd.size == 0:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    iters = np.arange(1, len(t_fwd) + 1)
    ax.bar(iters, t_fwd, color=_COLOR_G,
           label=r"Forward $\mathcal{F}$", width=0.8)
    ax.bar(iters, t_clip, bottom=t_fwd, color=_COLOR_ZSTAR,
           label=r"Clipping $\cap\,\mathcal{G}$", width=0.8)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Time [s]")
    ax.legend(loc="upper right", framealpha=0.92)
    ax.set_xlim(0.5, len(t_fwd) + 0.5)
    plt.tight_layout()
    _save(fig, save_dir, f"{pfx}_timing")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# [3] 2-D state projections (Z0 vs Z*)
# ═══════════════════════════════════════════════════════════════════════

def plot_state_projections_2d(data, K, n, m, w, save_dir, pfx="fig"):
    """2-D state-space projections: Z0 (dashed) vs Z* (filled)."""
    import matplotlib.pyplot as plt

    Zstar = data.get("Zstar", np.array([]))
    Z0    = data.get("Z0", np.array([]))
    if Zstar.size == 0 or Zstar.shape[0] < 4:
        return

    Zstar_x = Zstar[:, :n]
    Z0_x    = Z0[:, :n] if Z0.size > 0 else None
    tok     = ["px", "vx", "py", "vy"]
    pairs   = [(0,1,"px_vx"), (2,3,"py_vy"), (0,2,"px_py"), (1,3,"vx_vy")]

    for i, j, tag in pairs:
        if i >= n or j >= n:
            continue
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        _hull_2d(ax, Zstar_x[:,[i,j]], _COLOR_ZSTAR, alpha=0.25, label=r"$Z^*$")
        _hull_2d(ax, Zstar_x[:,[i,j]], _COLOR_ZSTAR, alpha=1.0, fill=False)
        if Z0_x is not None and Z0_x.shape[0] >= 3:
            _hull_2d(ax, Z0_x[:,[i,j]], _COLOR_Z0, alpha=0.7,
                     fill=False, ls="--", label=r"$Z_0$")
        ax.set_xlabel(_tok_label(tok[i]))
        ax.set_ylabel(_tok_label(tok[j]))
        ax.legend(loc="best", framealpha=0.92)
        ax.set_aspect("equal")
        plt.tight_layout()
        _save(fig, save_dir, f"{pfx}_{tag}")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# [4] 2-D G vs Z0 vs Z* comparison (with fills + outlines)
# ═══════════════════════════════════════════════════════════════════════

def plot_2d_set_comparison(data, K, n, m, w, save_dir, pfx="fig"):
    """2-D hull projections: G, Z0, Z* overlaid (px/vx, py/vy)."""
    import matplotlib.pyplot as plt

    G_V  = data.get("G_vset", np.array([]))
    Z0_V = data.get("Z0", np.array([]))
    Zs_V = data.get("Zstar", np.array([]))
    if G_V.size == 0:
        print("  [2D comparison] No G_vset, skipping.")
        return

    X_G,_,_,_ = _parts(G_V, K, n, m, w)
    X_0 = Z0_V[:,:n] if Z0_V.size > 0 else None
    X_s = Zs_V[:,:n] if Zs_V.size > 0 else None

    tok    = ["px", "vx", "py", "vy"]
    pairs  = [(0,1,"px_vx_full"), (2,3,"py_vy_full")]

    for i, j, tag in pairs:
        if i >= n or j >= n:
            continue
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        _hull_2d(ax, X_G[:,[i,j]], _COLOR_G, alpha=0.15,
                 label=r"$\mathcal{G}$")
        if X_0 is not None and X_0.shape[0] >= 3:
            _hull_2d(ax, X_0[:,[i,j]], _COLOR_Z0, alpha=0.20, label=r"$Z_0$")
            _hull_2d(ax, X_0[:,[i,j]], _COLOR_Z0, alpha=0.9, fill=False, ls="--")
        if X_s is not None and X_s.shape[0] >= 3:
            _hull_2d(ax, X_s[:,[i,j]], _COLOR_ZSTAR, alpha=0.30, label=r"$Z^*$")
            _hull_2d(ax, X_s[:,[i,j]], _COLOR_ZSTAR, alpha=1.0, fill=False)
        ax.set_xlabel(_tok_label(tok[i]))
        ax.set_ylabel(_tok_label(tok[j]))
        ax.legend(loc="upper right", framealpha=0.92)
        plt.tight_layout()
        _save(fig, save_dir, f"{pfx}_2d_{tag}")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# [5] 3-D G vs Z0 vs Z* comparison
# ═══════════════════════════════════════════════════════════════════════

def plot_3d_set_comparison(data, K, n, m, w, save_dir, pfx="fig"):
    """3-D trisurf of G vs Z0 vs Z* in several coordinate triplets."""
    import matplotlib.pyplot as plt

    G_V  = data.get("G_vset", np.array([]))
    Z0_V = data.get("Z0", np.array([]))
    Zs_V = data.get("Zstar", np.array([]))
    if G_V.size == 0 or Z0_V.size == 0 or Zs_V.size == 0:
        print("  [3D comparison] Missing G_vset/Z0/Zstar, skipping.")
        return

    cols_G  = _token_cols(G_V, K, n, m, w)
    cols_Z0 = _token_cols(Z0_V, K, n, m, w)
    cols_Zs = _token_cols(Zs_V, K, n, m, w)

    # Coordinate triplets for 3-D comparison
    triplets_3d = [
        # (tok_x, tok_y, tok_z, elev, azim, z_aspect, filename_tag)
        ("px", "py", "vx",  20,  45, 0.80, "px_py_vx"),
        ("vx", "vy", "wx",  25, -60, 0.65, "vx_vy_wx"),
        ("px", "vx", "ux",  22,  35, 0.80, "px_vx_ux"),
        ("py", "vy", "wy",  25, -55, 0.70, "py_vy_wy"),
    ]

    for tx, ty, tz, elev, azim, zasp, tag in triplets_3d:
        fig = plt.figure(figsize=(4.2, 3.6))
        ax  = fig.add_subplot(111, projection="3d")

        def _pts(cols):
            return np.c_[cols[tx], cols[ty], cols[tz]]

        _hull_3d(ax, _pts(cols_G),  _COLOR_G, alpha=0.20,
                 label=r"$\mathcal{G}$", edge_color=_COLOR_G)
        _hull_3d(ax, _pts(cols_Z0), _COLOR_Z0, alpha=0.25,
                 label=r"$Z_0$", edge_color=_COLOR_Z0)
        _hull_3d(ax, _pts(cols_Zs), _COLOR_ZSTAR, alpha=0.35,
                 label=r"$Z^*$", edge_color=_COLOR_ZSTAR)

        ax.set_xlabel(_tok_label(tx), labelpad=8)
        ax.set_ylabel(_tok_label(ty), labelpad=8)
        ax.set_zlabel(_tok_label(tz), labelpad=8)
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc="upper right", framealpha=0.92)
        ax.set_box_aspect([1, 1, zasp])
        plt.tight_layout()
        _save(fig, save_dir, f"{pfx}_3d_{tag}")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# [6] 3-D convergence evolution
# ═══════════════════════════════════════════════════════════════════════

def plot_convergence_evolution_3d(data, K, n, m, w, save_dir, pfx="fig"):
    """3-D convergence layers: G → Z0 (faint) → intermediate → Z* (bold)."""
    import matplotlib.pyplot as plt

    G_V      = data.get("G_vset", np.array([]))
    snap_idx = data.get("history_snap_indices", np.array([]))
    if G_V.size == 0 or snap_idx.size == 0:
        print("  [3D evolution] Missing G_vset or history snapshots, skipping.")
        return

    # Gather history snapshots
    history_V = []
    for i in snap_idx:
        key = f"history_{int(i)}"
        if key in data and data[key].size > 0:
            history_V.append(data[key])
    if len(history_V) < 2:
        print("  [3D evolution] Not enough history snapshots, skipping.")
        return

    cols_G = _token_cols(G_V, K, n, m, w)

    # Coordinate triplets for convergence evolution
    triplets = [
        ("px", "ux", "wx"),
        ("py", "uy", "wy"),
        ("vx", "ux", "wx"),
        ("vy", "uy", "wy"),
        ("px", "vx", "wx"),
        ("py", "vy", "wy"),
    ]

    for ta, tb, tc in triplets:
        fig = plt.figure(figsize=(4.6, 3.8))
        ax  = fig.add_subplot(111, projection="3d")

        # G background
        PG = np.c_[cols_G[ta], cols_G[tb], cols_G[tc]]
        _hull_3d(ax, PG, _COLOR_G, alpha=0.18,
                 label=r"$\mathcal{G}$", edge_color=_COLOR_G)

        # Intermediate layers (faint → less faint)
        alphas = np.linspace(0.05, 0.12, len(history_V))
        for idx_k, (V_k, a) in enumerate(zip(history_V[:-1], alphas[:-1])):
            cols_k = _token_cols(V_k, K, n, m, w)
            Pk     = np.c_[cols_k[ta], cols_k[tb], cols_k[tc]]
            col    = _COLOR_Z0 if idx_k == 0 else _COLOR_MID
            lbl    = r"$Z_0$" if idx_k == 0 else None
            _hull_3d(ax, Pk, col, alpha=a, label=lbl, edge_color=col)

        # Final Z*
        cols_s = _token_cols(history_V[-1], K, n, m, w)
        Ps = np.c_[cols_s[ta], cols_s[tb], cols_s[tc]]
        _hull_3d(ax, Ps, _COLOR_ZSTAR, alpha=0.80,
                 label=r"$Z^*$", edge_color=_COLOR_ZSTAR)

        ax.set_xlabel(_tok_label(ta), labelpad=8)
        ax.set_ylabel(_tok_label(tb), labelpad=8)
        ax.set_zlabel(_tok_label(tc), labelpad=8)
        ax.legend(loc="upper right", framealpha=0.92)
        ax.view_init(elev=22, azim=42)
        plt.tight_layout()
        _save(fig, save_dir, f"{pfx}_conv_{ta}_{tb}_{tc}")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Simulation data + GP helper  (shared by steps 7, 8)
# ═══════════════════════════════════════════════════════════════════════

def _load_or_run_simulation(data):
    """Load simulation histories from NPZ, or re-run the MPC simulation.

    The NPZ stores state_history, control_history, disturbance_history
    when the pipeline was run with the GP-learned graph set.  If those
    keys are missing (e.g. older NPZ or analytical-drag run), re-run
    the simulation using the same parameters as the paper.

    Returns
    -------
    state_history : ndarray (N, 4)
    control_history : ndarray (N, 2)
    disturbance_history : ndarray (N, 4)
    """
    sh = data.get("state_history")
    ch = data.get("control_history")
    dh = data.get("disturbance_history")

    if sh is not None and ch is not None and dh is not None:
        print("    Loaded simulation histories from NPZ")
        return np.asarray(sh), np.asarray(ch), np.asarray(dh)

    # Re-run the lightweight MPC simulation (~2-5 min)
    print("    Simulation data not in NPZ, re-running MPC simulation ...")
    from lifted_rpi.simulation import simulate_trajectory_with_realistic_drag
    from scipy.linalg import solve_discrete_are

    dt = 0.02
    A_p = np.array([[1, dt, 0, 0],[0, 1, 0, 0],[0, 0, 1, dt],[0, 0, 0, 1]])
    B_p = np.array([[0.5*dt**2, 0],[dt, 0],[0, 0.5*dt**2],[0, dt]])
    Q = np.diag([1000, 0.1, 1000, 0.1])
    R = np.diag([0.1, 0.1])
    P = solve_discrete_are(A_p, B_p, Q, R)

    sh, ch, dh, _, sim_t, fps = simulate_trajectory_with_realistic_drag(
        trajectory_type='linear',
        x_initial=np.array([0, 0, 4, 0]),
        x_target=np.array([10, 0, 10, 0]),
        n_steps=2500, L_k=5,
        A_p=A_p, B_p=B_p, Q=Q, R=R, P=P,
        beta1=0.05, beta2=0.02, mass=1.0, noise_std=0.01, dt=dt,
    )
    print(f"    Simulation done ({sim_t:.1f}s, {fps:.0f} fps)")
    return sh, ch, dh


def _fit_gp_for_viz(state_history, control_history, disturbance_history,
                    downsample_factor=5, kernel_choice="RBF * SinExp + White"):
    """Fit GPs on simulation data for visualisation.

    Downsamples the trajectory, extracts velocity/control/disturbance
    features, and fits two independent GaussianProcessRegressors (one per
    disturbance axis) using the specified kernel.

    Parameters
    ----------
    state_history : ndarray (N, 4)
    control_history : ndarray (N, 2)
    disturbance_history : ndarray (N, 4)
    downsample_factor : int
    kernel_choice : str

    Returns
    -------
    dict with keys:
        vx, vy, ux, uy, wx, wy           -- downsampled raw arrays
        y_pred_x, y_pred_y                -- GP mean predictions
        sigma_x, sigma_y                  -- GP posterior std
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF, WhiteKernel, ConstantKernel, ExpSineSquared,
    )

    n_steps = len(state_history)
    indices = np.linspace(0, n_steps - 1,
                          n_steps // downsample_factor, dtype=int)

    states_ds = state_history[indices]
    controls_ds = control_history[indices]
    disturbances_ds = disturbance_history[indices]

    vx = states_ds[:, 1]
    vy = states_ds[:, 3]
    v_mag = np.sqrt(vx**2 + vy**2)
    ux = controls_ds[:, 0]
    uy = controls_ds[:, 1]
    wx = disturbances_ds[:, 1]
    wy = disturbances_ds[:, 3]

    # Kernel selection matching paper configuration
    if kernel_choice == "RBF + White":
        kernel = RBF(length_scale=1.0) + WhiteKernel(0.1)
    elif kernel_choice == "RBF * SinExp + White":
        kernel = (
            RBF(length_scale=1.0)
            * ExpSineSquared(1.0, 5.0, periodicity_bounds=(0.1, 10.0))
            + WhiteKernel(0.1)
        )
    else:
        kernel = RBF(length_scale=1.0) + WhiteKernel(0.1)

    X_feat = np.column_stack([vx, vy, v_mag, ux, uy, indices.astype(float)])

    gp_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                    alpha=1e-6)
    gp_x.fit(X_feat, wx)

    gp_y = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                    alpha=1e-6)
    gp_y.fit(X_feat, wy)

    y_pred_x, sigma_x = gp_x.predict(X_feat, return_std=True)
    y_pred_y, sigma_y = gp_y.predict(X_feat, return_std=True)

    return dict(
        vx=vx, vy=vy, ux=ux, uy=uy, wx=wx, wy=wy,
        y_pred_x=y_pred_x, y_pred_y=y_pred_y,
        sigma_x=sigma_x, sigma_y=sigma_y,
    )


def _griddata_fill(pts, vals, grid_x, grid_y):
    """Interpolate scattered data onto a regular grid.

    Uses cubic griddata for smooth results, then fills any remaining
    NaN values at grid edges with nearest-neighbour interpolation.
    """
    from scipy.interpolate import griddata
    Z = griddata(pts, vals, (grid_x, grid_y), method="cubic")
    mask = np.isnan(Z)
    if mask.any():
        Z_nn = griddata(pts, vals, (grid_x, grid_y), method="nearest")
        Z[mask] = Z_nn[mask]
    return Z


# ═══════════════════════════════════════════════════════════════════════
# [7] 3-D acceleration-disturbance scatter + GP surfaces
#     (matches paper: plot_acceleration_disturbance_spaces)
# ═══════════════════════════════════════════════════════════════════════

def plot_accel_dist_3d(data, K, n, m, w, save_dir, pfx="fig",
                       gp_results=None):
    """3-D scatter of simulation data coloured by GP uncertainty.

    Four figures corresponding to
    ``plot_acceleration_disturbance_spaces``:
        (vx, ux, wx),  (vy, uy, wy),  (ux, uy, wx),  (ux, uy, wy).

    Scatter points show the downsampled simulation trajectory.  The red
    surface is the GP mean prediction interpolated via griddata; the
    orange surfaces are the +/- 2-sigma confidence bounds.
    """
    import matplotlib.pyplot as plt

    if gp_results is None:
        print("  [3D accel-dist] No GP results -- skipping.")
        return

    gd = gp_results
    grid_res = 30

    panels = [
        # x_data, y_data, z_data, colour, gp_pred, gp_sigma,
        # x_tok, y_tok, z_tok, cmap, tag
        (gd["vx"], gd["ux"], gd["wx"], gd["sigma_x"],
         gd["y_pred_x"], gd["sigma_x"],
         "vx", "ux", "wx", "viridis", "vx_ux_wx"),
        (gd["vy"], gd["uy"], gd["wy"], gd["sigma_y"],
         gd["y_pred_y"], gd["sigma_y"],
         "vy", "uy", "wy", "plasma", "vy_uy_wy"),
        (gd["ux"], gd["uy"], gd["wx"], gd["sigma_x"],
         gd["y_pred_x"], gd["sigma_x"],
         "ux", "uy", "wx", "inferno", "ux_uy_wx"),
        (gd["ux"], gd["uy"], gd["wy"], gd["sigma_y"],
         gd["y_pred_y"], gd["sigma_y"],
         "ux", "uy", "wy", "coolwarm", "ux_uy_wy"),
    ]

    for xd, yd, zd, cd, w_pred, sig, xt, yt, zt, cmap, tag in panels:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(xd, yd, zd, c=cd, cmap=cmap, s=20, alpha=0.7)

        # Interpolate GP mean and +/- 2-sigma bounds onto a regular grid
        xr = np.linspace(xd.min(), xd.max(), grid_res)
        yr = np.linspace(yd.min(), yd.max(), grid_res)
        XG, YG = np.meshgrid(xr, yr)
        pts = np.column_stack([xd, yd])

        Z_mean  = _griddata_fill(pts, w_pred,          XG, YG)
        Z_upper = _griddata_fill(pts, w_pred + 2 * sig, XG, YG)
        Z_lower = _griddata_fill(pts, w_pred - 2 * sig, XG, YG)

        ax.plot_surface(XG, YG, Z_mean,  alpha=0.3, color="red")
        ax.plot_surface(XG, YG, Z_upper, alpha=0.15, color="orange")
        ax.plot_surface(XG, YG, Z_lower, alpha=0.15, color="orange")

        ax.set_xlabel(_tok_label(xt), labelpad=6)
        ax.set_ylabel(_tok_label(yt), labelpad=6)
        ax.set_zlabel(_tok_label(zt), labelpad=6)
        ax.tick_params(labelsize=6)

        cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.08)
        sigma_sym = r"$\sigma_x$" if "x" in zt else r"$\sigma_y$"
        cbar.set_label(f"Uncertainty {sigma_sym}", fontsize=8)
        cbar.ax.tick_params(labelsize=6)

        plt.tight_layout()
        _save(fig, save_dir, f"{pfx}_3d_gp_{tag}")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# [8] 2-D disturbance heatmap projections (contourf + scatter)
#     (matches paper: plot_2d_projections_acceleration_space)
# ═══════════════════════════════════════════════════════════════════════

def plot_2d_heatmap_projections(data, K, n, m, w, save_dir, pfx="fig",
                                gp_results=None):
    """2-D contourf projections of disturbance and GP uncertainty.

    Six figures matching
    ``plot_2d_projections_acceleration_space``:
        Row 1 (x-axis): (vx, ux) -> wx,  (vx, ux) -> sigma_x, (ux, uy) -> wx
        Row 2 (y-axis): (vy, uy) -> wy,  (vy, uy) -> sigma_y, (ux, uy) -> wy

    Each panel shows a filled contour interpolated from simulated data
    points, with the data overlaid as a scatter for reference.
    """
    import matplotlib.pyplot as plt

    if gp_results is None:
        print("  [2D heatmaps] No GP results -- skipping.")
        return

    gd = gp_results
    grid_res = 50

    panels = [
        # x_data, y_data, z_data, x_tok, y_tok, cbar_label, cmap, tag
        (gd["vx"], gd["ux"], gd["wx"],
         "vx", "ux", _tok_label("wx"), "RdBu_r", "vx_ux_wx"),
        (gd["vx"], gd["ux"], gd["sigma_x"],
         "vx", "ux", r"$\sigma_x$", "viridis", "vx_ux_sigma_x"),
        (gd["ux"], gd["uy"], gd["wx"],
         "ux", "uy", _tok_label("wx"), "RdBu_r", "ux_uy_wx"),
        (gd["vy"], gd["uy"], gd["wy"],
         "vy", "uy", _tok_label("wy"), "RdBu_r", "vy_uy_wy"),
        (gd["vy"], gd["uy"], gd["sigma_y"],
         "vy", "uy", r"$\sigma_y$", "plasma", "vy_uy_sigma_y"),
        (gd["ux"], gd["uy"], gd["wy"],
         "ux", "uy", _tok_label("wy"), "RdBu_r", "ux_uy_wy"),
    ]

    for xd, yd, zd, xt, yt, clabel, cmap, tag in panels:
        fig, ax = plt.subplots(figsize=(9, 7))

        xr = np.linspace(xd.min(), xd.max(), grid_res)
        yr = np.linspace(yd.min(), yd.max(), grid_res)
        XG, YG = np.meshgrid(xr, yr)
        ZG = _griddata_fill(np.column_stack([xd, yd]), zd, XG, YG)

        cs = ax.contourf(XG, YG, ZG, levels=20, cmap=cmap)
        ax.scatter(xd, yd, c=zd, s=10, cmap=cmap,
                   edgecolors="white", linewidth=0.3)

        cbar = plt.colorbar(cs, ax=ax)
        cbar.set_label(clabel, fontsize=8)
        cbar.ax.tick_params(labelsize=6)

        ax.set_xlabel(_tok_label(xt))
        ax.set_ylabel(_tok_label(yt))
        ax.grid(True, alpha=0.3, linewidth=0.3)
        plt.tight_layout()
        _save(fig, save_dir, f"{pfx}_2d_gp_{tag}")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--results", type=str,
                        default="results/pipeline_paper_exact.npz",
                        help="Path to saved .npz results file")
    parser.add_argument("--save-dir", type=str, default="results/figures",
                        help="Output directory for figures")
    parser.add_argument("--prefix", type=str, default="fig",
                        help="Filename prefix for all figures")
    parser.add_argument("--no-tex", action="store_true",
                        help="Disable LaTeX rendering (faster, less polished)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load the saved pipeline results
    if not os.path.exists(args.results):
        print(f"ERROR: Results file not found: {args.results}")
        print("Run 'python scripts/run_pipeline.py' first.")
        sys.exit(1)

    data = dict(np.load(args.results, allow_pickle=True))
    print(f"Loaded results from {args.results}")
    print(f"  Z*  shape : {data.get('Zstar', np.array([])).shape}")
    print(f"  Z0  shape : {data.get('Z0', np.array([])).shape}")
    print(f"  G   shape : {data.get('G_vset', np.array([])).shape}")
    print(f"  Iterations: {len(data.get('hausdorff', []))}")

    n = int(data.get("n", 4))
    m = int(data.get("m", 2))
    w = int(data.get("w", 2))
    K = data.get("K", np.zeros((m, n)))

    apply_ieee_style(use_tex=not args.no_tex)
    warnings.filterwarnings("ignore", category=UserWarning)

    print(f"\nGenerating figures in {args.save_dir}/ ...")

    print("\n[1/8] Convergence metrics")
    plot_convergence_metrics(data, args.save_dir, args.prefix)

    print("[2/8] Timing breakdown")
    plot_timing_breakdown(data, args.save_dir, args.prefix)

    print("[3/8] 2D state projections")
    plot_state_projections_2d(data, K, n, m, w, args.save_dir, args.prefix)

    print("[4/8] 2D set comparison (G vs Z0 vs Z*)")
    plot_2d_set_comparison(data, K, n, m, w, args.save_dir, args.prefix)

    print("[5/8] 3D set comparison (G vs Z0 vs Z*)")
    plot_3d_set_comparison(data, K, n, m, w, args.save_dir, args.prefix)

    print("[6/8] 3D convergence evolution")
    plot_convergence_evolution_3d(data, K, n, m, w, args.save_dir, args.prefix)

    # Steps 7-8 require simulation trajectory data + fresh GP fit
    # Steps 7-8 require simulation trajectory data + fresh GP fit.
    print("\n[7-8] Loading simulation data and fitting GP ...")
    sh, ch, dh = _load_or_run_simulation(data)
    print("    Fitting GP on downsampled simulation data ...")
    gp_results = _fit_gp_for_viz(sh, ch, dh, downsample_factor=5,
                                  kernel_choice="RBF * SinExp + White")
    print(f"    GP fit complete: {len(gp_results['vx'])} points")

    print("[7/8] 3D acceleration-disturbance spaces")
    plot_accel_dist_3d(data, K, n, m, w, args.save_dir, args.prefix,
                       gp_results=gp_results)

    print("[8/8] 2D disturbance heatmap projections")
    plot_2d_heatmap_projections(data, K, n, m, w, args.save_dir, args.prefix,
                                gp_results=gp_results)

    n_files = len([f for f in os.listdir(args.save_dir)
                   if f.endswith(('.pdf', '.png'))])
    print(f"\nDone. {n_files} figure files in {args.save_dir}/")


if __name__ == "__main__":
    main()
