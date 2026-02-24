"""
GP analysis visualisation utilities.

Provides 3-D acceleration-disturbance scatter+surface plots and
2-D heatmap projections with GP uncertainty overlays.  Each
subplot is saved as an individual PDF for paper inclusion.
"""
from __future__ import annotations

import logging
import os
import numpy as np
import matplotlib.pyplot as plt

_log = logging.getLogger(__name__)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    ConstantKernel,
    ExpSineSquared,
)
from scipy.interpolate import griddata
from scipy.stats import pearsonr

# ─── font-size helpers ───

TICK_LABEL_RATIO = 8.0 / 12.0  # tick : label font-size ratio


def _resolved_label_fs(scale: float = 2.0) -> float:
    """Resolve the base ``axes.labelsize`` and multiply by *scale*."""
    base = plt.rcParams.get("axes.labelsize", plt.rcParams.get("font.size", 10))
    if isinstance(base, str):
        from matplotlib.font_manager import FontProperties
        base = FontProperties(size=base).get_size_in_points()
    return float(base) * float(scale)


# ─── GP fitting ───

def extract_gp_results_for_visualization(
    state_history: np.ndarray,
    control_history: np.ndarray,
    disturbance_history: np.ndarray,
    downsample_factor: int = 10,
    kernel_choice: str = "RBF + White",
) -> dict:
    """
    Fit per-axis GPs on (vx, vy, |v|, ax, ay, t) features and return
    predictions + uncertainties for downstream plotting.
    """
    n_steps = len(state_history)
    indices = np.linspace(0, n_steps - 1, n_steps // downsample_factor, dtype=int)

    states_ds = state_history[indices]
    controls_ds = control_history[indices]
    disturbances_ds = disturbance_history[indices]
    time_steps = indices

    vx = states_ds[:, 1]
    vy = states_ds[:, 3]
    v_mag = np.sqrt(vx**2 + vy**2)
    ax = controls_ds[:, 0]
    ay = controls_ds[:, 1]

    d_x = disturbances_ds[:, 1]
    d_y = disturbances_ds[:, 3]

    if kernel_choice == "RBF + White":
        kernel = RBF(length_scale=1.0) + WhiteKernel(0.1)
    elif kernel_choice == "SinExp + White + Const":
        kernel = (
            ConstantKernel(1.0)
            * ExpSineSquared(1.0, 5.0, periodicity_bounds=(0.1, 10.0))
            + WhiteKernel(0.1)
        )
    elif kernel_choice == "RBF * SinExp + White":
        kernel = (
            RBF(length_scale=1.0)
            * ExpSineSquared(1.0, 5.0, periodicity_bounds=(0.1, 10.0))
            + WhiteKernel(0.1)
        )
    else:
        raise ValueError(f"Unknown kernel_choice: {kernel_choice}")

    X = np.column_stack([vx, vy, v_mag, ax, ay, time_steps])

    _log.info("Fitting GP model with %s kernel...", kernel_choice)

    gp_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
    gp_x.fit(X, d_x)

    gp_y = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
    gp_y.fit(X, d_y)

    y_pred_x, sigma_x = gp_x.predict(X, return_std=True)
    y_pred_y, sigma_y = gp_y.predict(X, return_std=True)

    return {
        "y_pred_x": y_pred_x,
        "y_pred_y": y_pred_y,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "vx": vx,
        "vy": vy,
        "d_x_true": d_x,
        "d_y_true": d_y,
        "gp_x_model": gp_x,
        "gp_y_model": gp_y,
        "kernel_used": kernel_choice,
        "indices": indices,
    }


# ─── 3-D scatter + surface figures ───

def plot_acceleration_disturbance_spaces(
    state_history: np.ndarray,
    control_history: np.ndarray,
    disturbance_history: np.ndarray,
    gp_results: dict,
    downsample_factor: int = 10,
    grid_resolution: int = 30,
    save_dir: str = "figures",
    label_scale: float = 2.0,
    save: bool = True,
):
    """Save four 3-D figures (vx-ux-wx, vy-uy-wy, ux-uy-wx, ux-uy-wy)."""
    os.makedirs(save_dir, exist_ok=True)
    label_fs = _resolved_label_fs(label_scale)
    tick_fs = label_fs * TICK_LABEL_RATIO

    n_steps = len(state_history)
    indices = np.linspace(0, n_steps - 1, n_steps // downsample_factor, dtype=int)

    states_ds = state_history[indices]
    controls_ds = control_history[indices]
    disturbances_ds = disturbance_history[indices]

    vx = states_ds[:, 1]
    vy = states_ds[:, 3]
    ux = controls_ds[:, 0]
    uy = controls_ds[:, 1]
    wx = disturbances_ds[:, 1]
    wy = disturbances_ds[:, 3]

    wx_pred = gp_results["y_pred_x"]
    wy_pred = gp_results["y_pred_y"]
    sigma_x = gp_results["sigma_x"]
    sigma_y = gp_results["sigma_y"]

    def _plot_3d(xv, yv, zv, color_vals, cmap, pred, upper, lower,
                 xlabel, ylabel, zlabel, cbar_label, fname):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(xv, yv, zv, c=color_vals, cmap=cmap, s=20, alpha=0.7)

        xr = np.linspace(xv.min(), xv.max(), grid_resolution)
        yr = np.linspace(yv.min(), yv.max(), grid_resolution)
        XG, YG = np.meshgrid(xr, yr)
        pts = np.column_stack([xv, yv])

        Zp = griddata(pts, pred, (XG, YG), method="cubic")
        Zu = griddata(pts, upper, (XG, YG), method="cubic")
        Zl = griddata(pts, lower, (XG, YG), method="cubic")

        ax.plot_surface(XG, YG, Zp, alpha=0.3, color="red")
        ax.plot_surface(XG, YG, Zu, alpha=0.15, color="orange")
        ax.plot_surface(XG, YG, Zl, alpha=0.15, color="orange")

        ax.set_xlabel(xlabel, fontsize=label_fs)
        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.set_zlabel(zlabel, fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)

        cb = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cb.set_label(cbar_label, fontsize=label_fs)
        cb.ax.tick_params(labelsize=tick_fs)

        if save:
            fig.savefig(os.path.join(save_dir, fname), bbox_inches="tight")
        plt.close(fig)

    # Plot 1: (vx, ux, wx)
    _plot_3d(vx, ux, wx, sigma_x, "viridis",
             wx_pred, wx_pred + 2 * sigma_x, wx_pred - 2 * sigma_x,
             r"$v_x$ (X Velocity)", r"$u_x$ (X Control/Accel)",
             r"$w_x$ (X Disturbance)", r"Uncertainty $\sigma_x$",
             "vx_ux_wx.pdf")

    # Plot 2: (vy, uy, wy)
    _plot_3d(vy, uy, wy, sigma_y, "plasma",
             wy_pred, wy_pred + 2 * sigma_y, wy_pred - 2 * sigma_y,
             r"$v_y$ (Y Velocity)", r"$u_y$ (Y Control/Accel)",
             r"$w_y$ (Y Disturbance)", r"Uncertainty $\sigma_y$",
             "vy_uy_wy.pdf")

    # Plot 3: (ux, uy, wx)
    _plot_3d(ux, uy, wx, sigma_x, "inferno",
             wx_pred, wx_pred + 2 * sigma_x, wx_pred - 2 * sigma_x,
             r"$u_x$ (X Control/Accel)", r"$u_y$ (Y Control/Accel)",
             r"$w_x$ (X Disturbance)", r"Uncertainty $\sigma_x$",
             "ux_uy_wx.pdf")

    # Plot 4: (ux, uy, wy)
    _plot_3d(ux, uy, wy, sigma_y, "coolwarm",
             wy_pred, wy_pred + 2 * sigma_y, wy_pred - 2 * sigma_y,
             r"$u_x$ (X Control/Accel)", r"$u_y$ (Y Control/Accel)",
             r"$w_y$ (Y Disturbance)", r"Uncertainty $\sigma_y$",
             "ux_uy_wy.pdf")

    # Data coverage log
    _log.info("Data Coverage Analysis:")
    _log.info("  vx in [%.3f, %.3f], vy in [%.3f, %.3f]", vx.min(), vx.max(), vy.min(), vy.max())
    _log.info("  ux in [%.3f, %.3f], uy in [%.3f, %.3f]", ux.min(), ux.max(), uy.min(), uy.max())
    _log.info("  wx in [%.3f, %.3f], wy in [%.3f, %.3f]", wx.min(), wx.max(), wy.min(), wy.max())

    corr_vx_wx, _ = pearsonr(vx, wx)
    corr_ux_wx, _ = pearsonr(ux, wx)
    corr_vy_wy, _ = pearsonr(vy, wy)
    corr_uy_wy, _ = pearsonr(uy, wy)
    _log.info("Correlation Analysis:")
    _log.info("  vx-wx: %.3f,  ux-wx: %.3f", corr_vx_wx, corr_ux_wx)
    _log.info("  vy-wy: %.3f,  uy-wy: %.3f", corr_vy_wy, corr_uy_wy)


# ─── 2-D heatmap projections ───

def plot_2d_projections_acceleration_space(
    state_history: np.ndarray,
    control_history: np.ndarray,
    disturbance_history: np.ndarray,
    gp_results: dict,
    downsample_factor: int = 10,
    grid_resolution: int = 50,
    save_dir: str = "figures",
    label_scale: float = 2.0,
    save: bool = True,
):
    """Save six 2-D heatmap figures as individual PDFs."""
    os.makedirs(save_dir, exist_ok=True)
    label_fs = _resolved_label_fs(label_scale)
    tick_fs = label_fs * TICK_LABEL_RATIO

    n_steps = len(state_history)
    indices = np.linspace(0, n_steps - 1, n_steps // downsample_factor, dtype=int)

    states_ds = state_history[indices]
    controls_ds = control_history[indices]
    disturbances_ds = disturbance_history[indices]

    vx = states_ds[:, 1]
    vy = states_ds[:, 3]
    ux = controls_ds[:, 0]
    uy = controls_ds[:, 1]
    wx = disturbances_ds[:, 1]
    wy = disturbances_ds[:, 3]

    sigma_x = gp_results["sigma_x"]
    sigma_y = gp_results["sigma_y"]

    # helper grids
    vx_range = np.linspace(vx.min(), vx.max(), grid_resolution)
    ux_range = np.linspace(ux.min(), ux.max(), grid_resolution)
    vy_range = np.linspace(vy.min(), vy.max(), grid_resolution)
    uy_range = np.linspace(uy.min(), uy.max(), grid_resolution)

    VX_grid, UX_grid = np.meshgrid(vx_range, ux_range)
    VY_grid, UY_grid = np.meshgrid(vy_range, uy_range)
    UX_ctrl, UY_ctrl = np.meshgrid(ux_range, uy_range)

    pts_vxux = np.column_stack([vx, ux])
    pts_vyuy = np.column_stack([vy, uy])
    pts_ctrl = np.column_stack([ux, uy])

    def _heatmap(xg, yg, pts, vals, scatter_x, scatter_y, scatter_c,
                 cmap, cbar_lbl, xlabel, ylabel, fname):
        fig, ax = plt.subplots(figsize=(9, 7))
        grid_vals = griddata(pts, vals, (xg, yg), method="cubic")
        im = ax.contourf(xg, yg, grid_vals, levels=20, cmap=cmap)
        ax.scatter(scatter_x, scatter_y, c=scatter_c, s=10, cmap=cmap,
                   edgecolors="white", linewidth=0.3)
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(cbar_lbl, fontsize=label_fs)
        cb.ax.tick_params(labelsize=tick_fs)
        ax.set_xlabel(xlabel, fontsize=label_fs)
        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.grid(True, alpha=0.3)
        if save:
            fig.savefig(os.path.join(save_dir, fname), bbox_inches="tight")
        plt.close(fig)

    _heatmap(VX_grid, UX_grid, pts_vxux, wx, vx, ux, wx,
             "RdBu_r", r"$w_x$", r"$v_x$", r"$u_x$", "vx_ux__wx.pdf")

    _heatmap(VX_grid, UX_grid, pts_vxux, sigma_x, vx, ux, sigma_x,
             "viridis", r"$\sigma_x$", r"$v_x$", r"$u_x$", "vx_ux__sigmax.pdf")

    _heatmap(UX_ctrl, UY_ctrl, pts_ctrl, wx, ux, uy, wx,
             "RdBu_r", r"$w_x$", r"$u_x$", r"$u_y$", "ux_uy__wx.pdf")

    _heatmap(VY_grid, UY_grid, pts_vyuy, wy, vy, uy, wy,
             "RdBu_r", r"$w_y$", r"$v_y$", r"$u_y$", "vy_uy__wy.pdf")

    _heatmap(VY_grid, UY_grid, pts_vyuy, sigma_y, vy, uy, sigma_y,
             "plasma", r"$\sigma_y$", r"$v_y$", r"$u_y$", "vy_uy__sigmay.pdf")

    _heatmap(UX_ctrl, UY_ctrl, pts_ctrl, wy, ux, uy, wy,
             "RdBu_r", r"$w_y$", r"$u_x$", r"$u_y$", "ux_uy__wy.pdf")
