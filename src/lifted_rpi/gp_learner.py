"""
Gaussian-process-based learned disturbance graph set.

``LearnedGraphSetGP`` wraps two per-axis sklearn GP regressors (one for
w_x, one for w_y) and provides a ``bounds_fn(X, U)`` compatible with the
engine's ``clip_with_graph``.  It produces interval bounds
    [mu - k*sigma, mu + k*sigma]
for each disturbance component, with safeguards:
    - Points far from training data fall back to a conservative prior box.
    - A minimum half-width prevents degenerate zero-width intervals.
    - Bounds are always clamped to the prior box limits.

``train_disturbance_gps`` fits the two GPs from MPC simulation data and
returns them together with a KDTree for coverage-distance queries.

"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    ConstantKernel,
    ExpSineSquared,
)
from sklearn.neighbors import KDTree
import joblib


# ───────────────────── learned GraphSet ─────────────────────

class LearnedGraphSetGP:
    """GP-based learned disturbance graph set.

    Provides ``bounds_fn(X, U) -> (lbW, ubW)`` with shape (N, 2) that
    returns per-sample interval bounds [lb, ub] for the two disturbance
    components w_x, w_y.

    The bounds are computed as mu +/- k_sigma * sigma from each GP.
    Three safeguards are applied in sequence:

    1. If the half-width is below ``min_halfwidth``, expand to that minimum.
    2. If the query point is farther than ``far_radius`` from any training
       point (measured in feature space via KDTree), replace with the
       conservative ``prior_box`` bounds.
    3. Clamp the final bounds to ``prior_box`` limits.

    Parameters
    ----------
    gp_x, gp_y : GaussianProcessRegressor
        Fitted sklearn GP models for w_x and w_y respectively.
    kdtree : KDTree
        Nearest-neighbour tree over the GP training features, used to
        detect out-of-distribution queries.
    dist_rows : tuple of int
        State-vector row indices where disturbance is injected (default: velocity rows).
    prior_box : tuple of tuple
        Conservative ((lb_x, lb_y), (ub_x, ub_y)) used as fallback.
    k_sigma : float
        Number of GP standard deviations for the interval width.
    min_halfwidth : float
        Floor on the half-width to prevent degenerate intervals.
    far_radius : float
        KDTree distance threshold beyond which the prior box is used.
    tol : float
        Tolerance passed to the engine for graph-set clipping.
    """

    def __init__(
        self,
        gp_x,
        gp_y,
        kdtree,
        *,
        dist_rows: Tuple[int, ...] = (1, 3),
        prior_box: Tuple[Tuple[float, ...], Tuple[float, ...]] = ((-3.0, -3.0), (3.0, 3.0)),
        k_sigma: float = 3.5,
        min_halfwidth: float = 1e-3,
        far_radius: float = 0.75,
        tol: float = 1e-2,
        name: str = "G_learned",
    ):
        self.gp_x = gp_x
        self.gp_y = gp_y
        self.kdt = kdtree
        self.k = float(k_sigma)
        self.lb_prior = np.array(prior_box[0], dtype=float)
        self.ub_prior = np.array(prior_box[1], dtype=float)
        self.min_hw = float(min_halfwidth)
        self.far_r = float(far_radius)
        self.tol = float(tol)
        self.name = name
        self.dist_rows = dist_rows

    # ── feature construction ──

    @staticmethod
    def _features(X, U, t_index=None):
        """Build the GP feature vector from state and input samples.

        Features: [vx, vy, |v|, ux, uy, t_index].  The speed magnitude
        |v| is included because the drag model depends on it.
        """
        vx, vy = X[:, 1], X[:, 3]
        vmag = np.sqrt(vx**2 + vy**2)
        ax, ay = U[:, 0], U[:, 1]
        if t_index is None:
            t_index = np.arange(len(vx), dtype=float)
        return np.column_stack([vx, vy, vmag, ax, ay, t_index])

    # -- core bounds_fn --

    def bounds_fn(self, X, U, t_index=None):
        """Compute interval disturbance bounds at query points.

        Parameters
        ----------
        X : ndarray (N, n)
            State samples.
        U : ndarray (N, m)
            Input samples.
        t_index : ndarray (N,), optional
            Time indices (defaults to arange).

        Returns
        -------
        lb, ub : ndarray (N, 2)
            Lower and upper disturbance bounds per sample.
        """
        F = self._features(X, U, t_index=t_index)
        d, _ = self.kdt.query(F, k=1, return_distance=True)
        d = d.ravel()

        mu_x, std_x = self.gp_x.predict(F, return_std=True)
        mu_y, std_y = self.gp_y.predict(F, return_std=True)

        lb = np.stack([mu_x - self.k * std_x, mu_y - self.k * std_y], axis=1)
        ub = np.stack([mu_x + self.k * std_x, mu_y + self.k * std_y], axis=1)

        # enforce min half-width
        hw = 0.5 * (ub - lb)
        too_narrow = hw < self.min_hw
        if np.any(too_narrow):
            mid = 0.5 * (ub + lb)
            hw[too_narrow] = self.min_hw
            lb = mid - hw
            ub = mid + hw

        # fall back to prior in uncovered (far) regions
        far = d > self.far_r
        if np.any(far):
            lb[far] = self.lb_prior
            ub[far] = self.ub_prior

        # always cap to the prior box
        lb = np.maximum(lb, self.lb_prior)
        ub = np.minimum(ub, self.ub_prior)
        return lb, ub

    # ── serialisation ──

    def save(self, path):
        joblib.dump(
            {
                "gp_x": self.gp_x,
                "gp_y": self.gp_y,
                "kdt": self.kdt,
                "k": self.k,
                "lb_prior": self.lb_prior,
                "ub_prior": self.ub_prior,
                "min_hw": self.min_hw,
                "far_r": self.far_r,
                "tol": self.tol,
                "dist_rows": self.dist_rows,
                "name": self.name,
            },
            path,
        )

    @staticmethod
    def load(path) -> "LearnedGraphSetGP":
        d = joblib.load(path)
        obj = LearnedGraphSetGP(
            d["gp_x"],
            d["gp_y"],
            d["kdt"],
            prior_box=(d["lb_prior"], d["ub_prior"]),
            k_sigma=d["k"],
            min_halfwidth=d["min_hw"],
            far_radius=d["far_r"],
            tol=d.get("tol", 1e-2),
            name=d["name"],
        )
        obj.dist_rows = tuple(d["dist_rows"])
        return obj


# ─────────────── feature extraction from history ───────────────

def _features_from_hist(state_history, control_history, time_idx=None):
    """Build GP feature matrix from trajectory history."""
    vx = state_history[:, 1]
    vy = state_history[:, 3]
    vmag = np.sqrt(vx**2 + vy**2)
    ax = control_history[:, 0]
    ay = control_history[:, 1]
    if time_idx is None:
        time_idx = np.arange(len(vx))
    return np.column_stack([vx, vy, vmag, ax, ay, time_idx.astype(float)])


# ─────────────── GP training ───────────────

def train_disturbance_gps(
    state_history: np.ndarray,
    control_history: np.ndarray,
    disturbance_history: np.ndarray,
    *,
    downsample: int = 10,
    kernel_choice: str = "RBF * SinExp + White",
    gp_restarts: int = 2,
    alpha: float = 1e-6,
    random_state: int = 0,
) -> Tuple[GaussianProcessRegressor, GaussianProcessRegressor, KDTree, Dict]:
    """Train two independent GPs for w_x, w_y from MPC simulation data.

    The disturbance is injected into the velocity rows of the state
    vector, so w_x = disturbance_history[:, 1] and
    w_y = disturbance_history[:, 3].

    Parameters
    ----------
    state_history : ndarray (T, n)
        Recorded state trajectory.
    control_history : ndarray (T, m)
        Recorded control inputs.
    disturbance_history : ndarray (T, n)
        Recorded disturbance vectors (only velocity rows are used).
    downsample : int
        Take every ``downsample``-th sample for training. Use 1 for the
        full dataset (slower but more accurate).
    kernel_choice : str
        GP kernel specification. Supported: 'RBF + White',
        'SinExp + White + Const', 'RBF * SinExp + White'.
    gp_restarts : int
        Number of random restarts for the kernel hyperparameter optimiser.
    alpha : float
        Tikhonov regularisation added to the GP diagonal.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    gp_x, gp_y : GaussianProcessRegressor
        Fitted GP models.
    kdt : KDTree
        Nearest-neighbour tree over the training features.
    meta : dict
        Training metadata (kernel string, downsample factor, feature names).
    """
    n = len(state_history)
    idx = np.linspace(0, n - 1, max(1, n // downsample), dtype=int)

    states_ds = state_history[idx]
    controls_ds = control_history[idx]
    d_ds = disturbance_history[idx]
    t_ds = idx

    X = _features_from_hist(states_ds, controls_ds, time_idx=t_ds)
    yx = d_ds[:, 1]  # w_x component (disturbs v_x row)
    yy = d_ds[:, 3]  # w_y component (disturbs v_y row)

    if kernel_choice == "RBF + White":
        kernel = RBF(1.0) + WhiteKernel(0.1)
    elif kernel_choice == "SinExp + White + Const":
        kernel = (
            ConstantKernel(1.0)
            * ExpSineSquared(1.0, 5.0, periodicity_bounds=(0.1, 10.0))
            + WhiteKernel(0.1)
        )
    else:  # "RBF * SinExp + White"
        kernel = (
            RBF(1.0)
            * ExpSineSquared(1.0, 5.0, periodicity_bounds=(0.1, 10.0))
            + WhiteKernel(0.1)
        )

    gp_x = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=gp_restarts,
        alpha=alpha,
        random_state=random_state,
    )
    gp_y = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=gp_restarts,
        alpha=alpha,
        random_state=random_state,
    )

    gp_x.fit(X, yx)
    gp_y.fit(X, yy)

    kdt = KDTree(X, leaf_size=64)

    meta = dict(
        kernel_used=str(kernel),
        downsample=downsample,
        feat_names=["vx", "vy", "|v|", "ux", "uy", "t_index"],
    )
    return gp_x, gp_y, kdt, meta
