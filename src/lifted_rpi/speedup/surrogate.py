"""
Surrogate-accelerated bounds functions for graph-set clipping.

This module provides ``SurrogateGraphSet``, a drop-in replacement for
``LearnedGraphSetGP`` (or any ``GraphSet``) that uses a fast surrogate
model instead of calling ``sklearn.gaussian_process.predict``.

Two surrogate backends are implemented:

Nystroem (default)
    Uses ``sklearn.kernel_approximation.Nystroem`` to approximate the
    RBF kernel component of the GP, then fits ``Ridge`` regressors for
    the four target surfaces (mu_x, mu_y, std_x, std_y).  The RBF basis
    decays to zero outside the training domain, making extrapolation
    safe (predictions revert to zero, and the safeguard chain clamps
    them to the prior box).

Poly-3
    Fits degree-3 polynomial features (with ``StandardScaler`` pre-
    processing) through ``Ridge`` regression.  Significantly faster
    inference but the polynomial can diverge on far-OOD queries.
    Suitable when the query domain is guaranteed to stay within or
    near the GP training domain.

Both backends are built from the GP training data and GP predictions
(for std targets).  Build cost is negligible (< 1 s for 2 500 training
points) and amortised over all 57+ clipping iterations.

"""
from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal
from time import perf_counter

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# ────────────────────── helper: batched GP predict ──────────────────────

def _batched_gp_predict(gp, F, return_std=True, batch_size=5000):
    """Predict in batches to avoid OOM on large kernel matrices."""
    N = F.shape[0]
    mu = np.empty(N)
    std = np.empty(N) if return_std else None
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        if return_std:
            mu[i:j], std[i:j] = gp.predict(F[i:j], return_std=True)
        else:
            mu[i:j] = gp.predict(F[i:j], return_std=False)
    return (mu, std) if return_std else mu


# ────────────────────── surrogate builders ──────────────────────

def build_nystroem_surrogate(
    gp_x,
    gp_y,
    *,
    n_components: int = 200,
    gamma: Optional[float] = None,
    alpha: float = 1e-4,
    random_state: int = 0,
    verbose: bool = False,
) -> dict:
    """Build a Nystroem + Ridge surrogate from fitted GP models.

    Parameters
    ----------
    gp_x, gp_y : GaussianProcessRegressor
        Fitted sklearn GP models for w_x and w_y.
    n_components : int
        Number of Nystroem components (rank of the kernel approximation).
    gamma : float or None
        RBF gamma for Nystroem.  If None, extracted from gp_x's kernel.
    alpha : float
        Ridge regularisation parameter.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        Print build diagnostics.

    Returns
    -------
    dict
        Keys: nys, ridge_mu_x, ridge_mu_y, ridge_std_x, ridge_std_y,
        build_time, backend.
    """
    t0 = perf_counter()
    X_tr = gp_x.X_train_

    # Extract gamma from the GP's RBF component if not provided
    if gamma is None:
        from sklearn.gaussian_process.kernels import RBF
        kernel = gp_x.kernel_
        gamma = _extract_rbf_gamma(kernel)
        if gamma is None:
            # Fallback: use median heuristic
            from scipy.spatial.distance import pdist
            gamma = 1.0 / (2.0 * np.median(pdist(X_tr, "sqeuclidean")))

    # Get GP std predictions on training data (targets for std surrogates)
    _, std_x_tr = _batched_gp_predict(gp_x, X_tr, return_std=True)
    _, std_y_tr = _batched_gp_predict(gp_y, X_tr, return_std=True)

    # Nystroem feature map
    nys = Nystroem(
        kernel="rbf", gamma=gamma,
        n_components=min(n_components, len(X_tr)),
        random_state=random_state,
    )
    nys.fit(X_tr)
    Phi = nys.transform(X_tr)

    # Ridge fits
    ridge_mu_x = Ridge(alpha=alpha).fit(Phi, gp_x.y_train_)
    ridge_mu_y = Ridge(alpha=alpha).fit(Phi, gp_y.y_train_)
    ridge_std_x = Ridge(alpha=alpha).fit(Phi, std_x_tr)
    ridge_std_y = Ridge(alpha=alpha).fit(Phi, std_y_tr)

    build_time = perf_counter() - t0

    if verbose:
        res_mu_x = np.sqrt(np.mean((gp_x.y_train_ - ridge_mu_x.predict(Phi)) ** 2))
        res_mu_y = np.sqrt(np.mean((gp_y.y_train_ - ridge_mu_y.predict(Phi)) ** 2))
        res_std_x = np.sqrt(np.mean((std_x_tr - ridge_std_x.predict(Phi)) ** 2))
        res_std_y = np.sqrt(np.mean((std_y_tr - ridge_std_y.predict(Phi)) ** 2))
        print(f"  [Nystroem-{n_components}] built in {build_time:.3f}s  "
              f"RMSE mu=({res_mu_x:.6f},{res_mu_y:.6f})  "
              f"std=({res_std_x:.6f},{res_std_y:.6f})")

    return dict(
        nys=nys,
        ridge_mu_x=ridge_mu_x,
        ridge_mu_y=ridge_mu_y,
        ridge_std_x=ridge_std_x,
        ridge_std_y=ridge_std_y,
        build_time=build_time,
        backend="nystroem",
    )


def build_poly_surrogate(
    gp_x,
    gp_y,
    *,
    degree: int = 3,
    alpha: float = 1e-2,
    verbose: bool = False,
) -> dict:
    """Build a Polynomial + Ridge surrogate from fitted GP models.

    Parameters
    ----------
    gp_x, gp_y : GaussianProcessRegressor
        Fitted sklearn GP models for w_x and w_y.
    degree : int
        Polynomial degree (3 recommended; 4 increases features significantly).
    alpha : float
        Ridge regularisation.  Use >= 1e-2 to prevent ill-conditioning.
    verbose : bool
        Print build diagnostics.

    Returns
    -------
    dict
        Keys: scaler, poly, ridge_mu_x, ridge_mu_y, ridge_std_x,
        ridge_std_y, build_time, backend.
    """
    t0 = perf_counter()
    X_tr = gp_x.X_train_

    _, std_x_tr = _batched_gp_predict(gp_x, X_tr, return_std=True)
    _, std_y_tr = _batched_gp_predict(gp_y, X_tr, return_std=True)

    scaler = StandardScaler().fit(X_tr)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Phi = poly.fit_transform(scaler.transform(X_tr))

    ridge_mu_x = Ridge(alpha=alpha).fit(Phi, gp_x.y_train_)
    ridge_mu_y = Ridge(alpha=alpha).fit(Phi, gp_y.y_train_)
    ridge_std_x = Ridge(alpha=alpha).fit(Phi, std_x_tr)
    ridge_std_y = Ridge(alpha=alpha).fit(Phi, std_y_tr)

    build_time = perf_counter() - t0

    if verbose:
        res_mu_x = np.sqrt(np.mean((gp_x.y_train_ - ridge_mu_x.predict(Phi)) ** 2))
        res_mu_y = np.sqrt(np.mean((gp_y.y_train_ - ridge_mu_y.predict(Phi)) ** 2))
        print(f"  [Poly-{degree}] built in {build_time:.3f}s  "
              f"({Phi.shape[1]} features)  "
              f"RMSE mu=({res_mu_x:.6f},{res_mu_y:.6f})")

    return dict(
        scaler=scaler,
        poly=poly,
        ridge_mu_x=ridge_mu_x,
        ridge_mu_y=ridge_mu_y,
        ridge_std_x=ridge_std_x,
        ridge_std_y=ridge_std_y,
        build_time=build_time,
        backend="poly",
    )


def _extract_rbf_gamma(kernel):
    """Walk kernel tree to find the RBF length-scale and return gamma."""
    from sklearn.gaussian_process.kernels import RBF, Sum, Product
    if isinstance(kernel, RBF):
        ls = np.atleast_1d(kernel.length_scale)
        return float(1.0 / (2.0 * ls[0] ** 2))
    if isinstance(kernel, (Sum, Product)):
        left = _extract_rbf_gamma(kernel.k1)
        if left is not None:
            return left
        return _extract_rbf_gamma(kernel.k2)
    return None


# ────────────────────── SurrogateGraphSet ──────────────────────

class SurrogateGraphSet:
    """Drop-in replacement for ``LearnedGraphSetGP`` using fast surrogates.

    Conforms to the ``GraphSet`` protocol: has ``bounds_fn``, ``name``,
    and ``tol`` attributes.  Can be passed directly to
    ``LiftedSetOpsGPU_NoHull.clip_with_graph``.

    The surrogate is built once (sub-second) and then used for all
    iterations.  After the GP is retrained in an online setting, call
    ``rebuild()`` to update the surrogate from the new GP.

    Parameters
    ----------
    learned_gp : LearnedGraphSetGP
        The original GP-based graph set (source of training data and
        safeguard parameters).
    backend : {"nystroem", "poly"}
        Which surrogate to use.  Default "nystroem".
    verbose : bool
        Print build diagnostics.
    **surrogate_kw
        Passed to ``build_nystroem_surrogate`` or ``build_poly_surrogate``.
    """

    def __init__(
        self,
        learned_gp,
        *,
        backend: Literal["nystroem", "poly"] = "nystroem",
        verbose: bool = True,
        **surrogate_kw,
    ):
        self._gp_source = learned_gp
        self._backend = backend
        self._surr_kw = surrogate_kw
        self._verbose = verbose

        # Copy safeguard parameters from the GP source
        self.k = learned_gp.k
        self.lb_prior = learned_gp.lb_prior.copy()
        self.ub_prior = learned_gp.ub_prior.copy()
        self.min_hw = learned_gp.min_hw
        self.far_r = learned_gp.far_r
        self.tol = learned_gp.tol
        self.name = f"{learned_gp.name}_surrogate_{backend}"
        self.dist_rows = learned_gp.dist_rows
        self.kdt = learned_gp.kdt

        # Build the surrogate
        self._surr = None
        self.build_time = 0.0
        self.rebuild()

    def rebuild(self):
        """(Re)build the surrogate from the current GP source."""
        gp_x = self._gp_source.gp_x
        gp_y = self._gp_source.gp_y

        if self._backend == "nystroem":
            self._surr = build_nystroem_surrogate(
                gp_x, gp_y, verbose=self._verbose, **self._surr_kw
            )
        elif self._backend == "poly":
            self._surr = build_poly_surrogate(
                gp_x, gp_y, verbose=self._verbose, **self._surr_kw
            )
        else:
            raise ValueError(f"Unknown backend: {self._backend!r}")
        self.build_time = self._surr["build_time"]

    # ── feature construction (same as LearnedGraphSetGP) ──

    @staticmethod
    def _features(X, U, t_index=None):
        vx, vy = X[:, 1], X[:, 3]
        vmag = np.sqrt(vx ** 2 + vy ** 2)
        ax, ay = U[:, 0], U[:, 1]
        if t_index is None:
            t_index = np.arange(len(vx), dtype=float)
        return np.column_stack([vx, vy, vmag, ax, ay, t_index])

    # ── core bounds_fn ──

    def bounds_fn(self, X, U, t_index=None):
        """Compute interval disturbance bounds using the surrogate.

        Applies the same three-layer safeguard chain as
        ``LearnedGraphSetGP.bounds_fn``:
          1. Minimum half-width floor.
          2. Far-fallback to prior box for OOD queries.
          3. Clamp to prior box.
        """
        F = self._features(X, U, t_index=t_index)

        # KDTree distance check (fast: O(N log N_train))
        d, _ = self.kdt.query(F, k=1, return_distance=True)
        d = d.ravel()

        # Surrogate prediction
        if self._surr["backend"] == "nystroem":
            Phi = self._surr["nys"].transform(F)
            mu_x = self._surr["ridge_mu_x"].predict(Phi)
            mu_y = self._surr["ridge_mu_y"].predict(Phi)
            std_x = np.maximum(self._surr["ridge_std_x"].predict(Phi), 0.0)
            std_y = np.maximum(self._surr["ridge_std_y"].predict(Phi), 0.0)
        else:  # poly
            F_sc = self._surr["scaler"].transform(F)
            Phi = self._surr["poly"].transform(F_sc)
            mu_x = self._surr["ridge_mu_x"].predict(Phi)
            mu_y = self._surr["ridge_mu_y"].predict(Phi)
            std_x = np.maximum(self._surr["ridge_std_x"].predict(Phi), 0.0)
            std_y = np.maximum(self._surr["ridge_std_y"].predict(Phi), 0.0)

        lb = np.stack([mu_x - self.k * std_x, mu_y - self.k * std_y], axis=1)
        ub = np.stack([mu_x + self.k * std_x, mu_y + self.k * std_y], axis=1)

        # Safeguard 1: minimum half-width
        hw = 0.5 * (ub - lb)
        too_narrow = hw < self.min_hw
        if np.any(too_narrow):
            mid = 0.5 * (ub + lb)
            hw[too_narrow] = self.min_hw
            lb = mid - hw
            ub = mid + hw

        # Safeguard 2: far-fallback
        far = d > self.far_r
        if np.any(far):
            lb[far] = self.lb_prior
            ub[far] = self.ub_prior

        # Safeguard 3: clamp to prior box
        lb = np.maximum(lb, self.lb_prior)
        ub = np.minimum(ub, self.ub_prior)
        return lb, ub
