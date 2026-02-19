"""
Fixed-point iterator with convergence diagnostics.

Provides ``fixed_point_reach`` which implements the outside-in recursion
    Z_{k+1} = F(Z_k) = (Ã Z_k ⊕ B̃ ΔV ⊕ D̃ W) ∩ G
and monitors convergence via one of four selectable metrics:
    - Hausdorff distance between successive iterates (point-cloud KD-tree)
    - Support-function gap (random projection directions, full or state-only)
    - AABB volume change (axis-aligned bounding box)

Also includes per-iteration tracking of constraint violation rate,
closed-loop cost, facet counts, and a detailed timing breakdown.

"""
from __future__ import annotations

import math
import numpy as np
from time import perf_counter
from typing import Optional, Dict, Tuple, List
from scipy.spatial import cKDTree, ConvexHull

from .vset import VSet
from .convergence import (
    aabb_volume,
    support_gap_with_dirs,
    make_dirs,
    hausdorff_pointcloud,
    count_facets,
)


# ══════════════ violation / cost helpers ══════════════

def compute_violation_rate(
    Z: VSet,
    G,
    eng,
    n_samples: int = 10000,
    seed: int = 0,
) -> Tuple[float, int, int]:
    """
    Estimate violation rate by sampling points from Z and checking G.

    Returns (violation_rate, n_violations, n_total).
    """
    if Z.nverts == 0:
        return 0.0, 0, 0

    rng = np.random.default_rng(seed)

    n_samples = min(n_samples, Z.nverts)
    if Z.nverts <= n_samples:
        samples = Z.V
    else:
        indices = rng.choice(Z.nverts, size=n_samples, replace=False)
        samples = Z.V[indices]

    n_total = samples.shape[0]

    if hasattr(G, "A") and hasattr(G, "b"):
        violations = np.any(G.A @ samples.T > G.b[:, None] + 1e-6, axis=0)
        n_violations = int(np.sum(violations))
    elif callable(getattr(G, "bounds_fn", None)):
        X = samples[:, : eng.n]
        V = samples[:, eng.n : eng.n + eng.m]
        W = samples[:, eng.n + eng.m :]
        U = (eng.K @ X.T).T + V

        lbW, ubW = G.bounds_fn(X, U)
        violations = np.any((W < lbW - 1e-6) | (W > ubW + 1e-6), axis=1)
        n_violations = int(np.sum(violations))
    else:
        raise TypeError(f"Unknown constraint type: {type(G)}")

    violation_rate = n_violations / n_total if n_total > 0 else 0.0
    return violation_rate, n_violations, n_total


def compute_closed_loop_cost(
    Z: VSet,
    eng,
    Q: np.ndarray,
    R: np.ndarray,
    cost_type: str = "expected",
    n_samples: Optional[int] = None,
) -> float:
    """
    Compute closed-loop cost for reachable set Z.

    cost_type: "expected" | "max" | "trace"
    """
    if Z.nverts == 0:
        return float("inf")

    X = Z.V[:, : eng.n]
    V = Z.V[:, eng.n : eng.n + eng.m]
    U = (eng.K @ X.T).T + V

    if n_samples is not None and Z.nverts > n_samples:
        rng = np.random.default_rng(0)
        indices = rng.choice(Z.nverts, size=n_samples, replace=False)
        X = X[indices]
        U = U[indices]

    state_costs = np.sum(X @ Q * X, axis=1)
    input_costs = np.sum(U @ R * U, axis=1)
    total_costs = state_costs + input_costs

    if cost_type == "expected":
        return float(np.mean(total_costs))
    elif cost_type == "max":
        return float(np.max(total_costs))
    elif cost_type == "trace":
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        cov = (X_centered.T @ X_centered) / X.shape[0]
        return float(np.trace(Q @ cov))
    else:
        raise ValueError(f"Unknown cost_type: {cost_type}")


def compute_expected_stage_cost(
    X: np.ndarray, U: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> float:
    """Compute E[x^T Q x + u^T R u] over samples."""
    state_costs = np.sum(X @ Q * X, axis=1)
    input_costs = np.sum(U @ R * U, axis=1)
    return float(np.mean(state_costs + input_costs))


def simulate_trajectory_cost(
    x0: np.ndarray,
    eng,
    Q: np.ndarray,
    R: np.ndarray,
    horizon: int = 50,
    noise_samples: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Simulate a single trajectory and compute cumulative cost."""
    n, m = eng.n, eng.m
    X_traj = np.zeros((horizon + 1, n))
    U_traj = np.zeros((horizon, m))
    X_traj[0] = x0

    if noise_samples is None:
        rng = np.random.default_rng(seed)
        noise_samples = rng.normal(0, 0.1, size=(horizon, eng.w))

    cumulative_cost = 0.0
    for k in range(horizon):
        x = X_traj[k]
        u = eng.K @ x
        stage_cost = x @ Q @ x + u @ R @ u
        cumulative_cost += stage_cost

        w = noise_samples[k]
        x_next = eng.A @ x + eng.B @ u + w[:n]
        X_traj[k + 1] = x_next
        U_traj[k] = u

    return cumulative_cost, X_traj, U_traj


# ══════════════ ENHANCED FIXED-POINT ITERATOR ══════════════

def fixed_point_reach(
    eng,
    Z0: VSet,
    DV: VSet,
    W: VSet,
    G,
    # Cost matrices (optional)
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    # iteration & compression
    max_iters: int = 300,
    compress_every: int = 1,
    max_points: Optional[int] = None,
    dedupe_each_iter: bool = True,
    # convergence control
    convergence_metric: str = "support_state",
    metric_tol: float = 1e-3,
    metric_patience: int = 5,
    # support-gap options
    support_ndirs_all: Optional[int] = None,
    support_ndirs_state: Optional[int] = None,
    support_seed: int = 0,
    cache_dirs: bool = True,
    # hausdorff options
    hausdorff_dims: str = "state",
    hausdorff_normalize: bool = True,
    hausdorff_scale: Optional[float] = None,
    # volume options
    volume_on: str = "state_aabb",
    volume_rel_tol: Optional[float] = None,
    volume_patience: Optional[int] = None,
    # robustness / UX
    stop_if_empty: bool = True,
    verbose: bool = True,
    track_facets: bool = True,
    detail_timing: bool = True,
    # violation and cost tracking
    track_violations: bool = True,
    violation_samples: int = 5000,
    track_cost: bool = True,
    cost_type: str = "expected",
    cost_samples: Optional[int] = None,
) -> Tuple[List[VSet], Dict]:
    """
    Iterate  Z_{k+1} = (Ã Z_k ⊕ B̃ ΔV ⊕ D̃ W) ∩ G  until convergence.

    Parameters
    ----------
    eng : LiftedSetOpsGPU_NoHull
        The lifted set-operations engine (holds system matrices).
    Z0 : VSet
        Initial set (vertex cloud in the augmented space).
    DV, W : VSet
        Input perturbation set and disturbance outer-approximation.
    G : Polytope or GraphSet
        Constraint set for the intersection step.
    Q, R : ndarray, optional
        State and input cost matrices for optional cost tracking.
    max_iters : int
        Upper bound on the number of fixed-point iterations.
    convergence_metric : str
        One of 'hausdorff', 'volume', 'support_all', 'support_state'.
    metric_tol : float
        Threshold below which the metric is considered converged.
    metric_patience : int
        Number of consecutive below-threshold readings before stopping.

    Returns
    -------
    history : list of VSet
        Sequence [Z0, Z1, ..., Z*] of iterates.
    stats : dict
        Convergence metrics, timing, violation rates, etc.
    """

    def _compress_cloud(V: np.ndarray) -> np.ndarray:
        if V.size == 0:
            return V
        if dedupe_each_iter:
            V = np.unique(V, axis=0)
        if (max_points is not None) and (V.shape[0] > max_points):
            step = max(1, V.shape[0] // max_points)
            V = V[::step]
        return V

    def _slice_dims(Valike: np.ndarray, which: str) -> np.ndarray:
        if which == "state":
            return Valike[:, : eng.n]
        elif which == "all":
            return Valike
        raise ValueError(f"Unknown dims={which!r}")

    def _is_graphset(obj):
        return callable(getattr(obj, "bounds_fn", None))

    def _is_polytope(obj):
        return hasattr(obj, "A") and hasattr(obj, "b")

    vol_eps = (
        volume_rel_tol
        if (convergence_metric == "volume" and volume_rel_tol is not None)
        else metric_tol
    )
    vol_pat = (
        volume_patience
        if (convergence_metric == "volume" and volume_patience is not None)
        else metric_patience
    )

    # direction banks
    D_all = D_state = None
    if cache_dirs and convergence_metric in ("support_all", "support_state"):
        if convergence_metric == "support_all":
            q_all = support_ndirs_all or int(
                math.ceil(
                    2
                    * (eng.n + eng.m + eng.w)
                    * math.log(max(4, Z0.nverts) + 1)
                )
            )
            D_all = make_dirs(eng.n + eng.m + eng.w, q=q_all, seed=support_seed)
        if convergence_metric == "support_state":
            q_x = support_ndirs_state or int(
                math.ceil(2 * eng.n * math.log(max(4, Z0.nverts) + 1))
            )
            D_state = make_dirs(eng.n, q=q_x, seed=support_seed)

    # ── iteration state ──
    history: List[VSet] = [Z0]
    times: List[float] = []
    metric_values: List[float] = []
    support_all_hist: List[float] = []
    support_state_hist: List[float] = []
    hausdorff_hist: List[float] = []
    volumes_state: List[float] = []
    volumes_all: List[float] = []
    facets_state: List[int] = []
    facets_all: List[int] = []
    violation_rates: List[float] = []
    costs: List[float] = []
    time_forward: List[float] = []
    time_clip: List[float] = []
    time_compress: List[float] = []
    time_metric: List[float] = []

    Zk = Z0
    patience_count = 0
    termination_reason = None

    # init
    volumes_state.append(aabb_volume(Z0.V[:, : eng.n]))
    volumes_all.append(aabb_volume(Z0.V))
    if track_facets:
        facets_state.append(count_facets(Z0.V[:, : eng.n]))
        facets_all.append(count_facets(Z0.V))
    if track_violations:
        vr, _, _ = compute_violation_rate(
            Z0, G, eng, n_samples=violation_samples, seed=support_seed
        )
        violation_rates.append(vr)
    if track_cost and Q is not None and R is not None:
        c = compute_closed_loop_cost(
            Z0, eng, Q, R, cost_type=cost_type, n_samples=cost_samples
        )
        costs.append(c)

    for k in range(1, max_iters + 1):
        t_iter_start = perf_counter()

        # forward operator
        t0 = perf_counter()
        base = eng.F_no_intersection(Zk, DV, W, name="FZ")
        t_forward_op = perf_counter() - t0
        time_forward.append(t_forward_op)

        # clipping
        t0 = perf_counter()
        if _is_graphset(G):
            Znext = eng.clip_with_graph(base, G, name=f"Z{k}")
        elif _is_polytope(G):
            Znext = eng.clip_with_polytope(base, G, name=f"Z{k}")
        else:
            raise TypeError(
                "G must be a polytope (A,b) or an object with bounds_fn(X,U)."
            )
        t_clip_op = perf_counter() - t0
        time_clip.append(t_clip_op)

        dt = perf_counter() - t_iter_start
        times.append(dt)
        last_base = base

        if verbose:
            timing_str = (
                f"[fwd={t_forward_op:.3f}s, clip={t_clip_op:.3f}s]"
                if detail_timing
                else ""
            )
            print(
                f"[iter {k:03d}] clip kept {Znext.nverts} / {base.nverts} points {timing_str}"
            )

        # compression
        t0 = perf_counter()
        if compress_every > 0 and (k % compress_every == 0):
            Znext = VSet(_compress_cloud(Znext.V), name=Znext.name)
        t_compress_op = perf_counter() - t0
        time_compress.append(t_compress_op)

        if Znext.nverts == 0 and stop_if_empty:
            termination_reason = "empty_after_clip"
            if verbose:
                print(f"[iter {k:03d}] Z became empty after clipping → stopping.")
            history.append(Znext)
            break

        # volumes & facets
        volumes_state.append(aabb_volume(Znext.V[:, : eng.n]))
        volumes_all.append(aabb_volume(Znext.V))
        if track_facets:
            nf_state = count_facets(Znext.V[:, : eng.n])
            nf_all = count_facets(Znext.V)
            facets_state.append(nf_state)
            facets_all.append(nf_all)
            if verbose:
                print(f"[iter {k:03d}] facets: state={nf_state}, all={nf_all}")

        # violations
        if track_violations:
            vr, n_viol, n_tot = compute_violation_rate(
                Znext, G, eng,
                n_samples=violation_samples,
                seed=support_seed + k,
            )
            violation_rates.append(vr)
            if verbose:
                print(f"[iter {k:03d}] violation rate: {vr:.4f} ({n_viol}/{n_tot})")

        # cost
        if track_cost and Q is not None and R is not None:
            c = compute_closed_loop_cost(
                Znext, eng, Q, R, cost_type=cost_type, n_samples=cost_samples
            )
            costs.append(c)
            if verbose:
                print(f"[iter {k:03d}] {cost_type} cost: {c:.4f}")

        # convergence metric
        t0 = perf_counter()
        if convergence_metric == "volume":
            prev = (
                volumes_state[-2] if volume_on == "state_aabb" else volumes_all[-2]
            )
            curr = (
                volumes_state[-1] if volume_on == "state_aabb" else volumes_all[-1]
            )
            denom = max(prev, 1e-12)
            mval = abs(curr - prev) / denom
            metric_values.append(mval)
            if verbose:
                print(
                    f"[iter {k:03d}] volume({volume_on})={curr:.3e}  Δrel={mval:.3e}"
                )
            if mval <= vol_eps:
                patience_count += 1
            else:
                patience_count = 0
            if patience_count >= vol_pat:
                termination_reason = (
                    f"converged_volume(≤{vol_eps} for {vol_pat} iters)"
                )
                if verbose:
                    print(f"{termination_reason}.")
                time_metric.append(perf_counter() - t0)
                history.append(Znext)
                break

        elif convergence_metric in ("support_all", "support_state"):
            if convergence_metric == "support_all":
                if D_all is None:
                    q_all = support_ndirs_all or int(
                        math.ceil(
                            2
                            * (eng.n + eng.m + eng.w)
                            * math.log(max(4, Znext.nverts) + 1)
                        )
                    )
                    D_all = make_dirs(
                        eng.n + eng.m + eng.w, q=q_all, seed=support_seed
                    )
                mval = support_gap_with_dirs(Zk.V, Znext.V, D_all)
                support_all_hist.append(mval)
            else:
                if D_state is None:
                    q_x = support_ndirs_state or int(
                        math.ceil(
                            2 * eng.n * math.log(max(4, Znext.nverts) + 1)
                        )
                    )
                    D_state = make_dirs(eng.n, q=q_x, seed=support_seed)
                mval = support_gap_with_dirs(
                    Zk.V[:, : eng.n], Znext.V[:, : eng.n], D_state
                )
                support_state_hist.append(mval)

            metric_values.append(mval)
            if verbose:
                print(f"[iter {k:03d}] {convergence_metric}: gap={mval:.3e}")
            if mval <= metric_tol:
                patience_count += 1
            else:
                patience_count = 0
            if patience_count >= metric_patience:
                termination_reason = (
                    f"converged_{convergence_metric}"
                    f"(≤{metric_tol} for {metric_patience} iters)"
                )
                if verbose:
                    print(f"{termination_reason}.")
                time_metric.append(perf_counter() - t0)
                history.append(Znext)
                break

        elif convergence_metric == "hausdorff":
            A = _slice_dims(Zk.V, hausdorff_dims)
            B = _slice_dims(Znext.V, hausdorff_dims)
            mval = hausdorff_pointcloud(
                A, B, normalize=hausdorff_normalize, scale=hausdorff_scale
            )
            hausdorff_hist.append(mval)
            metric_values.append(mval)
            if verbose:
                normtxt = " (normalized)" if hausdorff_normalize else ""
                print(
                    f"[iter {k:03d}] hausdorff[{hausdorff_dims}]{normtxt} = {mval:.3e}"
                )
            if mval <= metric_tol:
                patience_count += 1
            else:
                patience_count = 0
            if patience_count >= metric_patience:
                termination_reason = (
                    f"converged_hausdorff(≤{metric_tol} for {metric_patience} iters)"
                )
                if verbose:
                    print(f"{termination_reason}.")
                time_metric.append(perf_counter() - t0)
                history.append(Znext)
                break
        else:
            raise ValueError(f"Unknown convergence_metric={convergence_metric!r}")

        t_metric_op = perf_counter() - t0
        time_metric.append(t_metric_op)

        history.append(Znext)
        Zk = Znext

    if termination_reason is None:
        termination_reason = "max_iters_reached"

    v0 = volumes_state[0] if volumes_state[0] > 0 else (max(volumes_state) or 1.0)
    volumes_state_norm = [v / v0 for v in volumes_state]

    stats: Dict = dict(
        iters=len(history) - 1,
        times=times,
        termination_reason=termination_reason,
        metric=convergence_metric,
        metric_tol=metric_tol,
        metric_patience=metric_patience,
        metric_values=metric_values,
        volumes_state=volumes_state,
        volumes_all=volumes_all,
        volumes_state_norm=volumes_state_norm,
        facets_state=facets_state,
        facets_all=facets_all,
        support_all=support_all_hist,
        support_state=support_state_hist,
        hausdorff=hausdorff_hist,
        last_base=locals().get("last_base", history[-1]),
        time_forward=time_forward,
        time_clip=time_clip,
        time_compress=time_compress,
        time_metric=time_metric,
        violation_rates=violation_rates,
        costs=costs,
        cost_type=cost_type if track_cost else None,
    )
    return history, stats


# ══════════════ iteration-specific plots ══════════════

def plot_violations_and_cost(
    stats: Dict, title: str = "Violation Rate and Cost Evolution"
):
    """Plot violation rate and cost on dual y-axes."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    iters = list(range(len(stats["violation_rates"])))
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if stats["violation_rates"]:
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=stats["violation_rates"],
                mode="lines+markers",
                name="Violation Rate",
                line=dict(color="red", width=2),
            ),
            secondary_y=False,
        )
    if stats["costs"]:
        cost_label = (
            f"{stats['cost_type']} cost" if stats["cost_type"] else "cost"
        )
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=stats["costs"],
                mode="lines+markers",
                name=cost_label.capitalize(),
                line=dict(color="blue", width=2),
            ),
            secondary_y=True,
        )

    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Violation Rate", secondary_y=False)
    fig.update_yaxes(title_text="Cost", secondary_y=True)
    fig.update_layout(title=title, template="plotly_white")
    fig.show()
    return fig


def plot_normalized_volume(
    volumes_norm: List[float],
    title: str = "Normalized state AABB volume vs iteration",
):
    """Line chart of normalised AABB volume over iterations."""
    import plotly.graph_objects as go

    x = list(range(len(volumes_norm)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=volumes_norm, mode="lines+markers", name="normalized volume")
    )
    fig.update_layout(
        title=title,
        xaxis_title="iteration k",
        yaxis_title="normalized volume (state AABB)",
        template="plotly_white",
        margin=dict(l=10, r=10, b=30, t=50),
    )
    fig.show()
    return fig


def plot_timing_breakdown(
    stats: Dict, title: str = "Iteration timing breakdown"
):
    """Stacked bar chart of per-iteration timing."""
    import plotly.graph_objects as go

    iters = list(range(1, len(stats["time_forward"]) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=iters, y=stats["time_forward"], name="Forward operator", marker_color="steelblue")
    )
    fig.add_trace(
        go.Bar(x=iters, y=stats["time_clip"], name="Clipping", marker_color="orange")
    )
    fig.add_trace(
        go.Bar(x=iters, y=stats["time_compress"], name="Compression", marker_color="green")
    )
    fig.add_trace(
        go.Bar(x=iters, y=stats["time_metric"], name="Metric computation", marker_color="red")
    )
    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Time (seconds)",
        barmode="stack",
        template="plotly_white",
        margin=dict(l=10, r=10, b=30, t=50),
    )
    fig.show()
    return fig
