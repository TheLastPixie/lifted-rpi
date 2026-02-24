#!/usr/bin/env python
"""
End-to-end lifted-RPI pipeline runner.

Reproduces the double-integrator experiment from the paper,
matching every parameter value for reproducibility.

System
------
- 2-D position/velocity double integrator, n=4, m=2, w=2, dt=0.02
- LQR gain K from Q=diag([1000, 0.1, 1000, 0.1]), R=diag([0.1, 0.1])
- GraphSet intersection via point-cloud clipping (no vertex enumeration)

Pipeline stages
---------------
 1. Build system matrices (A, B) and compute LQR gain K.
 2. Construct or load the disturbance graph set G (GP-learned or analytical).
 3. Build the operating sets: DV (input perturbation), W (disturbance
    outer-approximation), Z0 (initial set inside G).
 4. Run the fixed-point iteration Z_{k+1} = F(Z_k) cap G until
    Hausdorff convergence.
 5. Save all results (Z*, G decomposition, metrics, timing) to NPZ.

Usage
-----
    python scripts/run_pipeline.py                          # full GP pipeline
    python scripts/run_pipeline.py --graphset analytical    # analytical drag
    python scripts/run_pipeline.py --gp-model results/G_learned.joblib  # reuse saved GP
    python scripts/run_pipeline.py --max-iters 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path so we can import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lifted_rpi.polytope import Polytope
from lifted_rpi.vset import VSet, box_corners, GraphSet
from lifted_rpi.engine import LiftedSetOpsGPU_NoHull
from lifted_rpi.disturbance import make_drag_graphset
from lifted_rpi.initialization import (
    build_Z0_inside_G,
    make_W_from_learned_G_envelope,
    make_DV_from_u_box,
)
from lifted_rpi.iteration import fixed_point_reach
from lifted_rpi.convergence import aabb_volume


def _parts(V, K, n, m, w):
    """Split augmented vertex matrix into (X, Vc, U, W)."""
    X  = V[:, :n]
    Vc = V[:, n:n + m]
    W  = V[:, n + m:n + m + w]
    U  = (K @ X.T).T + Vc
    return X, Vc, U, W


def _build_G_vset_from_last_base(eng, G, last_base_V, n, m, w,
                                  *, max_pairs=1_000_000, seed=0):
    """Build the graph-set point cloud restricted to Z*'s operating region.

    ``last_base_V`` is the pre-clip cloud F(Z_{k-1}) from the final
    iteration.  For each (x, v) pair, the function queries
    G.bounds_fn(x, u) to get the disturbance box and expands it into
    2^w corner vertices.  This scopes G to the region Z* actually
    occupies, matching the paper's approach.

    Parameters
    ----------
    eng : LiftedSetOpsGPU_NoHull
        Engine instance (provides K, dimensions).
    G : GraphSet
        Disturbance graph set with bounds_fn.
    last_base_V : ndarray
        Pre-clip augmented vertex cloud from the final iteration.
    n, m, w : int
        State, input, disturbance dimensions.
    max_pairs : int
        Subsample limit for the number of (x, v) pairs to expand.

    Returns
    -------
    ndarray
        Downsampled point cloud of shape (N, n+m+w).
    """
    from lifted_rpi.vset import downsample_cloud

    V = np.asarray(last_base_V)
    if V.size == 0:
        return np.zeros((0, n + m + w))

    X  = V[:, :n]
    Vv = V[:, n:n + m]
    U  = (eng.K @ X.T).T + Vv

    # subsample if huge
    N = X.shape[0]
    take = min(max_pairs, N)
    idx = np.linspace(0, N - 1, take).astype(int)

    lbW, ubW = G.bounds_fn(X[idx], U[idx])  # (take, w)

    corner_bits = np.array(
        np.meshgrid(*[[0, 1]] * w, indexing="ij")
    ).reshape(w, -1).T

    cloud = []
    for i in range(take):
        Wi = lbW[i] + corner_bits * (ubW[i] - lbW[i])
        Xi = np.repeat(X[idx[i]:idx[i] + 1], Wi.shape[0], axis=0)
        Vi = np.repeat(Vv[idx[i]:idx[i] + 1], Wi.shape[0], axis=0)
        cloud.append(np.hstack([Xi, Vi, Wi]))
    G_cloud = np.vstack(cloud)
    G_cloud = downsample_cloud(G_cloud, max_points=50_000, method="grid")
    return G_cloud


def build_system(dt: float = 0.02):
    """Double integrator: state = [px, vx, py, vy], input = [ax, ay]."""
    A = np.array([
        [1, dt, 0, 0],
        [0, 1,  0, 0],
        [0, 0,  1, dt],
        [0, 0,  0, 1],
    ])
    B = np.array([
        [0.5 * dt**2, 0],
        [dt,          0],
        [0, 0.5 * dt**2],
        [0,          dt],
    ])
    return A, B


def build_lqr_gain(A, B):
    """Compute LQR gain K.

    Q = diag([1000, 0.1, 1000, 0.1])
    R = diag([0.1,  0.1])
    K = -inv(R + B'PB)(B'PA)
    """
    from scipy.linalg import solve_discrete_are
    Q_lqr = np.diag([1000.0, 0.1, 1000.0, 0.1])
    R_lqr = np.diag([0.1, 0.1])
    P = solve_discrete_are(A, B, Q_lqr, R_lqr)
    K = -np.linalg.inv(R_lqr + B.T @ P @ B) @ (B.T @ P @ A)
    return K, Q_lqr, R_lqr


def build_drag_graphset():
    """Velocity-dependent drag disturbance model.

    When running with the GP-learned GraphSet (G_learned), we use
    the analytical drag model as fallback, with the SAME physical
    parameters:
        mass=1.0, beta1=0.05, beta2=0.02,
        sigma_wx=0.01, sigma_wy=0.01, chi2_tau=6.635 (99.5%)
    and tol=1e-2 (matching G_learned.tol).
    """
    return make_drag_graphset(
        mass=1.0,
        beta1=0.05,
        beta2=0.02,
        sigma_wx=0.01,
        sigma_wy=0.01,
        chi2_tau=6.635,       # 99.5% confidence
        vx_index=1,
        vy_index=3,
        name="G_drag",
        tol=1e-2,             # matches G_learned.tol default
    )


def build_gp_learned_graphset(save_dir: str = "results"):
    """Reproduce the GP-learned graph set G_learned.

    This runs three stages:
    1. MPC simulation: 2500-step trajectory to collect
       state/control/disturbance data under the drag model.
    2. GP training: fit two independent GPs (w_x, w_y)
       with kernel = RBF * SinExp + White, using all 2500 data points.
    3. Assemble LearnedGraphSetGP and save to disk.

    The trained model is saved to ``{save_dir}/G_learned.joblib``.

    Parameters matching the paper exactly:
      trajectory: 'linear', x0=[0,0,4,0], xt=[10,0,10,0], n_steps=2500
      drag: beta1=0.05, beta2=0.02, mass=1.0, noise_std=0.01
      GP: kernel='RBF * SinExp + White', downsample=1, restarts=2
      G: prior_box=+/-3, k_sigma=3.5, far_radius=0.75, tol=1e-2
    """
    from lifted_rpi.simulation import simulate_trajectory_with_realistic_drag
    from lifted_rpi.gp_learner import (
        LearnedGraphSetGP,
        train_disturbance_gps,
    )
    from scipy.linalg import solve_discrete_are

    dt = 0.02
    A_p = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    B_p = np.array([[0.5*dt**2, 0], [dt, 0], [0, 0.5*dt**2], [0, dt]])
    Q = np.diag([1000, 0.1, 1000, 0.1])
    R = np.diag([0.1, 0.1])
    P = solve_discrete_are(A_p, B_p, Q, R)

    # Step 1: MPC simulation
    print("\n--- Step 1: MPC simulation (2500 steps, ~2-5 min) ---")
    t0 = time.time()
    state_history, control_history, disturbance_history, _, sim_time, fps = \
        simulate_trajectory_with_realistic_drag(
            trajectory_type='linear',
            x_initial=np.array([0, 0, 4, 0]),
            x_target=np.array([10, 0, 10, 0]),
            n_steps=2500,
            L_k=5,
            A_p=A_p, B_p=B_p, Q=Q, R=R, P=P,
            beta1=0.05, beta2=0.02, mass=1.0, noise_std=0.01,
            dt=dt,
        )
    print(f"  Simulation done in {time.time()-t0:.1f}s (MPC: {sim_time:.1f}s, {fps:.1f} fps)")
    print(f"  States shape: {state_history.shape}")
    print(f"  Disturbance range: w_x=[{disturbance_history[:,1].min():.4f}, {disturbance_history[:,1].max():.4f}]"
          f" w_y=[{disturbance_history[:,3].min():.4f}, {disturbance_history[:,3].max():.4f}]")

    # Step 2: Train GPs
    # Using downsample=1 (all 2500 pts).  GP fit is O(n³) so this
    # takes ~10-20 min but produces accurate bounds (critical for W tightness).
    print("\n--- Step 2: Training GPs (downsample=1) ---")
    t0 = time.time()
    gp_x, gp_y, kdt, meta = train_disturbance_gps(
        state_history, control_history, disturbance_history,
        downsample=1,              # use ALL 2500 points
        kernel_choice="RBF * SinExp + White",
        gp_restarts=2,
    )
    print(f"  GP training done in {time.time()-t0:.1f}s")
    print(f"  Kernel: {meta['kernel_used'][:80]}...")

    # Step 3: Build G_learned
    G_learned = LearnedGraphSetGP(
        gp_x, gp_y, kdtree=kdt,
        prior_box=((-3.0, -3.0), (3.0, 3.0)),
        k_sigma=3.5,
        min_halfwidth=1e-3,
        far_radius=0.75,
        tol=1e-2,
        name="G_learned",
    )

    # Save for reuse
    gp_path = os.path.join(save_dir, "G_learned.joblib")
    G_learned.save(gp_path)
    print(f"  G_learned saved to {gp_path}")

    # Return simulation histories alongside G so they can be saved for
    # figure generation (the heatmap / 3-D plots use these).
    sim_data = dict(
        state_history=state_history,
        control_history=control_history,
        disturbance_history=disturbance_history,
    )
    return G_learned, sim_data


def main():
    parser = argparse.ArgumentParser(description="Lifted-RPI pipeline runner")
    parser.add_argument("--graphset", choices=["gp_learned", "analytical"],
                        default="gp_learned",
                        help="gp_learned: train GP | analytical: drag model")
    parser.add_argument("--gp-model", type=str, default=None,
                        help="Path to pre-trained G_learned.joblib (skips simulation+training)")
    parser.add_argument("--max-iters", type=int, default=1000,
                        help="Max fixed-point iterations (default: 1000)")
    parser.add_argument("--metric-tol", type=float, default=3e-2,
                        help="Hausdorff convergence tolerance (default: 3e-2)")
    parser.add_argument("--metric-patience", type=int, default=3,
                        help="Patience for convergence (default: 3)")
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--no-gpu-clip", action="store_true")
    parser.add_argument("--surrogate", choices=["nystroem", "poly", "none"],
                        default="none",
                        help="Surrogate backend for GP clipping speedup "
                             "(default: none = raw GP)")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU acceleration (PyTorch CUDA) for "
                             "unique-rows, KNN, Nystroem, etc.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # -- 0. GPU acceleration (optional) --
    if args.gpu:
        from lifted_rpi.speedup import init_gpu
        gpu_ok = init_gpu(verbose=True)
        if not gpu_ok:
            print("WARNING: --gpu requested but CUDA unavailable; continuing on CPU")

    # -- 1. System matrices --
    print("=" * 60)
    print("LIFTED-RPI PIPELINE  (paper-exact parameters)")
    print("=" * 60)
    n, m, w, dt = 4, 2, 2, 0.02
    A, B = build_system(dt)
    K, Q_lqr, R_lqr = build_lqr_gain(A, B)
    Acl = A + B @ K
    eigs = np.linalg.eigvals(Acl)
    rho_plus  = max(abs(np.linalg.eigvals(A + B @ (-K))))
    rho_minus = max(abs(np.linalg.eigvals(A + B @ K)))
    print(f"System: n={n}, m={m}, w={w}, dt={dt}")
    print(f"Q_lqr = diag{list(np.diag(Q_lqr))}")
    print(f"R_lqr = diag{list(np.diag(R_lqr))}")
    print(f"rho(A+BK)={rho_minus:.6f}, rho(A-BK)={rho_plus:.6f}")
    print(f"Acl eigenvalues: {np.abs(eigs).round(6)}  (all < 1? {all(abs(e) < 1 for e in eigs)})")
    print(f"K =\n{K}")

    # -- 2. Disturbance model --
    sim_data = None  # filled when GP pipeline runs the simulation
    if args.graphset == "gp_learned":
        if args.gp_model and os.path.exists(args.gp_model):
            from lifted_rpi.gp_learner import LearnedGraphSetGP
            print(f"\nLoading pre-trained GP model from {args.gp_model}")
            G = LearnedGraphSetGP.load(args.gp_model)
        else:
            G, sim_data = build_gp_learned_graphset(save_dir=args.save_dir)
    else:
        G = build_drag_graphset()
    # -- 2b. Optional surrogate wrapper --
    if args.surrogate != "none" and args.graphset == "gp_learned":
        from lifted_rpi.speedup import SurrogateGraphSet
        print(f"\nWrapping GP in SurrogateGraphSet (backend={args.surrogate})")
        G = SurrogateGraphSet(G, backend=args.surrogate, verbose=True)
        print(f"  Surrogate build time: {G.build_time:.3f}s")

    print(f"\nGraphSet: {G.name}, tol={G.tol}")

    # -- 3. Operating region: xv_box = +/-0.5 --
    xv_box_half = 0.5
    xv_box = (
        -xv_box_half * np.ones(n + m),
        xv_box_half * np.ones(n + m),
    )
    print(f"xv_box: +/-{xv_box_half}")

    # -- 4. Engine, paper_exact mode (alpha_v=1.0, beta_w=0.0) --
    #    All engine params match the paper exactly:
    #      mink_tol=1e-5, downsample_method="stride",
    #      downsample_max_points=50000, batch_target_points=120_000
    alpha_v = 1.0
    beta_w = 0.0
    print(f"\n-> Paper-exact mode: alpha_v={alpha_v}, beta_w={beta_w}")

    eng = LiftedSetOpsGPU_NoHull.paper_exact(
        A=A, B=B, K=K, n=n, m=m, w=w,
        dist_rows=(1, 3),
        use_paper_aug=True,
        compute_filter=False,
        mink_tol=1e-5,                   # paper: 1e-5
        downsample_method="stride",      # paper: "stride"
        downsample_max_points=50_000,    # paper: 50000
        batch_target_points=120_000,     # paper: 120_000
        seed=0,
        use_gpu_clipping=not args.no_gpu_clip,
    )

    print(f"  alpha_v={eng.alpha_v}, beta_w={eng.beta_w}")
    print(f"  A_tilde shape: {eng.A_tilde.shape}")
    print(f"  B_tilde shape: {eng.B_tilde.shape}")
    print(f"  D_tilde shape: {eng.D_tilde.shape}")
    print(f"  D_top:\n{eng.D_top}")

    # -- 5. DV from input constraints --
    u_min = np.array([-5.0, -5.0])
    u_max = np.array([ 5.0,  5.0])
    DV, hv, dv_half, R_abs = make_DV_from_u_box(
        K, xv_box, u_min, u_max,
        alpha_v=eng.alpha_v,   # 1.0 → d = (1-1)*hv = 0 → DV is origin
        safety=0.9,
    )
    print(f"\nKx bound R: {R_abs}")
    print(f"v halfwidth hv: {hv}")
    print(f"DV halfwidth d: {dv_half}")
    print(f"DV verts: {DV.nverts}")

    # -- 6. W from GraphSet envelope --
    W = make_W_from_learned_G_envelope(
        eng, G,
        xv_box=xv_box,
        x_levels=5,
        v_levels=5,
        add_local_grid=True,
        local_levels=3,
        name="W_from_G",
    )
    print(f"\nW verts: {W.nverts}")
    print(f"W range: lb={W.V.min(0).round(4)}, ub={W.V.max(0).round(4)}")

    # -- 7. Z0 = box_corners(+/-0.3) clipped to G --
    #    With the GP-learned G the wide prior_box fallback ensures most
    #    box corners survive clipping.  With the analytical model the
    #    bounds are tighter and may clip to empty; in that case, fall
    #    back to build_Z0_inside_G which queries G for valid w-bounds.
    n_aug = n + m + w
    Z0_box = VSet(box_corners(-0.3 * np.ones(n_aug), 0.3 * np.ones(n_aug)),
                  name="Z0_box")
    Z0 = eng.clip_with_graph(Z0_box, G, name="Z0_in_G")
    print(f"\nZ0_box verts: {Z0_box.nverts}")
    print(f"Z0 (after G clip) verts: {Z0.nverts}")

    if Z0.nverts == 0:
        print("  -> box_corners empty after clip, using build_Z0_inside_G")
        Z0 = build_Z0_inside_G(
            eng, G,
            xv_box=xv_box,
            x_levels=3,
            v_levels=3,
            w_mode="corners",
            name="Z_0",
        )
        print(f"  -> Z0 from build_Z0_inside_G: {Z0.nverts} verts")

    assert Z0.nverts > 0, "Z0 is empty even with build_Z0_inside_G!"
    print(f"Z0 state-AABB volume: {aabb_volume(Z0.V[:, :n]):.6e}")

    # -- 8. One-step test --
    print("\n--- One-step test ---")
    t0 = time.perf_counter()
    FZ = eng.F_no_intersection(Z0, DV, W, name="FZ_test")
    t1 = time.perf_counter()
    FGZ = eng.F_with_intersection(Z0, DV, W, G, name="FGZ_test")
    t2 = time.perf_counter()
    print(f"F(Z0)       verts: {FZ.nverts:>8,}  ({t1 - t0:.3f}s)")
    fgz_pct = FGZ.nverts / max(1, FZ.nverts)
    print(f"F(Z0) cap G verts: {FGZ.nverts:>8,}  ({t2 - t1:.3f}s)")
    print(f"Clipping kept {FGZ.nverts}/{FZ.nverts} = {fgz_pct:.1%}")

    if FGZ.nverts == 0:
        print("\nWARNING: F(Z0) cap G is empty! Intersection too aggressive.")
        print("Cannot proceed with empty set. Check W box and GraphSet config.")
        return

    # -- 9. Fixed-point iteration --
    print("\n" + "=" * 60)
    print("FIXED-POINT ITERATION")
    print("=" * 60)

    history, stats = fixed_point_reach(
        eng=eng,
        Z0=Z0,
        DV=DV,
        W=W,
        G=G,
        # Convergence: hausdorff on state dims, normalized
        convergence_metric="hausdorff",
        hausdorff_dims="state",
        hausdorff_normalize=True,
        metric_tol=args.metric_tol,       # default 3e-2
        metric_patience=args.metric_patience,  # default 3
        max_iters=args.max_iters,         # default 1000
        # Match paper exactly, no extra tracking
        stop_if_empty=True,
        verbose=True,
        track_facets=False,      # ConvexHull in 8D is too slow
        track_violations=False,  # compute_violation_rate misses back-projection
        track_cost=False,
        detail_timing=True,
    )

    # -- 10. Summary --
    Zstar = history[-1]
    n_iters = stats["iters"]
    total_time = sum(stats["times"])

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Termination:     {stats['termination_reason']}")
    print(f"Iterations:      {n_iters}")
    print(f"Total time:      {total_time:.2f}s  ({total_time/max(1,n_iters):.2f}s/iter)")
    print(f"Z* verts:        {Zstar.nverts:,}")
    print(f"Z* dim:          {Zstar.dim}")
    vol_state_0 = stats["volumes_state"][0]
    vol_state_f = stats["volumes_state"][-1]
    print(f"State AABB vol:  {vol_state_0:.4e} -> {vol_state_f:.4e}  "
          f"(x{vol_state_f / max(1e-30, vol_state_0):.2f})")

    # Hausdorff metric history
    if stats.get("hausdorff"):
        print(f"\nLast hausdorff: {stats['hausdorff'][-1]:.3e}")

    # Average timing breakdown
    if stats['time_forward']:
        print(f"\n=== Average timing per iteration ===")
        print(f"Forward operator: {np.mean(stats['time_forward']):.4f}s")
        print(f"Clipping:         {np.mean(stats['time_clip']):.4f}s")
        print(f"Total per iter:   {np.mean(stats['times']):.4f}s")

    # -- 11. Project to state space --
    Zstar_x = Zstar.V[:, :n] if Zstar.nverts > 0 else np.zeros((0, n))
    print(f"\nProjected state cloud: {Zstar_x.shape[0]} points in R^{n}")
    for i, label in enumerate(["px", "vx", "py", "vy"]):
        if Zstar_x.shape[0] > 0:
            print(f"  {label}: [{Zstar_x[:, i].min():.4f}, {Zstar_x[:, i].max():.4f}]")

    # -- 12. Build G_vset for 3-D plotting --
    #    Use the pre-clip cloud from the final iteration (last_base) so
    #    that G is restricted to the region Z* actually occupies.  This
    #    matches the paper's approach.
    print("\n--- Building G_vset for visualisation ---")
    last_base = stats.get("last_base", None)
    last_base_V = last_base.V if hasattr(last_base, "V") else last_base
    if last_base_V is not None and last_base_V.size > 0:
        G_vset_V = _build_G_vset_from_last_base(eng, G, last_base_V, n, m, w)
    else:
        # fallback: random sampling (should not happen)
        print("  WARNING: last_base not available; falling back to random sampling")
        rng = np.random.default_rng(0)
        lb_xv, ub_xv = xv_box
        X_r = rng.uniform(lb_xv[:n], ub_xv[:n], size=(500, n))
        V_r = rng.uniform(lb_xv[n:], ub_xv[n:], size=(500, m))
        U_r = (eng.K @ X_r.T).T + V_r
        lbW, ubW = G.bounds_fn(X_r, U_r)
        bits = np.array(np.meshgrid(*[[0,1]]*w, indexing="ij")).reshape(w,-1).T
        cloud = []
        for i in range(200):
            Wi = lbW[i] + bits*(ubW[i]-lbW[i])
            cloud.append(np.hstack([np.repeat(X_r[i:i+1],4,0), np.repeat(V_r[i:i+1],4,0), Wi]))
        G_vset_V = np.vstack(cloud)
    print(f"  G_vset cloud: {G_vset_V.shape[0]} points in R^{n + m + w}")

    # -- 13. Decompose G_vset into coordinate arrays --
    #    This lets figures be generated without re-computing K @ X + V etc.
    print("\n--- Decomposing G_vset into (X, Vc, U, W) ---")
    X_G, Vc_G, U_G, W_G = _parts(G_vset_V, eng.K, n, m, w)
    print(f"  G_X  : {X_G.shape}  (state columns)")
    print(f"  G_Vc : {Vc_G.shape}  (controller auxiliary)")
    print(f"  G_U  : {U_G.shape}  (reconstructed input)")
    print(f"  G_W  : {W_G.shape}  (disturbance columns)")

    # -- 14. Collect history snapshots --
    #    Save every n/10 iterations (plus first and last) for the
    #    3-D convergence evolution figures.
    n_hist = len(history)
    step = max(1, n_hist // 10)          # ≈ 10 % of total iterations
    snap_indices = sorted(set(
        list(range(0, n_hist, step)) + [0, n_hist - 1]
    ))
    history_snaps = {f"history_{i}": history[i].V for i in snap_indices}
    history_snap_indices = np.array(snap_indices)
    print(f"  Saved {len(snap_indices)} history snapshots at indices "
          f"{snap_indices} (of {n_hist} total, step={step})")

    # -- 15. Save results --
    save_path = os.path.join(args.save_dir, "pipeline_paper_exact.npz")
    save_dict = dict(
        # ── Core sets ──
        Z0=Z0.V,                       # first base (initial set)
        Zstar=Zstar.V,                  # fixed-point Z*
        Zstar_x=Zstar_x,               # Z* projected to state space

        # ── Graph G ──
        G_vset=G_vset_V,               # full augmented G cloud
        last_base_V=last_base_V,        # pre-clip cloud from final iter (source of G_vset)
        G_X=X_G,                        # G state columns [px, vx, py, vy]
        G_Vc=Vc_G,                      # G controller-aux columns
        G_U=U_G,                        # G reconstructed input  u = Kx + v
        G_W=W_G,                        # G disturbance columns  [wx, wy]

        # ── Convergence history ──
        history_snap_indices=history_snap_indices,

        # ── Scalar metrics per iteration ──
        volumes_state=np.array(stats["volumes_state"]),
        volumes_state_norm=np.array(stats["volumes_state_norm"]),
        metric_values=np.array(stats["metric_values"]),
        hausdorff=np.array(stats.get("hausdorff", [])),

        # ── Timing per iteration ──
        times=np.array(stats["times"]),
        time_forward=np.array(stats["time_forward"]),
        time_clip=np.array(stats["time_clip"]),
        time_compress=np.array(stats["time_compress"]),

        # ── System / engine parameters ──
        alpha_v=eng.alpha_v,
        beta_w=eng.beta_w,
        A_tilde=eng.A_tilde,
        B_tilde=eng.B_tilde,
        D_tilde=eng.D_tilde,
        K=eng.K,
        n=n, m=m, w=w,
    )
    # Include simulation trajectory histories when available (needed by
    # generate_figures.py for the acceleration-disturbance heatmaps).
    if sim_data is not None:
        save_dict["state_history"] = sim_data["state_history"]
        save_dict["control_history"] = sim_data["control_history"]
        save_dict["disturbance_history"] = sim_data["disturbance_history"]
    save_dict.update(history_snaps)
    np.savez_compressed(save_path, **save_dict)
    fsize_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\nResults saved to {save_path}  ({fsize_mb:.2f} MB)")
    print(f"  Keys: {sorted(save_dict.keys())}")

    # vertex counts per iteration
    vert_counts = [h.nverts for h in history]
    print(f"\nVertex counts per iteration:")
    for i, vc in enumerate(vert_counts):
        tag = " <- Z0" if i == 0 else (" <- Z*" if i == len(vert_counts) - 1 else "")
        print(f"  k={i:3d}: {vc:>8,} verts{tag}")

    # -- 13. Quick matplotlib plots --
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Lifted-RPI Pipeline [paper_exact]", fontsize=14)

        # Volume
        ax = axes[0, 0]
        ax.plot(stats["volumes_state_norm"], "o-", markersize=3, color="steelblue")
        ax.set_xlabel("Iteration k")
        ax.set_ylabel("Normalized state AABB volume")
        ax.set_title("Convergence (volume)")
        ax.grid(True, alpha=0.3)

        # Metric (hausdorff)
        ax = axes[0, 1]
        if stats["metric_values"]:
            ax.semilogy(stats["metric_values"], "s-", markersize=3, color="darkorange")
            ax.axhline(args.metric_tol, ls="--", color="red", alpha=0.6,
                        label=f"tol={args.metric_tol}")
            ax.legend()
        ax.set_xlabel("Iteration k")
        ax.set_ylabel("Hausdorff (state, normalized)")
        ax.set_title("Convergence (hausdorff metric)")
        ax.grid(True, alpha=0.3)

        # State projection (px vs py)
        ax = axes[1, 0]
        if Zstar_x.shape[0] > 0:
            ax.scatter(Zstar_x[:, 0], Zstar_x[:, 2], s=1, alpha=0.3, c="navy")
        ax.set_xlabel("$p_x$ [m]")
        ax.set_ylabel("$p_y$ [m]")
        ax.set_title(f"Z* projected (px, py) -- {Zstar_x.shape[0]} pts")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # State projection (vx vs vy)
        ax = axes[1, 1]
        if Zstar_x.shape[0] > 0:
            ax.scatter(Zstar_x[:, 1], Zstar_x[:, 3], s=1, alpha=0.3, c="darkred")
        ax.set_xlabel("$v_x$ [m/s]")
        ax.set_ylabel("$v_y$ [m/s]")
        ax.set_title(f"Z* projected (vx, vy) -- {Zstar_x.shape[0]} pts")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(args.save_dir, "pipeline_paper_exact.png")
        fig.savefig(fig_path, dpi=150)
        print(f"Figure saved to {fig_path}")
        plt.close(fig)

    except ImportError:
        print("matplotlib not available, skipping plots")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
