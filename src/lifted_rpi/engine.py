"""
Lifted set-operations engine (no convex hull in the computational loop).

Implements the one-step operator
    F(Z) = (Ã Z  ⊕  B̃ ΔV  ⊕  D̃ W) ∩ G
using vertex-cloud arithmetic on GPU (JAX + CuPy).

The augmented state is z = [x; v; w] where:
    x  -- physical state  (R^n)
    v  -- controller auxiliary / perturbation  (R^m)
    w  -- disturbance realisation  (R^w)

The augmented matrices are constructed from:
    Ã = [[Acl, B, 0], [0, alpha_v*I_m, 0], [0, 0, beta_w*I_w]]
    B̃ = [B; I_m; 0]
    D̃ = [D_top; 0; I_w]
where D_top maps disturbance into the state equation rows specified
by ``dist_rows`` (default: velocity rows 1 and 3 for a 2-D double
integrator).

"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from time import perf_counter as _perf_counter

try:
    import cupy as cp
except ImportError:
    cp = None

from .polytope import Polytope
from .minkowski_gpu import MinkowskiGPU
from .vset import (
    VSet, GraphSet, box_corners,
    apply_linear_to_cloud, clip_cloud_by_polytope,
    clip_cloud_by_polytope_gpu, downsample_cloud,
)


# ----------------------------- main engine ------------------------------

@dataclass
class LiftedSetOpsGPU_NoHull:
    """
    Lifted set-ops engine that avoids ConvexHull in the computational loop.

    Parameters
    ----------
    A, B, K : plant matrices and stabilizing feedback
    n, m, w : dims (state, input, disturbance)
    alpha_v : leak factor for v-block (0 < alpha_v <= 1)
    beta_w  : leak factor for stored w-block; 0.0 = memoryless
    use_paper_aug : build (Ã, B̃, D̃) in canonical form
    Edist : optional custom n×w injection for D̃ top block
    """

    A: np.ndarray
    B: np.ndarray
    K: np.ndarray
    n: int
    m: int
    w: int
    alpha_v: float = 0.6
    beta_w: float = 0.8
    use_paper_aug: bool = True
    Edist: Optional[np.ndarray] = None
    dist_rows: Optional[Tuple[int, ...]] = None
    compute_filter: bool = False
    downsample_max_points: int = 80_000
    downsample_method: str = "grid"
    downsample_grid_eps: float = None
    engine: Optional[MinkowskiGPU] = None
    mink_tol: float = 1e-7
    mink_num_dirs: Optional[int] = None
    batch_target_points: int = 750_000
    seed: int = 0
    store_dtype: str = "float32"
    n_filter_passes: int = 3
    use_gpu_clipping: bool = True

    @classmethod
    def paper_exact(cls, A, B, K, n, m, w, **kw):
        """Convenience: paper-exact mode with α_v=1, β_w=0 (memoryless w).

        This matches the canonical augmented system from the paper:
            Ã = [[Acl, B, 0], [0, I_m, 0], [0, 0, 0]]
            B̃ = [B; I_m; 0]
            D̃ = [E; 0; I_w]
        No leaky contractions.
        """
        kw.setdefault("alpha_v", 1.0)
        kw.setdefault("beta_w", 0.0)
        return cls(A=A, B=B, K=K, n=n, m=m, w=w, **kw)

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        self.K = np.asarray(self.K, dtype=float)
        assert self.A.shape == (self.n, self.n)
        assert self.B.shape == (self.n, self.m)
        assert self.K.shape == (self.m, self.n)

        self.n_aug = self.n + self.m + self.w
        self._build_augmented_mats()

        if self.engine is None:
            self.engine = MinkowskiGPU(
                num_dirs=self.mink_num_dirs,
                tol=self.mink_tol,
                seed=self.seed,
                force_host_copy=False,
                n_filter_passes=self.n_filter_passes,
            )

    def _build_augmented_mats(self):
        """Build Ã (n_aug x n_aug), B_tilde (n_aug x m), D_tilde (n_aug x w).

        The v-block uses ``alpha_v`` as a leak/contraction factor.
        The w-block uses ``beta_w`` (0 = memoryless, 1 = full memory).
        D_top is determined by ``Edist`` > ``dist_rows`` > default identity.
        """
        Acl = self.A + self.B @ self.K  # closed-loop state matrix
        I_m = np.eye(self.m)
        I_w = np.eye(self.w)

        A_tl = Acl
        A_tm = self.B
        A_tr = np.zeros((self.n, self.w))

        A_ml = np.zeros((self.m, self.n))
        A_mm = self.alpha_v * I_m
        A_mr = np.zeros((self.m, self.w))

        A_bl = np.zeros((self.w, self.n))
        A_bm = np.zeros((self.w, self.m))
        A_br = self.beta_w * I_w

        self.A_tilde = np.block([
            [A_tl, A_tm, A_tr],
            [A_ml, A_mm, A_mr],
            [A_bl, A_bm, A_br],
        ])

        self.B_tilde = np.vstack([
            self.B,
            I_m,
            np.zeros((self.w, self.m)),
        ])

        if self.Edist is not None:
            D_top = np.asarray(self.Edist, dtype=float)
            assert D_top.shape == (self.n, self.w), "Edist must be (n × w)"
        else:
            if self.dist_rows is not None:
                assert len(self.dist_rows) == self.w
                D_top = np.zeros((self.n, self.w), dtype=float)
                for j, r in enumerate(self.dist_rows):
                    assert 0 <= r < self.n
                    D_top[r, j] = 1.0
            else:
                r = min(self.n, self.w)
                D_top = np.zeros((self.n, self.w), dtype=float)
                D_top[:r, :r] = np.eye(r)

        self.D_tilde = np.vstack([
            D_top,
            np.zeros((self.m, self.w)),
            np.eye(self.w),
        ])
        self.D_top = D_top

    # ------------------- GPU Minkowski (NO HULL) --------------------

    def _minkowski_gpu_batched_nohull(self, V1: np.ndarray, V2: np.ndarray):
        """Batched Minkowski sum with optional multi-pass extreme-point filtering.

        Splits V1 into chunks of size ``batch_target_points / |V2|`` to keep
        GPU memory usage bounded.  When ``compute_filter`` is True, each batch
        is filtered by multi-pass random projection before concatenation.
        Otherwise, the raw pairwise sums are downsampled to
        ``downsample_max_points`` via grid/random/stride methods.

        Returns (filtered_points, stats_dict).
        """
        V1 = np.asarray(V1, dtype=np.float32 if self.store_dtype == "float32" else np.float64)
        V2 = np.asarray(V2, dtype=np.float32 if self.store_dtype == "float32" else np.float64)

        if V1.size == 0 or V2.size == 0:
            return V1[:0], dict(total_candidates=0, batch_kept=0, final_kept=0,
                                t_batch=0.0, t_global_filter=0.0)

        d = V1.shape[1]
        n1, n2 = V1.shape[0], V2.shape[0]
        total_candidates = n1 * n2
        chunk = max(1, self.batch_target_points // max(1, n2))

        kept_batches = []
        t_batch = 0.0

        for s in range(0, n1, chunk):
            e = min(n1, s + chunk)
            C_j = self.engine.candidates(V1[s:e], V2)

            if self.compute_filter:
                t0 = _perf_counter()
                C_cp = self.engine._jax_to_cupy(C_j)
                mask_cp = self.engine._filter_extreme_points_cupy_multipass(
                    C_cp, self.mink_num_dirs, self.mink_tol,
                    self.seed, self.n_filter_passes,
                )
                kept = C_cp[mask_cp]
                kept_batches.append(kept)
                t_batch += (_perf_counter() - t0)
            else:
                kept_batches.append(np.asarray(C_j))

            del C_j

        if self.compute_filter:
            if cp is not None:
                big = cp.concatenate(kept_batches, axis=0) if len(kept_batches) > 1 else kept_batches[0]
                if big.size == 0:
                    return np.empty((0, d), dtype=V1.dtype), dict(
                        total_candidates=int(total_candidates), batch_kept=0, final_kept=0,
                        t_batch=float(t_batch), t_global_filter=0.0)

                mask_g = self.engine._filter_extreme_points_cupy_multipass(
                    big, self.mink_num_dirs, self.mink_tol,
                    self.seed, self.n_filter_passes,
                )
                filtered = cp.asnumpy(big[mask_g])
            else:
                big = np.concatenate([np.asarray(k) for k in kept_batches], axis=0)
                filtered = big

            filtered = np.unique(filtered, axis=0)
            if self.n_filter_passes > 1:
                print(f"  Multi-pass filter ({self.n_filter_passes} passes): "
                      f"{total_candidates} candidates → {filtered.shape[0]} kept")

            return filtered, dict(
                total_candidates=int(total_candidates),
                batch_kept=int(sum(int(k.shape[0]) for k in kept_batches)),
                final_kept=int(filtered.shape[0]),
                t_batch=float(t_batch), t_global_filter=0.0)
        else:
            if cp is not None and isinstance(kept_batches[0], cp.ndarray):
                big = cp.concatenate(kept_batches, axis=0).get()
            else:
                big = np.concatenate([np.asarray(k) for k in kept_batches], axis=0)

            big = np.unique(big, axis=0)
            big = downsample_cloud(
                big, max_points=self.downsample_max_points,
                method=self.downsample_method,
                grid_eps=self.downsample_grid_eps,
                seed=self.seed if self.seed is not None else 0,
            )

            return big.astype(V1.dtype, copy=False), dict(
                total_candidates=int(total_candidates),
                batch_kept=int(big.shape[0]),
                final_kept=int(big.shape[0]),
                t_batch=float(t_batch), t_global_filter=0.0)

    # ------------------- public ops on V-sets -----------------------

    def linmap(self, S: VSet, M: np.ndarray, name: Optional[str] = None) -> VSet:
        """Pointwise linear map S ↦ { M v : v ∈ S }."""
        Y = apply_linear_to_cloud(
            S.V, M,
            out_dtype="float32" if self.store_dtype == "float32" else "float64",
        )
        return VSet(Y, name=name or (S.name and f"{S.name}_mapped"))

    def minkowski(self, S: VSet, T: VSet, name: Optional[str] = None) -> VSet:
        """GPU Minkowski (no hull): filter-only cloud."""
        F, _stats = self._minkowski_gpu_batched_nohull(S.V, T.V)
        return VSet(F, name=name or (S.name and T.name and f"{S.name}_oplus_{T.name}") or name)

    def clip_with_polytope(self, S: VSet, G: Polytope,
                           tol: float = 1e-9, name: Optional[str] = None) -> VSet:
        """UNDER-approx intersection: keep only points inside A_G x ≤ b_G."""
        if self.use_gpu_clipping and cp is not None:
            return self.clip_with_polytope_gpu(S, G, tol, name)
        Vc = clip_cloud_by_polytope(S.V, G, tol=tol)
        return VSet(Vc, name=name or (S.name and f"{S.name}_cap_G"))

    def clip_with_polytope_gpu(self, S: VSet, G: Polytope,
                               tol: float = 1e-9, name: Optional[str] = None) -> VSet:
        Vc = clip_cloud_by_polytope_gpu(S.V, G, tol=tol)
        return VSet(Vc, name=name or (S.name and f"{S.name}_cap_G"))

    def project_state(self, S_aug: VSet, name: Optional[str] = None) -> VSet:
        """Project augmented cloud onto the first n coordinates (x)."""
        if S_aug.V.size == 0:
            Z = np.zeros(
                (0, self.n),
                dtype=np.float32 if self.store_dtype == "float32" else np.float64,
            )
            return VSet(Z, name=name or "Proj_x")
        Vx = S_aug.V[:, :self.n].astype(
            np.float32 if self.store_dtype == "float32" else np.float64, copy=False,
        )
        return VSet(Vx, name=name or (S_aug.name and f"{S_aug.name}_projx"))

    # -------------- one-step operator (NO HULL) ---------------------

    def F_no_intersection(self, Z: VSet, DeltaV: VSet, W: VSet,
                          name: str = "FZ") -> VSet:
        """F(Z) = Ã Z ⊕ B̃ ΔV ⊕ D̃ W  (all as vertex clouds)."""
        AZ = self.linmap(Z, self.A_tilde, name="ÃZ")
        BDV = self.linmap(DeltaV, self.B_tilde, name="B̃ΔV")
        DW = self.linmap(W, self.D_tilde, name="D̃W")

        AZpBDV = self.minkowski(AZ, BDV, name="ÃZ⊕B̃ΔV")
        FZ = self.minkowski(AZpBDV, DW, name=name)
        return FZ

    def clip_with_graph(self, S: VSet, Ggraph: GraphSet,
                        name: Optional[str] = None) -> VSet:
        """Point-filtering intersection with a state-dependent graph set.

        For each point z = [x; v; w] in S, reconstruct the pre-disturbance
        state  x_pre = x - D_top @ w  and the applied input  u = K x_pre + v.
        Keep z only if  lb(x_pre, u) <= w <= ub(x_pre, u)  according to
        the graph-set bounds function.
        """
        V = np.asarray(S.V)
        if V.size == 0:
            return VSet(V, name=name or S.name)

        x = V[:, :self.n]
        v = V[:, self.n:self.n + self.m]
        w = V[:, self.n + self.m:self.n + self.m + self.w]

        x_pre = x - w @ self.D_top.T
        u_pre = (self.K @ x_pre.T).T + v

        lb, ub = Ggraph.bounds_fn(x_pre, u_pre)
        mask = ((w <= ub + Ggraph.tol).all(axis=1)
                & (w >= lb - Ggraph.tol).all(axis=1))
        return VSet(V[mask], name=name or (S.name and f"{S.name}_cap_{Ggraph.name}"))

    def clip_with_graph_gpu(self, S: VSet, Ggraph: GraphSet,
                            name: Optional[str] = None) -> VSet:
        """GPU-accelerated graph-set clipping."""
        V = np.asarray(S.V)
        if V.size == 0:
            return VSet(V, name=name or S.name)

        V_gpu = cp.asarray(V, dtype=cp.float32)
        x = V_gpu[:, :self.n]
        v = V_gpu[:, self.n:self.n + self.m]
        w = V_gpu[:, self.n + self.m:self.n + self.m + self.w]

        D_top_gpu = cp.asarray(self.D_top, dtype=cp.float32)
        K_gpu = cp.asarray(self.K, dtype=cp.float32)

        x_pre = x - w @ D_top_gpu.T
        u_pre = (K_gpu @ x_pre.T).T + v

        x_pre_cpu = cp.asnumpy(x_pre)
        u_pre_cpu = cp.asnumpy(u_pre)
        lb, ub = Ggraph.bounds_fn(x_pre_cpu, u_pre_cpu)

        lb_gpu = cp.asarray(lb, dtype=cp.float32)
        ub_gpu = cp.asarray(ub, dtype=cp.float32)

        mask = (cp.all(w <= ub_gpu + Ggraph.tol, axis=1)
                & cp.all(w >= lb_gpu - Ggraph.tol, axis=1))
        V_clipped = V_gpu[mask]

        return VSet(cp.asnumpy(V_clipped),
                    name=name or (S.name and f"{S.name}_cap_{Ggraph.name}"))

    def F_with_intersection(self, Z, DV, W, G, name="FZ∩G") -> VSet:
        """Full one-step operator: F(Z) = (Ã Z ⊕ B̃ ΔV ⊕ D̃ W) ∩ G.

        G may be a Polytope (halfspace clipping), a GraphSet (bounds-based
        point filtering), or a list/tuple of both.
        """
        base = self.F_no_intersection(Z, DV, W, name="FZ")
        if G is None:
            return base
        out = base
        if isinstance(G, (list, tuple)):
            for g in G:
                if isinstance(g, Polytope):
                    out = self.clip_with_polytope(out, g)
                else:
                    out = self.clip_with_graph(out, g)
            return VSet(out.V, name=name)
        if isinstance(G, Polytope):
            return self.clip_with_polytope(out, G, name=name)
        return self.clip_with_graph(out, G, name=name)


# ---------------------- optional: hull for plotting ----------------------

def hull_for_plot(vset: VSet):
    """
    Compute a convex hull of a vertex cloud for visualization.
    Returns (hull_vertices, scipy_hull) or (vset.V, None) if degenerate.
    """
    from scipy.spatial import ConvexHull
    V = np.asarray(vset.V)
    d = V.shape[1] if V.ndim == 2 else 0
    if V.size == 0 or d == 0 or V.shape[0] < d + 1:
        return V, None
    try:
        hull = ConvexHull(V, qhull_options="QJ")
        return V[hull.vertices], hull
    except Exception:
        return np.unique(V, axis=0), None
