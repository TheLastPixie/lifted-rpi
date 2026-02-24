"""Unit tests for lifted_rpi core modules (Phase 1 hardening)."""
from __future__ import annotations

import numpy as np
import pytest

from lifted_rpi.polytope import Polytope
from lifted_rpi.vset import VSet, GraphSet, box_corners, downsample_cloud
from lifted_rpi.convergence import (
    aabb_volume,
    support_gap,
    hausdorff_pointcloud,
    count_facets,
    make_dirs,
    support_gap_with_dirs,
)
from lifted_rpi.engine import LiftedSetOpsGPU_NoHull, hull_for_plot


# ═══════════════════ Polytope ═══════════════════


class TestPolytope:
    """Test H-rep polytope construction, membership, intersection."""

    def test_unit_square(self):
        # 2-D unit square: -1 <= x,y <= 1
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        b = np.array([1, 1, 1, 1], dtype=float)
        P = Polytope(A, b, name="unit_sq")
        assert P.dim == 2
        assert P.m == 4
        assert P.name == "unit_sq"

    def test_contains(self):
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        b = np.ones(4)
        P = Polytope(A, b)
        assert P.contains([0, 0])
        assert P.contains([0.99, 0.99])
        assert not P.contains([1.1, 0])

    def test_intersection(self):
        A1 = np.array([[1, 0], [-1, 0]], dtype=float)
        b1 = np.array([2, 0], dtype=float)  # 0 <= x <= 2
        A2 = np.array([[1, 0], [-1, 0]], dtype=float)
        b2 = np.array([3, -1], dtype=float)  # 1 <= x <= 3
        P1 = Polytope(A1, b1)
        P2 = Polytope(A2, b2)
        P12 = P1.intersect(P2, name="inter")
        # intersection is 1 <= x <= 2
        assert P12.contains([1.5, 0])
        assert not P12.contains([0.5, 0])
        assert not P12.contains([2.5, 0])

    def test_interior_point(self):
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        b = np.ones(4)
        P = Polytope(A, b)
        result = P.interior_point()
        assert result is not None
        x0, t = result
        assert t > 0
        assert P.contains(x0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="row counts"):
            Polytope(np.eye(2), np.ones(3))

    def test_a_eq_without_b_eq_raises(self):
        with pytest.raises(ValueError, match="Both A_eq and b_eq"):
            Polytope(np.eye(2), np.ones(2), A_eq=np.eye(2))

    def test_random_blob(self):
        P = Polytope.random_blob(3, num_facets=20, seed=42)
        assert P.dim == 3
        assert P.m == 20
        assert P.is_feasible()


# ═══════════════════ VSet ═══════════════════


class TestVSet:
    """Test vertex-cloud wrapper."""

    def test_basic(self):
        V = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        vs = VSet(V, name="tri")
        assert vs.dim == 2
        assert vs.nverts == 3
        assert vs.name == "tri"

    def test_dedup(self):
        V = np.array([[1, 2], [1, 2], [3, 4]], dtype=float)
        vs = VSet(V)
        assert vs.nverts == 2  # duplicate removed

    def test_empty(self):
        V = np.zeros((0, 3))
        vs = VSet(V)
        assert vs.nverts == 0
        assert vs.dim == 3

    def test_1d_promoted(self):
        """A 1-D vector should be promoted to (1, d)."""
        vs = VSet(np.array([1.0, 2.0, 3.0]))
        assert vs.V.ndim == 2
        assert vs.nverts == 1
        assert vs.dim == 3

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="dtype"):
            VSet(np.eye(2), dtype="int32")

    def test_3d_array_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            VSet(np.ones((2, 2, 2)))


# ═══════════════════ box_corners ═══════════════════


class TestBoxCorners:
    def test_2d(self):
        V = box_corners(np.array([-1, -2]), np.array([1, 2]))
        assert V.shape == (4, 2)  # 2^2 corners
        assert V.min(axis=0)[0] == pytest.approx(-1.0)
        assert V.max(axis=0)[1] == pytest.approx(2.0)

    def test_3d(self):
        V = box_corners(np.zeros(3), np.ones(3))
        assert V.shape == (8, 3)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            box_corners(np.array([0, 0]), np.array([1, 1, 1]))

    def test_lb_gt_ub_raises(self):
        with pytest.raises(ValueError, match="<= ub"):
            box_corners(np.array([2.0]), np.array([1.0]))


# ═══════════════════ downsample_cloud ═══════════════════


class TestDownsample:
    def test_noop_when_small(self):
        V = np.random.default_rng(0).normal(size=(10, 2)).astype(np.float32)
        V2 = downsample_cloud(V, max_points=100)
        assert V2.shape[0] <= 100

    def test_grid_reduces(self):
        V = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
        V2 = downsample_cloud(V, max_points=50, method="grid")
        assert V2.shape[0] <= 50 + 5  # some slack for grid snapping

    def test_random_reduces(self):
        V = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
        V2 = downsample_cloud(V, max_points=50, method="random")
        assert V2.shape[0] == 50

    def test_stride_reduces(self):
        V = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
        V2 = downsample_cloud(V, max_points=50, method="stride")
        assert V2.shape[0] <= 50


# ═══════════════════ Convergence metrics ═══════════════════


class TestConvergence:
    def test_aabb_volume_box(self):
        V = box_corners(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        vol = aabb_volume(V)
        assert vol == pytest.approx(8.0, rel=1e-6)

    def test_aabb_volume_empty(self):
        assert aabb_volume(np.zeros((0, 3))) == 0.0

    def test_support_gap_identical(self):
        rng = np.random.default_rng(42)
        V = rng.normal(size=(50, 3)).astype(np.float32)
        assert support_gap(V, V) == pytest.approx(0.0, abs=1e-6)

    def test_support_gap_different(self):
        V1 = np.array([[0, 0], [1, 0]], dtype=float)
        V2 = np.array([[0, 0], [2, 0]], dtype=float)
        gap = support_gap(V1, V2)
        assert gap > 0

    def test_hausdorff_identical(self):
        V = np.array([[0, 0], [1, 1]], dtype=float)
        assert hausdorff_pointcloud(V, V, normalize=False) == pytest.approx(0.0)

    def test_hausdorff_shifted(self):
        V1 = np.array([[0, 0], [1, 0]], dtype=float)
        V2 = V1 + 0.5
        hd = hausdorff_pointcloud(V1, V2, normalize=False)
        expected = np.sqrt(0.5)  # shift by [0.5, 0.5]
        assert hd == pytest.approx(expected, rel=1e-6)

    def test_make_dirs_shape(self):
        D = make_dirs(4, q=64, seed=0)
        assert D.shape == (64, 4)
        # rows should be unit-length
        norms = np.linalg.norm(D, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_support_gap_with_dirs(self):
        D = make_dirs(2, q=32, seed=0)
        V = np.array([[0, 0], [1, 1]], dtype=np.float32)
        gap = support_gap_with_dirs(V, V, D)
        assert gap == pytest.approx(0.0, abs=1e-5)

    def test_count_facets_cube(self):
        V = box_corners(np.zeros(3), np.ones(3))
        nf = count_facets(V)
        assert nf > 0  # a cube has 12 triangular facets (or 6 quads → 12 simplices)


# ═══════════════════ Engine construction & validation ═══════════════════


def _make_simple_engine(n=2, m=1, w=1, **kw):
    """Tiny 2-state, 1-input, 1-disturbance engine for testing."""
    A = np.eye(n) * 0.9
    B = np.ones((n, m)) * 0.1
    K = -0.5 * np.ones((m, n))
    return LiftedSetOpsGPU_NoHull(A=A, B=B, K=K, n=n, m=m, w=w, **kw)


class TestEngine:
    def test_construction(self):
        eng = _make_simple_engine()
        assert eng.n_aug == 4  # n+m+w = 2+1+1
        assert eng.A_tilde.shape == (4, 4)
        assert eng.B_tilde.shape == (4, 1)
        assert eng.D_tilde.shape == (4, 1)

    def test_paper_exact(self):
        eng = LiftedSetOpsGPU_NoHull.paper_exact(
            A=np.eye(2) * 0.9, B=np.ones((2, 1)) * 0.1,
            K=-0.5 * np.ones((1, 2)), n=2, m=1, w=1,
        )
        assert eng.alpha_v == 1.0
        assert eng.beta_w == 0.0

    def test_A_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="A must be"):
            LiftedSetOpsGPU_NoHull(
                A=np.eye(3), B=np.ones((2, 1)), K=np.ones((1, 2)),
                n=2, m=1, w=1,
            )

    def test_B_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="B must be"):
            LiftedSetOpsGPU_NoHull(
                A=np.eye(2), B=np.ones((3, 1)), K=np.ones((1, 2)),
                n=2, m=1, w=1,
            )

    def test_K_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="K must be"):
            LiftedSetOpsGPU_NoHull(
                A=np.eye(2), B=np.ones((2, 1)), K=np.ones((2, 2)),
                n=2, m=1, w=1,
            )

    def test_dist_rows_mismatch_raises(self):
        with pytest.raises(ValueError, match="dist_rows"):
            LiftedSetOpsGPU_NoHull(
                A=np.eye(2), B=np.ones((2, 1)), K=np.ones((1, 2)),
                n=2, m=1, w=1, dist_rows=(0, 1),  # len=2 but w=1
            )

    def test_dist_rows_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            LiftedSetOpsGPU_NoHull(
                A=np.eye(2), B=np.ones((2, 1)), K=np.ones((1, 2)),
                n=2, m=1, w=1, dist_rows=(5,),
            )

    def test_Edist_shape_raises(self):
        with pytest.raises(ValueError, match="Edist must be"):
            LiftedSetOpsGPU_NoHull(
                A=np.eye(2), B=np.ones((2, 1)), K=np.ones((1, 2)),
                n=2, m=1, w=1, Edist=np.ones((3, 1)),
            )

    def test_linmap(self):
        eng = _make_simple_engine()
        V = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        S = VSet(V)
        mapped = eng.linmap(S, eng.A_tilde, name="test")
        assert mapped.name == "test"
        assert mapped.dim == 4

    def test_F_no_intersection(self):
        eng = _make_simple_engine()
        Z = VSet(box_corners(-np.ones(4), np.ones(4)))
        DV = VSet(box_corners(-0.1 * np.ones(1), 0.1 * np.ones(1)))
        W = VSet(box_corners(-0.1 * np.ones(1), 0.1 * np.ones(1)))
        result = eng.F_no_intersection(Z, DV, W)
        assert result.nverts > 0


# ═══════════════════ hull_for_plot ═══════════════════


class TestHullForPlot:
    def test_cube_hull(self):
        V = box_corners(np.zeros(3), np.ones(3))
        vs = VSet(V)
        verts, hull = hull_for_plot(vs)
        assert hull is not None
        assert verts.shape[0] == 8

    def test_empty(self):
        vs = VSet(np.zeros((0, 3)))
        verts, hull = hull_for_plot(vs)
        assert hull is None


# ═══════════════════ GraphSet ═══════════════════


class TestGraphSet:
    def test_basic(self):
        def bounds(X, U):
            lb = -np.ones((X.shape[0], 1))
            ub = np.ones((X.shape[0], 1))
            return lb, ub

        gs = GraphSet(bounds_fn=bounds, name="test_G", tol=1e-6)
        assert gs.name == "test_G"
        lb, ub = gs.bounds_fn(np.zeros((5, 2)), np.zeros((5, 1)))
        assert lb.shape == (5, 1)


# ═══════════════════ __version__ ═══════════════════


class TestVersion:
    def test_version_exists(self):
        import lifted_rpi
        assert hasattr(lifted_rpi, "__version__")
        assert isinstance(lifted_rpi.__version__, str)
        assert len(lifted_rpi.__version__) > 0


# ═══════════════════ Initialization validation ═══════════════════


class TestInitialization:
    def test_make_W_w_not_2_raises(self):
        from lifted_rpi.initialization import make_W_from_learned_G_envelope

        eng = _make_simple_engine(n=2, m=1, w=3)

        class FakeG:
            def bounds_fn(self, X, U):
                return -np.ones((X.shape[0], 3)), np.ones((X.shape[0], 3))

        with pytest.raises(NotImplementedError, match="w=2"):
            make_W_from_learned_G_envelope(eng, FakeG())


# ═══════════════════ Iteration validation ═══════════════════


class TestIteration:
    def test_invalid_metric_raises(self):
        from lifted_rpi.iteration import fixed_point_reach

        eng = _make_simple_engine()
        Z0 = VSet(box_corners(-np.ones(4), np.ones(4)))
        DV = VSet(box_corners(-0.1 * np.ones(1), 0.1 * np.ones(1)))
        W = VSet(box_corners(-0.1 * np.ones(1), 0.1 * np.ones(1)))

        A_g = np.vstack([np.eye(4), -np.eye(4)])
        b_g = 5 * np.ones(8)
        G = Polytope(A_g, b_g)

        with pytest.raises(ValueError, match="convergence_metric"):
            fixed_point_reach(
                eng, Z0, DV, W, G,
                convergence_metric="bogus",
                max_iters=1,
            )

    def test_negative_max_iters_raises(self):
        from lifted_rpi.iteration import fixed_point_reach

        eng = _make_simple_engine()
        Z0 = VSet(box_corners(-np.ones(4), np.ones(4)))
        DV = VSet(box_corners(-0.1 * np.ones(1), 0.1 * np.ones(1)))
        W = VSet(box_corners(-0.1 * np.ones(1), 0.1 * np.ones(1)))

        A_g = np.vstack([np.eye(4), -np.eye(4)])
        b_g = 5 * np.ones(8)
        G = Polytope(A_g, b_g)

        with pytest.raises(ValueError, match="max_iters"):
            fixed_point_reach(
                eng, Z0, DV, W, G,
                max_iters=0,
            )

    def test_one_iteration_runs(self):
        """Smoke test: a single iteration should complete without error."""
        from lifted_rpi.iteration import fixed_point_reach

        eng = _make_simple_engine()
        Z0 = VSet(box_corners(-np.ones(4), np.ones(4)))
        DV = VSet(box_corners(-0.1 * np.ones(1), 0.1 * np.ones(1)))
        W = VSet(box_corners(-0.1 * np.ones(1), 0.1 * np.ones(1)))

        A_g = np.vstack([np.eye(4), -np.eye(4)])
        b_g = 5 * np.ones(8)
        G = Polytope(A_g, b_g)

        history, stats = fixed_point_reach(
            eng, Z0, DV, W, G,
            max_iters=1,
            verbose=False,
            track_facets=False,
            track_violations=False,
            track_cost=False,
        )
        assert len(history) == 2  # Z0 + Z1
        assert stats["iters"] == 1
        assert stats["termination_reason"] == "max_iters_reached"
