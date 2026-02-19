"""
Polytope (H-representation) utilities for n-dimensions.

Represents convex polytopes as  P = { x in R^d : A x <= b,  A_eq x = b_eq }.
Supports intersection, feasibility checking, strict interior-point
computation, vertex enumeration (via SciPy HalfspaceIntersection), random
bounded polytope generation, and 2-D/3-D plotting.

The ``Polytope`` dataclass is frozen (immutable after creation).

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull


# ------------------------------- helpers ------------------------------------

def _as_2d(a: ArrayLike) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x


def _normalize_rows(A: np.ndarray, eps: float = 1e-15) -> Tuple[np.ndarray, np.ndarray]:
    """Return A_scaled, row_norms; each row i: A_scaled[i] = A[i] / max(||A[i]||, eps)."""
    A = np.asarray(A, dtype=float)
    norms = np.linalg.norm(A, axis=1)
    norms = np.where(norms < eps, 1.0, norms)
    return A / norms[:, None], norms


# ------------------------------ main class ----------------------------------

@dataclass(frozen=True)
class Polytope:
    """H-representation polytope: { x : Ax <= b, A_eq x = b_eq }."""

    A: np.ndarray               # (m, d)
    b: np.ndarray               # (m,)
    A_eq: Optional[np.ndarray] = None  # (k, d)
    b_eq: Optional[np.ndarray] = None  # (k,)
    tol: float = 1e-9
    name: Optional[str] = None

    def __post_init__(self):
        A = _as_2d(self.A)
        b = np.asarray(self.b, dtype=float).reshape(-1)
        if A.shape[0] != b.shape[0]:
            raise ValueError("A and b row counts must match: A is (m,d), b is (m,)")
        object.__setattr__(self, 'A', A)
        object.__setattr__(self, 'b', b)

        if self.A_eq is not None or self.b_eq is not None:
            if self.A_eq is None or self.b_eq is None:
                raise ValueError("Both A_eq and b_eq must be provided (or neither).")
            A_eq = _as_2d(self.A_eq)
            b_eq = np.asarray(self.b_eq, dtype=float).reshape(-1)
            if A_eq.shape[0] != b_eq.shape[0]:
                raise ValueError("A_eq and b_eq row counts must match.")
            if A_eq.shape[1] != A.shape[1]:
                raise ValueError("A_eq must have same number of columns as A.")
            object.__setattr__(self, 'A_eq', A_eq)
            object.__setattr__(self, 'b_eq', b_eq)

    # ----------------------------- basic props ------------------------------
    @property
    def dim(self) -> int:
        return self.A.shape[1]

    @property
    def m(self) -> int:
        return self.A.shape[0]

    @property
    def k(self) -> int:
        return 0 if self.A_eq is None else self.A_eq.shape[0]

    # ----------------------------- operations -------------------------------
    def intersect(self, other: "Polytope", name: Optional[str] = None) -> "Polytope":
        if self.dim != other.dim:
            raise ValueError("Dimension mismatch in intersection.")

        A = np.vstack([self.A, other.A])
        b = np.concatenate([self.b, other.b])

        if self.A_eq is None and other.A_eq is None:
            A_eq = None; b_eq = None
        else:
            A_eq = np.vstack([x for x in [self.A_eq, other.A_eq] if x is not None])
            b_eq = np.concatenate([x for x in [self.b_eq, other.b_eq] if x is not None])

        return Polytope(A, b, A_eq, b_eq, tol=max(self.tol, other.tol), name=name)

    # ------------------------- feasibility & interior ------------------------
    def is_feasible(self) -> bool:
        res = linprog(c=np.zeros(self.dim), A_ub=self.A, b_ub=self.b,
                      A_eq=self.A_eq, b_eq=self.b_eq, method="highs")
        return res.success

    def interior_point(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Find a *strict* interior point by maximizing a common margin t:
            maximize    t
            subject to  (A_i / ||A_i||) x <= (b_i / ||A_i||) - t    for all i
                        A_eq x = b_eq (if present)
        Returns (x, t). If infeasible or no positive-margin interior exists,
        returns None.
        """
        if self.m == 0:
            if self.A_eq is None or self.k == 0:
                return np.zeros(self.dim), np.inf
            res = linprog(c=np.zeros(self.dim), A_eq=self.A_eq, b_eq=self.b_eq, method="highs")
            if res.success:
                return res.x, np.inf
            return None

        A_scaled, norms = _normalize_rows(self.A)
        b_scaled = self.b / norms

        c = np.zeros(self.dim + 1); c[-1] = -1.0  # maximize t -> minimize -t

        A_ub = np.hstack([A_scaled, np.ones((self.m, 1))])
        b_ub = b_scaled.copy()

        if self.A_eq is not None and self.k > 0:
            A_eq_aug = np.hstack([self.A_eq, np.zeros((self.k, 1))])
            res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq_aug, b_eq=self.b_eq, method="highs")
        else:
            res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, method="highs")

        if not res.success:
            return None
        x = res.x[:-1]
        t = res.x[-1]
        if t <= self.tol:
            return None
        return x, t

    # ----------------------------- membership -------------------------------
    def contains(self, x: ArrayLike, tol: Optional[float] = None) -> bool:
        tol = self.tol if tol is None else tol
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.dim:
            raise ValueError("x has wrong dimension")
        if self.m and np.any(self.A @ x - self.b > tol):
            return False
        if self.k:
            if np.linalg.norm(self.A_eq @ x - self.b_eq, ord=np.inf) > tol:
                return False
        return True

    # ---------------------------- bounds (LP) --------------------------------
    def axis_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-axis min/max via LP. Returns (lower, upper) with shape (d,)."""
        lowers = np.empty(self.dim)
        uppers = np.empty(self.dim)
        for i in range(self.dim):
            ei = np.zeros(self.dim); ei[i] = 1.0
            rmin = linprog(c=ei, A_ub=self.A, b_ub=self.b, A_eq=self.A_eq, b_eq=self.b_eq, method="highs")
            rmax = linprog(c=-ei, A_ub=self.A, b_ub=self.b, A_eq=self.A_eq, b_eq=self.b_eq, method="highs")
            lowers[i] = rmin.x[i] if rmin.success else -np.inf
            uppers[i] = rmax.x[i] if rmax.success else np.inf
        return lowers, uppers

    # -------------------- vertices via halfspace intersection -----------------
    def vertices(self) -> np.ndarray:
        """Enumerate vertices using SciPy's HalfspaceIntersection (requires interior point)."""
        if self.dim < 2:
            l, u = self.axis_bounds()
            return np.array([[l[0]], [u[0]]])

        interior = self.interior_point()
        if interior is None:
            raise RuntimeError("Polytope is infeasible or has no strictly interior point.")
        x0, _ = interior

        halfspaces = np.hstack([self.A, -self.b.reshape(-1, 1)])
        hs = HalfspaceIntersection(halfspaces, x0)
        return hs.intersections

    # -------------------------- random blob factory --------------------------
    @staticmethod
    def random_blob(dim: int, num_facets: int = 64, center: Optional[ArrayLike] = None,
                    radius: float = 1.0, jitter: float = 0.25, seed: Optional[int] = None,
                    name: Optional[str] = None, tol: float = 1e-9) -> "Polytope":
        """
        Create a random *bounded* polytope around `center` by intersecting `num_facets`
        random halfspaces with outward normals sampled uniformly on the sphere.
        """
        rng = np.random.default_rng(seed)
        c = np.zeros(dim) if center is None else np.asarray(center, dtype=float).reshape(-1)
        if c.size != dim:
            raise ValueError("center has wrong dimension")

        G = rng.normal(size=(num_facets, dim))
        G /= np.linalg.norm(G, axis=1, keepdims=True)

        r = radius * (1.0 + jitter * rng.random(num_facets))
        b = G @ c + r
        return Polytope(G, b, tol=tol, name=name)

    # ------------------------------- plotting --------------------------------
    def plot2d(self, ax=None, **scatter_kwargs):
        """Plot a 2D polygon (requires dim=2)."""
        if self.dim != 2:
            raise ValueError("plot2d only supports dim=2")
        import matplotlib.pyplot as plt
        V = self.vertices()
        if V.size == 0:
            raise RuntimeError("No vertices to plot (empty intersection?)")
        hull = ConvexHull(V)
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        pts = V[hull.vertices]
        pts = np.vstack([pts, pts[0]])
        ax.plot(pts[:, 0], pts[:, 1], lw=1.5)
        ax.scatter(V[:, 0], V[:, 1], s=10, **scatter_kwargs)
        ax.set_aspect('equal', 'box')
        ax.set_title(self.name or "Polytope (2D)")
        return ax

    def plot3d(self, ax=None):
        """Plot a 3D polytope via its convex hull of vertices (dim=3)."""
        if self.dim != 3:
            raise ValueError("plot3d only supports dim=3")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        V = self.vertices()
        if V.size == 0:
            raise RuntimeError("No vertices to plot (empty intersection?)")
        hull = ConvexHull(V)
        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=8)
        for tri in hull.simplices:
            T = V[tri]
            T = np.vstack([T, T[0]])
            ax.plot(T[:, 0], T[:, 1], T[:, 2], lw=0.8)
        ax.set_title(self.name or "Polytope (3D)")
        return ax
