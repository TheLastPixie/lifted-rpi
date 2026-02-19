"""
Plotly-based 3D convex-hull visualization for lifted polytope sets.

Builds robust convex hulls per 3-D projection (with coplanar
fall-back), renders Mesh3d + edge overlays, and provides a scatter
fallback if the hull fails entirely.
"""
from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import Dict, Tuple, List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull
from numpy.linalg import svd


# ─────────────── projection helpers ───────────────

def pick_axis_triples(dim: int, k: int = 4) -> List[Tuple[int, int, int]]:
    """Return *k* 3-element index triples from *dim* dimensions."""
    if dim < 3:
        raise ValueError("Need dim >= 3 for 3D projections.")
    triples = list(combinations(range(dim), 3))
    if len(triples) >= k:
        return triples[:k]
    return (triples * ((k + len(triples) - 1) // len(triples)))[:k]


# ─────────────── hull builders ───────────────

def _build_hull_3d(V3: np.ndarray):
    V3u = np.unique(V3, axis=0)
    if V3u.shape[0] < 4:
        raise ValueError("Not enough points for 3D hull")
    hull = ConvexHull(V3u, qhull_options="QJ")
    return V3u, hull.simplices


def _build_hull_coplanar(V3: np.ndarray):
    C = V3.mean(axis=0, keepdims=True)
    X = V3 - C
    U, S, VT = svd(X, full_matrices=False)
    if (S < 1e-12).sum() >= 2 or V3.shape[0] < 3:
        raise ValueError("Too few points for a 2D hull on a plane")
    B = VT[:2, :].T
    P2 = X @ B
    hull2 = ConvexHull(P2, qhull_options="QJ")
    order = hull2.vertices
    if order.size < 3:
        raise ValueError("Degenerate 2D hull")
    poly3 = V3[order]
    tris = [[0, i, i + 1] for i in range(1, poly3.shape[0] - 1)]
    return poly3, np.array(tris, dtype=int)


def build_hull_mesh(V3: np.ndarray):
    """Build convex-hull vertices + simplices, falling back to coplanar."""
    V3 = np.asarray(V3)
    try:
        return _build_hull_3d(V3)
    except Exception:
        return _build_hull_coplanar(V3)


def _edge_coordinates(Vh: np.ndarray, simplices: np.ndarray):
    """Build x/y/z arrays with ``None`` separators for unique triangle edges."""
    edges = set()
    for tri in np.asarray(simplices, dtype=int):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            if u == v:
                continue
            i, j = (u, v) if u < v else (v, u)
            edges.add((i, j))

    xs, ys, zs = [], [], []
    for i, j in sorted(edges):
        xs.extend([Vh[i, 0], Vh[j, 0], None])
        ys.extend([Vh[i, 1], Vh[j, 1], None])
        zs.extend([Vh[i, 2], Vh[j, 2], None])
    return xs, ys, zs


# ─────────────── main plotting function ───────────────

def plot_polytopes_hulls_3d(
    vsets: Dict[str, object],
    dim: int,
    projections: Optional[List[Tuple[int, int, int]]] = None,
    opacity: float = 0.5,
    showlegend_once: bool = True,
    fallback_scatter: bool = True,
    marker_size: int = 2,
    marker_opacity: float = 0.6,
    height: int = 900,
    width: int = 1200,
    title: str = "Convex Hulls (alpha=0.5) in 3D projections",
    draw_edges: bool = True,
    edge_width: int = 2,
):
    """
    Plot named V-sets as Plotly Mesh3d convex hulls across 3-D projections.

    Parameters
    ----------
    vsets : dict[str, VSet]
        Mapping from name to a VSet whose ``.V`` is an (N, dim) array.
    dim : int
        Full dimensionality of the augmented space.
    projections : list of 3-tuples, optional
        Which axis triples to show (defaults to 4 auto-picked triples).
    """
    projs = projections or pick_axis_triples(dim, 4)
    n = len(projs)

    cols = max(1, (n + 1) // 2)
    rows = 2

    specs = [[{"type": "scene"} for _ in range(cols)] for _ in range(rows)]
    titles = [f"axes {p}" for p in projs] + [""] * (rows * cols - n)

    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=titles)

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]
    names = list(vsets.keys())

    for idx, axes in enumerate(projs):
        row = 1 + (idx // cols)
        col = 1 + (idx % cols)

        for j, name in enumerate(names):
            V = np.asarray(vsets[name].V)
            if V.ndim != 2 or V.shape[1] != dim:
                raise ValueError(
                    f"VSet '{name}' has shape {V.shape}, expected (*, {dim})."
                )

            Vp = np.unique(V[:, axes], axis=0)
            if Vp.shape[0] < 3:
                if fallback_scatter and Vp.shape[0] > 0:
                    fig.add_trace(
                        go.Scatter3d(
                            x=Vp[:, 0], y=Vp[:, 1], z=Vp[:, 2],
                            mode="markers",
                            name=name,
                            legendgroup=name,
                            showlegend=(idx == 0) if showlegend_once else True,
                            marker=dict(
                                size=marker_size,
                                opacity=marker_opacity,
                                color=palette[j % len(palette)],
                            ),
                        ),
                        row=row, col=col,
                    )
                continue

            try:
                Vh, simplices = build_hull_mesh(Vp)

                fig.add_trace(
                    go.Mesh3d(
                        x=Vh[:, 0], y=Vh[:, 1], z=Vh[:, 2],
                        i=simplices[:, 0], j=simplices[:, 1], k=simplices[:, 2],
                        name=name,
                        legendgroup=name,
                        showlegend=(idx == 0) if showlegend_once else True,
                        opacity=opacity,
                        color=palette[j % len(palette)],
                        flatshading=True,
                        lighting=dict(
                            ambient=0.7, diffuse=0.9, specular=0.1, roughness=0.9
                        ),
                        showscale=False,
                    ),
                    row=row, col=col,
                )

                if draw_edges:
                    ex, ey, ez = _edge_coordinates(Vh, simplices)
                    fig.add_trace(
                        go.Scatter3d(
                            x=ex, y=ey, z=ez,
                            mode="lines",
                            name=f"{name} edges",
                            legendgroup=name,
                            showlegend=False,
                            line=dict(
                                color=palette[j % len(palette)], width=edge_width
                            ),
                            opacity=1.0,
                            hoverinfo="skip",
                        ),
                        row=row, col=col,
                    )

            except Exception:
                if fallback_scatter:
                    fig.add_trace(
                        go.Scatter3d(
                            x=Vp[:, 0], y=Vp[:, 1], z=Vp[:, 2],
                            mode="markers",
                            name=name,
                            legendgroup=name,
                            showlegend=(idx == 0) if showlegend_once else True,
                            marker=dict(
                                size=marker_size,
                                opacity=marker_opacity,
                                color=palette[j % len(palette)],
                            ),
                        ),
                        row=row, col=col,
                    )

        fig.update_scenes(
            dict(
                xaxis_title=f"x[{axes[0]}]",
                yaxis_title=f"x[{axes[1]}]",
                zaxis_title=f"x[{axes[2]}]",
                aspectmode="data",
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title=title,
        height=height,
        width=width,
        margin=dict(l=10, r=10, b=30, t=50),
        legend=dict(itemsizing="constant"),
    )
    fig.show()
    return fig
