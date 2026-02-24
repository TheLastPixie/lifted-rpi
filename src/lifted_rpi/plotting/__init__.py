"""Plotting subpackage for the lifted-RPI project.

Provides publication-quality matplotlib figures, Plotly 3-D convex-hull
viewers, GP uncertainty visualisation, and an interactive HTML explorer.
"""

from .hull_3d import (
    pick_axis_triples,
    build_hull_mesh,
    plot_polytopes_hulls_3d,
)
from .publication import (
    parts_from_vset,
    plot_hull_3d,
    plot_hull_2d,
    PUBLICATION_RCPARAMS,
)
from .convergence_plots import (
    plot_convergence_triplet,
    triplet_points,
)
from .gp_analysis import (
    extract_gp_results_for_visualization,
    plot_acceleration_disturbance_spaces,
    plot_2d_projections_acceleration_space,
)
from .interactive import build_interactive_explorer

__all__ = [
    # hull_3d
    "pick_axis_triples",
    "build_hull_mesh",
    "plot_polytopes_hulls_3d",
    # publication
    "parts_from_vset",
    "plot_hull_3d",
    "plot_hull_2d",
    "PUBLICATION_RCPARAMS",
    # convergence
    "plot_convergence_triplet",
    "triplet_points",
    # gp_analysis
    "extract_gp_results_for_visualization",
    "plot_acceleration_disturbance_spaces",
    "plot_2d_projections_acceleration_space",
    # interactive
    "build_interactive_explorer",
]
