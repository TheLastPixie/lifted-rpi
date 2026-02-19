"""
lifted_rpi -- GPU-accelerated Robust Positively Invariant set computation.

Computes RPI sets for discrete-time LTI systems with state- and
input-dependent disturbances via an outside-in fixed-point iteration
in a lifted (augmented) space.

Reference
---------
A. Ramadan and S. Givigi, "Learning-Based Shrinking Disturbance-Invariant
Tubes for State- and Input-Dependent Uncertainty," IEEE Control Systems
Letters, vol. 9, pp. 2699-2704, 2025, doi: 10.1109/LCSYS.2025.3641128.

Modules
-------
polytope         H-representation convex polytope utilities
minkowski_gpu    JAX + CuPy GPU-accelerated Minkowski sum
vset             Vertex-cloud (VSet), GraphSet, box_corners helpers
engine           LiftedSetOpsGPU_NoHull -- main lifted-space engine
disturbance      Analytical drag disturbance model (make_drag_graphset)
gp_learner       GP-based learned disturbance graph set
convergence      Convergence metrics (support gap, Hausdorff, AABB volume)
iteration        Fixed-point iterator (fixed_point_reach)
initialization   Z0, W, DV construction helpers
simulation       epsilon-MRPI, MPC, trajectory generation, drag simulation
plotting         Visualisation sub-package (hull_3d, publication, convergence,
                 gp_analysis, interactive)
"""

from .polytope import Polytope
from .vset import VSet, GraphSet, box_corners
from .engine import LiftedSetOpsGPU_NoHull
from .convergence import (
    support_gap, support_gap_with_dirs, make_dirs,
    hausdorff_pointcloud, aabb_volume, count_facets,
)
from .iteration import fixed_point_reach
from .disturbance import make_drag_graphset, suggest_W_box_for_drag
from .gp_learner import LearnedGraphSetGP, train_disturbance_gps
from .initialization import (
    build_Z0_inside_G, make_W_from_learned_G_envelope, make_DV_from_u_box,
)
from .simulation import (
    eps_MRPI, generate_trajectory, simulate_trajectory_with_realistic_drag,
    test_realistic_drag,
)
from .speedup import SurrogateGraphSet, build_nystroem_surrogate, build_poly_surrogate

__all__ = [
    # core geometry
    "Polytope",
    "VSet", "GraphSet", "box_corners",
    "LiftedSetOpsGPU_NoHull",
    # convergence
    "support_gap", "support_gap_with_dirs", "make_dirs",
    "hausdorff_pointcloud", "aabb_volume", "count_facets",
    # iteration
    "fixed_point_reach",
    # disturbance
    "make_drag_graphset", "suggest_W_box_for_drag",
    # GP learner
    "LearnedGraphSetGP", "train_disturbance_gps",
    # initialization
    "build_Z0_inside_G", "make_W_from_learned_G_envelope", "make_DV_from_u_box",
    # simulation
    "eps_MRPI", "generate_trajectory", "simulate_trajectory_with_realistic_drag",
    "test_realistic_drag",
    # speedup / surrogates
    "SurrogateGraphSet", "build_nystroem_surrogate", "build_poly_surrogate",
]
