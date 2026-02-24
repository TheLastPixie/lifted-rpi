"""
Surrogate-accelerated clipping for the lifted-RPI pipeline.

Replaces the expensive sklearn GP ``predict(return_std=True)`` calls inside
``LearnedGraphSetGP.bounds_fn`` with lightweight surrogates that reproduce
the same ``(lb, ub)`` bounds 80--300x faster.

Two surrogate backends are provided:

* **Nystroem** (default) -- Nystroem kernel approximation of the RBF
  component followed by Ridge regression.  Robust to mild extrapolation
  because the RBF basis functions decay gracefully outside the training
  domain.

* **Poly-3** -- Degree-3 polynomial expansion with standardised features
  and Ridge regression.  Faster inference but can diverge on
  out-of-distribution queries.

Both surrogates approximate the four GP surfaces
``(mu_x, mu_y, std_x, std_y)`` and slot into the existing ``GraphSet``
protocol via ``SurrogateGraphSet``, a drop-in replacement for
``LearnedGraphSetGP``.

See ``README.md`` in this directory for a detailed explanation.
"""

from .surrogate import (
    SurrogateGraphSet,
    build_nystroem_surrogate,
    build_poly_surrogate,
)
from . import gpu_ops


def init_gpu(verbose: bool = True) -> bool:
    """Initialise GPU acceleration for the entire pipeline.

    Calls ``gpu_ops.init()`` and, if successful, hooks the GPU
    ``unique_rows`` function into ``VSet.__post_init__`` and the
    engine's Minkowski dedup path so that all subsequent VSet
    and Minkowski operations use the GPU automatically.

    Returns True if GPU is ready.
    """
    ok = gpu_ops.init(verbose=verbose)
    if ok:
        # Hook GPU unique into VSet and engine
        from ..vset import set_gpu_unique
        from .. import engine as _eng
        set_gpu_unique(gpu_ops.unique_rows)
        _eng._gpu_unique_rows = gpu_ops.unique_rows
    return ok


__all__ = [
    "SurrogateGraphSet",
    "build_nystroem_surrogate",
    "build_poly_surrogate",
    "gpu_ops",
    "init_gpu",
]
