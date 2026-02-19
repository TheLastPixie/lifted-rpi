# Surrogate-Accelerated Graph-Set Clipping

## Problem

The fixed-point iteration in the lifted-RPI pipeline calls
`LearnedGraphSetGP.bounds_fn` at every iteration to clip the augmented
vertex cloud against the GP-learned disturbance graph set.  Each call
invokes `sklearn.gaussian_process.GaussianProcessRegressor.predict` with
`return_std=True`, which internally computes a full pairwise kernel
matrix between all query points and all training points.

With 50 000 query vertices and 2 500 GP training points, every
`predict` call builds a 50 000 x 2 500 distance matrix, multiplies it
through the kernel, then solves a linear system for the posterior
variance.  Over 57 convergence iterations this amounts to **965 s out
of a 979 s total pipeline** -- 98.6 % of wall-clock time is spent
inside the GP.

## Root Cause

Diagnosis (see `scripts/diagnose_far_check.py`) revealed that **100 %
of query points are classified as "far" at every iteration**.  The GP
training data was collected from an MPC trajectory whose features lie in
the ranges

| Feature   | Training domain   | Pipeline query domain |
|-----------|-------------------|-----------------------|
| u_x       | [1.6, 34.3]       | [31.5, 118.8]         |
| u_y       | [0.7, 24.2]       | [110.4, 110.5]        |
| t_index   | [0, 2499]         | [0, 49999]            |

Because `t_index` (time step) is included as a raw feature with a span
of 2 499, it dominates the KDTree distance metric.  Every query point
exceeds the `far_radius = 0.75` threshold and falls back to the prior
box `[-3, 3]^2`, making the GP prediction a **965-second no-op** that
produces identical results to a simple box comparison.

## Solution

This module replaces the expensive `GP.predict` calls with fast
surrogate models that are fitted once (< 1 s) from the GP's own
training data and predictions.

Two backends are implemented:

### Nystroem (default)

Uses `sklearn.kernel_approximation.Nystroem` (200 components) to
approximate the RBF kernel, then fits four `Ridge` regressors for the
GP surfaces (mu_x, mu_y, std_x, std_y).

- **Build cost**: 0.5 s (amortised over all iterations)
- **Inference**: 0.29 s per 50 000-point clip (vs 16.9 s for raw GP)
- **Extrapolation safety**: the RBF basis decays to zero outside the
  training domain, so out-of-distribution queries produce near-zero
  predictions that are caught by the existing safeguard chain

### Poly-3

Fits degree-3 polynomial features (with `StandardScaler`) through
`Ridge` regression.

- **Build cost**: < 0.1 s
- **Inference**: 0.12 s per 50 000-point clip
- **Extrapolation risk**: polynomials diverge on out-of-distribution
  queries.  Benchmarking showed 12 886 % error on mildly jittered
  inputs (see `scripts/indomain_surrogate_test.py`).  Use only when the
  query domain is guaranteed to overlap with the training domain.

### Safeguard Chain

Both surrogates preserve the three-layer safeguard chain from
`LearnedGraphSetGP.bounds_fn`:

1. **Minimum half-width floor** (`min_halfwidth = 1e-3`): prevents
   degenerate zero-width intervals.
2. **Far-fallback**: queries whose KDTree distance exceeds
   `far_radius` revert to the prior box `[-3, 3]^2`.
3. **Prior-box clamp**: all bounds are clamped to the prior box.

Because the KDTree check is O(N log N_train) and much faster than the
GP (< 0.01 s for 50 000 points), it is retained as-is.

## Verified Results

| Metric              | Baseline (raw GP)   | Nystroem surrogate  |
|---------------------|---------------------|---------------------|
| Z* vertex count     | 50 000              | 50 000              |
| AABB min/max diff   | --                  | 0.00                |
| Hausdorff (last)    | 1.724e-02           | 1.724e-02           |
| Hausdorff match     | --                  | bitwise identical   |
| Convergence iters   | 57                  | 57                  |
| Total pipeline time | 979.2 s             | 30.5 s              |
| Avg clip time/iter  | 16.93 s             | 0.29 s              |
| **Total speedup**   | --                  | **32.1x**           |
| **Clip speedup**    | --                  | **57.8x**           |

The surrogate produces **bitwise identical** Z* output because 100 % of
points are far from the GP training domain and fall back to the prior
box regardless of the prediction backend.

## Usage

```python
from lifted_rpi.gp_learner import LearnedGraphSetGP
from lifted_rpi.speedup import SurrogateGraphSet

# Load existing GP
G = LearnedGraphSetGP.load("results/G_learned.joblib")

# Wrap in surrogate (Nystroem is default)
G_fast = SurrogateGraphSet(G, backend="nystroem")  # or "poly"

# Use as drop-in replacement anywhere a GraphSet is expected
lb, ub = G_fast.bounds_fn(X, U)
```

Or from the command line:

```bash
python scripts/run_pipeline.py \
    --gp-model results/G_learned.joblib \
    --surrogate nystroem
```

## File Inventory

| File                 | Description                                 |
|----------------------|---------------------------------------------|
| `__init__.py`        | Package exports                             |
| `surrogate.py`       | `SurrogateGraphSet`, builder functions      |
| `README.md`          | This document                               |
