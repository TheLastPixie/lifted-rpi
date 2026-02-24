#!/usr/bin/env python
"""
Standalone benchmark: compare clip_with_graph acceleration strategies.

Measures wall-clock time and accuracy (mask agreement) of each approach
against the baseline GP-based bounds_fn on a 5k-point subset (baseline
is ~100x too slow for 50k inline benchmarking).

Solutions tested
----------------
  [Baseline]  Full sklearn GP predict (return_std=True), 2 GPs
  [A]  RegularGridInterpolator over (vx, vy, ux, uy), batched GP build
  [B]  Nystroem kernel approx + Ridge
  [C]  Mean-only GP (return_std=False) + fixed sigma_max
  [D]  Polynomial surrogate (degree 3-4)
  [AC] Grid interpolation for mean only + fixed sigma
  [+noKDT] Variants that skip the per-query KDTree far-check
  [+E] Reduced cloud sizes (5k, 10k, 20k, 50k)

Usage:
    python scripts/bench_clip_solutions.py
"""
from __future__ import annotations

import sys, os, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib
from scipy.interpolate import RegularGridInterpolator

# ────────────────────── helpers ──────────────────────

def load_setup():
    """Load GP model, engine matrices, and a representative cloud."""
    gp_data = joblib.load("results/G_learned.joblib")
    npz = np.load("results/pipeline_paper_exact.npz", allow_pickle=True)

    return dict(
        gp_x=gp_data["gp_x"], gp_y=gp_data["gp_y"], kdt=gp_data["kdt"],
        k_sigma=gp_data["k"], lb_prior=gp_data["lb_prior"],
        ub_prior=gp_data["ub_prior"], min_hw=gp_data["min_hw"],
        far_r=gp_data["far_r"], tol_G=gp_data["tol"],
        K=npz["K"],
        D_top=np.array([[0,0],[1,0],[0,0],[0,1]], dtype=np.float64),
        V=npz["Zstar"].astype(np.float64),
    )


def build_features(X, U, t_index=None):
    """Replicate LearnedGraphSetGP._features."""
    vx, vy = X[:, 1], X[:, 3]
    vmag = np.sqrt(vx**2 + vy**2)
    if t_index is None:
        t_index = np.arange(len(vx), dtype=float)
    return np.column_stack([vx, vy, vmag, U[:, 0], U[:, 1], t_index])


def extract_xuw(V, K, D_top):
    """From V=[x;v;w], get (x_pre, u_pre, w)."""
    x = V[:, :4]; v = V[:, 4:6]; ww = V[:, 6:8]
    x_pre = x - ww @ D_top.T
    u_pre = (K @ x_pre.T).T + v
    return x_pre, u_pre, ww


def safeguards(lb, ub, d_kdt, s):
    """Apply min-halfwidth, far-fallback, prior-clamp."""
    hw = 0.5 * (ub - lb)
    too_narrow = hw < s["min_hw"]
    if np.any(too_narrow):
        mid = 0.5 * (ub + lb)
        hw[too_narrow] = s["min_hw"]
        lb = mid - hw; ub = mid + hw
    far = d_kdt > s["far_r"]
    if np.any(far):
        lb[far] = s["lb_prior"]; ub[far] = s["ub_prior"]
    return np.maximum(lb, s["lb_prior"]), np.minimum(ub, s["ub_prior"])


def safeguards_noKDT(lb, ub, s):
    """Apply min-halfwidth and prior-clamp only (skip KDTree far-check)."""
    hw = 0.5 * (ub - lb)
    too_narrow = hw < s["min_hw"]
    if np.any(too_narrow):
        mid = 0.5 * (ub + lb)
        hw[too_narrow] = s["min_hw"]
        lb = mid - hw; ub = mid + hw
    return np.maximum(lb, s["lb_prior"]), np.minimum(ub, s["ub_prior"])


def clip_mask(ww, lb, ub, tol):
    return ((ww <= ub + tol).all(axis=1) & (ww >= lb - tol).all(axis=1))


def accuracy(m_test, m_ref):
    agree = (m_test == m_ref).sum()
    return dict(
        pct=agree/len(m_ref)*100,
        false_in=int((m_test & ~m_ref).sum()),
        false_out=int((~m_test & m_ref).sum()),
        kept_ref=int(m_ref.sum()), kept_test=int(m_test.sum()),
    )


def batched_gp_predict(gp, F, return_std=True, batch_size=5000):
    """Predict in batches to avoid OOM on large kernel matrices."""
    N = F.shape[0]
    mu_all = np.empty(N)
    std_all = np.empty(N) if return_std else None
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        if return_std:
            mu_all[i:j], std_all[i:j] = gp.predict(F[i:j], return_std=True)
        else:
            mu_all[i:j] = gp.predict(F[i:j], return_std=False)
    return (mu_all, std_all) if return_std else mu_all


# ────────────────────── BASELINE ──────────────────────

def baseline_clip(V, s):
    x_pre, u_pre, ww = extract_xuw(V, s["K"], s["D_top"])
    F = build_features(x_pre, u_pre)
    d, _ = s["kdt"].query(F, k=1, return_distance=True); d = d.ravel()
    mu_x, std_x = s["gp_x"].predict(F, return_std=True)
    mu_y, std_y = s["gp_y"].predict(F, return_std=True)
    lb = np.stack([mu_x - s["k_sigma"]*std_x, mu_y - s["k_sigma"]*std_y], axis=1)
    ub = np.stack([mu_x + s["k_sigma"]*std_x, mu_y + s["k_sigma"]*std_y], axis=1)
    lb, ub = safeguards(lb, ub, d, s)
    return clip_mask(ww, lb, ub, s["tol_G"])


# ────────────────────── A: Grid Interpolation ──────────────────────

def build_grid(s, grid_res=15, t_fixed=1250.0, mean_only=False):
    """Build RegularGridInterpolator lookup tables by batched GP eval."""
    X_tr = s["gp_x"].X_train_
    pad_v, pad_u = 0.15, 1.5
    axes = [
        np.linspace(X_tr[:,0].min()-pad_v, X_tr[:,0].max()+pad_v, grid_res),  # vx
        np.linspace(X_tr[:,1].min()-pad_v, X_tr[:,1].max()+pad_v, grid_res),  # vy
        np.linspace(X_tr[:,3].min()-pad_u, X_tr[:,3].max()+pad_u, grid_res),  # ux
        np.linspace(X_tr[:,4].min()-pad_u, X_tr[:,4].max()+pad_u, grid_res),  # uy
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    shape = mesh[0].shape
    N_grid = mesh[0].size
    vx = mesh[0].ravel(); vy = mesh[1].ravel()
    ux = mesh[2].ravel(); uy = mesh[3].ravel()
    vmag = np.sqrt(vx**2 + vy**2)
    F = np.column_stack([vx, vy, vmag, ux, uy, np.full(N_grid, t_fixed)])

    tag = "mean-only" if mean_only else "mu+std"
    print(f"  Grid {tag}: {grid_res}^4 = {N_grid:,} pts ... ", end="", flush=True)
    t0 = time.perf_counter()
    if mean_only:
        mu_x = batched_gp_predict(s["gp_x"], F, return_std=False)
        mu_y = batched_gp_predict(s["gp_y"], F, return_std=False)
        std_x = std_y = None
    else:
        mu_x, std_x = batched_gp_predict(s["gp_x"], F, return_std=True)
        mu_y, std_y = batched_gp_predict(s["gp_y"], F, return_std=True)
    t1 = time.perf_counter()
    print(f"{t1-t0:.1f}s")

    kw = dict(bounds_error=False, fill_value=None)
    out = dict(
        i_mu_x=RegularGridInterpolator(axes, mu_x.reshape(shape), **kw),
        i_mu_y=RegularGridInterpolator(axes, mu_y.reshape(shape), **kw),
        build_time=t1-t0, grid_res=grid_res, mean_only=mean_only,
    )
    if not mean_only:
        out["i_std_x"] = RegularGridInterpolator(axes, std_x.reshape(shape), **kw)
        out["i_std_y"] = RegularGridInterpolator(axes, std_y.reshape(shape), **kw)
    return out


def grid_clip(V, s, g, sigma_x=None, sigma_y=None, use_kdt=True, t_fixed=1250.0):
    """Clip via grid interpolation."""
    x_pre, u_pre, ww = extract_xuw(V, s["K"], s["D_top"])
    pts = np.column_stack([x_pre[:,1], x_pre[:,3], u_pre[:,0], u_pre[:,1]])
    mu_x = g["i_mu_x"](pts); mu_y = g["i_mu_y"](pts)
    sx = g["i_std_x"](pts) if "i_std_x" in g else np.full(len(pts), sigma_x)
    sy = g["i_std_y"](pts) if "i_std_y" in g else np.full(len(pts), sigma_y)
    lb = np.stack([mu_x - s["k_sigma"]*sx, mu_y - s["k_sigma"]*sy], axis=1)
    ub = np.stack([mu_x + s["k_sigma"]*sx, mu_y + s["k_sigma"]*sy], axis=1)
    if use_kdt:
        F = build_features(x_pre, u_pre, t_index=np.full(len(x_pre), t_fixed))
        d, _ = s["kdt"].query(F, k=1, return_distance=True); d = d.ravel()
        lb, ub = safeguards(lb, ub, d, s)
    else:
        lb, ub = safeguards_noKDT(lb, ub, s)
    return clip_mask(ww, lb, ub, s["tol_G"])


# ────────────────────── B: Nystroem Sparse ──────────────────────

def build_nystroem(s, n_components=200):
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import Ridge
    t0 = time.perf_counter()
    X_tr = s["gp_x"].X_train_
    nys = Nystroem(kernel="rbf", gamma=1.0/(2*197**2),
                   n_components=n_components, random_state=0)
    nys.fit(X_tr)
    Phi = nys.transform(X_tr)
    rx = Ridge(alpha=1e-4).fit(Phi, s["gp_x"].y_train_)
    ry = Ridge(alpha=1e-4).fit(Phi, s["gp_y"].y_train_)
    res_x = float(np.std(s["gp_x"].y_train_ - rx.predict(Phi)))
    res_y = float(np.std(s["gp_y"].y_train_ - ry.predict(Phi)))
    t1 = time.perf_counter()
    print(f"  Nystroem-{n_components}: {t1-t0:.3f}s, resid_std=({res_x:.5f},{res_y:.5f})")
    return dict(nys=nys, rx=rx, ry=ry, res_x=res_x, res_y=res_y, build=t1-t0)


def nystroem_clip(V, s, nd):
    x_pre, u_pre, ww = extract_xuw(V, s["K"], s["D_top"])
    F = build_features(x_pre, u_pre)
    Phi = nd["nys"].transform(F)
    mu_x = nd["rx"].predict(Phi); mu_y = nd["ry"].predict(Phi)
    sx = nd["res_x"]; sy = nd["res_y"]
    lb = np.stack([mu_x - s["k_sigma"]*sx, mu_y - s["k_sigma"]*sy], axis=1)
    ub = np.stack([mu_x + s["k_sigma"]*sx, mu_y + s["k_sigma"]*sy], axis=1)
    d, _ = s["kdt"].query(F, k=1, return_distance=True); d = d.ravel()
    lb, ub = safeguards(lb, ub, d, s)
    return clip_mask(ww, lb, ub, s["tol_G"])


# ────────────────────── C: Mean-only GP ──────────────────────

def meanonly_clip(V, s, sigma_x, sigma_y):
    x_pre, u_pre, ww = extract_xuw(V, s["K"], s["D_top"])
    F = build_features(x_pre, u_pre)
    mu_x = s["gp_x"].predict(F, return_std=False)
    mu_y = s["gp_y"].predict(F, return_std=False)
    lb = np.stack([mu_x - s["k_sigma"]*sigma_x, mu_y - s["k_sigma"]*sigma_y], axis=1)
    ub = np.stack([mu_x + s["k_sigma"]*sigma_x, mu_y + s["k_sigma"]*sigma_y], axis=1)
    d, _ = s["kdt"].query(F, k=1, return_distance=True); d = d.ravel()
    lb, ub = safeguards(lb, ub, d, s)
    return clip_mask(ww, lb, ub, s["tol_G"])


# ────────────────────── D: Polynomial Surrogate ──────────────────────

def build_poly(s, degree=3):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    t0 = time.perf_counter()
    X_tr = s["gp_x"].X_train_
    _, std_x_tr = batched_gp_predict(s["gp_x"], X_tr, return_std=True)
    _, std_y_tr = batched_gp_predict(s["gp_y"], X_tr, return_std=True)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Phi = poly.fit_transform(X_tr)
    ridge_mu_x = Ridge(alpha=1e-6).fit(Phi, s["gp_x"].y_train_)
    ridge_mu_y = Ridge(alpha=1e-6).fit(Phi, s["gp_y"].y_train_)
    ridge_std_x = Ridge(alpha=1e-6).fit(Phi, std_x_tr)
    ridge_std_y = Ridge(alpha=1e-6).fit(Phi, std_y_tr)
    t1 = time.perf_counter()
    # Training RMSE
    rmse_mu = np.sqrt(np.mean((s["gp_x"].y_train_ - ridge_mu_x.predict(Phi))**2))
    rmse_std = np.sqrt(np.mean((std_x_tr - ridge_std_x.predict(Phi))**2))
    print(f"  Poly-{degree}: {Phi.shape[1]} features, {t1-t0:.2f}s, RMSE mu={rmse_mu:.5f} std={rmse_std:.6f}")
    return dict(poly=poly, rm_x=ridge_mu_x, rm_y=ridge_mu_y,
                rs_x=ridge_std_x, rs_y=ridge_std_y, build=t1-t0)


def poly_clip(V, s, pd):
    x_pre, u_pre, ww = extract_xuw(V, s["K"], s["D_top"])
    F = build_features(x_pre, u_pre)
    Phi = pd["poly"].transform(F)
    mu_x = pd["rm_x"].predict(Phi); mu_y = pd["rm_y"].predict(Phi)
    sx = np.maximum(pd["rs_x"].predict(Phi), 0.0)
    sy = np.maximum(pd["rs_y"].predict(Phi), 0.0)
    lb = np.stack([mu_x - s["k_sigma"]*sx, mu_y - s["k_sigma"]*sy], axis=1)
    ub = np.stack([mu_x + s["k_sigma"]*sx, mu_y + s["k_sigma"]*sy], axis=1)
    d, _ = s["kdt"].query(F, k=1, return_distance=True); d = d.ravel()
    lb, ub = safeguards(lb, ub, d, s)
    return clip_mask(ww, lb, ub, s["tol_G"])


# ────────────────────── BENCHMARK RUNNER ──────────────────────

def bench(fn, V, n=3):
    times = []
    for _ in range(n):
        t0 = time.perf_counter(); m = fn(V); times.append(time.perf_counter()-t0)
    return m, min(times), np.mean(times)


def main():
    SEP = "=" * 78
    print(SEP)
    print("CLIPPING ACCELERATION BENCHMARK")
    print(SEP)

    s = load_setup()
    V = s["V"]
    N = V.shape[0]
    rng = np.random.default_rng(42)
    V5k = V[rng.choice(N, 5000, replace=False)]
    V10k = V[rng.choice(N, 10000, replace=False)]
    V20k = V[rng.choice(N, 20000, replace=False)]
    print(f"Cloud: {N:,} pts x {V.shape[1]}d")

    # ── Baseline on 5k ──
    print(f"\n--- Baseline (5k subset) ---")
    t0 = time.perf_counter()
    mask_ref = baseline_clip(V5k, s)
    t_base_5k = time.perf_counter() - t0
    t_base_50k_est = t_base_5k * 10  # linear scaling
    print(f"  Time(5k)={t_base_5k:.3f}s  est(50k)={t_base_50k_est:.1f}s  kept={mask_ref.sum()}/{len(mask_ref)}")

    # ── Pre-compute sigma_max ──
    print("\n--- sigma_max from training data ---")
    _, sx_tr = s["gp_x"].predict(s["gp_x"].X_train_, return_std=True)
    _, sy_tr = s["gp_y"].predict(s["gp_y"].X_train_, return_std=True)
    sig_x, sig_y = float(sx_tr.max()), float(sy_tr.max())
    print(f"  sigma_max: x={sig_x:.6f} y={sig_y:.6f}")

    # ── Build acceleration structures ──
    print("\n--- Build Phase ---")
    g15 = build_grid(s, grid_res=15, mean_only=False)
    g15m = build_grid(s, grid_res=15, mean_only=True)
    g10 = build_grid(s, grid_res=10, mean_only=False)
    g10m = build_grid(s, grid_res=10, mean_only=True)

    nys200 = build_nystroem(s, 200)
    nys500 = build_nystroem(s, 500)

    pd3 = build_poly(s, 3)
    pd4 = build_poly(s, 4)

    # ════════════════════ ACCURACY on 5k ════════════════════
    print(f"\n{SEP}")
    print("ACCURACY vs BASELINE (5k pts, baseline = full GP)")
    print(SEP)

    methods = [
        ("Baseline (full GP)",             lambda V: baseline_clip(V, s)),
        ("A:  Grid-15 mu+std",             lambda V: grid_clip(V, s, g15)),
        ("A:  Grid-10 mu+std",             lambda V: grid_clip(V, s, g10)),
        ("A-noKDT: Grid-15 mu+std",        lambda V: grid_clip(V, s, g15, use_kdt=False)),
        ("AC: Grid-15 mean+fixSig",        lambda V: grid_clip(V, s, g15m, sig_x, sig_y)),
        ("AC-noKDT: Grid-15 mean+fixSig",  lambda V: grid_clip(V, s, g15m, sig_x, sig_y, use_kdt=False)),
        ("AC: Grid-10 mean+fixSig",        lambda V: grid_clip(V, s, g10m, sig_x, sig_y)),
        ("B:  Nystroem-200",               lambda V: nystroem_clip(V, s, nys200)),
        ("B:  Nystroem-500",               lambda V: nystroem_clip(V, s, nys500)),
        ("C:  Mean-only GP + fixSig",      lambda V: meanonly_clip(V, s, sig_x, sig_y)),
        ("D:  Poly-3",                     lambda V: poly_clip(V, s, pd3)),
        ("D:  Poly-4",                     lambda V: poly_clip(V, s, pd4)),
    ]

    print(f"\n{'Method':<36} {'t(5k)':>8} {'Kept':>6} {'Agree':>7} {'F+':>5} {'F-':>5}")
    print("-" * 70)
    for name, fn in methods:
        m, tb, ta = bench(fn, V5k, n=2)
        a = accuracy(m, mask_ref) if name != methods[0][0] else accuracy(m, m)
        tag = "*ref*" if name == methods[0][0] else ""
        print(f"{name:<36} {tb:>7.4f}s {a['kept_test']:>6} {a['pct']:>6.2f}% "
              f"{a['false_in']:>5} {a['false_out']:>5} {tag}")

    # ════════════════════ SPEED on all cloud sizes ════════════════════
    print(f"\n{SEP}")
    print("SPEED BENCHMARK (best of 3)")
    print(SEP)

    fast_methods = [
        ("A:  Grid-15 mu+std",             lambda V: grid_clip(V, s, g15)),
        ("A:  Grid-10 mu+std",             lambda V: grid_clip(V, s, g10)),
        ("A-noKDT: Grid-15",              lambda V: grid_clip(V, s, g15, use_kdt=False)),
        ("AC: Grid-15 mean+fixSig",        lambda V: grid_clip(V, s, g15m, sig_x, sig_y)),
        ("AC-noKDT: Grid-15",             lambda V: grid_clip(V, s, g15m, sig_x, sig_y, use_kdt=False)),
        ("B:  Nystroem-200",               lambda V: nystroem_clip(V, s, nys200)),
        ("B:  Nystroem-500",               lambda V: nystroem_clip(V, s, nys500)),
        ("C:  Mean-only + fixSig",         lambda V: meanonly_clip(V, s, sig_x, sig_y)),
        ("D:  Poly-3",                     lambda V: poly_clip(V, s, pd3)),
        ("D:  Poly-4",                     lambda V: poly_clip(V, s, pd4)),
    ]

    print(f"\n{'Method':<36} {'5k':>8} {'10k':>8} {'20k':>8} {'50k':>8} {'Speedup':>8}")
    print("-" * 78)

    for name, fn in fast_methods:
        _, t5,  _ = bench(fn, V5k,  n=3)
        _, t10, _ = bench(fn, V10k, n=3)
        _, t20, _ = bench(fn, V20k, n=3)
        _, t50, _ = bench(fn, V,    n=3)
        su = t_base_50k_est / max(t50, 1e-9)
        print(f"{name:<36} {t5:>7.4f}s {t10:>7.4f}s {t20:>7.4f}s {t50:>7.4f}s {su:>7.0f}x")

    # ════════════════════ PROJECTED PIPELINE ════════════════════
    print(f"\n{SEP}")
    print("PROJECTED 57-ITERATION PIPELINE TIME")
    print(SEP)
    fwd = 0.25  # avg forward op per iter from saved data
    base_total = 979.2
    print(f"Baseline: {base_total:.0f}s (fwd={fwd*57:.0f}s + clip={t_base_50k_est*57:.0f}s)")
    print()
    for name, fn in fast_methods:
        _, t50, _ = bench(fn, V, n=1)
        total = fwd*57 + t50*57
        su = base_total / total
        print(f"  {name:<36} clip/iter={t50:.4f}s  pipeline={total:>7.1f}s  {su:>6.0f}x")

    # ── Build cost summary ──
    print(f"\n{SEP}")
    print("ONE-TIME BUILD COSTS (amortised over 57 iters)")
    print(SEP)
    builds = [
        ("Grid-15 mu+std",   g15["build_time"]),
        ("Grid-10 mu+std",   g10["build_time"]),
        ("Grid-15 mean-only", g15m["build_time"]),
        ("Grid-10 mean-only", g10m["build_time"]),
        ("Nystroem-200",     nys200["build"]),
        ("Nystroem-500",     nys500["build"]),
        ("Poly-3",           pd3["build"]),
        ("Poly-4",           pd4["build"]),
    ]
    for name, t in builds:
        print(f"  {name:<24} {t:>8.2f}s")

    print(f"\n{SEP}")
    print("DONE")
    print(SEP)


if __name__ == "__main__":
    main()
