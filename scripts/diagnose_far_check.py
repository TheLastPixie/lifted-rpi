#!/usr/bin/env python
"""Diagnostic: investigate whether GP bounds are ever actually used."""
import numpy as np, joblib, sys
sys.path.insert(0, "src")

d = joblib.load("results/G_learned.joblib")
gp_x, gp_y = d["gp_x"], d["gp_y"]
kdt = d["kdt"]
far_r = d["far_r"]
lb_prior, ub_prior = d["lb_prior"], d["ub_prior"]
k_sigma = d["k"]
min_hw = d["min_hw"]
tol = d["tol"]

npz = np.load("results/pipeline_paper_exact.npz", allow_pickle=True)
V = npz["Zstar"].astype(np.float64)
K = npz["K"]
D_top = np.array([[0,0],[1,0],[0,0],[0,1]], dtype=np.float64)

# Reconstruct x_pre, u_pre (same as engine.clip_with_graph)
x = V[:, :4]; v = V[:, 4:6]; w = V[:, 6:8]
x_pre = x - w @ D_top.T
u_pre = (K @ x_pre.T).T + v
N = len(x_pre)

# Build features WITH default t_index (arange) - as the pipeline does
vx, vy = x_pre[:, 1], x_pre[:, 3]
vmag = np.sqrt(vx**2 + vy**2)
t_default = np.arange(N, dtype=float)  # 0..49999
F_default = np.column_stack([vx, vy, vmag, u_pre[:, 0], u_pre[:, 1], t_default])

# Query KDTree with default t_index
dist_default, _ = kdt.query(F_default, k=1, return_distance=True)
dist_default = dist_default.ravel()
far_default = dist_default > far_r

print("=== WITH DEFAULT t_index = arange(N) [as pipeline does] ===")
print(f"far_radius threshold: {far_r}")
print(f"N total: {N}")
print(f"N far: {far_default.sum()} ({far_default.mean()*100:.1f}%)")
print(f"N not-far: {(~far_default).sum()} ({(~far_default).mean()*100:.1f}%)")
print(f"dist: min={dist_default.min():.1f} median={np.median(dist_default):.1f} max={dist_default.max():.1f}")
print()

# Training t_index range
t_train = gp_x.X_train_[:, 5]
print(f"Training t_index range: {t_train.min():.0f} - {t_train.max():.0f}")
print(f"Query t_index range:    {t_default.min():.0f} - {t_default.max():.0f}")
print(f"Scale of t_index gap:   query max - train max = {t_default.max() - t_train.max():.0f}")
print()

# Feature-by-feature comparison
print("=== FEATURE RANGES: TRAINING vs QUERY ===")
feat_names = ["vx", "vy", "|v|", "ux", "uy", "t_index"]
X_tr = gp_x.X_train_
for i, fn in enumerate(feat_names):
    tr_lo, tr_hi = X_tr[:, i].min(), X_tr[:, i].max()
    q_lo, q_hi = F_default[:, i].min(), F_default[:, i].max()
    tr_rng = tr_hi - tr_lo
    print(f"  {fn:>8}: train=[{tr_lo:>10.3f}, {tr_hi:>10.3f}] rng={tr_rng:>10.3f}"
          f"  query=[{q_lo:>10.3f}, {q_hi:>10.3f}]")
print()

# With fixed t_index in training range
t_mid = float(np.median(t_train))
F_fixed = np.column_stack([vx, vy, vmag, u_pre[:, 0], u_pre[:, 1], np.full(N, t_mid)])
dist_fixed, _ = kdt.query(F_fixed, k=1, return_distance=True)
dist_fixed = dist_fixed.ravel()
far_fixed = dist_fixed > far_r

print(f"=== WITH FIXED t_index = {t_mid:.0f} (median of training) ===")
print(f"N far: {far_fixed.sum()} ({far_fixed.mean()*100:.1f}%)")
print(f"N not-far: {(~far_fixed).sum()} ({(~far_fixed).mean()*100:.1f}%)")
print(f"dist: min={dist_fixed.min():.3f} median={np.median(dist_fixed):.3f} max={dist_fixed.max():.3f}")
print()

# GP predictions for close points (fixed t)
if (~far_fixed).sum() > 0:
    F_close = F_fixed[~far_fixed][:min(1000, (~far_fixed).sum())]
    mu_x, std_x = gp_x.predict(F_close, return_std=True)
    print(f"=== GP PREDICTIONS (not-far points, fixed t) ===")
    print(f"N sampled: {len(F_close)}")
    print(f"mu_x:  [{mu_x.min():.6f}, {mu_x.max():.6f}]")
    print(f"std_x: [{std_x.min():.6f}, {std_x.max():.6f}]")
    hw_x = k_sigma * std_x
    print(f"half-width ({k_sigma}*std): [{hw_x.min():.6f}, {hw_x.max():.6f}]")
    print(f"bounds: [{(mu_x - hw_x).min():.6f}, {(mu_x + hw_x).max():.6f}]")
    print(f"prior box: [{lb_prior[0]:.1f}, {ub_prior[0]:.1f}]")
    print()

# Full simulation on 5k subset to verify final == prior-only
print("=== SIMULATING FULL bounds_fn ON 5k SUBSET ===")
rng = np.random.RandomState(42)
idx5k = rng.choice(N, 5000, replace=False)
F5k = F_default[idx5k]; d5k = dist_default[idx5k]; w5k = w[idx5k]

mu_x, std_x = gp_x.predict(F5k, return_std=True)
mu_y, std_y = gp_y.predict(F5k, return_std=True)

lb = np.stack([mu_x - k_sigma*std_x, mu_y - k_sigma*std_y], axis=1)
ub = np.stack([mu_x + k_sigma*std_x, mu_y + k_sigma*std_y], axis=1)

# min-hw
hw = 0.5 * (ub - lb)
too_narrow = hw < min_hw
if np.any(too_narrow):
    mid = 0.5 * (ub + lb)
    hw[too_narrow] = min_hw
    lb = mid - hw; ub = mid + hw

lb_pre, ub_pre = lb.copy(), ub.copy()

# far fallback
far5k = d5k > far_r
if np.any(far5k):
    lb[far5k] = lb_prior
    ub[far5k] = ub_prior

# clamp
lb = np.maximum(lb, lb_prior)
ub = np.minimum(ub, ub_prior)

# Masks
mask_raw_gp = ((w5k <= ub_pre + tol).all(axis=1) & (w5k >= lb_pre - tol).all(axis=1))
mask_final = ((w5k <= ub + tol).all(axis=1) & (w5k >= lb - tol).all(axis=1))
mask_prior = ((w5k <= ub_prior + tol).all(axis=1) & (w5k >= lb_prior - tol).all(axis=1))

print(f"far on 5k: {far5k.sum()} ({far5k.mean()*100:.1f}%)")
print(f"Kept by raw GP bounds (pre-safeguards): {mask_raw_gp.sum()}")
print(f"Kept by final bounds (far-fallback+clamp): {mask_final.sum()}")
print(f"Kept by prior-box only (skip GP entirely):  {mask_prior.sum()}")
print(f"Final == Prior-only? {(mask_final == mask_prior).all()}")
print()

# Disturbance range
print("=== DISTURBANCE VALUES ===")
print(f"w range: x=[{w[:, 0].min():.6f}, {w[:, 0].max():.6f}]  y=[{w[:, 1].min():.6f}, {w[:, 1].max():.6f}]")
print(f"prior box: lb={lb_prior}  ub={ub_prior}")
print(f"|w| max: {np.abs(w).max():.6f}")
print()

# CRITICAL: Check what happens at INTERMEDIATE iterations
# where the cloud is smaller and t_index might stay in-range
print("=== CHECK ACROSS ITERATIONS (from saved history) ===")
hist = npz.get("history", None)
if hist is not None and hasattr(hist, '__len__'):
    for it_idx in [0, 5, 10, 20, 30, 56]:
        try:
            snap = hist[it_idx] if it_idx < len(hist) else None
            if snap is not None:
                V_it = np.asarray(snap)
                n_it = V_it.shape[0]
                x_it = V_it[:, :4]; v_it = V_it[:, 4:6]; w_it = V_it[:, 6:8]
                xp_it = x_it - w_it @ D_top.T
                up_it = (K @ xp_it.T).T + v_it
                vx_it, vy_it = xp_it[:, 1], xp_it[:, 3]
                vmag_it = np.sqrt(vx_it**2 + vy_it**2)
                t_it = np.arange(n_it, dtype=float)
                F_it = np.column_stack([vx_it, vy_it, vmag_it, up_it[:, 0], up_it[:, 1], t_it])
                d_it, _ = kdt.query(F_it, k=1, return_distance=True)
                d_it = d_it.ravel()
                far_it = (d_it > far_r).sum()
                print(f"  iter {it_idx:>3}: N={n_it:>6}, far={far_it:>6} ({far_it/n_it*100:.1f}%)")
        except Exception as e:
            print(f"  iter {it_idx}: error - {e}")
else:
    print("  No iteration history in NPZ. Checking NPZ keys...")
    print(f"  Available keys: {list(npz.keys())}")
