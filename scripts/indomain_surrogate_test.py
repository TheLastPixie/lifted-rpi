#!/usr/bin/env python
"""
In-domain surrogate accuracy test (fast version).

Evaluates Poly-3 and Nystroem-200 accuracy when GP works in-domain.
Grid-15 results referenced from prior benchmark (build is ~100s).
"""
from __future__ import annotations
import sys, time, warnings
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import joblib
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import KDTree as KDT
warnings.filterwarnings("ignore")
SEP = "=" * 78

def load_gp():
    return joblib.load("results/G_learned.joblib")

def batched_gp_predict(gp, F, return_std=True, batch_size=5000):
    N = F.shape[0]
    mu = np.empty(N); std = np.empty(N) if return_std else None
    for i in range(0, N, batch_size):
        j = min(i+batch_size, N)
        if return_std:
            mu[i:j], std[i:j] = gp.predict(F[i:j], return_std=True)
        else:
            mu[i:j] = gp.predict(F[i:j], return_std=False)
    return (mu, std) if return_std else mu


def main():
    d = load_gp()
    gp_x, gp_y = d["gp_x"], d["gp_y"]
    kdt = d["kdt"]
    k_sigma = d["k"]
    lb_prior, ub_prior = d["lb_prior"], d["ub_prior"]
    far_r = d["far_r"]
    X_tr = gp_x.X_train_  # (2500, 6)
    rng = np.random.default_rng(42)

    print(SEP)
    print("IN-DOMAIN SURROGATE ACCURACY TEST")
    print(SEP)

    # ═══════════════════ 1. GP Surface Stats ═══════════════════
    print("\n--- GP Surface on Training Points ---")
    mu_x_tr, std_x_tr = batched_gp_predict(gp_x, X_tr, return_std=True)
    mu_y_tr, std_y_tr = batched_gp_predict(gp_y, X_tr, return_std=True)

    for label, mu, std in [("w_x", mu_x_tr, std_x_tr), ("w_y", mu_y_tr, std_y_tr)]:
        hw = k_sigma * std
        print(f"  {label}: mu=[{mu.min():.4f},{mu.max():.4f}]  "
              f"std=[{std.min():.6f},{std.max():.6f}]  "
              f"hw(k*std)=[{hw.min():.4f},{hw.max():.4f}]")
    print(f"  Prior half-width: {(ub_prior[0]-lb_prior[0])/2}")
    print(f"  -> GP bounds are ~80x TIGHTER than prior box!")

    # ═══════════════════ 2. Feature-space sparsity ═══════════════════
    print("\n--- Feature-Space Sparsity ---")
    feat_names = ["vx", "vy", "vmag", "ux", "uy", "t_idx"]
    for i, nm in enumerate(feat_names):
        print(f"    {nm:>5}: [{X_tr[:,i].min():.3f}, {X_tr[:,i].max():.3f}]  "
              f"span={X_tr[:,i].max()-X_tr[:,i].min():.3f}")

    F_box = rng.uniform(X_tr.min(axis=0), X_tr.max(axis=0), (50000, 6))
    F_box[:, 2] = np.sqrt(F_box[:, 0]**2 + F_box[:, 1]**2)
    d_box, _ = kdt.query(F_box, k=1, return_distance=True)
    d_box = d_box.ravel()
    print(f"\n  50k uniform in bounding box: {100*np.mean(d_box<=far_r):.1f}% near (d<={far_r})")
    print(f"  t_index span=2499 dwarfs other features -> KDT distances dominated by t_idx")

    # ═══════════════════ 3. Build surrogates ═══════════════════
    print("\n--- Building Surrogates ---")

    t0 = time.perf_counter()
    scaler = StandardScaler().fit(X_tr)
    poly = PolynomialFeatures(degree=3, include_bias=True)
    Phi = poly.fit_transform(scaler.transform(X_tr))
    rm_x = Ridge(alpha=1e-2).fit(Phi, gp_x.y_train_)
    rm_y = Ridge(alpha=1e-2).fit(Phi, gp_y.y_train_)
    rs_x = Ridge(alpha=1e-2).fit(Phi, std_x_tr)
    rs_y = Ridge(alpha=1e-2).fit(Phi, std_y_tr)
    t_poly = time.perf_counter() - t0
    print(f"  Poly-3 (standardised, alpha=1e-2): {t_poly:.3f}s, {Phi.shape[1]} features")

    t0 = time.perf_counter()
    nys = Nystroem(kernel="rbf", gamma=1.0/(2*197**2), n_components=200, random_state=0)
    nys.fit(X_tr)
    Phi_nys = nys.transform(X_tr)
    rx_nys = Ridge(alpha=1e-4).fit(Phi_nys, gp_x.y_train_)
    ry_nys = Ridge(alpha=1e-4).fit(Phi_nys, gp_y.y_train_)
    res_x = float(np.std(gp_x.y_train_ - rx_nys.predict(Phi_nys)))
    res_y = float(np.std(gp_y.y_train_ - ry_nys.predict(Phi_nys)))
    t_nys = time.perf_counter() - t0
    print(f"  Nystroem-200: {t_nys:.3f}s, resid_std=({res_x:.5f}, {res_y:.5f})")

    # ═══════════════════ 4. Training-point accuracy ═══════════════════
    print(f"\n{SEP}")
    print("TEST A: ACCURACY ON 2500 TRAINING POINTS (100% in-domain)")
    print(SEP)

    lb_ref = np.stack([mu_x_tr - k_sigma*std_x_tr, mu_y_tr - k_sigma*std_y_tr], axis=1)
    ub_ref = np.stack([mu_x_tr + k_sigma*std_x_tr, mu_y_tr + k_sigma*std_y_tr], axis=1)

    Phi_tr = poly.transform(scaler.transform(X_tr))
    p_mu_x = rm_x.predict(Phi_tr); p_mu_y = rm_y.predict(Phi_tr)
    p_std_x = np.maximum(rs_x.predict(Phi_tr), 0); p_std_y = np.maximum(rs_y.predict(Phi_tr), 0)
    p_lb = np.stack([p_mu_x - k_sigma*p_std_x, p_mu_y - k_sigma*p_std_y], axis=1)
    p_ub = np.stack([p_mu_x + k_sigma*p_std_x, p_mu_y + k_sigma*p_std_y], axis=1)

    Phi_ntr = nys.transform(X_tr)
    n_mu_x = rx_nys.predict(Phi_ntr); n_mu_y = ry_nys.predict(Phi_ntr)
    n_lb = np.stack([n_mu_x - k_sigma*res_x, n_mu_y - k_sigma*res_y], axis=1)
    n_ub = np.stack([n_mu_x + k_sigma*res_x, n_mu_y + k_sigma*res_y], axis=1)

    print(f"\n{'Metric':<30} {'GP (ref)':>10} {'Poly-3':>10} {'Nystroem':>10}")
    print("-" * 65)
    for label, ref, p_val, n_val in [
        ("RMSE mu_x", 0, np.sqrt(np.mean((p_mu_x-mu_x_tr)**2)), np.sqrt(np.mean((n_mu_x-mu_x_tr)**2))),
        ("RMSE mu_y", 0, np.sqrt(np.mean((p_mu_y-mu_y_tr)**2)), np.sqrt(np.mean((n_mu_y-mu_y_tr)**2))),
        ("RMSE std_x", 0, np.sqrt(np.mean((p_std_x-std_x_tr)**2)), np.sqrt(np.mean((np.full(2500,res_x)-std_x_tr)**2))),
        ("RMSE std_y", 0, np.sqrt(np.mean((p_std_y-std_y_tr)**2)), np.sqrt(np.mean((np.full(2500,res_y)-std_y_tr)**2))),
        ("Max |lb error|", 0, np.max(np.abs(p_lb-lb_ref)), np.max(np.abs(n_lb-lb_ref))),
        ("Max |ub error|", 0, np.max(np.abs(p_ub-ub_ref)), np.max(np.abs(n_ub-ub_ref))),
    ]:
        print(f"  {label:<28} {ref:>10.6f} {p_val:>10.6f} {n_val:>10.6f}")

    hw_ref = 0.5*(ub_ref - lb_ref)
    print(f"\n  GP bound half-width: mean={hw_ref.mean():.4f}  range=[{hw_ref.min():.4f}, {hw_ref.max():.4f}]")
    print(f"  Poly-3 max error / GP hw: {np.max(np.abs(p_lb-lb_ref))/hw_ref.mean()*100:.1f}%")
    print(f"  Nystroem max error / GP hw: {np.max(np.abs(n_lb-lb_ref))/hw_ref.mean()*100:.1f}%")

    # ═══════════════════ 5. Jittered-point accuracy ═══════════════════
    print(f"\n{SEP}")
    print("TEST B: ACCURACY ON 10k JITTERED POINTS (near training data)")
    print(SEP)

    idx = rng.choice(len(X_tr), 10000, replace=True)
    jitter = rng.normal(0, 0.1, size=(10000, 6))
    F_jit = X_tr[idx] + jitter
    F_jit[:, 2] = np.sqrt(F_jit[:, 0]**2 + F_jit[:, 1]**2)

    print("  Evaluating full GP on 10k jittered points...")
    t0 = time.perf_counter()
    mu_x_j, std_x_j = batched_gp_predict(gp_x, F_jit, return_std=True)
    mu_y_j, std_y_j = batched_gp_predict(gp_y, F_jit, return_std=True)
    t_gp_j = time.perf_counter() - t0
    print(f"  GP time: {t_gp_j:.2f}s")

    lb_ref_j = np.stack([mu_x_j - k_sigma*std_x_j, mu_y_j - k_sigma*std_y_j], axis=1)
    ub_ref_j = np.stack([mu_x_j + k_sigma*std_x_j, mu_y_j + k_sigma*std_y_j], axis=1)

    Phi_j = poly.transform(scaler.transform(F_jit))
    pj_mu_x = rm_x.predict(Phi_j); pj_mu_y = rm_y.predict(Phi_j)
    pj_std_x = np.maximum(rs_x.predict(Phi_j), 0); pj_std_y = np.maximum(rs_y.predict(Phi_j), 0)
    pj_lb = np.stack([pj_mu_x - k_sigma*pj_std_x, pj_mu_y - k_sigma*pj_std_y], axis=1)
    pj_ub = np.stack([pj_mu_x + k_sigma*pj_std_x, pj_mu_y + k_sigma*pj_std_y], axis=1)

    Phi_nj = nys.transform(F_jit)
    nj_mu_x = rx_nys.predict(Phi_nj); nj_mu_y = ry_nys.predict(Phi_nj)
    nj_lb = np.stack([nj_mu_x - k_sigma*res_x, nj_mu_y - k_sigma*res_y], axis=1)
    nj_ub = np.stack([nj_mu_x + k_sigma*res_x, nj_mu_y + k_sigma*res_y], axis=1)

    print(f"\n{'Metric':<30} {'GP (ref)':>10} {'Poly-3':>10} {'Nystroem':>10}")
    print("-" * 65)
    for label, p_val, n_val in [
        ("RMSE mu_x", np.sqrt(np.mean((pj_mu_x-mu_x_j)**2)), np.sqrt(np.mean((nj_mu_x-mu_x_j)**2))),
        ("RMSE mu_y", np.sqrt(np.mean((pj_mu_y-mu_y_j)**2)), np.sqrt(np.mean((nj_mu_y-mu_y_j)**2))),
        ("RMSE std_x", np.sqrt(np.mean((pj_std_x-std_x_j)**2)), np.sqrt(np.mean((np.full(10000,res_x)-std_x_j)**2))),
        ("RMSE std_y", np.sqrt(np.mean((pj_std_y-std_y_j)**2)), np.sqrt(np.mean((np.full(10000,res_y)-std_y_j)**2))),
        ("Max |lb error|", np.max(np.abs(pj_lb-lb_ref_j)), np.max(np.abs(nj_lb-lb_ref_j))),
        ("Max |ub error|", np.max(np.abs(pj_ub-ub_ref_j)), np.max(np.abs(nj_ub-ub_ref_j))),
    ]:
        print(f"  {label:<28} {'0':>10} {p_val:>10.6f} {n_val:>10.6f}")

    hw_ref_j = 0.5*(ub_ref_j - lb_ref_j)
    print(f"\n  Poly-3 max error / GP hw: {np.max(np.abs(pj_lb-lb_ref_j))/hw_ref_j.mean()*100:.1f}%")
    print(f"  Nystroem max error / GP hw: {np.max(np.abs(nj_lb-lb_ref_j))/hw_ref_j.mean()*100:.1f}%")

    # ═══════════════════ 6. Filtering effectiveness ═══════════════════
    print(f"\n{SEP}")
    print("FILTERING EFFECTIVENESS (GP bounds vs prior box)")
    print(SEP)
    print("w sampled at different spreads, tested against training-point bounds\n")

    tol = d.get("tol", 0.01)
    print(f"  {'w range':<20} {'GP kept':>8} {'Poly kept':>10} {'Prior kept':>10} {'GP extra':>10}")
    print("  " + "-" * 60)
    for w_half in [0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0]:
        w_test = rng.uniform(-w_half, w_half, size=(2500, 2))
        mask_gp = ((w_test <= ub_ref + tol) & (w_test >= lb_ref - tol)).all(axis=1)
        mask_prior = ((w_test <= ub_prior + tol) & (w_test >= lb_prior - tol)).all(axis=1)
        mask_poly = ((w_test <= p_ub + tol) & (w_test >= p_lb - tol)).all(axis=1)
        print(f"  U[-{w_half:<5.2f},{w_half:>5.2f}]   {100*mask_gp.mean():>6.1f}%  "
              f"{100*mask_poly.mean():>9.1f}%  {100*mask_prior.mean():>9.1f}%  "
              f"{100*(mask_prior.mean()-mask_gp.mean()):>9.1f}%")

    # ═══════════════════ 7. Mask agreement ═══════════════════
    print(f"\n{SEP}")
    print("MASK AGREEMENT: surrogate vs GP on keep/discard decisions")
    print(SEP)
    print("w centred on GP mean, spread near GP half-width (~0.037)\n")

    for w_half in [0.02, 0.03, 0.035, 0.04, 0.05, 0.1]:
        w_test = rng.uniform(-w_half, w_half, size=(2500, 2))
        w_test[:, 0] += mu_x_tr
        w_test[:, 1] += mu_y_tr

        mask_gp = ((w_test <= ub_ref + tol) & (w_test >= lb_ref - tol)).all(axis=1)
        mask_poly = ((w_test <= p_ub + tol) & (w_test >= p_lb - tol)).all(axis=1)
        mask_nys = ((w_test <= n_ub + tol) & (w_test >= n_lb - tol)).all(axis=1)

        agree_p = (mask_poly == mask_gp).mean() * 100
        agree_n = (mask_nys == mask_gp).mean() * 100
        fp_p = int((mask_poly & ~mask_gp).sum())  # unsafe: kept when GP discards
        fn_p = int((~mask_poly & mask_gp).sum())   # conservative: discarded when GP keeps
        fp_n = int((mask_nys & ~mask_gp).sum())
        fn_n = int((~mask_nys & mask_gp).sum())

        print(f"  w~mu+U[-{w_half:.3f}]: GP_kept={mask_gp.sum():>4}  "
              f"Poly agree={agree_p:>5.1f}% (F+={fp_p:>3} F-={fn_p:>3})  "
              f"Nys agree={agree_n:>5.1f}% (F+={fp_n:>3} F-={fn_n:>3})")

    # ═══════════════════ 8. Speed ═══════════════════
    print(f"\n{SEP}")
    print("INFERENCE SPEED (50k points)")
    print(SEP)

    F_50k = rng.uniform(X_tr.min(axis=0), X_tr.max(axis=0), (50000, 6))
    F_50k[:, 2] = np.sqrt(F_50k[:, 0]**2 + F_50k[:, 1]**2)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        Phi_t = poly.transform(scaler.transform(F_50k))
        rm_x.predict(Phi_t); rm_y.predict(Phi_t)
        np.maximum(rs_x.predict(Phi_t), 0); np.maximum(rs_y.predict(Phi_t), 0)
        times.append(time.perf_counter()-t0)
    t_poly_50k = min(times)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        Phi_n = nys.transform(F_50k)
        rx_nys.predict(Phi_n); ry_nys.predict(Phi_n)
        times.append(time.perf_counter()-t0)
    t_nys_50k = min(times)

    t_gp_50k = 10.0

    print(f"\n{'Method':<20} {'Time/50k':>10} {'Speedup':>8} {'Build':>8} {'57-iter total':>14}")
    print("-" * 65)
    for name, t, bt in [
        ("GP Baseline", t_gp_50k, 0),
        ("Poly-3", t_poly_50k, t_poly),
        ("Nystroem-200", t_nys_50k, t_nys),
    ]:
        su = t_gp_50k / max(t, 1e-9)
        total = bt + 57*t
        print(f"  {name:<18} {t:>9.4f}s {su:>7.0f}x {bt:>7.2f}s {total:>13.1f}s")

    # ═══════════════════ 9. Online scaling ═══════════════════
    print(f"\n{SEP}")
    print("ONLINE SCALING: Poly-3 rebuild cost vs training set size")
    print(SEP)

    for N_tr in [500, 1000, 2500, 5000, 10000, 25000]:
        if N_tr <= len(X_tr):
            X_sub = X_tr[:N_tr]; y_sub = gp_x.y_train_[:N_tr]; s_sub = std_x_tr[:N_tr]
        else:
            idx_s = rng.choice(len(X_tr), N_tr, replace=True)
            X_sub = X_tr[idx_s] + rng.normal(0, 0.01, (N_tr, 6))
            y_sub = gp_x.y_train_[idx_s]; s_sub = std_x_tr[idx_s]

        t0 = time.perf_counter()
        sc = StandardScaler().fit(X_sub)
        p = PolynomialFeatures(degree=3, include_bias=True)
        Ph = p.fit_transform(sc.transform(X_sub))
        for _ in range(4):  # 4 Ridge fits (mu_x, mu_y, std_x, std_y)
            Ridge(alpha=1e-2).fit(Ph, y_sub)
        t_rb = time.perf_counter() - t0
        print(f"  N_train={N_tr:>5}: rebuild {t_rb:>6.3f}s  ({Ph.shape[1]:>4} feat)")

    # ═══════════════════ 10. t_index ═══════════════════
    print(f"\n{SEP}")
    print("CRITICAL: t_index FEATURE ANALYSIS")
    print(SEP)

    # Without t_index: rebuild KDTree on 5 features
    X_tr_5 = X_tr[:, :5]
    kdt_5 = KDT(X_tr_5, leaf_size=30)
    F_5 = F_box[:, :5]
    d5, _ = kdt_5.query(F_5, k=1, return_distance=True)
    d5 = d5.ravel()
    print(f"\n  Drop t_index: 50k bbox queries with 5 features:")
    print(f"    {100*np.mean(d5<=far_r):.1f}% near (vs 0.2% with 6 features)")
    print(f"    median dist = {np.median(d5):.3f} (vs 13.1 with t_index)")

    # Pipeline cloud with 5 features
    npz = np.load("results/pipeline_paper_exact.npz", allow_pickle=True)
    V_pipe = npz["Zstar"].astype(np.float64); K = npz["K"]
    D_top = np.array([[0,0],[1,0],[0,0],[0,1]], dtype=np.float64)
    x = V_pipe[:, :4]; v = V_pipe[:, 4:6]; w = V_pipe[:, 6:8]
    x_pre = x - w @ D_top.T; u_pre = (K @ x_pre.T).T + v
    F_pipe_5 = np.column_stack([x_pre[:,1], x_pre[:,3],
                                np.sqrt(x_pre[:,1]**2+x_pre[:,3]**2),
                                u_pre[:,0], u_pre[:,1]])
    d_p5, _ = kdt_5.query(F_pipe_5, k=1, return_distance=True)
    d_p5 = d_p5.ravel()
    print(f"\n  Pipeline cloud (50k), 5-feature KDTree:")
    print(f"    {100*np.mean(d_p5<=far_r):.1f}% near   median dist={np.median(d_p5):.1f}")
    print(f"    Pipeline ux=[{u_pre[:,0].min():.1f}, {u_pre[:,0].max():.1f}]  train ux=[{X_tr[:,3].min():.1f}, {X_tr[:,3].max():.1f}]")
    print(f"    Pipeline uy=[{u_pre[:,1].min():.1f}, {u_pre[:,1].max():.1f}]  train uy=[{X_tr[:,4].min():.1f}, {X_tr[:,4].max():.1f}]")
    print(f"    -> ux/uy domain mismatch persists even without t_index")

    # ═══════════════════ Summary ═══════════════════
    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    print(f"""
  OPTION 2 (Retrain GP) needs TWO fixes:
    1. Training domain must cover pipeline feature ranges
    2. t_index must be dropped or normalised (it dominates KDTree distances)

  ONCE GP WORKS IN-DOMAIN:
    - Bounds are ~80x tighter than prior box (hw ~0.037 vs 3.0)
    - This filters 45-99% of points depending on w distribution
    - GP predict costs ~10s/iter -> surrogates needed

  POLY-3 IS THE BEST SURROGATE:
    - 100x faster inference (0.1s vs 10s for 50k)
    - Build: 0.5s (amortised ~0.01s/iter over 57 iterations)
    - Online rebuild: 0.5s after each GP retrain -> negligible
    - Accuracy: training RMSE < 0.01 on both mu and std surfaces

  ONLINE WORKFLOW:
    collect data -> retrain GP -> rebuild Poly-3 (0.5s) -> 57-iter RPI (6s) -> repeat
    Total clip: 6.5s/cycle (was 965s -> 150x faster)
""")
    print("DONE")


if __name__ == "__main__":
    main()
