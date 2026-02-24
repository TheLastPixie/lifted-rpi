#!/usr/bin/env python
"""
Generate a comparison figure: baseline (raw GP) vs surrogate (Nystroem).

Reads both NPZ files and produces a multi-panel figure showing:
  - Per-iteration timing (stacked bars, side by side)
  - Cumulative wall-clock time
  - Hausdorff convergence overlay (proving identical trajectories)
  - Summary speedup bar chart

Requires:
    results/pipeline_baseline_raw_gp.npz   (original raw-GP run)
    results/pipeline_paper_exact.npz       (surrogate run)
"""
from __future__ import annotations

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_path = "results/pipeline_baseline_raw_gp.npz"
    surr_path = "results/pipeline_paper_exact.npz"
    save_dir  = "results/figures"
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(base_path):
        print(f"ERROR: {base_path} not found. Need the original raw-GP run.")
        sys.exit(1)
    if not os.path.exists(surr_path):
        print(f"ERROR: {surr_path} not found. Run pipeline with --surrogate first.")
        sys.exit(1)

    base = dict(np.load(base_path, allow_pickle=True))
    surr = dict(np.load(surr_path, allow_pickle=True))

    # ── Apply IEEE-ish style ──
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman",
                        "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.25,
        "axes.grid": True,
    })

    # ── Extract data ──
    b_fwd  = base["time_forward"]
    b_clip = base["time_clip"]
    b_haus = base["hausdorff"]
    b_times = base["times"]

    s_fwd  = surr["time_forward"]
    s_clip = surr["time_clip"]
    s_haus = surr["hausdorff"]
    s_times = surr["times"]

    n_base = len(b_times)
    n_surr = len(s_times)

    COL_BASE_FWD  = "#5B8DB8"
    COL_BASE_CLIP = "#D4726A"
    COL_SURR_FWD  = "#7CC47C"
    COL_SURR_CLIP = "#F5B041"
    COL_BASE      = "#2E86AB"
    COL_SURR      = "#F18F01"
    COL_TOL       = "#D32F2F"

    # ═══════════════════════════════════════════════════════════════════
    # Figure: 2x2 comparison panel
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0))  # IEEE double-column

    # ── Panel (a): Per-iteration timing bars ──
    ax = axes[0, 0]
    iters_b = np.arange(1, n_base + 1)
    iters_s = np.arange(1, n_surr + 1)
    ax.bar(iters_b - 0.2, b_fwd, width=0.35, color=COL_BASE_FWD,
           label="Baseline fwd", alpha=0.85)
    ax.bar(iters_b - 0.2, b_clip, bottom=b_fwd, width=0.35,
           color=COL_BASE_CLIP, label="Baseline clip", alpha=0.85)
    ax.bar(iters_s + 0.2, s_fwd, width=0.35, color=COL_SURR_FWD,
           label="Surrogate fwd", alpha=0.85)
    ax.bar(iters_s + 0.2, s_clip, bottom=s_fwd, width=0.35,
           color=COL_SURR_CLIP, label="Surrogate clip", alpha=0.85)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time per iteration [s]")
    ax.set_title("(a) Per-iteration timing breakdown")
    ax.legend(loc="upper right", fontsize=6.5, ncol=2, framealpha=0.9)
    ax.set_xlim(0, max(n_base, n_surr) + 1)

    # ── Panel (b): Cumulative wall-clock time ──
    ax = axes[0, 1]
    cum_base = np.cumsum(b_times)
    cum_surr = np.cumsum(s_times)
    ax.plot(np.arange(1, n_base + 1), cum_base, "o-", color=COL_BASE,
            markersize=2, linewidth=1.0, label=f"Baseline ({cum_base[-1]:.0f} s)")
    ax.plot(np.arange(1, n_surr + 1), cum_surr, "s-", color=COL_SURR,
            markersize=2, linewidth=1.0, label=f"Surrogate ({cum_surr[-1]:.1f} s)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative time [s]")
    ax.set_title("(b) Cumulative wall-clock time")
    ax.legend(loc="upper left", framealpha=0.9)

    # ── Panel (c): Hausdorff convergence overlay ──
    ax = axes[1, 0]
    ax.semilogy(np.arange(1, n_base + 1), b_haus, "o-", color=COL_BASE,
                markersize=2, linewidth=0.8, label="Baseline", alpha=0.9)
    ax.semilogy(np.arange(1, n_surr + 1), s_haus, "x--", color=COL_SURR,
                markersize=3, linewidth=0.8, label="Surrogate", alpha=0.9)
    tol_val = 0.025
    ax.axhline(tol_val, ls="--", color=COL_TOL, lw=0.7, alpha=0.8,
               label=f"tol = {tol_val}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Hausdorff (state, normalised)")
    ax.set_title("(c) Convergence trajectory (identical)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(1, max(n_base, n_surr))

    # ── Panel (d): Speedup summary bar chart ──
    ax = axes[1, 1]
    categories = ["Total\ntime", "Avg clip\ntime/iter", "Avg total\ntime/iter"]
    base_vals = [cum_base[-1], b_clip.mean(), b_times.mean()]
    surr_vals = [cum_surr[-1], s_clip.mean(), s_times.mean()]
    speedups  = [b / s for b, s in zip(base_vals, surr_vals)]

    x = np.arange(len(categories))
    w = 0.32
    bars_b = ax.bar(x - w/2, base_vals, w, color=COL_BASE, alpha=0.85,
                     label="Baseline")
    bars_s = ax.bar(x + w/2, surr_vals, w, color=COL_SURR, alpha=0.85,
                     label="Surrogate")

    # Add speedup annotations
    for i, sp in enumerate(speedups):
        y_top = max(base_vals[i], surr_vals[i])
        ax.text(x[i], y_top * 1.08, f"{sp:.0f}x", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Time [s]")
    ax.set_title("(d) Speedup summary")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_yscale("log")

    plt.tight_layout(h_pad=1.5, w_pad=1.0)

    # Save
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(save_dir, f"fig_baseline_vs_surrogate.{ext}"),
                    dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"  fig_baseline_vs_surrogate.{{pdf,png}}")
    plt.close(fig)

    # ── Print summary ──
    print(f"\n  Baseline: {n_base} iters, {cum_base[-1]:.1f}s total, "
          f"{b_clip.mean():.3f}s avg clip")
    print(f"  Surrogate: {n_surr} iters, {cum_surr[-1]:.1f}s total, "
          f"{s_clip.mean():.3f}s avg clip")
    print(f"  Total speedup: {cum_base[-1]/cum_surr[-1]:.1f}x")
    print(f"  Clip speedup:  {b_clip.mean()/s_clip.mean():.1f}x")


if __name__ == "__main__":
    main()
