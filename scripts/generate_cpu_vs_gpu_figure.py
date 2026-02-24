#!/usr/bin/env python
"""
Generate a comparison figure: CPU Surrogate vs GPU Surrogate.

Reads both NPZ files and produces a multi-panel figure showing:
  (a) Per-iteration timing breakdown (stacked bars, side by side)
  (b) Cumulative wall-clock time
  (c) Hausdorff convergence overlay (proving identical trajectories)
  (d) Speedup summary bar chart

Requires:
    results/pipeline_paper_exact.npz           (CPU surrogate run)
    results/gpu_bench/pipeline_paper_exact.npz  (GPU surrogate run)
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

    cpu_path = "results/pipeline_paper_exact.npz"
    gpu_path = "results/gpu_bench/pipeline_paper_exact.npz"
    save_dir = "results/figures"
    os.makedirs(save_dir, exist_ok=True)

    for p, label in [(cpu_path, "CPU surrogate"), (gpu_path, "GPU surrogate")]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found ({label} run required).")
            sys.exit(1)

    cpu = dict(np.load(cpu_path, allow_pickle=True))
    gpu = dict(np.load(gpu_path, allow_pickle=True))

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
    c_fwd  = cpu["time_forward"]
    c_clip = cpu["time_clip"]
    c_haus = cpu["hausdorff"]
    c_times = cpu["times"]

    g_fwd  = gpu["time_forward"]
    g_clip = gpu["time_clip"]
    g_haus = gpu["hausdorff"]
    g_times = gpu["times"]

    n_cpu = len(c_times)
    n_gpu = len(g_times)

    # Colours
    COL_CPU_FWD  = "#5B8DB8"
    COL_CPU_CLIP = "#D4726A"
    COL_GPU_FWD  = "#7CC47C"
    COL_GPU_CLIP = "#F5B041"
    COL_CPU      = "#2E86AB"
    COL_GPU      = "#8E44AD"
    COL_TOL      = "#D32F2F"

    # ═══════════════════════════════════════════════════════════════════
    # Figure: 2x2 comparison panel
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0))

    # ── Panel (a): Per-iteration timing bars ──
    ax = axes[0, 0]
    iters_c = np.arange(1, n_cpu + 1)
    iters_g = np.arange(1, n_gpu + 1)
    ax.bar(iters_c - 0.2, c_fwd, width=0.35, color=COL_CPU_FWD,
           label="CPU fwd", alpha=0.85)
    ax.bar(iters_c - 0.2, c_clip, bottom=c_fwd, width=0.35,
           color=COL_CPU_CLIP, label="CPU clip", alpha=0.85)
    ax.bar(iters_g + 0.2, g_fwd, width=0.35, color=COL_GPU_FWD,
           label="GPU fwd", alpha=0.85)
    ax.bar(iters_g + 0.2, g_clip, bottom=g_fwd, width=0.35,
           color=COL_GPU_CLIP, label="GPU clip", alpha=0.85)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time per iteration [s]")
    ax.set_title("(a) Per-iteration timing breakdown")
    ax.legend(loc="upper right", fontsize=6.5, ncol=2, framealpha=0.9)
    ax.set_xlim(0, max(n_cpu, n_gpu) + 1)

    # ── Panel (b): Cumulative wall-clock time ──
    ax = axes[0, 1]
    cum_cpu = np.cumsum(c_times)
    cum_gpu = np.cumsum(g_times)
    ax.plot(np.arange(1, n_cpu + 1), cum_cpu, "o-", color=COL_CPU,
            markersize=2, linewidth=1.0,
            label=f"CPU ({cum_cpu[-1]:.1f} s)")
    ax.plot(np.arange(1, n_gpu + 1), cum_gpu, "s-", color=COL_GPU,
            markersize=2, linewidth=1.0,
            label=f"GPU ({cum_gpu[-1]:.1f} s)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative time [s]")
    ax.set_title("(b) Cumulative wall-clock time")
    ax.legend(loc="upper left", framealpha=0.9)

    # ── Panel (c): Hausdorff convergence overlay ──
    ax = axes[1, 0]
    ax.semilogy(np.arange(1, n_cpu + 1), c_haus, "o-", color=COL_CPU,
                markersize=2, linewidth=0.8, label="CPU surrogate", alpha=0.9)
    ax.semilogy(np.arange(1, n_gpu + 1), g_haus, "x--", color=COL_GPU,
                markersize=3, linewidth=0.8, label="GPU surrogate", alpha=0.9)
    tol_val = 0.025
    ax.axhline(tol_val, ls="--", color=COL_TOL, lw=0.7, alpha=0.8,
               label=f"tol = {tol_val}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Hausdorff (state, normalised)")
    ax.set_title("(c) Convergence trajectory (identical)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(1, max(n_cpu, n_gpu))

    # ── Panel (d): Speedup summary bar chart ──
    ax = axes[1, 1]
    categories = ["Total\ntime", "Avg fwd\ntime/iter", "Avg clip\ntime/iter",
                  "Avg total\ntime/iter"]
    cpu_vals = [cum_cpu[-1], c_fwd.mean(), c_clip.mean(), c_times.mean()]
    gpu_vals = [cum_gpu[-1], g_fwd.mean(), g_clip.mean(), g_times.mean()]
    speedups = [c / g if g > 0 else float("inf")
                for c, g in zip(cpu_vals, gpu_vals)]

    x = np.arange(len(categories))
    w = 0.32
    ax.bar(x - w / 2, cpu_vals, w, color=COL_CPU, alpha=0.85, label="CPU")
    ax.bar(x + w / 2, gpu_vals, w, color=COL_GPU, alpha=0.85, label="GPU")

    for i, sp in enumerate(speedups):
        y_top = max(cpu_vals[i], gpu_vals[i])
        ax.text(x[i], y_top * 1.15, f"{sp:.1f}x", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Time [s]")
    ax.set_title("(d) Speedup summary")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_yscale("log")

    plt.tight_layout(h_pad=1.5, w_pad=1.0)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(save_dir, f"fig_cpu_vs_gpu.{ext}"),
                    dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"  fig_cpu_vs_gpu.{{pdf,png}}")
    plt.close(fig)

    # ── Print summary ──
    print(f"\n  CPU surrogate:  {n_cpu} iters, {cum_cpu[-1]:.1f}s total, "
          f"fwd={c_fwd.mean():.4f}s  clip={c_clip.mean():.4f}s  "
          f"total/iter={c_times.mean():.4f}s")
    print(f"  GPU surrogate:  {n_gpu} iters, {cum_gpu[-1]:.1f}s total, "
          f"fwd={g_fwd.mean():.4f}s  clip={g_clip.mean():.4f}s  "
          f"total/iter={g_times.mean():.4f}s")
    print(f"  Total speedup: {cum_cpu[-1]/cum_gpu[-1]:.1f}x")
    print(f"  Forward speedup: {c_fwd.mean()/g_fwd.mean():.1f}x")
    print(f"  Clip speedup: {c_clip.mean()/g_clip.mean():.1f}x")
    print(f"  Frequency: {1.0/g_times.mean():.1f} Hz (GPU) vs "
          f"{1.0/c_times.mean():.1f} Hz (CPU)")


if __name__ == "__main__":
    main()
