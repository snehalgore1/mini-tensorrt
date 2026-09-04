"""Plot the GEMM benchmark results: the GFLOP/s optimization ladder and a
roofline. Reads the JSON emitted by benchmarks/bench_gemm.

Run:  ./build/benchmarks/bench_gemm --sizes 256,512,1024,2048 --out gemm_results.json
      python python/plot_gemm.py gemm_results.json
Writes: docs/images/gemm_ladder.png, docs/images/roofline.png
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
IMG = os.path.join(ROOT, "docs", "images")

LADDER = ["naive", "reorder", "register", "tiled", "packed", "neon", "threaded"]
P_CORES = 8  # M1 Pro performance cores (for the all-core NEON ceiling)


def load(path):
    with open(path) as f:
        return json.load(f)


def plot_ladder(d, size_idx):
    n = d["sizes"][size_idx]
    vals = [d["variants"][v][size_idx] for v in LADDER]
    accel = d["accelerate"][size_idx]

    labels = LADDER + ["Accelerate"]
    heights = vals + [accel]
    colors = ["#9aa7b8"] * len(LADDER) + ["#e07a5f"]
    colors[LADDER.index("neon")] = "#3d8bfd"
    colors[LADDER.index("threaded")] = "#2a6fdb"

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, heights, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("GFLOP/s (log scale)")
    ax.set_title(f"GEMM optimization ladder  (N={n}, FP32, Apple M1 Pro)")
    for b, h in zip(bars, heights):
        if h and h == h:
            ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(IMG, "gemm_ladder.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)


def plot_roofline(d, size_idx):
    n = d["sizes"][size_idx]
    peak_fma = d["peak_fma_gflops"]
    peak_bw = d["peak_bw_gbps"]
    allcore = peak_fma * P_CORES

    # Estimated arithmetic intensity (FLOP/byte): naive reloads operands every
    # FMA (~0.25); a blocked GEMM reads each element ~once (~N/6).
    ai_naive = 0.25
    ai_blocked = n / 6.0

    import numpy as np
    xs = np.logspace(-2, 3, 200)
    mem = peak_bw * xs  # memory-bound ceiling (GB/s * FLOP/byte = GFLOP/s)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, np.minimum(mem, allcore), "k-", lw=1.5, label="roofline (all-core NEON)")
    ax.axhline(peak_fma, color="#888", ls="--", lw=1, label=f"1-core NEON peak ({peak_fma:.0f})")
    ax.axhline(allcore, color="#444", ls="--", lw=1, label=f"{P_CORES}-core NEON peak ({allcore:.0f})")

    def pt(ai, g, name, color):
        if g and g == g:
            ax.plot(ai, g, "o", color=color, ms=9)
            ax.annotate(f"{name} ({g:.0f})", (ai, g), textcoords="offset points",
                        xytext=(8, 4), fontsize=9)

    pt(ai_naive, d["variants"]["naive"][size_idx], "naive", "#9aa7b8")
    pt(ai_blocked, d["variants"]["neon"][size_idx], "neon", "#3d8bfd")
    pt(ai_blocked, d["variants"]["threaded"][size_idx], "threaded", "#2a6fdb")
    pt(ai_blocked, d["accelerate"][size_idx], "Accelerate (AMX)", "#e07a5f")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte, estimated)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"Roofline  (N={n}, FP32, Apple M1 Pro)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(IMG, "roofline.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)


def largest_idx(d, need):
    """Largest size index where all variants in `need` have a value."""
    for i in range(len(d["sizes"]) - 1, -1, -1):
        if all(d["variants"][v][i] is not None for v in need):
            return i
    return 0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "gemm_results.json")
    d = load(path)
    os.makedirs(IMG, exist_ok=True)
    idx = largest_idx(d, LADDER)  # largest size where every variant (incl. naive) ran
    plot_ladder(d, idx)
    plot_roofline(d, idx)


if __name__ == "__main__":
    main()
