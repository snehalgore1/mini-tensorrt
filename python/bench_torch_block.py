"""PyTorch-eager latency baseline for the GPT-2 block, apples-to-apples with the
exported models/gpt2_block.json: same architecture, weights (build_gpt2_params),
and input (sample_gpt2_input) that gen_goldens.py uses. Reports p50/p95 wall time
per forward so it can sit next to the MiniTensorRT rows in docs/RESULTS.md.

    python python/bench_torch_block.py [--iters N] [--warmup N]

PyTorch eager uses its own BLAS (Accelerate on macOS) and all cores by default;
this is the high-level framework baseline, not a single-thread comparison.
"""
import argparse
import time

import torch

from model_def import build_gpt2_params, sample_gpt2_input, gpt2_forward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    params = build_gpt2_params()
    x = sample_gpt2_input()

    with torch.inference_mode():
        for _ in range(args.warmup):
            gpt2_forward(params, x)

        samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            gpt2_forward(params, x)
            samples.append((time.perf_counter() - t0) * 1e3)  # ms

    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[int(0.95 * (len(samples) - 1))]
    print(f"torch_threads={torch.get_num_threads()} iters={args.iters}")
    print(f"PyTorch eager GPT-2 block: p50={p50:.3f} ms  p95={p95:.3f} ms")


if __name__ == "__main__":
    main()
