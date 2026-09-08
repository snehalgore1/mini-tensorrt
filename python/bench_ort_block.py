"""ONNX Runtime CPU latency baseline for the GPT-2 block — the `ONNX Runtime CPU`
row of the Week 8 matrix in docs/RESULTS.md, apples-to-apples with the PyTorch and
MiniTensorRT rows (same block: 2 layers, S=128, D=768, H=12, from build_gpt2_params).

The block is exported to ONNX with torch.onnx.export (so the graph provably matches
the PyTorch block — torch does the standard-op decomposition ORT runs), then timed in
ONNX Runtime. This is a reference baseline like the PyTorch row; it does not go through
MiniTensorRT's ONNX frontend (that is proven separately on the MLP in export_onnx.py).

    python python/bench_ort_block.py [--iters N] [--warmup N]

Needs `pip install onnxscript` (torch.onnx's exporter backend). The exported
models/gpt2_block.onnx(.data) is large and regenerated, not committed.
"""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_def import GPT2, build_gpt2_params, sample_gpt2_input, gpt2_forward

HERE = os.path.dirname(__file__)
ONNX_PATH = os.path.join(HERE, "..", "models", "gpt2_block.onnx")


class Gpt2Block(nn.Module):
    """nn.Module mirroring model_def.gpt2_forward so torch.onnx.export can trace it."""

    def __init__(self, params):
        super().__init__()
        for k, v in params.items():
            self.register_buffer(k, torch.from_numpy(v))

    def forward(self, x):
        S, D, H, d = GPT2["S"], GPT2["D"], GPT2["H"], GPT2["d"]
        eps, n_layers = GPT2["eps"], GPT2["n_layers"]
        b = self._buffers
        for i in range(n_layers):
            p = lambda name: b[f"l{i}_{name}"]  # noqa: E731
            h = F.layer_norm(x, [D], p("ln1_g"), p("ln1_b"), eps)
            Q, K, V = h @ p("Wq"), h @ p("Wk"), h @ p("Wv")
            Qh = Q.reshape(S, H, d).permute(1, 0, 2)
            Kh = K.reshape(S, H, d).permute(1, 0, 2)
            Vh = V.reshape(S, H, d).permute(1, 0, 2)
            scores = (Qh @ Kh.transpose(1, 2)) * (1.0 / (d ** 0.5))
            attn = F.softmax(scores, dim=-1)
            ctxM = (attn @ Vh).permute(1, 0, 2).reshape(S, D)
            x1 = x + ctxM @ p("Wo")
            h2 = F.layer_norm(x1, [D], p("ln2_g"), p("ln2_b"), eps)
            m = F.gelu(h2 @ p("W1") + p("b1"), approximate="tanh")
            x = x1 + m @ p("W2") + p("b2")
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    params = build_gpt2_params()
    x = torch.from_numpy(sample_gpt2_input())
    model = Gpt2Block(params).eval()

    with torch.inference_mode():
        torch.onnx.export(model, (x,), ONNX_PATH, input_names=["x"],
                          output_names=["out"], opset_version=17)
    print(f"exported {ONNX_PATH} ({os.path.getsize(ONNX_PATH) // 1024} KB)")

    so = ort.SessionOptions()
    sess = ort.InferenceSession(ONNX_PATH, so, providers=["CPUExecutionProvider"])
    xn = sample_gpt2_input()

    # Correctness: ORT matches the numpy reference forward within FP32 tolerance.
    ort_out = sess.run(["out"], {"x": xn})[0]
    ref = gpt2_forward(params, xn)
    print(f"[CHECK] ONNX Runtime vs reference forward: max abs err "
          f"{np.abs(ort_out - ref).max():.2e}")

    for _ in range(args.warmup):
        sess.run(["out"], {"x": xn})
    samples = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(["out"], {"x": xn})
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[int(0.95 * (len(samples) - 1))]
    print(f"ort_threads={sess.get_session_options().intra_op_num_threads} iters={args.iters}")
    print(f"ONNX Runtime CPU GPT-2 block: p50={p50:.3f} ms  p95={p95:.3f} ms")


if __name__ == "__main__":
    main()
