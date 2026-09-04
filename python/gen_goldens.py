"""Generate golden .npy fixtures from PyTorch (DESIGN D8).

Per-operator fixtures (inputs + expected output) let each C++ kernel be tested
in isolation against a real reference; the whole-model fixture validates the full
graph. C++ tests load these with a small .npy reader, so the test suite needs no
Python at build/test time.

Run:  python python/gen_goldens.py
Writes: models/goldens/*.npy
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from model_def import build_mlp, sample_input, to_f32

GOLDENS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "goldens")


def save(name, arr):
    path = os.path.join(GOLDENS_DIR, name + ".npy")
    np.save(path, np.ascontiguousarray(arr, dtype=np.float32))


def gen_op_goldens():
    g = torch.Generator().manual_seed(123)

    # Add: identical-shape elementwise.
    a = torch.randn(3, 4, generator=g)
    b = torch.randn(3, 4, generator=g)
    save("op_add_a", to_f32(a))
    save("op_add_b", to_f32(b))
    save("op_add_out", to_f32(a + b))

    # Relu.
    x = torch.randn(3, 4, generator=g)
    save("op_relu_in", to_f32(x))
    save("op_relu_out", to_f32(F.relu(x)))

    # MatMul: A[M,K] @ B[K,N].
    A = torch.randn(2, 3, generator=g)
    B = torch.randn(3, 5, generator=g)
    save("op_matmul_a", to_f32(A))
    save("op_matmul_b", to_f32(B))
    save("op_matmul_out", to_f32(A @ B))

    # Gelu: exact, erf-based.
    x = torch.randn(3, 4, generator=g)
    save("op_gelu_in", to_f32(x))
    save("op_gelu_out", to_f32(F.gelu(x)))

    # Softmax over the last axis.
    x = torch.randn(3, 4, generator=g)
    save("op_softmax_in", to_f32(x))
    save("op_softmax_out", to_f32(torch.softmax(x, dim=-1)))


def gen_model_goldens():
    model = build_mlp()
    x = sample_input()
    with torch.no_grad():
        y = model(x)
    save("mlp_input", to_f32(x))
    save("mlp_output", to_f32(y))


def main():
    os.makedirs(GOLDENS_DIR, exist_ok=True)
    gen_op_goldens()
    gen_model_goldens()
    print(f"wrote golden fixtures to {GOLDENS_DIR}")


if __name__ == "__main__":
    main()
