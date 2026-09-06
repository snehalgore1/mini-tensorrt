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

from model_def import (
    TF,
    build_gpt2_params,
    build_mlp,
    build_transformer_params,
    gpt2_forward,
    sample_gpt2_input,
    sample_input,
    sample_transformer_input,
    to_f32,
    transformer_forward,
)

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


def gen_transformer_op_goldens():
    g = torch.Generator().manual_seed(456)

    # LayerNorm over last dim.
    x = torch.randn(4, 8, generator=g)
    gamma = torch.randn(8, generator=g)
    beta = torch.randn(8, generator=g)
    save("op_layernorm_in", to_f32(x))
    save("op_layernorm_gamma", to_f32(gamma))
    save("op_layernorm_beta", to_f32(beta))
    save("op_layernorm_out", to_f32(F.layer_norm(x, [8], gamma, beta, 1e-5)))

    # Scale by 0.25.
    x = torch.randn(3, 4, generator=g)
    save("op_scale_in", to_f32(x))
    save("op_scale_out", to_f32(x * 0.25))

    # Reshape [2,6] -> [3,4] (contiguous).
    x = torch.randn(2, 6, generator=g)
    save("op_reshape_in", to_f32(x))
    save("op_reshape_out", to_f32(x.reshape(3, 4)))

    # Transpose [2,3,4] with perm (1,0,2) -> [3,2,4].
    x = torch.randn(2, 3, 4, generator=g)
    save("op_transpose_in", to_f32(x))
    save("op_transpose_out", to_f32(x.permute(1, 0, 2).contiguous()))

    # BatchedMatMul [2,3,4] @ [2,4,5] -> [2,3,5].
    A = torch.randn(2, 3, 4, generator=g)
    B = torch.randn(2, 4, 5, generator=g)
    save("op_bmm_a", to_f32(A))
    save("op_bmm_b", to_f32(B))
    save("op_bmm_out", to_f32(A @ B))


def gen_transformer_model_goldens():
    params = build_transformer_params()
    x = sample_transformer_input()
    save("tf_input", x)
    save("tf_output", transformer_forward(params, x))


# GPT-specific op goldens. The Gather indices are hard-coded in the C++ test
# (the .npy reader is f32-only), so keep this list in sync there.
GATHER_IDS = [3, 1, 4, 1, 5]


def gen_gpt_op_goldens():
    g = torch.Generator().manual_seed(789)

    # tanh-approx GELU (GPT-2 gelu_new).
    x = torch.randn(3, 4, generator=g)
    save("op_gelutanh_in", to_f32(x))
    save("op_gelutanh_out", to_f32(F.gelu(x, approximate="tanh")))

    # Embedding gather: table[10,4], out = table[ids].
    table = torch.randn(10, 4, generator=g)
    ids = torch.tensor(GATHER_IDS, dtype=torch.long)
    save("op_gather_table", to_f32(table))
    save("op_gather_out", to_f32(table[ids]))

    # CausalSoftmax: [H,S,S] scores, softmax over keys with j>i masked to 0.
    H, S = 2, 4
    scores = torch.randn(H, S, S, generator=g)
    mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
    out = torch.softmax(scores + mask, dim=-1)  # masked rows -> 0 after softmax
    save("op_causalsoftmax_in", to_f32(scores))
    save("op_causalsoftmax_out", to_f32(out))


def gen_gpt2_model_goldens():
    """Whole-model golden for the stacked GPT-2 block. Large (S*D floats) and
    regenerable, so gitignored -- the C++ test that uses it skips when absent."""
    params = build_gpt2_params()
    x = sample_gpt2_input()
    save("gpt2_input", x)
    save("gpt2_output", gpt2_forward(params, x))


def main():
    os.makedirs(GOLDENS_DIR, exist_ok=True)
    gen_op_goldens()
    gen_model_goldens()
    gen_transformer_op_goldens()
    gen_transformer_model_goldens()
    gen_gpt_op_goldens()
    gen_gpt2_model_goldens()
    print(f"wrote golden fixtures to {GOLDENS_DIR}")


if __name__ == "__main__":
    main()
