"""Measure the accuracy/size tradeoff of INT8 weight quantization on real GPT-2.

Symmetric per-tensor INT8 quantization of the matmul weight matrices (the same
scheme the C++ MatMulQ kernel implements): scale = max|W| / 127, q = round(W/scale)
clamped to [-127, 127]. We then run the reference forward with the *dequantized*
weights and compare logits to FP32 -- the accuracy half -- and report the size drop.

Run:  python python/quantize_gpt2.py
(reuses export_gpt2_hf.load_hf_gpt2 / reference_forward; needs torch + transformers)
"""

import numpy as np

from export_gpt2_hf import load_hf_gpt2, reference_forward

# The 2D matmul weights (per layer + tied LM head). Embeddings (wte/wpe) stay FP32.
QUANT_SUFFIXES = ("Wq", "Wk", "Wv", "Wo", "Wfc", "Wproj")


def quant_per_tensor(W):
    """Symmetric per-tensor int8; return the dequantized approximation."""
    scale = float(np.abs(W).max()) / 127.0
    if scale == 0.0:
        return W.copy()
    q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return q.astype(np.float32) * scale


def quant_per_channel(W):
    """Symmetric per-output-column int8 (a scale per column of [K,N])."""
    scale = np.abs(W).max(axis=0, keepdims=True) / 127.0  # [1,N]
    scale[scale == 0.0] = 1.0
    q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return q.astype(np.float32) * scale


def is_quantized(name):
    return name == "lm_head" or any(name.endswith("_" + s) for s in QUANT_SUFFIXES)


def eval_scheme(conf, w, ids, fp32, quant):
    wq = {n: (quant(W) if is_quantized(n) else W) for n, W in w.items()}
    out = reference_forward(conf, wq, ids)
    err = np.abs(fp32 - out)
    return float(err.mean()), float(err.max())


def main():
    conf, w, _ = load_hf_gpt2()
    ids = np.arange(10, 10 + 16, dtype=np.int32)
    fp32 = reference_forward(conf, w, ids)

    # Size: quantized matmul weights become int8 + a per-column scale vector.
    fp32_bytes = sum(W.size * 4 for W in w.values())
    int8_bytes = sum((W.size + (W.shape[-1] * 4 if is_quantized(n) else 0))
                     if is_quantized(n) else W.size * 4 for n, W in w.items())
    q_fp32 = sum(W.size * 4 for n, W in w.items() if is_quantized(n))
    q_int8 = sum(W.size + W.shape[-1] * 4 for n, W in w.items() if is_quantized(n))

    pt = eval_scheme(conf, w, ids, fp32, quant_per_tensor)
    pc = eval_scheme(conf, w, ids, fp32, quant_per_channel)

    print("=== INT8 weight quantization on real GPT-2 (124M) ===")
    print("quantized: Wq/Wk/Wv/Wo/Wfc/Wproj + lm_head; embeddings/LayerNorm/biases FP32")
    print(f"model size:   FP32 {fp32_bytes/1e6:6.1f} MB -> INT8 {int8_bytes/1e6:6.1f} MB "
          f"({fp32_bytes/int8_bytes:.2f}x); quantized weights {q_fp32/1e6:.1f} -> "
          f"{q_int8/1e6:.1f} MB ({q_fp32/q_int8:.1f}x)")
    print(f"per-TENSOR  int8 logit error vs FP32:  mean {pt[0]:.4f}  max {pt[1]:.3f}")
    print(f"per-CHANNEL int8 logit error vs FP32:  mean {pc[0]:.4f}  max {pc[1]:.3f}"
          f"   ({pt[1]/pc[1]:.1f}x lower max error than per-tensor)")


if __name__ == "__main__":
    main()
