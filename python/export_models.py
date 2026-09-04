"""Export the MLP to a JSON topology + raw float32 weights blob.

This is the input format for the C++ JSON frontend (DESIGN D3). The JSON
describes tensors (by name) and nodes; weight tensors carry a byte offset/size
into the companion .bin file.

Run:  python python/export_models.py
Writes: models/mlp.json, models/mlp.bin
"""

import json
import os
import struct

import numpy as np

from model_def import TF, build_mlp, build_transformer_params, to_f32

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    model = build_mlp()

    # PyTorch Linear computes x @ W^T + b with W of shape [out, in]. Our MatMul
    # computes A[M,K] @ B[K,N], so we export the transposed weight [in, out] and
    # the bias reshaped to [1, out] (batch=1 -> identical-shape Add, no bcast).
    weights = [
        ("W0", to_f32(model.fc0.weight.t().contiguous()), [8, 16]),
        ("b0", to_f32(model.fc0.bias.reshape(1, 16)), [1, 16]),
        ("W1", to_f32(model.fc1.weight.t().contiguous()), [16, 4]),
        ("b1", to_f32(model.fc1.bias.reshape(1, 4)), [1, 4]),
    ]

    # Pack weights contiguously into the .bin, recording (offset, nbytes).
    blob = bytearray()
    weight_meta = {}
    for name, arr, _shape in weights:
        offset = len(blob)
        data = arr.tobytes(order="C")
        blob += data
        weight_meta[name] = (offset, len(data))

    tensors = []

    def add_tensor(name, shape, kind):
        t = {"name": name, "dtype": "f32", "shape": shape, "kind": kind}
        if kind == "weight":
            off, nbytes = weight_meta[name]
            t["offset"] = off
            t["nbytes"] = nbytes
        tensors.append(t)

    add_tensor("x", [1, 8], "input")
    add_tensor("W0", [8, 16], "weight")
    add_tensor("b0", [1, 16], "weight")
    add_tensor("t0", [1, 16], "intermediate")
    add_tensor("t1", [1, 16], "intermediate")
    add_tensor("t2", [1, 16], "intermediate")
    add_tensor("W1", [16, 4], "weight")
    add_tensor("b1", [1, 4], "weight")
    add_tensor("t3", [1, 4], "intermediate")
    add_tensor("t4", [1, 4], "intermediate")
    add_tensor("out", [1, 4], "output")

    nodes = [
        {"op": "MatMul", "inputs": ["x", "W0"], "outputs": ["t0"], "attrs": {}},
        {"op": "Add", "inputs": ["t0", "b0"], "outputs": ["t1"], "attrs": {}},
        {"op": "Gelu", "inputs": ["t1"], "outputs": ["t2"], "attrs": {}},
        {"op": "MatMul", "inputs": ["t2", "W1"], "outputs": ["t3"], "attrs": {}},
        {"op": "Add", "inputs": ["t3", "b1"], "outputs": ["t4"], "attrs": {}},
        {"op": "Softmax", "inputs": ["t4"], "outputs": ["out"], "attrs": {}},
    ]

    topo = {
        "name": "mlp",
        "weights_file": "mlp.bin",
        "tensors": tensors,
        "nodes": nodes,
        "inputs": ["x"],
        "outputs": ["out"],
    }

    json_path = os.path.join(MODELS_DIR, "mlp.json")
    bin_path = os.path.join(MODELS_DIR, "mlp.bin")
    with open(json_path, "w") as f:
        json.dump(topo, f, indent=2)
    with open(bin_path, "wb") as f:
        f.write(blob)

    print(f"wrote {json_path}")
    print(f"wrote {bin_path} ({len(blob)} bytes)")


def export_transformer():
    """Export the multi-head transformer block to transformer.json + .bin."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    p = build_transformer_params()
    S, D, H, d, hidden = TF["S"], TF["D"], TF["H"], TF["d"], TF["hidden"]

    # Weight blob, in a fixed order.
    order = ["ln1_g", "ln1_b", "Wq", "Wk", "Wv", "Wo",
             "ln2_g", "ln2_b", "W1", "b1", "W2", "b2"]
    blob = bytearray()
    meta = {}
    for name in order:
        off = len(blob)
        data = p[name].tobytes(order="C")
        blob += data
        meta[name] = (off, len(data))

    tensors = []

    def T(name, shape, kind):
        t = {"name": name, "dtype": "f32", "shape": shape, "kind": kind}
        if kind == "weight":
            t["offset"], t["nbytes"] = meta[name]
        tensors.append(t)

    T("x", [S, D], "input")
    T("ln1_g", [D], "weight")
    T("ln1_b", [D], "weight")
    T("h", [S, D], "intermediate")
    for w in ("Wq", "Wk", "Wv", "Wo"):
        T(w, [D, D], "weight")
    for t in ("Q", "K", "V"):
        T(t, [S, D], "intermediate")
    for t in ("Qr", "Kr", "Vr"):
        T(t, [S, H, d], "intermediate")
    for t in ("Qh", "Kh", "Vh"):
        T(t, [H, S, d], "intermediate")
    T("KhT", [H, d, S], "intermediate")
    T("scores", [H, S, S], "intermediate")
    T("scaled", [H, S, S], "intermediate")
    T("attn", [H, S, S], "intermediate")
    T("ctx", [H, S, d], "intermediate")
    T("ctxT", [S, H, d], "intermediate")
    T("ctxM", [S, D], "intermediate")
    T("ao", [S, D], "intermediate")
    T("x1", [S, D], "intermediate")
    T("ln2_g", [D], "weight")
    T("ln2_b", [D], "weight")
    T("h2", [S, D], "intermediate")
    T("W1", [D, hidden], "weight")
    T("b1", [1, hidden], "weight")
    T("m1", [S, hidden], "intermediate")
    T("m1b", [S, hidden], "intermediate")
    T("g", [S, hidden], "intermediate")
    T("W2", [hidden, D], "weight")
    T("b2", [1, D], "weight")
    T("m2", [S, D], "intermediate")
    T("m2b", [S, D], "intermediate")
    T("out", [S, D], "output")

    eps = TF["eps"]
    scale = 1.0 / (d ** 0.5)
    nodes = [
        {"op": "LayerNorm", "inputs": ["x", "ln1_g", "ln1_b"], "outputs": ["h"],
         "attrs": {"eps": eps}},
        {"op": "MatMul", "inputs": ["h", "Wq"], "outputs": ["Q"], "attrs": {}},
        {"op": "MatMul", "inputs": ["h", "Wk"], "outputs": ["K"], "attrs": {}},
        {"op": "MatMul", "inputs": ["h", "Wv"], "outputs": ["V"], "attrs": {}},
        {"op": "Reshape", "inputs": ["Q"], "outputs": ["Qr"], "attrs": {"shape": [S, H, d]}},
        {"op": "Reshape", "inputs": ["K"], "outputs": ["Kr"], "attrs": {"shape": [S, H, d]}},
        {"op": "Reshape", "inputs": ["V"], "outputs": ["Vr"], "attrs": {"shape": [S, H, d]}},
        {"op": "Transpose", "inputs": ["Qr"], "outputs": ["Qh"], "attrs": {"perm": [1, 0, 2]}},
        {"op": "Transpose", "inputs": ["Kr"], "outputs": ["Kh"], "attrs": {"perm": [1, 0, 2]}},
        {"op": "Transpose", "inputs": ["Vr"], "outputs": ["Vh"], "attrs": {"perm": [1, 0, 2]}},
        {"op": "Transpose", "inputs": ["Kh"], "outputs": ["KhT"], "attrs": {"perm": [0, 2, 1]}},
        {"op": "BatchedMatMul", "inputs": ["Qh", "KhT"], "outputs": ["scores"], "attrs": {}},
        {"op": "Scale", "inputs": ["scores"], "outputs": ["scaled"], "attrs": {"scale": scale}},
        {"op": "Softmax", "inputs": ["scaled"], "outputs": ["attn"], "attrs": {}},
        {"op": "BatchedMatMul", "inputs": ["attn", "Vh"], "outputs": ["ctx"], "attrs": {}},
        {"op": "Transpose", "inputs": ["ctx"], "outputs": ["ctxT"], "attrs": {"perm": [1, 0, 2]}},
        {"op": "Reshape", "inputs": ["ctxT"], "outputs": ["ctxM"], "attrs": {"shape": [S, D]}},
        {"op": "MatMul", "inputs": ["ctxM", "Wo"], "outputs": ["ao"], "attrs": {}},
        {"op": "Add", "inputs": ["x", "ao"], "outputs": ["x1"], "attrs": {}},
        {"op": "LayerNorm", "inputs": ["x1", "ln2_g", "ln2_b"], "outputs": ["h2"],
         "attrs": {"eps": eps}},
        {"op": "MatMul", "inputs": ["h2", "W1"], "outputs": ["m1"], "attrs": {}},
        {"op": "Add", "inputs": ["m1", "b1"], "outputs": ["m1b"], "attrs": {}},
        {"op": "Gelu", "inputs": ["m1b"], "outputs": ["g"], "attrs": {}},
        {"op": "MatMul", "inputs": ["g", "W2"], "outputs": ["m2"], "attrs": {}},
        {"op": "Add", "inputs": ["m2", "b2"], "outputs": ["m2b"], "attrs": {}},
        {"op": "Add", "inputs": ["x1", "m2b"], "outputs": ["out"], "attrs": {}},
    ]

    topo = {"name": "transformer_block", "weights_file": "transformer.bin",
            "tensors": tensors, "nodes": nodes, "inputs": ["x"], "outputs": ["out"]}

    with open(os.path.join(MODELS_DIR, "transformer.json"), "w") as f:
        json.dump(topo, f, indent=2)
    with open(os.path.join(MODELS_DIR, "transformer.bin"), "wb") as f:
        f.write(blob)
    print(f"wrote transformer.json + transformer.bin ({len(blob)} bytes)")


if __name__ == "__main__":
    main()
    export_transformer()
