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

from model_def import build_mlp, to_f32

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


if __name__ == "__main__":
    main()
