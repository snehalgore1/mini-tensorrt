"""Export the shared MLP to a real .onnx file and generate an ONNX Runtime
reference output — the fixtures for the C++ ONNX frontend test (Week 6).

The ONNX graph is emitted with `onnx.helper` using ops that map 1:1 onto
MiniTensorRT kernels (MatMul, Add, Gelu, Softmax), from the *same* seeded weights
as the JSON export (model_def.build_mlp). So models/mlp.onnx and models/mlp.json
are the identical function, and the SAME .onnx runs in both ONNX Runtime and our
runtime — an apples-to-apples frontend-agnostic check.

`onnx.shape_inference.infer_shapes` is run so every intermediate carries a static
shape (MiniTensorRT resolves all shapes at load and does no C++ shape inference).

Run:  python python/export_onnx.py
Writes: models/mlp.onnx, models/goldens/onnx_mlp_output.npy
"""
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, helper, shape_inference

from model_def import build_mlp, sample_input, to_f32

HERE = os.path.dirname(__file__)
MODELS = os.path.join(HERE, "..", "models")
GOLDENS = os.path.join(MODELS, "goldens")


def f32_initializer(name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return helper.make_tensor(name, TensorProto.FLOAT, list(arr.shape),
                              arr.tobytes(), raw=True)


def build_onnx_mlp():
    m = build_mlp()
    sd = m.state_dict()
    # nn.Linear stores weight [out, in]; our MatMul computes x @ W, so W = weightᵀ.
    # Bias stored as [1, out] so the runtime's Add is elementwise (batch=1, DESIGN D1).
    W0 = to_f32(sd["fc0.weight"].T)          # [8, 16]
    b0 = to_f32(sd["fc0.bias"]).reshape(1, 16)
    W1 = to_f32(sd["fc1.weight"].T)          # [16, 4]
    b1 = to_f32(sd["fc1.bias"]).reshape(1, 4)

    inits = [f32_initializer("W0", W0), f32_initializer("b0", b0),
             f32_initializer("W1", W1), f32_initializer("b1", b1)]
    nodes = [
        helper.make_node("MatMul", ["x", "W0"], ["t0"]),
        helper.make_node("Add", ["t0", "b0"], ["t1"]),
        helper.make_node("Gelu", ["t1"], ["t2"]),          # opset-20, erf-based
        helper.make_node("MatMul", ["t2", "W1"], ["t3"]),
        helper.make_node("Add", ["t3", "b1"], ["t4"]),
        helper.make_node("Softmax", ["t4"], ["out"], axis=-1),
    ]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(nodes, "mlp", [x], [out], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model = shape_inference.infer_shapes(model)
    checker.check_model(model)
    return model


def main():
    model = build_onnx_mlp()
    onnx_path = os.path.join(MODELS, "mlp.onnx")
    onnx.save(model, onnx_path)
    print(f"wrote {onnx_path} ({os.path.getsize(onnx_path)} bytes, "
          f"{len(model.graph.node)} nodes)")

    # ONNX Runtime reference on the same input the C++ test uses (mlp_input golden).
    x = sample_input().numpy().astype(np.float32)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(["out"], {"x": x})[0].astype(np.float32)
    np.save(os.path.join(GOLDENS, "onnx_mlp_output.npy"),
            np.ascontiguousarray(ort_out))
    print("wrote models/goldens/onnx_mlp_output.npy", ort_out.shape)

    # Sanity: ORT on the ONNX graph matches PyTorch eager on the source MLP.
    torch_out = to_f32(build_mlp()(sample_input()))
    print(f"[CHECK] ONNX Runtime vs PyTorch eager: max abs err "
          f"{np.abs(ort_out - torch_out).max():.2e}")


if __name__ == "__main__":
    main()
