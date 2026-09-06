"""Export real GPT-2 (124M) from HuggingFace to the MiniTensorRT JSON+bin format.

This makes the runtime run a *real* network, not a random-weight block. It:
  1. loads HF `GPT2LMHeadModel` ("gpt2"),
  2. maps its weights into our [in, out] MatMul layout (GPT-2 uses Conv1D, whose
     weight is already [in, out] -- so, unlike nn.Linear, NO transpose is needed;
     this is the classic layout trap, handled here once),
  3. writes models/gpt2_124m.bin (weights) + models/gpt2_124m.json (full LM graph
     for a fixed sequence length), and
  4. with --verify, runs an independent numpy/torch forward from the *exported*
     arrays and checks it matches HF logits (argmax + allclose). This proves the
     export is correct on its own, before any C++ runs.

Run:  python python/export_gpt2_hf.py --seq-len 16 --verify
Writes: models/gpt2_124m.json, models/gpt2_124m.bin  (~500 MB, gitignored)

The graph uses one op the CPU runtime gains in N1: `CausalSoftmax` (masked softmax
over the last dim). Everything else already exists (Gather, MatMul, Add w/ last-dim
bias broadcast, LayerNorm, Reshape, Transpose, Scale, BatchedMatMul, GeluTanh).
"""

import argparse
import json
import os

import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

EPS = 1e-5  # GPT-2 config.layer_norm_epsilon


def load_hf_gpt2():
    """Return (config dict, state) where state maps our weight names -> np.float32."""
    import torch
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    cfg = model.config
    D, H, L, V = cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.vocab_size
    ctx = cfg.n_positions

    def np32(t):
        return t.detach().numpy().astype(np.float32)

    tr = model.transformer
    w = {}
    w["wte"] = np32(tr.wte.weight)          # [V, D]
    w["wpe"] = np32(tr.wpe.weight)          # [ctx, D]
    w["ln_f_g"] = np32(tr.ln_f.weight)      # [D]
    w["ln_f_b"] = np32(tr.ln_f.bias)        # [D]
    # LM head is tied to wte: logits = h @ wte^T. Store wte^T = [D, V].
    w["lm_head"] = np.ascontiguousarray(w["wte"].T)  # [D, V]

    for i in range(L):
        blk = tr.h[i]
        p = f"l{i}_"
        w[p + "ln1_g"] = np32(blk.ln_1.weight)
        w[p + "ln1_b"] = np32(blk.ln_1.bias)
        # c_attn: Conv1D weight [D, 3D] already [in, out]; split into Q/K/V.
        cattn_w = np32(blk.attn.c_attn.weight)   # [D, 3D]
        cattn_b = np32(blk.attn.c_attn.bias)     # [3D]
        w[p + "Wq"], w[p + "Wk"], w[p + "Wv"] = (
            np.ascontiguousarray(cattn_w[:, 0 * D:1 * D]),
            np.ascontiguousarray(cattn_w[:, 1 * D:2 * D]),
            np.ascontiguousarray(cattn_w[:, 2 * D:3 * D]),
        )
        # Biases stored as [1, N] (rank-2): the CPU Add broadcast path expects
        # [1, D], not [D]. The reference forward below still broadcasts fine.
        w[p + "bq"], w[p + "bk"], w[p + "bv"] = (
            np.ascontiguousarray(cattn_b[0 * D:1 * D]).reshape(1, D),
            np.ascontiguousarray(cattn_b[1 * D:2 * D]).reshape(1, D),
            np.ascontiguousarray(cattn_b[2 * D:3 * D]).reshape(1, D),
        )
        w[p + "Wo"] = np32(blk.attn.c_proj.weight)   # [D, D]
        w[p + "bo"] = np32(blk.attn.c_proj.bias).reshape(1, D)     # [1, D]
        w[p + "ln2_g"] = np32(blk.ln_2.weight)
        w[p + "ln2_b"] = np32(blk.ln_2.bias)
        w[p + "Wfc"] = np32(blk.mlp.c_fc.weight)     # [D, 4D]
        w[p + "bfc"] = np32(blk.mlp.c_fc.bias).reshape(1, 4 * D)   # [1, 4D]
        w[p + "Wproj"] = np32(blk.mlp.c_proj.weight)  # [4D, D]
        w[p + "bproj"] = np32(blk.mlp.c_proj.bias).reshape(1, D)   # [1, D]

    conf = {"D": D, "H": H, "L": L, "V": V, "ctx": ctx, "d": D // H,
            "hidden": 4 * D, "eps": EPS}
    return conf, w, model


def build_graph_and_blob(conf, w, seq_len):
    """Serialize weights to a blob and build the full LM graph for `seq_len`.
    Returns (topo_dict, blob_bytes)."""
    D, H, L, V, d, hidden = (conf["D"], conf["H"], conf["L"], conf["V"],
                             conf["d"], conf["hidden"])
    S = seq_len

    # Positions constant [S] (i32) baked in as a weight.
    positions = np.arange(S, dtype=np.int32)

    blob = bytearray()
    meta = {}

    def put(name, arr):
        off = len(blob)
        data = np.ascontiguousarray(arr).tobytes(order="C")
        blob.extend(data)
        meta[name] = (off, len(data), list(arr.shape),
                      "i32" if arr.dtype == np.int32 else "f32")

    # Global weights first.
    put("wte", w["wte"]); put("wpe", w["wpe"])
    put("positions", positions)
    for i in range(L):
        p = f"l{i}_"
        for nm in ("ln1_g", "ln1_b", "Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo",
                   "ln2_g", "ln2_b", "Wfc", "bfc", "Wproj", "bproj"):
            put(p + nm, w[p + nm])
    put("ln_f_g", w["ln_f_g"]); put("ln_f_b", w["ln_f_b"]); put("lm_head", w["lm_head"])

    tensors, nodes = [], []

    def T(name, shape, kind, dtype="f32"):
        t = {"name": name, "dtype": dtype, "shape": shape, "kind": kind}
        if kind == "weight":
            off, nbytes, _, _ = meta[name]
            t["offset"], t["nbytes"] = off, nbytes
        tensors.append(t)

    def W(name):  # declare a weight tensor from its stored metadata
        _, _, shape, dt = meta[name]
        T(name, shape, "weight", dt)

    # Inputs / embeddings.
    T("ids", [S], "input", "i32")
    W("wte"); W("wpe"); W("positions")
    T("tok_emb", [S, D], "intermediate")
    T("pos_emb", [S, D], "intermediate")
    T("x0", [S, D], "intermediate")
    nodes += [
        {"op": "Gather", "inputs": ["wte", "ids"], "outputs": ["tok_emb"], "attrs": {}},
        {"op": "Gather", "inputs": ["wpe", "positions"], "outputs": ["pos_emb"], "attrs": {}},
        {"op": "Add", "inputs": ["tok_emb", "pos_emb"], "outputs": ["x0"], "attrs": {}},
    ]

    scale = 1.0 / (d ** 0.5)

    def emit_layer(i, xin, xout):
        p = f"l{i}_"
        for nm in ("ln1_g", "ln1_b", "Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo",
                   "ln2_g", "ln2_b", "Wfc", "bfc", "Wproj", "bproj"):
            W(p + nm)

        def iv(nm, shape):
            T(p + nm, shape, "intermediate"); return p + nm

        h = iv("h", [S, D])
        Qm, Km, Vm = iv("Qm", [S, D]), iv("Km", [S, D]), iv("Vm", [S, D])
        Q, K, Vv = iv("Q", [S, D]), iv("K", [S, D]), iv("V", [S, D])
        Qr, Kr, Vr = iv("Qr", [S, H, d]), iv("Kr", [S, H, d]), iv("Vr", [S, H, d])
        Qh, Kh, Vh = iv("Qh", [H, S, d]), iv("Kh", [H, S, d]), iv("Vh", [H, S, d])
        KhT = iv("KhT", [H, d, S])
        sc, scd, at = iv("sc", [H, S, S]), iv("scd", [H, S, S]), iv("at", [H, S, S])
        ctx = iv("ctx", [H, S, d]); ctxT = iv("ctxT", [S, H, d]); ctxM = iv("ctxM", [S, D])
        ao, aob = iv("ao", [S, D]), iv("aob", [S, D])
        x1 = iv("x1", [S, D])
        h2 = iv("h2", [S, D])
        m1, m1b, g = iv("m1", [S, hidden]), iv("m1b", [S, hidden]), iv("g", [S, hidden])
        m2, m2b = iv("m2", [S, D]), iv("m2b", [S, D])

        nodes.extend([
            {"op": "LayerNorm", "inputs": [xin, p+"ln1_g", p+"ln1_b"], "outputs": [h], "attrs": {"eps": EPS}},
            {"op": "MatMul", "inputs": [h, p+"Wq"], "outputs": [Qm], "attrs": {}},
            {"op": "Add", "inputs": [Qm, p+"bq"], "outputs": [Q], "attrs": {}},
            {"op": "MatMul", "inputs": [h, p+"Wk"], "outputs": [Km], "attrs": {}},
            {"op": "Add", "inputs": [Km, p+"bk"], "outputs": [K], "attrs": {}},
            {"op": "MatMul", "inputs": [h, p+"Wv"], "outputs": [Vm], "attrs": {}},
            {"op": "Add", "inputs": [Vm, p+"bv"], "outputs": [Vv], "attrs": {}},
            {"op": "Reshape", "inputs": [Q], "outputs": [Qr], "attrs": {"shape": [S, H, d]}},
            {"op": "Reshape", "inputs": [K], "outputs": [Kr], "attrs": {"shape": [S, H, d]}},
            {"op": "Reshape", "inputs": [Vv], "outputs": [Vr], "attrs": {"shape": [S, H, d]}},
            {"op": "Transpose", "inputs": [Qr], "outputs": [Qh], "attrs": {"perm": [1, 0, 2]}},
            {"op": "Transpose", "inputs": [Kr], "outputs": [Kh], "attrs": {"perm": [1, 0, 2]}},
            {"op": "Transpose", "inputs": [Vr], "outputs": [Vh], "attrs": {"perm": [1, 0, 2]}},
            {"op": "Transpose", "inputs": [Kh], "outputs": [KhT], "attrs": {"perm": [0, 2, 1]}},
            {"op": "BatchedMatMul", "inputs": [Qh, KhT], "outputs": [sc], "attrs": {}},
            {"op": "Scale", "inputs": [sc], "outputs": [scd], "attrs": {"scale": scale}},
            {"op": "CausalSoftmax", "inputs": [scd], "outputs": [at], "attrs": {}},
            {"op": "BatchedMatMul", "inputs": [at, Vh], "outputs": [ctx], "attrs": {}},
            {"op": "Transpose", "inputs": [ctx], "outputs": [ctxT], "attrs": {"perm": [1, 0, 2]}},
            {"op": "Reshape", "inputs": [ctxT], "outputs": [ctxM], "attrs": {"shape": [S, D]}},
            {"op": "MatMul", "inputs": [ctxM, p+"Wo"], "outputs": [ao], "attrs": {}},
            {"op": "Add", "inputs": [ao, p+"bo"], "outputs": [aob], "attrs": {}},
            {"op": "Add", "inputs": [xin, aob], "outputs": [x1], "attrs": {}},
            {"op": "LayerNorm", "inputs": [x1, p+"ln2_g", p+"ln2_b"], "outputs": [h2], "attrs": {"eps": EPS}},
            {"op": "MatMul", "inputs": [h2, p+"Wfc"], "outputs": [m1], "attrs": {}},
            {"op": "Add", "inputs": [m1, p+"bfc"], "outputs": [m1b], "attrs": {}},
            {"op": "GeluTanh", "inputs": [m1b], "outputs": [g], "attrs": {}},
            {"op": "MatMul", "inputs": [g, p+"Wproj"], "outputs": [m2], "attrs": {}},
            {"op": "Add", "inputs": [m2, p+"bproj"], "outputs": [m2b], "attrs": {}},
            {"op": "Add", "inputs": [x1, m2b], "outputs": [xout], "attrs": {}},
        ])

    prev = "x0"
    for i in range(L):
        xout = f"l{i}_out"
        T(xout, [S, D], "intermediate")
        emit_layer(i, prev, xout)
        prev = xout

    # Final LayerNorm + LM head -> logits [S, V].
    W("ln_f_g"); W("ln_f_b"); W("lm_head")
    T("hf", [S, D], "intermediate")
    T("logits", [S, V], "output")
    nodes += [
        {"op": "LayerNorm", "inputs": [prev, "ln_f_g", "ln_f_b"], "outputs": ["hf"], "attrs": {"eps": EPS}},
        {"op": "MatMul", "inputs": ["hf", "lm_head"], "outputs": ["logits"], "attrs": {}},
    ]

    topo = {"name": "gpt2_124m", "weights_file": "gpt2_124m.bin",
            "tensors": tensors, "nodes": nodes, "inputs": ["ids"], "outputs": ["logits"]}
    return topo, bytes(blob)


def reference_forward(conf, w, ids):
    """Independent numpy forward from the *exported* arrays (mirrors the graph),
    used to verify the mapping without the C++ runtime. Returns logits [S, V]."""
    D, H, L, d = conf["D"], conf["H"], conf["L"], conf["d"]
    S = len(ids)

    def layernorm(x, g, b):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + EPS) * g + b

    def gelu_tanh(x):
        return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))

    x = w["wte"][ids] + w["wpe"][np.arange(S)]
    mask = np.triu(np.full((S, S), -1e10, np.float32), 1)  # causal
    for i in range(L):
        p = f"l{i}_"
        h = layernorm(x, w[p+"ln1_g"], w[p+"ln1_b"])
        Q = h @ w[p+"Wq"] + w[p+"bq"]
        K = h @ w[p+"Wk"] + w[p+"bk"]
        V = h @ w[p+"Wv"] + w[p+"bv"]
        Qh = Q.reshape(S, H, d).transpose(1, 0, 2)
        Kh = K.reshape(S, H, d).transpose(1, 0, 2)
        Vh = V.reshape(S, H, d).transpose(1, 0, 2)
        sc = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(d) + mask
        sc = sc - sc.max(-1, keepdims=True)
        at = np.exp(sc); at /= at.sum(-1, keepdims=True)
        ctx = (at @ Vh).transpose(1, 0, 2).reshape(S, D)
        x = x + ctx @ w[p+"Wo"] + w[p+"bo"]
        h2 = layernorm(x, w[p+"ln2_g"], w[p+"ln2_b"])
        m = gelu_tanh(h2 @ w[p+"Wfc"] + w[p+"bfc"])
        x = x + m @ w[p+"Wproj"] + w[p+"bproj"]
    x = layernorm(x, w["ln_f_g"], w["ln_f_b"])
    return x @ w["lm_head"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    conf, w, hf_model = load_hf_gpt2()
    print(f"loaded gpt2: L={conf['L']} H={conf['H']} D={conf['D']} V={conf['V']}")

    topo, blob = build_graph_and_blob(conf, w, args.seq_len)
    with open(os.path.join(MODELS_DIR, "gpt2_124m.json"), "w") as f:
        json.dump(topo, f)
    with open(os.path.join(MODELS_DIR, "gpt2_124m.bin"), "wb") as f:
        f.write(blob)
    print(f"wrote gpt2_124m.json ({len(topo['nodes'])} nodes) + gpt2_124m.bin "
          f"({len(blob)/1e6:.1f} MB), seq_len={args.seq_len}")

    # Whole-model golden: HF logits for a canonical id sequence (kept in sync with
    # tests/gpt2_real_test.cpp, which hardcodes ids = arange(10, 10+S)).
    import torch
    goldens = os.path.join(MODELS_DIR, "goldens")
    os.makedirs(goldens, exist_ok=True)
    ids = np.arange(10, 10 + args.seq_len, dtype=np.int64)
    with torch.no_grad():
        hf_logits = hf_model(torch.tensor(ids).unsqueeze(0)).logits[0].numpy()
    np.save(os.path.join(goldens, "gpt2_124m_logits.npy"),
            np.ascontiguousarray(hf_logits, dtype=np.float32))
    print(f"wrote goldens/gpt2_124m_logits.npy {hf_logits.shape} (ids=arange(10,{10+args.seq_len}))")

    if args.verify:
        import torch
        ids = np.arange(10, 10 + args.seq_len, dtype=np.int64)  # arbitrary token ids
        ours = reference_forward(conf, w, ids.astype(np.int32))
        with torch.no_grad():
            hf = hf_model(torch.tensor(ids).unsqueeze(0)).logits[0].numpy()
        max_abs = float(np.abs(ours - hf).max())
        argmax_match = bool((ours.argmax(-1) == hf.argmax(-1)).all())
        print(f"[VERIFY] exported-weights forward vs HF: max_abs_err={max_abs:.4e} "
              f"argmax_match={argmax_match}")
        assert argmax_match, "argmax mismatch -- weight mapping is wrong"
        assert max_abs < 1e-2, f"logit mismatch too large: {max_abs}"
        print("[VERIFY] OK -- exported weights reproduce HuggingFace logits.")


if __name__ == "__main__":
    main()
