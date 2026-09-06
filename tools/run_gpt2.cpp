// Greedy text generation with real GPT-2 on MiniTensorRT, with two decode paths:
//
//   --mode graph : run the fixed-context static graph and read logits[t] each step
//                  (O(S) per token; recomputes the whole graph -- the N2 baseline).
//   --mode kv    : incremental decode with a KV-cache -- cache each layer's K/V for
//                  past positions, compute Q/K/V only for the new token, and attend
//                  over the cached keys (O(t) per token). This is a deliberate,
//                  bounded extension of the static-shape model: buffers are
//                  pre-allocated to a fixed max context and a runtime position
//                  tracks the valid length (N3).
//   --mode bench : run both, assert identical output ids, and report the speedup.
//
//   run_gpt2 --mode bench --ids "464 2068 7586" --max-new 20
//
// Reads prompt token ids (space-separated, --ids or stdin), prints generated ids.
// Tokenization is done in Python (python/gpt2_generate.py); this is the runtime.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "backends/cpu/gemm.h"
#include "mtrt/executor.h"
#include "mtrt/frontend/json_loader.h"
#include "mtrt/registry.h"

using namespace mtrt;
using namespace mtrt::frontend;

namespace {
constexpr float kEps = 1e-5f;

int64_t argmax(const float* row, int64_t n) {
  int64_t best = 0;
  for (int64_t j = 1; j < n; ++j)
    if (row[j] > row[best]) best = j;
  return best;
}

std::vector<int32_t> parse_ids(const std::string& s) {
  std::vector<int32_t> ids;
  std::istringstream iss(s);
  int64_t v;
  while (iss >> v) ids.push_back(static_cast<int32_t>(v));
  return ids;
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// ---- Graph decode (N2 baseline): fixed-context static graph, read logits[t]. ---
std::vector<int32_t> generate_graph(const LoadedModel& m, const KernelRegistry& reg,
                                    const std::vector<int32_t>& prompt, int max_new) {
  Executor exec(m.graph, reg);
  const TensorId in_id = m.graph.graph_inputs()[0];
  const int64_t S = m.graph.tensor(in_id).shape[0];
  const int64_t V = m.graph.tensor(m.graph.graph_outputs()[0]).shape[1];

  std::vector<int32_t> seq = prompt;
  for (int step = 0; step < max_new && (int64_t)seq.size() < S; ++step) {
    Tensor ids = Tensor::owning(DType::kI32, {S});
    int32_t* p = ids.data<int32_t>();
    const int64_t L = (int64_t)seq.size();
    for (int64_t i = 0; i < S; ++i) p[i] = i < L ? seq[(size_t)i] : 0;
    std::unordered_map<TensorId, Tensor> b = m.weights;
    b.emplace(in_id, ids);
    std::vector<Tensor> outs = exec.run(b);
    seq.push_back((int32_t)argmax(outs[0].data<float>() + (L - 1) * V, V));
  }
  return {seq.begin() + prompt.size(), seq.end()};
}

// ---- KV-cache incremental decode (N3). ------------------------------------------
// Reads the same weights by name and does the GPT-2 forward for one token at a
// time, caching K/V per layer so attention only touches the new query.
struct KVDecoder {
  const LoadedModel& m;
  std::unordered_map<std::string, const Tensor*> w;
  int64_t D, H, d, hidden, V, L;
  int max_ctx;
  std::vector<std::vector<float>> Kc, Vc;  // per layer: [max_ctx * D]

  KVDecoder(const LoadedModel& model, int heads, int max_context)
      : m(model), H(heads), max_ctx(max_context) {
    for (TensorId id = 0; id < m.graph.num_tensors(); ++id) {
      auto it = m.weights.find(id);
      if (it != m.weights.end()) w[m.graph.tensor(id).name] = &it->second;
    }
    D = w.at("wte")->shape()[1];
    V = w.at("wte")->shape()[0];
    hidden = w.at("l0_Wfc")->shape()[1];
    d = D / H;
    L = 0;
    while (w.count("l" + std::to_string(L) + "_ln1_g")) ++L;
    Kc.assign((size_t)L, std::vector<float>((size_t)max_ctx * D));
    Vc.assign((size_t)L, std::vector<float>((size_t)max_ctx * D));
  }

  const float* wp(const std::string& n) const { return w.at(n)->data<float>(); }

  void layernorm(const float* x, const float* g, const float* b, float* y) const {
    float mean = 0.f;
    for (int64_t i = 0; i < D; ++i) mean += x[i];
    mean /= D;
    float var = 0.f;
    for (int64_t i = 0; i < D; ++i) var += (x[i] - mean) * (x[i] - mean);
    var /= D;
    const float inv = 1.f / std::sqrt(var + kEps);
    for (int64_t i = 0; i < D; ++i) y[i] = (x[i] - mean) * inv * g[i] + b[i];
  }

  // y[N] = x[K] @ W[K,N] + b[N]  (b may be null).
  void linear(const float* x, const std::string& wn, const std::string& bn,
              float* y, int64_t K, int64_t N) const {
    cpu::gemm_auto(x, wp(wn), y, 1, (int)N, (int)K);
    if (!bn.empty()) {
      const float* b = wp(bn);
      for (int64_t j = 0; j < N; ++j) y[j] += b[j];
    }
  }

  // Decode one token at position `pos`; returns argmax(logits).
  int32_t step(int32_t token, int pos) {
    std::vector<float> x(D), h(D), q(D), k(D), v(D), ctx(D), ao(D), mh(hidden);
    const float* wte = wp("wte");
    const float* wpe = wp("wpe");
    for (int64_t i = 0; i < D; ++i) x[i] = wte[token * D + i] + wpe[pos * D + i];

    const float scale = 1.f / std::sqrt((float)d);
    for (int64_t l = 0; l < L; ++l) {
      const std::string p = "l" + std::to_string(l) + "_";
      layernorm(x.data(), wp(p + "ln1_g"), wp(p + "ln1_b"), h.data());
      linear(h.data(), p + "Wq", p + "bq", q.data(), D, D);
      linear(h.data(), p + "Wk", p + "bk", k.data(), D, D);
      linear(h.data(), p + "Wv", p + "bv", v.data(), D, D);
      // Append K/V for this position into the cache.
      float* Krow = &Kc[(size_t)l][(size_t)pos * D];
      float* Vrow = &Vc[(size_t)l][(size_t)pos * D];
      for (int64_t i = 0; i < D; ++i) { Krow[i] = k[i]; Vrow[i] = v[i]; }
      // Per-head attention over cached positions 0..pos.
      std::vector<float> scores((size_t)pos + 1);
      for (int64_t hd = 0; hd < H; ++hd) {
        const float* qh = q.data() + hd * d;
        float mx = -1e30f;
        for (int j = 0; j <= pos; ++j) {
          const float* kj = &Kc[(size_t)l][(size_t)j * D + hd * d];
          float s = 0.f;
          for (int64_t t = 0; t < d; ++t) s += qh[t] * kj[t];
          s *= scale;
          scores[(size_t)j] = s;
          if (s > mx) mx = s;
        }
        float sum = 0.f;
        for (int j = 0; j <= pos; ++j) { scores[(size_t)j] = std::exp(scores[(size_t)j] - mx); sum += scores[(size_t)j]; }
        const float inv = 1.f / sum;
        float* ch = ctx.data() + hd * d;
        for (int64_t t = 0; t < d; ++t) ch[t] = 0.f;
        for (int j = 0; j <= pos; ++j) {
          const float wgt = scores[(size_t)j] * inv;
          const float* vj = &Vc[(size_t)l][(size_t)j * D + hd * d];
          for (int64_t t = 0; t < d; ++t) ch[t] += wgt * vj[t];
        }
      }
      linear(ctx.data(), p + "Wo", p + "bo", ao.data(), D, D);
      for (int64_t i = 0; i < D; ++i) x[i] += ao[i];                 // residual
      layernorm(x.data(), wp(p + "ln2_g"), wp(p + "ln2_b"), h.data());
      linear(h.data(), p + "Wfc", p + "bfc", mh.data(), D, hidden);
      for (int64_t i = 0; i < hidden; ++i) {                          // gelu_new
        const float u = mh[i];
        mh[i] = 0.5f * u * (1.f + std::tanh(0.7978845608028654f * (u + 0.044715f * u * u * u)));
      }
      linear(mh.data(), p + "Wproj", p + "bproj", ao.data(), hidden, D);
      for (int64_t i = 0; i < D; ++i) x[i] += ao[i];                 // residual
    }
    layernorm(x.data(), wp("ln_f_g"), wp("ln_f_b"), h.data());
    std::vector<float> logits(V);
    cpu::gemm_auto(h.data(), wp("lm_head"), logits.data(), 1, (int)V, (int)D);
    return (int32_t)argmax(logits.data(), V);
  }
};

std::vector<int32_t> generate_kv(const LoadedModel& m, const std::vector<int32_t>& prompt,
                                 int max_new, int heads) {
  KVDecoder dec(m, heads, (int)prompt.size() + max_new + 1);
  std::vector<int32_t> gen;
  int pos = 0;
  int32_t next = 0;
  for (int32_t tok : prompt) next = dec.step(tok, pos++);  // prefill
  for (int i = 0; i < max_new; ++i) {
    gen.push_back(next);
    next = dec.step(next, pos++);
  }
  return gen;
}
}  // namespace

int main(int argc, char** argv) {
  std::string model = std::string(MTRT_MODELS_DIR) + "/gpt2_124m.json";
  std::string ids_arg, mode = "kv";
  int max_new = 20, heads = 12;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    else if (!std::strcmp(argv[i], "--ids") && i + 1 < argc) ids_arg = argv[++i];
    else if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
    else if (!std::strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--heads") && i + 1 < argc) heads = std::atoi(argv[++i]);
  }

  std::vector<int32_t> prompt;
  if (!ids_arg.empty()) {
    prompt = parse_ids(ids_arg);
  } else {
    std::string line, all;
    while (std::getline(std::cin, line)) all += line + " ";
    prompt = parse_ids(all);
  }
  if (prompt.empty()) { std::cerr << "run_gpt2: no prompt ids\n"; return 1; }

  LoadedModel m = load_json_model(model);
  KernelRegistry reg;
  register_builtin_kernels(reg);

  auto print_ids = [](const std::vector<int32_t>& g) {
    for (size_t i = 0; i < g.size(); ++i) std::cout << g[i] << (i + 1 < g.size() ? " " : "");
    std::cout << std::endl;
  };

  if (mode == "graph") {
    print_ids(generate_graph(m, reg, prompt, max_new));
  } else if (mode == "kv") {
    print_ids(generate_kv(m, prompt, max_new, heads));
  } else if (mode == "bench") {
    const double t0 = now_ms();
    std::vector<int32_t> g_graph = generate_graph(m, reg, prompt, max_new);
    const double t1 = now_ms();
    std::vector<int32_t> g_kv = generate_kv(m, prompt, max_new, heads);
    const double t2 = now_ms();
    const bool same = g_graph == g_kv;
    std::cerr << "[RESULTS] KV-cache vs full-recompute (" << max_new << " tokens, prompt "
              << prompt.size() << "): graph=" << (t1 - t0) << " ms, kv=" << (t2 - t1)
              << " ms, speedup=" << (t1 - t0) / (t2 - t1) << "x, identical_ids=" << same
              << std::endl;
    print_ids(g_kv);
    return same ? 0 : 1;
  } else {
    std::cerr << "run_gpt2: unknown --mode " << mode << "\n";
    return 1;
  }
  return 0;
}
