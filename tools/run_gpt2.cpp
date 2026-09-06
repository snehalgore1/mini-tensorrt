// Greedy text generation with the real GPT-2 running on MiniTensorRT.
//
//   run_gpt2 --model models/gpt2_124m.json --max-new 20 --ids "464 2068 7586"
//
// Reads prompt token ids (space-separated, from --ids or stdin), autoregressively
// generates up to --max-new tokens by greedy argmax, and prints the generated ids
// to stdout. Tokenization/detokenization is done in Python (python/gpt2_generate.py);
// this binary is the pure runtime loop.
//
// Static-shape generation (respecting core invariant 3): the graph has a fixed
// context length S. We keep the prefix in positions 0..L-1 (pad the rest with 0)
// and read logits[L-1]; causal masking makes positions >= L irrelevant to it. This
// recomputes the whole graph each step (O(S) per token) -- the KV-cache milestone
// (N3) is what removes that cost.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "mtrt/executor.h"
#include "mtrt/frontend/json_loader.h"
#include "mtrt/registry.h"

using namespace mtrt;
using namespace mtrt::frontend;

namespace {
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
}  // namespace

int main(int argc, char** argv) {
  std::string model = std::string(MTRT_MODELS_DIR) + "/gpt2_124m.json";
  std::string ids_arg;
  int max_new = 20;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    else if (!std::strcmp(argv[i], "--ids") && i + 1 < argc) ids_arg = argv[++i];
    else if (!std::strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = std::atoi(argv[++i]);
  }

  // Prompt ids from --ids or stdin.
  std::vector<int32_t> prompt;
  if (!ids_arg.empty()) {
    prompt = parse_ids(ids_arg);
  } else {
    std::string line, all;
    while (std::getline(std::cin, line)) all += line + " ";
    prompt = parse_ids(all);
  }
  if (prompt.empty()) {
    std::cerr << "run_gpt2: no prompt ids given (--ids or stdin)\n";
    return 1;
  }

  LoadedModel m = load_json_model(model);
  KernelRegistry reg;
  register_builtin_kernels(reg);
  Executor exec(m.graph, reg);

  const TensorId in_id = m.graph.graph_inputs()[0];
  const int64_t S = m.graph.tensor(in_id).shape[0];
  const int64_t V = m.graph.tensor(m.graph.graph_outputs()[0]).shape[1];

  if (static_cast<int64_t>(prompt.size()) >= S) {
    std::cerr << "run_gpt2: prompt (" << prompt.size() << ") >= context " << S << "\n";
    return 1;
  }

  std::vector<int32_t> seq = prompt;  // grows as we generate
  for (int step = 0; step < max_new && static_cast<int64_t>(seq.size()) < S; ++step) {
    // Fill a fixed [S] id buffer: prefix in 0..L-1, pad the rest with 0.
    Tensor ids = Tensor::owning(DType::kI32, {S});
    int32_t* p = ids.data<int32_t>();
    const int64_t L = static_cast<int64_t>(seq.size());
    for (int64_t i = 0; i < S; ++i) p[i] = i < L ? seq[static_cast<size_t>(i)] : 0;

    std::unordered_map<TensorId, Tensor> bindings = m.weights;
    bindings.emplace(in_id, ids);
    std::vector<Tensor> outs = exec.run(bindings);

    const float* logits = outs[0].data<float>();
    const int32_t next = static_cast<int32_t>(argmax(logits + (L - 1) * V, V));
    seq.push_back(next);
  }

  // Print generated ids (everything after the prompt).
  for (size_t i = prompt.size(); i < seq.size(); ++i)
    std::cout << seq[i] << (i + 1 < seq.size() ? " " : "");
  std::cout << std::endl;
  return 0;
}
