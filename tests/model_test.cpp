#include <iostream>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "mtrt/executor.h"
#include "mtrt/frontend/json_loader.h"
#include "mtrt/registry.h"
#include "mtrt/tensor.h"
#include "tests/support/golden.h"

using namespace mtrt;
using namespace mtrt::frontend;
using namespace mtrt::testing;

// Week 2 DoD: the exported MLP, loaded from JSON+bin, runs in MiniTensorRT and
// matches PyTorch within tolerance.
TEST(Model, MlpMatchesPyTorch) {
  LoadedModel m = load_json_model(std::string(MTRT_MODELS_DIR) + "/mlp.json");

  KernelRegistry reg;
  register_builtin_kernels(reg);
  Executor exec(m.graph, reg);

  // Bindings: all weights, plus the graph input.
  std::unordered_map<TensorId, Tensor> bindings = m.weights;
  Tensor x = tensor_from_npy(load_golden("mlp_input"));
  ASSERT_EQ(m.graph.graph_inputs().size(), 1u);
  bindings.emplace(m.graph.graph_inputs()[0], x);

  std::vector<Tensor> outputs = exec.run(bindings);
  ASSERT_EQ(outputs.size(), 1u);

  NpyArray expected = load_golden("mlp_output");
  ErrorStats err = compute_error(outputs[0].data<float>(), expected.data.data(),
                                 expected.numel());
  // Emit for docs/RESULTS.md (traceable to this test run).
  std::cout << "[RESULTS] MLP vs PyTorch eager: max_abs_err=" << err.max_abs
            << " max_rel_err=" << err.max_rel << std::endl;
  ExpectGolden(outputs[0], expected, kDefaultRtol, 1e-5f);  // ends in Softmax
}
