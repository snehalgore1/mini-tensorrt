#include "mtrt/frontend/onnx_loader.h"

#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mtrt/graph.h"
#include "mtrt/tensor.h"
#include "onnx.pb.h"

// Parses a real .onnx (protobuf) into MiniTensorRT IR. The onnx.pb types stay
// entirely inside this .cpp -- the header exposes only LoadedModel -- so the IR
// remains frontend-agnostic (core invariant 2). Shapes must be fully resolved in
// the file (static-shape invariant 3); we do not run shape inference in C++.

namespace mtrt::frontend {
namespace {

DType dtype_from_onnx(int32_t onnx_dtype, const std::string& tname) {
  switch (onnx_dtype) {
    case onnx::TensorProto::FLOAT: return DType::kF32;
    case onnx::TensorProto::INT32: return DType::kI32;
    case onnx::TensorProto::INT8: return DType::kI8;
    default:
      throw GraphError("ONNX tensor '" + tname + "' has unsupported dtype " +
                       std::to_string(onnx_dtype) + " (FP32 subset only)");
  }
}

// Map an ONNX op type to a registered MiniTensorRT kernel name. Strict subset:
// an unmapped op throws rather than silently misbehaving (Week 6 DoD). Most names
// coincide with ours; the map is the single place to extend coverage.
std::string op_from_onnx(const std::string& onnx_op) {
  static const std::unordered_map<std::string, std::string> kMap = {
      {"MatMul", "MatMul"},   {"Add", "Add"},
      {"Gelu", "Gelu"},       {"Relu", "Relu"},
      {"Softmax", "Softmax"}, {"Reshape", "Reshape"},
      {"Transpose", "Transpose"}, {"LayerNormalization", "LayerNorm"},
  };
  auto it = kMap.find(onnx_op);
  if (it == kMap.end())
    throw GraphError("ONNX op '" + onnx_op +
                     "' is not supported by MiniTensorRT (strict operator subset)");
  return it->second;
}

// Read a fully-resolved static shape from an ONNX type. Symbolic/missing dims are
// rejected: MiniTensorRT resolves all shapes at load (DESIGN D1, invariant 3).
std::vector<int64_t> shape_from_type(const onnx::TypeProto& type,
                                     const std::string& tname) {
  if (!type.has_tensor_type())
    throw GraphError("ONNX tensor '" + tname + "' is not a tensor type");
  const auto& sh = type.tensor_type().shape();
  std::vector<int64_t> dims;
  for (const auto& d : sh.dim()) {
    if (!d.has_dim_value())
      throw GraphError("ONNX tensor '" + tname +
                       "' has a dynamic/symbolic dimension; static shapes only");
    dims.push_back(d.dim_value());
  }
  return dims;
}

Attribute attr_from_onnx(const onnx::AttributeProto& a) {
  switch (a.type()) {
    case onnx::AttributeProto::INT: return static_cast<int64_t>(a.i());
    case onnx::AttributeProto::FLOAT: return static_cast<double>(a.f());
    case onnx::AttributeProto::INTS: {
      std::vector<int64_t> v(a.ints().begin(), a.ints().end());
      return v;
    }
    default:
      throw GraphError("ONNX attribute '" + a.name() +
                       "' has an unsupported type");
  }
}

// Copy an initializer's payload into an owning Tensor. ONNX stores data either in
// raw_data (little-endian bytes) or the typed repeated fields.
Tensor tensor_from_initializer(const onnx::TensorProto& t, DType dt,
                               const std::vector<int64_t>& shape) {
  Tensor w = Tensor::owning(dt, shape);
  const int64_t n = w.numel();
  if (!t.raw_data().empty()) {
    const size_t nbytes = static_cast<size_t>(n) * dtype_size(dt);
    if (t.raw_data().size() != nbytes)
      throw GraphError("ONNX initializer '" + t.name() + "' raw_data size mismatch");
    std::memcpy(w.data<float>(), t.raw_data().data(), nbytes);
  } else if (dt == DType::kF32 && t.float_data_size() == n) {
    std::memcpy(w.data<float>(), t.float_data().data(),
                static_cast<size_t>(n) * sizeof(float));
  } else if (dt == DType::kI32 && t.int32_data_size() == n) {
    std::memcpy(w.data<int32_t>(), t.int32_data().data(),
                static_cast<size_t>(n) * sizeof(int32_t));
  } else {
    throw GraphError("ONNX initializer '" + t.name() + "' has no readable data");
  }
  return w;
}

}  // namespace

LoadedModel load_onnx_model(const std::string& onnx_path) {
  std::ifstream in(onnx_path, std::ios::binary);
  if (!in) throw GraphError("cannot open ONNX model: " + onnx_path);
  onnx::ModelProto model_pb;
  if (!model_pb.ParseFromIstream(&in))
    throw GraphError("failed to parse ONNX protobuf: " + onnx_path);
  if (!model_pb.has_graph())
    throw GraphError("ONNX model has no graph: " + onnx_path);
  const onnx::GraphProto& gp = model_pb.graph();

  LoadedModel model;
  Graph& g = model.graph;
  std::unordered_map<std::string, TensorId> name_to_id;

  // Gather the static shape + dtype for every named tensor from the file's
  // value_info (inputs, outputs, and inferred intermediates) and initializers.
  struct Info { DType dtype; std::vector<int64_t> shape; };
  std::unordered_map<std::string, Info> meta;
  auto record = [&](const onnx::ValueInfoProto& vi) {
    meta[vi.name()] = {dtype_from_onnx(vi.type().tensor_type().elem_type(), vi.name()),
                       shape_from_type(vi.type(), vi.name())};
  };
  for (const auto& vi : gp.input()) record(vi);
  for (const auto& vi : gp.output()) record(vi);
  for (const auto& vi : gp.value_info()) record(vi);

  std::unordered_set<std::string> initializer_names;
  for (const auto& init : gp.initializer()) initializer_names.insert(init.name());
  std::unordered_set<std::string> output_names;
  for (const auto& o : gp.output()) output_names.insert(o.name());
  // Graph inputs that are not initializers are the real runtime inputs. They must
  // be TensorKind::kInput so the executor binds (not allocates) them -- otherwise
  // they would wrongly enter the memory plan as intermediates.
  std::unordered_set<std::string> input_names;
  for (const auto& vi : gp.input())
    if (!initializer_names.count(vi.name())) input_names.insert(vi.name());

  // Helper to add a tensor once, resolving its kind from its ONNX role. Kind is
  // fixed at creation, so the role sets must be known before any tensor is added.
  auto ensure_tensor = [&](const std::string& name) -> TensorId {
    auto it = name_to_id.find(name);
    if (it != name_to_id.end()) return it->second;
    auto mit = meta.find(name);
    if (mit == meta.end())
      throw GraphError("ONNX tensor '" + name +
                       "' has no shape; export with onnx.shape_inference.infer_shapes");
    TensorInfo info;
    info.name = name;
    info.dtype = mit->second.dtype;
    info.shape = mit->second.shape;
    if (initializer_names.count(name)) info.kind = TensorKind::kWeight;
    else if (input_names.count(name)) info.kind = TensorKind::kInput;
    else if (output_names.count(name)) info.kind = TensorKind::kOutput;
    else info.kind = TensorKind::kIntermediate;
    const TensorId id = g.add_tensor(info);
    name_to_id.emplace(name, id);
    return id;
  };

  // Initializers first (weights), with their data.
  for (const auto& init : gp.initializer()) {
    const std::vector<int64_t> shape(init.dims().begin(), init.dims().end());
    const DType dt = dtype_from_onnx(init.data_type(), init.name());
    meta[init.name()] = {dt, shape};  // initializers carry their own shape
    const TensorId id = ensure_tensor(init.name());
    model.weights.emplace(id, tensor_from_initializer(init, dt, shape));
  }

  // Nodes, adding any tensors they reference.
  for (const auto& np : gp.node()) {
    Node node;
    node.op_type = op_from_onnx(np.op_type());
    for (const auto& in_name : np.input())
      node.inputs.push_back(ensure_tensor(in_name));
    for (const auto& out_name : np.output())
      node.outputs.push_back(ensure_tensor(out_name));
    for (const auto& a : np.attribute()) node.attrs.emplace(a.name(), attr_from_onnx(a));
    g.add_node(std::move(node));
  }

  // Graph inputs that are not initializers are real inputs.
  for (const auto& vi : gp.input()) {
    if (initializer_names.count(vi.name())) continue;
    g.mark_input(ensure_tensor(vi.name()));
  }
  for (const auto& vi : gp.output()) g.mark_output(ensure_tensor(vi.name()));

  return model;
}

}  // namespace mtrt::frontend
