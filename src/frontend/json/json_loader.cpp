#include "mtrt/frontend/json_loader.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace mtrt::frontend {
namespace {

using nlohmann::json;

DType parse_dtype(const std::string& s) {
  if (s == "f32") return DType::kF32;
  if (s == "i32") return DType::kI32;  // integer index inputs (token/position ids)
  if (s == "i8") return DType::kI8;    // symmetric per-tensor INT8 weights (#5)
  throw GraphError("unsupported dtype in JSON model: " + s);
}

TensorKind parse_kind(const std::string& s) {
  if (s == "input") return TensorKind::kInput;
  if (s == "output") return TensorKind::kOutput;
  if (s == "weight") return TensorKind::kWeight;
  if (s == "intermediate") return TensorKind::kIntermediate;
  throw GraphError("unknown tensor kind in JSON model: " + s);
}

std::string dir_of(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

// Attributes are parsed generically so new ops can carry config without loader
// changes: integers -> int64_t, other numbers -> double, arrays -> vector<int64>.
Attribute parse_attr(const json& v) {
  if (v.is_number_integer() || v.is_number_unsigned()) {
    return static_cast<int64_t>(v.get<int64_t>());
  }
  if (v.is_number_float()) return v.get<double>();
  if (v.is_array()) {
    std::vector<int64_t> out;
    for (const auto& e : v) out.push_back(e.get<int64_t>());
    return out;
  }
  throw GraphError("unsupported attribute type in JSON model");
}

}  // namespace

LoadedModel load_json_model(const std::string& json_path) {
  std::ifstream f(json_path);
  if (!f) throw GraphError("cannot open JSON model: " + json_path);
  json j;
  try {
    f >> j;
  } catch (const std::exception& e) {
    throw GraphError(std::string("JSON parse error: ") + e.what());
  }

  LoadedModel model;
  Graph& g = model.graph;
  std::unordered_map<std::string, TensorId> name_to_id;

  // Load the weights blob once; weight tensors index into it.
  std::vector<char> blob;
  if (j.contains("weights_file")) {
    const std::string bin_path =
        dir_of(json_path) + "/" + j.at("weights_file").get<std::string>();
    std::ifstream bf(bin_path, std::ios::binary | std::ios::ate);
    if (!bf) throw GraphError("cannot open weights file: " + bin_path);
    const std::streamsize size = bf.tellg();
    bf.seekg(0);
    blob.resize(static_cast<std::size_t>(size));
    bf.read(blob.data(), size);
  }

  for (const auto& jt : j.at("tensors")) {
    TensorInfo info;
    info.name = jt.at("name").get<std::string>();
    info.dtype = parse_dtype(jt.at("dtype").get<std::string>());
    info.shape = jt.at("shape").get<std::vector<int64_t>>();
    info.kind = parse_kind(jt.at("kind").get<std::string>());

    const TensorId id = g.add_tensor(info);
    if (!name_to_id.emplace(info.name, id).second) {
      throw GraphError("duplicate tensor name in JSON model: " + info.name);
    }

    if (info.kind == TensorKind::kWeight) {
      const auto offset = jt.at("offset").get<std::size_t>();
      const auto nbytes = jt.at("nbytes").get<std::size_t>();
      if (offset + nbytes > blob.size()) {
        throw GraphError("weight '" + info.name + "' exceeds weights blob");
      }
      Tensor w = Tensor::owning(info.dtype, info.shape);
      const std::size_t expected =
          static_cast<std::size_t>(w.numel()) * dtype_size(info.dtype);
      if (nbytes != expected) {
        throw GraphError("weight '" + info.name + "' byte size mismatch");
      }
      std::memcpy(w.data<float>(), blob.data() + offset, nbytes);
      model.weights.emplace(id, std::move(w));
    }
  }

  auto id_of = [&](const std::string& name) -> TensorId {
    auto it = name_to_id.find(name);
    if (it == name_to_id.end()) {
      throw GraphError("node references unknown tensor: " + name);
    }
    return it->second;
  };

  for (const auto& jn : j.at("nodes")) {
    Node node;
    node.op_type = jn.at("op").get<std::string>();
    for (const auto& in : jn.at("inputs")) node.inputs.push_back(id_of(in));
    for (const auto& out : jn.at("outputs")) node.outputs.push_back(id_of(out));
    if (jn.contains("attrs")) {
      for (const auto& [k, v] : jn.at("attrs").items()) {
        node.attrs.emplace(k, parse_attr(v));
      }
    }
    g.add_node(std::move(node));
  }

  for (const auto& in : j.at("inputs")) g.mark_input(id_of(in));
  for (const auto& out : j.at("outputs")) g.mark_output(id_of(out));

  return model;
}

}  // namespace mtrt::frontend
