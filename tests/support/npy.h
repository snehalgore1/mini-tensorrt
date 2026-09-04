#pragma once

// Minimal .npy reader for tests (DESIGN D8): float32, C-contiguous only. Keeps
// the C++ test suite decoupled from Python -- fixtures are regenerated offline
// and loaded here. Not part of the shipping library; test support only.

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mtrt/tensor.h"

namespace mtrt::testing {

struct NpyArray {
  std::vector<int64_t> shape;
  std::vector<float> data;

  int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
};

inline NpyArray load_npy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open npy file: " + path);

  char magic[6];
  f.read(magic, 6);
  if (std::string(magic, 6) != std::string("\x93NUMPY", 6)) {
    throw std::runtime_error("not a .npy file: " + path);
  }
  std::uint8_t major = 0, minor = 0;
  f.read(reinterpret_cast<char*>(&major), 1);
  f.read(reinterpret_cast<char*>(&minor), 1);

  std::uint32_t header_len = 0;
  if (major == 1) {
    std::uint16_t len16 = 0;
    f.read(reinterpret_cast<char*>(&len16), 2);  // little-endian on our targets
    header_len = len16;
  } else {
    std::uint32_t len32 = 0;
    f.read(reinterpret_cast<char*>(&len32), 4);
    header_len = len32;
  }

  std::string header(header_len, '\0');
  f.read(&header[0], header_len);

  if (header.find("'<f4'") == std::string::npos) {
    throw std::runtime_error("npy reader supports only little-endian float32: " +
                             path);
  }
  if (header.find("'fortran_order': False") == std::string::npos) {
    throw std::runtime_error("npy reader supports only C-order arrays: " + path);
  }

  // Parse the shape tuple, e.g. "'shape': (3, 4)" or "(3,)" or "()".
  NpyArray arr;
  const std::size_t sp = header.find("'shape'");
  const std::size_t lp = header.find('(', sp);
  const std::size_t rp = header.find(')', lp);
  const std::string dims = header.substr(lp + 1, rp - lp - 1);
  std::string tok;
  for (char c : dims) {
    if (c == ',' || c == ' ') {
      if (!tok.empty()) {
        arr.shape.push_back(std::stoll(tok));
        tok.clear();
      }
    } else {
      tok.push_back(c);
    }
  }
  if (!tok.empty()) arr.shape.push_back(std::stoll(tok));

  arr.data.resize(static_cast<std::size_t>(arr.numel()));
  f.read(reinterpret_cast<char*>(arr.data.data()),
         static_cast<std::streamsize>(arr.data.size() * sizeof(float)));
  if (!f) throw std::runtime_error("npy file truncated: " + path);
  return arr;
}

// Load a golden fixture by base name from the compiled-in goldens directory.
inline NpyArray load_golden(const std::string& name) {
  return load_npy(std::string(MTRT_GOLDENS_DIR) + "/" + name + ".npy");
}

// Build a contiguous owning FP32 Tensor from an NpyArray.
inline Tensor tensor_from_npy(const NpyArray& a) {
  Tensor t = Tensor::owning(DType::kF32, a.shape);
  std::copy(a.data.begin(), a.data.end(), t.data<float>());
  return t;
}

}  // namespace mtrt::testing
