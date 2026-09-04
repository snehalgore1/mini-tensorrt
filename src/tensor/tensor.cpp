#include "mtrt/tensor.h"

#include <cstring>

namespace mtrt {

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size());
  int64_t acc = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = acc;
    acc *= shape[i];
  }
  return strides;
}

namespace {
int64_t product(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}
}  // namespace

Tensor Tensor::owning(DType dtype, std::vector<int64_t> shape) {
  Tensor t;
  t.dtype_ = dtype;
  t.shape_ = std::move(shape);
  t.strides_ = contiguous_strides(t.shape_);
  const std::size_t nbytes =
      static_cast<std::size_t>(product(t.shape_)) * dtype_size(dtype);
  std::byte* raw = new std::byte[nbytes];
  std::memset(raw, 0, nbytes);
  t.storage_ = std::shared_ptr<void>(
      raw, [](void* p) { delete[] static_cast<std::byte*>(p); });
  t.data_ = raw;
  return t;
}

Tensor Tensor::view(std::vector<int64_t> shape, std::vector<int64_t> strides,
                    int64_t offset_elems) const {
  MTRT_ASSERT(storage_ != nullptr, "view() on an empty tensor");
  MTRT_ASSERT(shape.size() == strides.size(), "view() rank/strides mismatch");
  Tensor v;
  v.dtype_ = dtype_;
  v.shape_ = std::move(shape);
  v.strides_ = std::move(strides);
  v.storage_ = storage_;  // refcount++, shares storage
  v.data_ = data_ + offset_elems * static_cast<int64_t>(dtype_size(dtype_));
  return v;
}

Tensor Tensor::reshape(std::vector<int64_t> shape) const {
  MTRT_ASSERT(is_contiguous(), "reshape() requires a contiguous tensor");
  MTRT_ASSERT(product(shape) == numel(), "reshape() element count mismatch");
  return view(shape, contiguous_strides(shape), 0);
}

Tensor Tensor::clone() const {
  MTRT_ASSERT(storage_ != nullptr, "clone() on an empty tensor");
  MTRT_ASSERT(is_contiguous(), "clone() currently supports contiguous tensors");
  Tensor c = owning(dtype_, shape_);
  const std::size_t nbytes =
      static_cast<std::size_t>(numel()) * dtype_size(dtype_);
  std::memcpy(c.data_, data_, nbytes);
  return c;
}

int64_t Tensor::numel() const { return product(shape_); }

bool Tensor::is_contiguous() const {
  return strides_ == contiguous_strides(shape_);
}

}  // namespace mtrt
