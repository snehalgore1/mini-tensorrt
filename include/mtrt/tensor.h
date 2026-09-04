#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "mtrt/dtype.h"
#include "mtrt/support/assert.h"

namespace mtrt {

// Ownership model (core invariant 1):
//   - owning()  allocates storage; this is the ONLY allocating path.
//   - view()    shares storage explicitly; never allocates.
//   - copy      is SHALLOW (refcount bump). Never a deep copy.
//   - move      steals the handle, leaves the source empty.
//   - clone()   is the one and only deep copy.
// A stray copy in a hot loop costs an atomic increment, never a malloc+memcpy.
class Tensor {
 public:
  Tensor() = default;

  // Copy = shallow (shares storage). Move = steal. Both defaulted.
  Tensor(const Tensor&) = default;
  Tensor& operator=(const Tensor&) = default;
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(Tensor&&) noexcept = default;

  // Allocate a fresh, contiguous, row-major tensor.
  static Tensor owning(DType dtype, std::vector<int64_t> shape);

  // Explicit view over this tensor's storage with new shape/strides and an
  // element offset. Shares storage (refcount++), never allocates.
  Tensor view(std::vector<int64_t> shape, std::vector<int64_t> strides,
              int64_t offset_elems = 0) const;

  // Contiguous reshape sharing storage. Requires *this contiguous and the same
  // element count.
  Tensor reshape(std::vector<int64_t> shape) const;

  // The only deep copy: fresh storage + element copy.
  Tensor clone() const;

  // Typed data access. Asserts dtype match and non-empty in Debug.
  template <class T>
  T* data() {
    check_dtype<T>();
    return reinterpret_cast<T*>(data_);
  }
  template <class T>
  const T* data() const {
    check_dtype<T>();
    return reinterpret_cast<const T*>(data_);
  }

  DType dtype() const { return dtype_; }
  const std::vector<int64_t>& shape() const { return shape_; }
  const std::vector<int64_t>& strides() const { return strides_; }
  int64_t rank() const { return static_cast<int64_t>(shape_.size()); }
  int64_t numel() const;
  bool is_contiguous() const;
  bool defined() const { return storage_ != nullptr; }

  // Number of times this tensor's storage buffer is referenced. Test aid.
  long use_count() const { return storage_.use_count(); }

 private:
  template <class T>
  void check_dtype() const {
    MTRT_ASSERT(storage_ != nullptr, "data() on an empty tensor");
    MTRT_ASSERT(sizeof(T) == dtype_size(dtype_), "data<T>() dtype size mismatch");
  }

  std::shared_ptr<void> storage_;  // keep-alive handle; shared_ptr only for storage
  std::byte* data_ = nullptr;      // points into storage_ (offset for a view)
  DType dtype_ = DType::kF32;
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;   // in elements, row-major default
};

// Contiguous row-major strides for a shape, in elements.
std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape);

}  // namespace mtrt
