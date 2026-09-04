#pragma once

#include <cstdint>
#include <vector>

#include "mtrt/support/assert.h"
#include "mtrt/tensor.h"

namespace mtrt {

// One contiguous buffer that planned intermediate tensors view into. Backed by a
// single owning Tensor, so arena views reuse the ordinary Tensor view mechanism
// (shared storage, refcount keeps the arena alive as long as any view lives).
class Arena {
 public:
  Arena() = default;

  explicit Arena(int64_t nbytes) {
    if (nbytes > 0) {
      const int64_t elems =
          nbytes / static_cast<int64_t>(dtype_size(DType::kF32));
      buffer_ = Tensor::owning(DType::kF32, {elems});
    }
  }

  // Contiguous view at a byte offset. FP32 only (matches the arena buffer).
  Tensor view_at(int64_t byte_offset, DType dtype,
                 const std::vector<int64_t>& shape) const {
    MTRT_ASSERT(dtype == DType::kF32, "arena is FP32 only");
    MTRT_ASSERT(buffer_.defined(), "view_at on an empty arena");
    const int64_t off_elems =
        byte_offset / static_cast<int64_t>(dtype_size(dtype));
    return buffer_.view(shape, contiguous_strides(shape), off_elems);
  }

  int64_t nbytes() const {
    return buffer_.defined()
               ? buffer_.numel() * static_cast<int64_t>(dtype_size(DType::kF32))
               : 0;
  }

 private:
  Tensor buffer_;
};

}  // namespace mtrt
