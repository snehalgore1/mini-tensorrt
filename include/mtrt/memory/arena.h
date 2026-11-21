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

  // Contiguous view of `dtype` at a byte offset. The buffer is untyped storage
  // (allocated as f32 only for its element count); the view carries the requested
  // dtype, so int32 intermediates (MoE idx/src/inv/offsets) are tagged kI32, not
  // reinterpreted as f32. Offsets come from the planner, which aligns every tensor
  // to kArenaAlignmentBytes (16), a multiple of every dtype's size.
  Tensor view_at(int64_t byte_offset, DType dtype,
                 const std::vector<int64_t>& shape) const {
    MTRT_ASSERT(buffer_.defined(), "view_at on an empty arena");
    int64_t nbytes_view = static_cast<int64_t>(dtype_size(dtype));
    for (int64_t d : shape) nbytes_view *= d;
    MTRT_ASSERT(byte_offset + nbytes_view <= nbytes(), "arena view out of bounds");
    (void)nbytes_view;
    return buffer_.view_bytes(dtype, shape, byte_offset);
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
