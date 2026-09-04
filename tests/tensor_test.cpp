#include "mtrt/tensor.h"

#include <gtest/gtest.h>

using mtrt::DType;
using mtrt::Tensor;

TEST(Tensor, OwningConstructionShapeStridesNumel) {
  Tensor t = Tensor::owning(DType::kF32, {2, 3});
  EXPECT_EQ(t.rank(), 2);
  EXPECT_EQ(t.numel(), 6);
  EXPECT_EQ(t.shape(), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(t.strides(), (std::vector<int64_t>{3, 1}));
  EXPECT_TRUE(t.is_contiguous());
  EXPECT_TRUE(t.defined());
}

TEST(Tensor, OwningIsZeroInitialized) {
  Tensor t = Tensor::owning(DType::kF32, {4});
  const float* p = t.data<float>();
  for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(p[i], 0.0f);
}

TEST(Tensor, CopyIsShallowSharesStorage) {
  Tensor a = Tensor::owning(DType::kF32, {3});
  Tensor b = a;  // shallow copy
  EXPECT_EQ(a.use_count(), 2);
  b.data<float>()[0] = 42.0f;
  EXPECT_FLOAT_EQ(a.data<float>()[0], 42.0f);  // same storage
}

TEST(Tensor, ViewSharesStorageWriteVisibleInOriginal) {
  Tensor a = Tensor::owning(DType::kF32, {2, 3});
  Tensor v = a.reshape({3, 2});
  EXPECT_EQ(v.shape(), (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(a.use_count(), 2);
  v.data<float>()[5] = 7.0f;
  EXPECT_FLOAT_EQ(a.data<float>()[5], 7.0f);
}

TEST(Tensor, CloneIsDeepIndependentStorage) {
  Tensor a = Tensor::owning(DType::kF32, {3});
  a.data<float>()[0] = 1.0f;
  Tensor c = a.clone();
  EXPECT_EQ(c.use_count(), 1);  // fresh storage
  c.data<float>()[0] = 99.0f;
  EXPECT_FLOAT_EQ(a.data<float>()[0], 1.0f);  // original untouched
}

TEST(Tensor, MoveLeavesSourceEmpty) {
  Tensor a = Tensor::owning(DType::kF32, {3});
  Tensor b = std::move(a);
  EXPECT_TRUE(b.defined());
  EXPECT_FALSE(a.defined());  // NOLINT(bugprone-use-after-move)
}

TEST(Tensor, ContiguousStridesHelper) {
  EXPECT_EQ(mtrt::contiguous_strides({2, 3, 4}),
            (std::vector<int64_t>{12, 4, 1}));
}

#ifndef NDEBUG
// MTRT_ASSERT is only active in Debug builds; skip these in Release.
TEST(TensorDeath, DataOnEmptyTensorAsserts) {
  Tensor t;
  EXPECT_DEATH({ (void)t.data<float>(); }, "empty tensor");
}
#endif
