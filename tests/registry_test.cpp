#include "mtrt/registry.h"

#include <gtest/gtest.h>

using namespace mtrt;

TEST(Registry, BuiltinsAreRegistered) {
  KernelRegistry reg;
  register_builtin_kernels(reg);
  EXPECT_NE(reg.lookup("Add", DType::kF32), nullptr);
  EXPECT_NE(reg.lookup("Relu", DType::kF32), nullptr);
}

TEST(Registry, UnknownOpReturnsNull) {
  KernelRegistry reg;
  register_builtin_kernels(reg);
  EXPECT_EQ(reg.lookup("Conv", DType::kF32), nullptr);
}

TEST(Registry, ManualRegistrationAndLookup) {
  KernelRegistry reg;
  KernelFn fn = [](const OpContext&) {};
  reg.register_kernel("Custom", DType::kF32, fn);
  EXPECT_EQ(reg.lookup("Custom", DType::kF32), fn);
}
