#pragma once

// Debug-only assertion macro (core invariant 7). Active when NDEBUG is not
// defined (Debug builds); compiles to nothing in Release. Use MTRT_ASSERT for
// shape, bounds, and dtype preconditions -- never raw assert.

#include <cstdio>
#include <cstdlib>

namespace mtrt::detail {

[[noreturn]] inline void assert_fail(const char* expr, const char* msg,
                                     const char* file, int line) {
  std::fprintf(stderr, "MTRT_ASSERT failed: %s\n  (%s)\n  at %s:%d\n",
               msg, expr, file, line);
  std::abort();
}

}  // namespace mtrt::detail

#ifdef NDEBUG
#define MTRT_ASSERT(cond, msg) ((void)0)
#else
#define MTRT_ASSERT(cond, msg)                                            \
  do {                                                                    \
    if (!(cond)) {                                                        \
      ::mtrt::detail::assert_fail(#cond, (msg), __FILE__, __LINE__);      \
    }                                                                     \
  } while (0)
#endif
