#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// A small std::thread pool with a blocking parallel_for. Own pool rather than
// OpenMP (DESIGN D6): Apple Clang ships no OpenMP by default, so this removes a
// setup tax for anyone who clones the project, and the pool is itself a small
// systems artifact. Reused by the threaded GEMM and later the executor.

namespace mtrt {

class ThreadPool {
 public:
  // num_threads <= 0 uses hardware_concurrency().
  explicit ThreadPool(int num_threads = 0);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return static_cast<int>(workers_.size()); }

  // Split [0, n) into contiguous chunks across the pool and run fn(begin, end)
  // on each. Blocks until all chunks finish. Templated with a captureless
  // trampoline so no std::function is constructed -- a lambda that captured more
  // than the small-buffer size would otherwise heap-allocate on every call, i.e.
  // inside Executor::run(). `fn` lives on the caller's stack for the (blocking)
  // duration, so storing a pointer to it is safe.
  template <class F>
  void parallel_for(int64_t n, F&& fn) {
    if (n <= 0) return;
    const int t = size();
    const int64_t chunk = (n + t - 1) / t;
    void (*tramp)(void*, int64_t, int64_t) = [](void* p, int64_t a, int64_t b) {
      (*static_cast<std::remove_reference_t<F>*>(p))(a, b);
    };
    {
      std::lock_guard<std::mutex> lock(mtx_);
      fn_ctx_ = static_cast<void*>(&fn);
      fn_tramp_ = tramp;
      ranges_.clear();  // keeps capacity: grow-only, no steady-state allocation
      for (int64_t b = 0; b < n; b += chunk)
        ranges_.emplace_back(b, std::min(b + chunk, n));
      next_ = 0;
      active_ = static_cast<int>(ranges_.size());
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> lock(mtx_);
    done_cv_.wait(lock, [this] { return active_ == 0; });
  }

 private:
  void worker_loop();

  std::vector<std::thread> workers_;
  // Dispatch is allocation-free: one job runs `fn_` over the contiguous index
  // ranges in `ranges_` (a reused, grow-only buffer), which workers claim via the
  // `next_` cursor. This replaced a per-chunk std::function heap-allocated onto a
  // task queue, so a threaded GEMM no longer allocates in Executor::run()
  // (invariant 4). Only one parallel_for runs at a time (called from the executor).
  void* fn_ctx_ = nullptr;                                 // -> caller's callable
  void (*fn_tramp_)(void*, int64_t, int64_t) = nullptr;    // invokes it
  std::vector<std::pair<int64_t, int64_t>> ranges_;
  std::size_t next_ = 0;  // next range index for a worker to claim
  std::mutex mtx_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  int active_ = 0;      // ranges currently running or unclaimed
  bool stop_ = false;
};

}  // namespace mtrt
