#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
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
  // on each. Blocks until all chunks finish.
  void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn);

 private:
  void worker_loop();

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  int active_ = 0;      // tasks currently running or queued
  bool stop_ = false;
};

}  // namespace mtrt
