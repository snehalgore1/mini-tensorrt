#include "backends/cpu/thread_pool.h"

#include <algorithm>

namespace mtrt {

ThreadPool::ThreadPool(int num_threads) {
  int n = num_threads > 0 ? num_threads
                          : static_cast<int>(std::thread::hardware_concurrency());
  if (n < 1) n = 1;
  for (int i = 0; i < n; ++i) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mtx_);
    stop_ = true;
  }
  cv_.notify_all();
  for (std::thread& t : workers_) t.join();
}

void ThreadPool::worker_loop() {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty()) return;
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (--active_ == 0) done_cv_.notify_all();
    }
  }
}

void ThreadPool::parallel_for(int64_t n,
                              const std::function<void(int64_t, int64_t)>& fn) {
  if (n <= 0) return;
  const int t = size();
  // One chunk per worker (contiguous ranges); simple and cache-friendly.
  const int64_t chunk = (n + t - 1) / t;
  std::vector<std::pair<int64_t, int64_t>> ranges;
  for (int64_t b = 0; b < n; b += chunk) ranges.emplace_back(b, std::min(b + chunk, n));

  {
    std::lock_guard<std::mutex> lock(mtx_);
    active_ += static_cast<int>(ranges.size());
    for (const auto& r : ranges) {
      tasks_.emplace([&fn, r] { fn(r.first, r.second); });
    }
  }
  cv_.notify_all();

  std::unique_lock<std::mutex> lock(mtx_);
  done_cv_.wait(lock, [this] { return active_ == 0; });
}

}  // namespace mtrt
