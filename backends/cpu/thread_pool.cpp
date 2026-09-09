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
    std::pair<int64_t, int64_t> range;
    {
      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait(lock, [this] { return stop_ || next_ < ranges_.size(); });
      if (stop_ && next_ >= ranges_.size()) return;
      range = ranges_[next_++];  // claim one range
    }
    fn_tramp_(fn_ctx_, range.first, range.second);  // run the claimed range
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (--active_ == 0) done_cv_.notify_all();
    }
  }
}
// parallel_for is a template, defined inline in thread_pool.h (no std::function).

}  // namespace mtrt
