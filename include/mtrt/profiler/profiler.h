#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Per-node profiler emitting Chrome Trace Event JSON (DESIGN D7), viewable in
// chrome://tracing or Perfetto. Near-zero implementation cost for a real flame
// chart -- the best single visual artifact for the README.

namespace mtrt {

struct TraceEvent {
  std::string name;   // op type
  int64_t ts_ns;      // start, nanoseconds from run start
  int64_t dur_ns;     // duration, nanoseconds
};

class Profiler {
 public:
  void record(std::string name, int64_t ts_ns, int64_t dur_ns) {
    events_.push_back(TraceEvent{std::move(name), ts_ns, dur_ns});
  }
  void clear() { events_.clear(); }
  const std::vector<TraceEvent>& events() const { return events_; }

  // Write Chrome Trace Event format (ts/dur in microseconds, fractional).
  void write_chrome_trace(const std::string& path) const;

 private:
  std::vector<TraceEvent> events_;
};

}  // namespace mtrt
