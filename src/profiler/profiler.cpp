#include "mtrt/profiler/profiler.h"

#include <fstream>
#include <stdexcept>

namespace mtrt {

void Profiler::write_chrome_trace(const std::string& path) const {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot open trace file for write: " + path);

  // Complete events ("ph":"X") with a duration; all on one pid/tid so they stack
  // into a single flame track in the order they ran.
  f << "{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n";
  for (std::size_t i = 0; i < events_.size(); ++i) {
    const TraceEvent& e = events_[i];
    const double ts_us = e.ts_ns / 1000.0;
    const double dur_us = e.dur_ns / 1000.0;
    f << "  {\"name\":\"" << e.name << "\",\"cat\":\"op\",\"ph\":\"X\","
      << "\"pid\":1,\"tid\":1,\"ts\":" << ts_us << ",\"dur\":" << dur_us << "}";
    if (i + 1 < events_.size()) f << ",";
    f << "\n";
  }
  f << "]}\n";
}

}  // namespace mtrt
