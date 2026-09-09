#include "mtrt/profiler/profiler.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace mtrt {

void Profiler::write_chrome_trace(const std::string& path) const {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot open trace file for write: " + path);

  // Emit microseconds with fixed 3-decimal precision, i.e. exact nanoseconds.
  // Default ostream precision is 6 significant figures, which for ts in the
  // thousands-of-us range silently drops the sub-10ns digits -- enough that a
  // later event's rounded start can fall below the prior event's rounded end,
  // producing phantom overlaps Perfetto can't nest (it spills them to overflow
  // tracks). The executor times ops strictly sequentially, so the true data
  // never overlaps; this keeps the written data faithful to that.
  f << std::fixed << std::setprecision(3);

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
