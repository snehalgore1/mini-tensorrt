#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "mtrt/graph.h"

// Static memory planning (DESIGN D4). Because all shapes are known at load time
// (D1), we can assign every intermediate a fixed offset in one contiguous arena
// ahead of execution, reusing space between tensors whose lifetimes don't
// overlap. The executor then allocates nothing at run time.

namespace mtrt {

// Byte alignment for arena slots. 16 bytes = NEON float32x4, so planned tensors
// stay vector-aligned for the Week 5 SIMD work.
constexpr int64_t kArenaAlignmentBytes = 16;

// Live interval of one planned tensor over the topological order: it is live
// from the step that writes it through the last step that reads it (inclusive).
struct LiveInterval {
  TensorId id;
  int first_write;  // step index in topo order
  int last_read;    // step index in topo order (== first_write if never read)
  int64_t nbytes;
};

struct MemoryPlan {
  std::unordered_map<TensorId, int64_t> offset;  // byte offset of each planned tensor
  int64_t total_bytes = 0;                       // arena size (peak intermediate bytes)
  std::vector<LiveInterval> intervals;           // for inspection / testing
};

// True if two live intervals overlap (inclusive endpoints). A node that reads A
// and writes B needs both at once, so touching endpoints count as overlap.
inline bool intervals_overlap(const LiveInterval& a, const LiveInterval& b) {
  return a.first_write <= b.last_read && b.first_write <= a.last_read;
}

// Plan arena offsets for all kIntermediate tensors of the graph using greedy-by-
// size assignment: sort by size descending, give each the lowest aligned offset
// that does not overlap an already-placed, lifetime-conflicting tensor.
MemoryPlan plan_memory(const Graph& graph);

}  // namespace mtrt
