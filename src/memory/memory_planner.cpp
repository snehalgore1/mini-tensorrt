#include "mtrt/memory/memory_planner.h"

#include <algorithm>

namespace mtrt {

namespace {

int64_t align_up(int64_t x, int64_t a) { return (x + a - 1) / a * a; }

int64_t tensor_bytes(const TensorInfo& info) {
  int64_t n = 1;
  for (int64_t d : info.shape) n *= d;
  return n * static_cast<int64_t>(dtype_size(info.dtype));
}

}  // namespace

MemoryPlan plan_memory(const Graph& graph) {
  const std::vector<NodeId> order = graph.topo_order();

  // Compute live intervals for intermediate tensors over the topo order.
  // Sentinels: first_write = -1 (unset) means not yet produced.
  std::unordered_map<TensorId, LiveInterval> live;
  for (TensorId id = 0; id < graph.num_tensors(); ++id) {
    if (graph.tensor(id).kind == TensorKind::kIntermediate) {
      live[id] = LiveInterval{id, -1, -1, tensor_bytes(graph.tensor(id))};
    }
  }

  for (int step = 0; step < static_cast<int>(order.size()); ++step) {
    const Node& n = graph.node(order[static_cast<size_t>(step)]);
    for (TensorId out : n.outputs) {
      auto it = live.find(out);
      if (it != live.end() && it->second.first_write < 0) {
        it->second.first_write = step;
        it->second.last_read = step;  // at least live at its birth
      }
    }
    for (TensorId in : n.inputs) {
      auto it = live.find(in);
      if (it != live.end()) it->second.last_read = step;
    }
  }

  MemoryPlan plan;
  plan.intervals.reserve(live.size());
  for (auto& [id, iv] : live) plan.intervals.push_back(iv);

  // Greedy by size descending; deterministic tie-break by id.
  std::sort(plan.intervals.begin(), plan.intervals.end(),
            [](const LiveInterval& a, const LiveInterval& b) {
              if (a.nbytes != b.nbytes) return a.nbytes > b.nbytes;
              return a.id < b.id;
            });

  struct Placed {
    int64_t begin, end;  // byte range [begin, end)
    const LiveInterval* iv;
  };
  std::vector<Placed> placed;

  for (const LiveInterval& iv : plan.intervals) {
    // Byte ranges already occupied by lifetime-conflicting tensors.
    std::vector<std::pair<int64_t, int64_t>> blocked;
    for (const Placed& p : placed) {
      if (intervals_overlap(iv, *p.iv)) blocked.emplace_back(p.begin, p.end);
    }
    std::sort(blocked.begin(), blocked.end());

    // Lowest aligned offset with a gap large enough for iv.nbytes.
    int64_t off = 0;
    for (const auto& [b, e] : blocked) {
      if (off + iv.nbytes <= b) break;             // fits before this block
      off = std::max(off, align_up(e, kArenaAlignmentBytes));
    }

    plan.offset[iv.id] = off;
    placed.push_back(Placed{off, off + iv.nbytes, &iv});
    plan.total_bytes = std::max(plan.total_bytes, off + iv.nbytes);
  }

  plan.total_bytes = align_up(plan.total_bytes, kArenaAlignmentBytes);
  return plan;
}

}  // namespace mtrt
