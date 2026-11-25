"""Simulates the PROPOSED GPU top-k and dispatch kernels (docs/moe/GPU_DISPATCH_DESIGN.md) at
warp granularity, and checks them against the CPU algorithms. The checkout has no GPU, so this
validates the LOGIC before any CUDA exists; the CUDA tests will compare device outputs with the
CPU kernels exactly.

Dispatch ("rank by counting", one thread block per expert, no atomics, no workspace):
    block e, pass 1:  start_e = #{a : idx[a] < e}          (= offsets[e])
    block e, pass 2:  walk a = 0..Tk-1 in chunks of blockDim. For the threads whose
                      assignment has idx[a] == e: rank = (# such threads before me in this
                      chunk, via warp ballot + popc + a scan of per-warp counts) + a running
                      cursor; j = start_e + rank;  src[j] = a;  inv[a] = j.
    offsets[e] = start_e (block e), offsets[E] = Tk (block E-1 also writes it).

Top-k ("k rounds of warp argmax", one warp per token, lane e holds logit e for E <= 32):
    each round: warp-reduce to the max under the key (value desc, index asc); the winner is
    recorded and its lane drops out; then softmax over the k winners.

Run: .venv/bin/python python/sim_gpu_dispatch.py
"""

from __future__ import annotations

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from verify_counting_sort import counting_sort  # the CPU algorithm (as in kernels_moe.cpp)

WARP = 32


def popc(x: int) -> int:
    return bin(x).count("1")


def gpu_dispatch_sim(flat: list[int], E: int, block: int = 256):
    """Returns src, inv, offsets exactly as the proposed kernel would write them, plus a
    per-slot write counter used to prove no slot is written twice and none is missed."""
    Tk = len(flat)
    src = [-1] * Tk
    inv = [-1] * Tk
    off = [-1] * (E + 1)
    writes_src = [0] * Tk
    writes_inv = [0] * Tk
    for e in range(E):                      # grid = E blocks; blocks share NOTHING
        # pass 1: start_e = count of assignments with a smaller expert id
        start = 0
        for c in range(0, Tk, block):
            for w in range(0, block, WARP):
                lanes = range(c + w, min(c + w + WARP, Tk))
                ballot = sum(1 << (a - c - w) for a in lanes if flat[a] < e)
                start += popc(ballot)       # (block-reduced in the kernel)
        off[e] = start
        if e == E - 1:
            off[E] = Tk
        # pass 2: stable compaction of this expert's assignments
        cursor = 0
        for c in range(0, Tk, block):
            warp_counts = []
            warp_ballots = []
            for w in range(0, block, WARP):
                lanes = range(c + w, min(c + w + WARP, Tk))
                b = sum(1 << (a - c - w) for a in lanes if flat[a] == e)
                warp_ballots.append(b)
                warp_counts.append(popc(b))
            warp_prefix = [sum(warp_counts[:i]) for i in range(len(warp_counts))]  # smem scan
            for wi, w in enumerate(range(0, block, WARP)):
                for lane in range(WARP):
                    a = c + w + lane
                    if a >= Tk or flat[a] != e:
                        continue
                    lower = warp_ballots[wi] & ((1 << lane) - 1)    # lanes before me
                    j = start + cursor + warp_prefix[wi] + popc(lower)
                    src[j] = a
                    inv[a] = j
                    writes_src[j] += 1
                    writes_inv[a] += 1
            cursor += sum(warp_counts)      # __syncthreads, then advance
    return src, inv, off, writes_src, writes_inv


def gpu_topk_sim(logits: list[float], k: int):
    """k rounds of a warp argmax with key (value desc, index asc). Lane e holds logits[e]."""
    taken = [False] * len(logits)
    idx = []
    for _ in range(k):
        best = None
        for e, v in enumerate(logits):      # = the butterfly reduction's result
            if taken[e]:
                continue
            if best is None or v > logits[best] or (v == logits[best] and e < best):
                best = e
        taken[best] = True
        idx.append(best)
    return idx


def cpu_topk(logits: list[float], k: int):
    """The CPU kernel's strict-> insertion (kernels_moe.cpp, moe_topk_f32)."""
    val, ids = [], []
    for e, v in enumerate(logits):
        if len(val) == k and not (v > val[-1]):
            continue
        p = len(val) if len(val) < k else k - 1
        if len(val) < k:
            val.append(v)
            ids.append(e)
        while p > 0 and v > val[p - 1]:
            val[p], ids[p] = val[p - 1], ids[p - 1]
            p -= 1
        val[p], ids[p] = v, e
    return ids


def main():
    rng = random.Random(0)
    print("== dispatch: proposed GPU kernel vs CPU counting sort ==")
    n = 0
    for E, k, Tmax in [(4, 1, 4), (4, 2, 3), (5, 2, 3), (4, 4, 3)]:
        for T in range(1, Tmax + 1):
            for routing in itertools.product(itertools.permutations(range(E), k), repeat=T):
                flat = [e for row in routing for e in row]
                s, i, o, ws, wi = gpu_dispatch_sim(flat, E, block=64)
                cs, ci, co, _, _ = counting_sort(flat, E)
                assert (s, i, o) == (cs, ci, co) and ws == [1] * len(flat) == wi
                n += 1
    for _ in range(1500):
        E, k = rng.choice([(32, 4), (8, 4), (128, 4)])
        T = rng.choice([1, 2, 3, 17, 64, 100, 300])
        hot = rng.sample(range(E), k) if rng.random() < 0.3 else None
        flat = []
        for _t in range(T):
            flat += hot if hot else rng.sample(range(E), k)
        block = rng.choice([32, 64, 256])
        s, i, o, ws, wi = gpu_dispatch_sim(flat, E, block)
        assert (s, i, o) == counting_sort(flat, E)[:3]
        assert ws == [1] * len(flat) == wi
        n += 1
    print(f"  {n} routings (exhaustive small grids + random incl. E=128, same-k, T=1, "
          f"partial last chunks): identical src/inv/offsets, every slot written exactly once")

    print("== top-k: warp argmax rounds vs CPU strict-> insertion ==")
    m = 0
    for _ in range(20000):
        E = rng.choice([8, 32])
        # small integer logits -> many ties on purpose
        logits = [float(rng.randint(-3, 3)) for _ in range(E)]
        assert gpu_topk_sim(logits, 4) == cpu_topk(logits, 4), logits
        m += 1
    print(f"  {m} tie-heavy logit rows: identical expert ids and order")
    print("\nALL PASSED")


if __name__ == "__main__":
    main()
