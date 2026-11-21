"""Checks the scratch-free stable counting sort used by MoEDispatch (D-MoE-5).

The algorithm is written here in plain Python, exactly as the C++ kernel will do it. It reuses
`offsets` as the write cursor, so it needs no extra memory. This script does three things:

  1. A step-by-step TRACE on a small example with empty experts.
  2. An EXHAUSTIVE check of every routing for small (T, E, k), plus random larger ones,
     asserting these invariants:
       - each output slot j is written exactly once (nothing overwritten, nothing lost)
       - while scattering, cursor[e] stays inside [start_e, end_e)  (it never runs into
         the next expert's block)
       - the final offsets: offsets[0] = 0, non-decreasing, offsets[E] = T*k,
         offsets[e+1] - offsets[e] = count of e, which is 0 for an empty expert
       - src is a permutation of 0..T*k-1, and inv[src[j]] == j
       - stability: within each expert, src is strictly ascending
       - row j of segment e really belongs to expert e: idx_flat[src[j]] == e
  3. A comparison against moe_ref.dispatch (an independent implementation, using
     bincount + cumsum).

Run: .venv/bin/python python/verify_counting_sort.py
"""

from __future__ import annotations

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))


def counting_sort(flat: list[int], E: int, trace=None):
    """flat[a] = expert of assignment a (a = token*k + slot). Returns (src, inv, offsets)."""
    n = len(flat)
    off = [0] * (E + 1)
    src = [-1] * n
    inv = [-1] * n
    writes = [0] * n                      # test instrumentation only: writes per slot

    for e in flat:                        # 1. histogram into off[e+1]
        off[e + 1] += 1
    counts = off[1:]
    if trace: trace("1 histogram  ", off)
    for e in range(E):                    # 2. inclusive scan -> off[e] = start of e
        off[e + 1] += off[e]
    starts = off[:E]
    ends = off[1:]
    if trace: trace("2 prefix sum ", off)
    for a, e in enumerate(flat):          # 3. scatter, a ascending -> stable
        j = off[e]
        assert starts[e] <= j < ends[e], "cursor left its expert's block"
        off[e] += 1
        src[j] = a
        inv[a] = j
        writes[j] += 1
    if trace: trace("3 after scatter (off[e] = END of e)", off)
    for e in range(E, 0, -1):             # 4. shift right; MUST go downward (reads before writes)
        off[e] = off[e - 1]
    off[0] = 0
    if trace: trace("4 shifted -> final offsets", off)
    return src, inv, off, writes, counts


def check(flat: list[int], E: int, k: int):
    n = len(flat)
    src, inv, off, writes, counts = counting_sort(flat, E)
    assert writes == [1] * n, f"slot written != once: {writes}"
    assert off[0] == 0 and off[E] == n
    assert all(off[e] <= off[e + 1] for e in range(E))
    assert all(off[e + 1] - off[e] == counts[e] for e in range(E))
    assert sorted(src) == list(range(n))
    assert all(inv[src[j]] == j for j in range(n))
    for e in range(E):
        seg = src[off[e]:off[e + 1]]
        assert all(flat[a] == e for a in seg), "row in wrong expert segment"
        assert all(seg[i] < seg[i + 1] for i in range(len(seg) - 1)), "not stable"
    return src, inv, off


def main():
    print("== 1. trace: T=3, k=2, E=5 (experts 1 and 4 receive nothing) ==")
    idx = [[0, 2], [3, 0], [2, 3]]           # rows = tokens, columns = slots
    flat = [e for row in idx for e in row]
    print(f"idx={idx}  flat (a = t*k+slot) = {flat}")
    src, inv, off, writes, _ = counting_sort(
        flat, 5, trace=lambda s, o: print(f"  {s:<38} offsets={o}"))
    print(f"  src={src}  (token = a//k: {[a // 2 for a in src]})")
    print(f"  inv={inv}  writes per slot={writes}")
    check(flat, 5, 2)

    print("\n== 2. exhaustive + random ==")
    total = 0
    for E, k, Tmax in [(4, 1, 4), (4, 2, 3), (5, 2, 3), (4, 4, 3), (6, 3, 2)]:
        choices = list(itertools.permutations(range(E), k))   # top-k: distinct experts
        for T in range(1, Tmax + 1):
            for routing in itertools.product(choices, repeat=T):
                check([e for row in routing for e in row], E, k)
                total += 1
    print(f"  exhaustive: {total} routings over (E,k,T) grids -- all invariants hold")
    rng = random.Random(0)
    for _ in range(2000):
        E, k = rng.choice([(8, 4), (32, 4), (32, 1), (16, 2)])
        T = rng.randint(1, 64)
        hot = rng.sample(range(E), k) if rng.random() < 0.3 else None  # skewed: same-k
        flat = []
        for _t in range(T):
            flat += hot if hot else rng.sample(range(E), k)
        check(flat, E, k)
    print("  random: 2000 routings (E up to 32, T up to 64, 30% all-same-experts) -- all hold")

    print("\n== 3. agrees with moe_ref.dispatch (bincount/cumsum implementation) ==")
    import torch
    import moe_ref
    for _ in range(300):
        E, k, T = 32, 4, rng.randint(1, 64)
        idx_t = torch.stack([torch.randperm(E)[:k] for _ in range(T)])
        src, inv, off = check(idx_t.reshape(-1).tolist(), E, k)
        _, s2, i2, o2 = moe_ref.dispatch(torch.zeros(T, 1), idx_t, E)
        assert (src, inv, off) == (s2.tolist(), i2.tolist(), o2.tolist())
    print("  300 random E=32,k=4 routings: identical src/inv/offsets")
    print("\nALL PASSED")


if __name__ == "__main__":
    main()
