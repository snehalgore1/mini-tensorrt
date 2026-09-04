---
description: Run a benchmark and record the measured result
argument-hint: <what changed>
---

Benchmark the current state and record it. What changed: $ARGUMENTS

1. **Verify the build type.** Confirm we are measuring a `Release` build. If the only
   build directory is Debug, rebuild Release first. Never report a Debug timing.

2. **Capture the environment**: machine, CPU, compiler version, and the exact
   optimization flags in use. These go into the table.

3. **Run the benchmark.** Warm up for at least 20 iterations, then measure at least 200.
   Report p50 and p95, not the mean. If variance across runs exceeds a few percent, say
   so rather than reporting a single number as if it were stable.

4. **Run the ablation.** Re-run with the change disabled via its runtime flag, on the
   same machine in the same session. A before/after from different sessions is not a
   valid comparison.

5. **Update `docs/RESULTS.md`** with the real numbers in the appropriate table. Replace
   `TBD` cells only with measured values.

6. **Interpret honestly.** If the change made things slower or made no difference, say
   that plainly and give your best explanation of why. Do not bury it. A well-explained
   negative result is worth more here than a flattering number.

7. **Propose a commit message** stating the optimization and its measured effect.
