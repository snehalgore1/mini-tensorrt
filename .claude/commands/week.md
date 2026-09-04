---
description: Check progress against the current week's definition of done
---

Audit where this project actually stands.

1. Read `docs/ROADMAP.md` and identify the current week and its definition of done.

2. For each DoD item, verify it **against the repository**, not against your memory of
   this conversation. Build the project, run the tests, check that the files and results
   claimed to exist actually exist. Report each item as done, partial, or not started,
   with the evidence you used.

3. Check the scope tripwires at the bottom of `docs/ROADMAP.md`. Flag any that have been
   crossed.

4. Check that `docs/RESULTS.md` contains no invented numbers. Every non-`TBD` cell should
   trace to a benchmark that was actually run.

5. Verify the core invariants in `CLAUDE.md` still hold, in particular:
   - `src/graph/` does not include anything from `src/frontend/`
   - the executor does not allocate during `run()`
   - every registered operator has a golden test

6. Give me a short honest status: what is genuinely done, what is thinner than it looks,
   and the single highest-value next action. If the week's DoD is met, say so and tell me
   what to update before moving on.

Do not be generous in this assessment. The point is to catch drift early.
