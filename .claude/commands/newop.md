---
description: Add a new operator, golden test first
argument-hint: <OpName> [notes]
---

Add the operator `$1` to MiniTensorRT. Notes from me: $ARGUMENTS

Follow this order exactly. Do not skip ahead to the kernel.

1. **Confirm it is in scope.** Check `docs/ROADMAP.md` for the current week and confirm
   some model in `models/` actually needs `$1`. If nothing needs it, stop and tell me
   this is a scope violation instead of implementing it.

2. **Generate the golden fixture.** Add `$1` to `python/gen_goldens.py` with a small
   representative shape and at least one awkward shape (non-power-of-two, batch > 1).
   Run it and confirm the `.npy` files land in `models/goldens/`.

3. **Write the failing test.** Add a GoogleTest case in `tests/ops/` that loads the
   fixture and compares. Build and confirm it fails for the right reason, not because
   of a missing symbol or a bad path.

4. **Implement the naive kernel.** Correct and obvious, in `src/ops/`. No optimization,
   no SIMD, no cleverness. Register it in the op registry keyed on
   `(op_type, dtype)`.

5. **Confirm the test passes.** Then add shape-inference support in `src/graph/` so the
   loader can resolve output shapes for `$1` statically.

6. **Report back** with: the tolerance used and why, any shape or layout assumptions
   you baked in, and whether this operator is likely to show up hot in the profile.

Do not optimize the kernel. That is a separate, profile-driven task.
