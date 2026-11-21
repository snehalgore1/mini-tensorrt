# Deep dive: Milestone 1A (questions only; you write the answers)

Answer each question in your own words, without looking at the code. Then check your answer
against the code, the tests, and `docs/ENGINEERING_DECISIONS.md`. A question you can't answer
cleanly marks the place to re-read.

## Semantics
1. Write the GPT-OSS MoE block as equations, from x to the output. Where does the residual
   come from, and which tensor is normalized?
2. Why softmax *after* top-k? What would change numerically, and in the model's behaviour, if
   you applied softmax over all 32 experts first and then took the top 4?
3. SwiGLU in GPT-OSS is interleaved and clamped asymmetrically. Write it out. What
   plausible-looking output would a first-half/second-half split produce, and which test
   would catch it?
4. Why does the FP64 comparison exist when the kernels run in FP32? What bug did it catch, and
   why could the FP32 comparison not have caught it?

## Static shapes and dispatch
5. "MoE's dynamism is in the values, not the shapes." Explain using the buffer sizes. What
   would break if a design dropped tokens over a capacity limit?
6. Trace the counting sort by hand for idx = [[0,2],[3,0],[2,3]] with E = 5. Give `offsets`
   after each of the four steps, then `src` and `inv`.
7. Prove that step 3 never overwrites a slot and never loses an assignment. What does an empty
   expert look like at each step?
8. Why must step 4 (the shift) loop downward? What happens if it loops upward?
9. Why keep both `src` and `inv`? Which stage uses each one? What would combine look like
   with only `src`, and why is that worse on a GPU?
10. Why is the dispatch stable, and why does that matter for testing and determinism?

## Kernels
11. Top-k: explain how strict `>` insertion with ascending expert order implements "lower
    index wins ties". What happens with NaN logits? (Check what the code does.)
12. `MoEGroupedMatMul`: why must each expert's rows be contiguous for this GEMM? What does the
    GEMM assume about strides, alignment, and the output buffer?
13. Why are the weights transposed at export rather than adding a transposed-B GEMM? Why does
    the transpose not break the SwiGLU interleaving?
14. Why does combine gather instead of scatter-add? What does that buy on CPU, and later on
    GPU?
15. Why does RMSNorm accumulate in double? What is the error, roughly, with an fp32 sum over
    2880 terms?

## Runtime invariants
16. Where could the MoE layer allocate inside `Executor::run()`? List every candidate and how
    each one was ruled out.
17. Explain the GEMM first-touch problem. Why did 17 allocations appear, why is the problem
    specific to routing-dependent workloads, and how does the barrier guarantee that every
    worker warms its scratch?
18. The arena bug: why did Release pass while Debug aborted? Why is "it works because both
    types are 4 bytes" a bug, not a non-issue?
19. Why does the `AllocInvariant` test pass in Release but fail in the Debug+sanitizer build,
    including before any MoE change? (Find out; this one is still open.)

## Numerics and testing
20. Define the combined tolerance. Why is max rel error reported only over |ref| ≥ 0.1?
21. Why does the whole-layer check use the summed-term magnitude instead of |ref|? Use the
    clamp case (output 6.07 from terms totalling 585) to explain. Why doesn't this hide real
    bugs?
22. Every stage passed with golden inputs, but the whole layer failed the |ref| rule on the
    clamp case. Why do per-stage and end-to-end errors differ?

## Performance (answer from the bench_moe numbers)
23. What is the derived weight-streaming bandwidth at T = 1? How does it compare with what the
    machine can stream? What in the GEMM explains the gap?
24. Why does grouped ≈ naive at T = 1 but pull ahead at larger T? Predict the speedup from the
    average rows per expert, then compare with the measurement.
25. How does the "same4" routing change the picture compared with uniform routing, for both
    time and experts touched? Which hypothesis in the 1B plan does this preview?
26. What is the planned arena size vs the sum of intermediates at T = 256? Which tensors
    dominate, and which decision would shrink it most?
