# Milestone 1B hypotheses: pre-registered before their experiments

Each hypothesis is written down **before** its data exists:
- the claim
- the experiment that tests it
- the decision rule
- my prediction and the reasoning behind it

The verdict is added only after the measured outputs are pasted. It is presented for approval
(Checkpoint 5), and negative results are kept.

---

## H1: making the routing host-visible is a large share of V0 at small T

- **Claim.** In V0, the D2H copy of `offsets` plus the stream sync, together with the GPU
  idling while the host waits and then issues the expert GEMMs, is a large share of layer time
  when T is small.
- **Experiment.** `bench_moe_cuda` (V0 vs V0-oracle):
  - T ∈ {1, 4, 16, 64, 256, 1024, 4096} × routing ∈ {uniform, zipf s=1.2, same4}
  - CUDA-event p50 over 200 iterations, interleaved
  - same forced segments, same kernels, same cuBLAS calls
- **Measured quantity.** Δ = p50(V0) − p50(V0-oracle), and the share Δ / p50(V0).
- **Decision rule (proposed, needs your approval).**
  - H1 is **supported** if the share is ≥ 10% for some routing at T ≤ 16.
  - It is **refuted** if the share is < 10% at every T ≤ 16.
  - Results whose Δ is smaller than the run-to-run spread (p95 − p50 of V0-oracle) are reported
    as "not resolved", not as a number.
- **My prediction (estimate, made before any data): H1 is likely refuted at real gpt-oss-20b
  dims.**
  1. **Where the time goes.** Even at T = 1 the layer reads every touched expert's weights:
     4 experts × 99.6 MB ≈ 398 MB.
  2. **GEMM time at T = 1.** At the T4's 320 GB/s spec bandwidth that is ≳ 1.2 ms. With
     M = 1 GEMMs, cuBLAS is unlikely to reach the spec, so probably more.
  3. **The sync's cost.** It is one tiny D2H copy (132 B), the host waking up, and then the
     first cuBLAS launch. I expect it in the **tens of microseconds**.
  4. **Resulting share.** About 1–5% at T = 1, falling as T grows.
  5. **Where it could fail.** If cuBLAS launch latency at small M is much worse than expected,
     or if the host-side loop over 2·U GEMM launches is slow, the oracle hides part of it. The
     oracle still issues those launches, though, so the loop is not part of Δ.
- **Why it is still worth measuring.**
  - The prediction depends on the expert weights being large.
  - The same experiment at small hidden sizes, or with a faster GEMM (V1/V2), would raise the
    share. The stage breakdown (`--stages`) shows how close the GEMMs are to weight-streaming
    speed.
  - A refuted H1 is a real result: "for a 20B-dim MoE layer on a T4, removing the host sync is
    not where the time is."

**Verdict:** pending. The data comes from the T4 run in `colab/README.md` ("MoE study" §4) or, equivalently, `kaggle/README.md` §4.

## H2–H5

These are pre-registered at their own variant checkpoints, before V1 and V2 exist.
