# Cross-row leakage investigation — RESOLVED 2026-04-21

## Problem

`verify_native_vs_pp.py` found that the native decoder failed its correctness
gate when chromosome 462 of `np.random.seed(42)` was evaluated as part of a
500-row batch for P50M2-0 (result ~433k, golden 448166.8336), while the same
chromosome in a single-row batch was exact.

## Investigation path

1. Bin-count instrumentation showed `n_over_mbps=0` across all runs — slot
   aliasing (the initial hypothesis) was NOT the cause.
2. Determinism test (Part C): the same 500-row batch evaluated twice produced
   **different** makespans for row 462 (e.g. 435139 → 479057). The decoder was
   non-deterministic.
3. `CUDA_LAUNCH_BLOCKING=1` made every test pass deterministically → confirmed
   a CUDA async race.

## Root cause

`batch_fft_all_tests`, `run_gpu_vacancy_check`, `recompute_vacancy_gpu`, and
`load_workspace_*` all used the same pattern:

```cpp
auto pinned = ensure_cpu_pinned_*(ws.ws_cpu_X, n);   // persistent pinned buffer
std::memcpy(pinned.data_ptr(), src, n * sizeof(T));  // host write
gpu_tensor.copy_(pinned, /*non_blocking=*/true);     // async GPU copy
```

`non_blocking=true` on pinned→GPU uses `cudaMemcpyAsync` on the current stream.
The GPU reads the pinned source **when the stream executes the copy**, not at
`copy_` enqueue time. But the CPU side continues immediately and on the next
wave / chunk writes to the same pinned buffer again — **racing with the
still-pending async copy from the previous iteration**.

The race is not always observable:
- Same-chromosome batches produce the same payload twice, so even with the
  race, the GPU reads the intended values (both "old" and "new" are identical).
- Different-chromosome batches produce different payloads wave-to-wave;
  whichever side wins the race determines what the GPU sees.

This explains every finding:
- Single-row → no reuse, no race → always correct.
- Tile[] (500 identical) → reuse happens but payloads identical → no
  observable race.
- Different chroms, different positions, different batch sizes → results vary
  with timing.

## Fix

All `.copy_(pinned, /*non_blocking=*/true)` pinned-host→GPU uploads in
`full_native_decoder.py` converted to `non_blocking=false`. This forces
`cudaMemcpy` (synchronous) for the copy, guaranteeing the GPU reads the
intended buffer contents before the CPU can overwrite them.

Sites changed:
- `load_workspace_long_from_i64` (line 787)
- `load_workspace_long_from_i32` (line 805)
- `load_workspace_i32_from_i32` (line 820)
- `batch_fft_all_tests` packed index upload (line 920)
- `run_gpu_vacancy_check` pair upload (line 1300)
- `recompute_vacancy_gpu` index upload (line 1950)
- `apply_gpu_updates` index upload (line 2027)

## Validation

`verify_native_vs_pp.py` now passes every test:
- Single-row: 448166.8336 ✓
- Full 500-row batch row 462: 448166.8336 ✓
- All batch sizes N ∈ {1, 100, 200, 250, 300, 400, 463, 500}: exact match ✓
- Determinism (same batch ×2): identical results ✓
- All swap / position variants: exact match ✓

### Full-population validation (2026-04-21)

`verify_full_population_vs_pp.py` runs pp on every chromosome of the seed-42
population (500 chroms × 100 genes for P50M2-0) and compares element-wise
against the native full-batch output:

- match (|diff|<1.0): **500 / 500**
- mismatch: **0**
- max |diff|: **0.000000**
- mean native fitness == mean pp fitness, byte-identical
- native full-batch: 2.2s; pp sequential: 13.7s

This is the strongest correctness gate available: every single chromosome in
the generation-0 population agrees with the golden sequential decoder.

## Performance

P50M2-0, seed 123, 5 reps:

| Config | Time/gen |
|--------|----------|
| Pre-fix baseline (CLAUDE.md) | 0.974s |
| Post-fix (this change) | **0.884s** |

Synchronous pinned→GPU copies are NOT a bottleneck here. The sum of all
"sync" blocks is ~610ms/5reps ≈ 122ms/gen, within the existing Phase 2 /
Phase 3 / Phase 4 budgets. Wall time actually improved slightly.

## Takeaways

1. **`non_blocking=true` on pinned→GPU is a footgun** whenever the pinned
   buffer is reused across iterations. Safe only if you explicitly
   `cudaStreamSynchronize` before reuse, or rotate multiple pinned buffers.
2. **Always keep a pp-anchored correctness gate.** The `profile_cpu_hotspots.py`
   fingerprint (seed-123 first-5 makespans) was passing throughout this bug
   because it was self-consistent, not anchored to ground truth. pp is the
   real reference.
3. **Determinism tests are cheap and extremely informative.** One Part-C test
   (same batch ×2) localized the bug class (race) in one run.
