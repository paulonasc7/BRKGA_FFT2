# Native Decoder Optimization Ideas — Single-Machine Runtime Reduction

**Created:** 2026-04-08
**Goal:** 25–50% reduction in per-generation wall time before tackling §2 (sparse grid allocation)
**Hardware:** RTX A4000 (16 GB VRAM), Paperspace Gradient
**Context:** These ideas target `full_native_decoder.py` (C++/CUDA extension). The old `OPTIMIZATION_IDEAS.md` covers the Python-era `wave_batch_evaluator.py` and is kept for historical reference.

---

## Current baselines (full_native_decoder, after §A–§D + ideas #1–#2)

| Instance | Original native | After #1 | After #2 (current) | Cumulative |
|----------|----------------|----------|--------------------|-----------| 
| P50M2-0 (500 ind.) | 1.907s | 1.753s | **1.438s** (std 0.039s) | **−24.6%** |
| P75M2-0 (750 ind.) | 3.776s | 3.211s | **2.692s** (std 0.047s) | **−28.7%** |

---

## Profiling summary (P50M2-0, per seed, 1.945s wall)

| Component | ms/seed | % of wall | Source |
|-----------|---------|-----------|--------|
| `aten::mul` (complex FFT multiply + irfft2 normalization) | 360 | 18.5% | Phase 4 |
| `aten::_fft_c2r` (irfft2) | 289 | 14.9% | Phase 4 |
| `aten::index_select` (gather grid FFTs + part FFTs) | 278 | 14.3% | Phase 2/4 |
| `Memcpy DtoD` (cuFFT internal) | 146 | 7.5% | Phase 4 |
| `aten::copy_` (CPU→GPU index transfers) | 147 | 7.6% | All phases |
| `rfft2` + `index_copy_` (grid FFT recompute) | 101 | 5.2% | Phase 2 |
| CUDA selector kernel | 65 | 3.3% | Phase 4 |
| Other GPU (fill_, grid_update) | 30 | 1.5% | Phase 5/6 |
| **CPU-only** (invisible to profiler) | **~675** | **34.7%** | Phase 1/3/5/6 |

Call counts per seed: 204 IFFT chunks, 70 rfft2 recomputes, 479 index_selects, 2289 copy_ calls.

The `mul` time (360ms) includes two distinct things: (a) the complex pointwise multiply of grid_fft × part_fft (217ms), and (b) a scalar float multiply that irfft2 applies to normalize the output by dividing by H×W (143ms). The normalization multiply is pure waste — it can be absorbed into initialization.

---

## Ideas

---

### 1. ~~Eliminate irfft2 normalization multiply~~ — DONE

**Result: −8.1% P50M2-0 (1.907s→1.753s), −14.9% P75M2-0 (3.776s→3.211s)**
**Expected savings:** ~144ms/seed (7.4%)
**Effort:** Very Low
**Risk:** Very Low

`torch::fft::irfft2` with default normalization (`"backward"`) divides the output by H×W. This is implemented as a separate full-tensor float scalar multiply kernel — 204 calls/seed at ~0.7ms each = 144ms/seed. That's 7.4% of wall time doing pure division.

**Fix:** pre-divide the part FFTs by H×W once during initialization, then use `irfft2(..., "forward")` which skips the normalization entirely:

```python
# In _pack_problem_data, after stacking machine_fft_dense:
H, W = mach.bin_length, mach.bin_width
machine_fft_dense[-1] /= (H * W)
```
```cpp
// In batch_fft_all_tests:
auto overlap_batch = torch::fft::irfft2(
    batch_grid_ffts * batch_part_ffts, {H, W},
    /*dim=*/-1, /*norm=*/"forward"
);
```

The math is identical: `IFFT_raw(G * P/N) = IFFT_raw(G * P) / N = IFFT_normalized(G * P)`. Zero overlap stays exactly 0. The selector kernel's `round(v) == 0` check is unaffected because the absolute values are unchanged — only the scale factor moves from post-IFFT to pre-multiply.

**Verification:** compare fitnesses between normalized and forward mode on seeds 123/321/777. Must match exactly (or within float32 epsilon).

---

### 2. ~~Fused gather-multiply CUDA kernel~~ — DONE

**Result: −18.0% P50M2-0 (1.753s→1.438s), −16.2% P75M2-0 (3.211s→2.692s)**
**Expected savings:** ~150–200ms/seed (8–10%)
**Effort:** High (custom CUDA kernel)
**Risk:** Medium (correctness of complex arithmetic)

Each IFFT chunk currently does three separate ops:

1. `grid_ffts.index_select(0, grid_idx)` — scattered gather of complex `(chunk_n, H, W/2+1)` tensors → temp1
2. `machine_ffts_dense_.index_select(0, rot_idx)` — scattered gather of complex tensors → temp2
3. `temp1 * temp2` — pointwise complex multiply → temp3

This reads and writes the full data 3 times (6 memory passes total for read+write). A single fused CUDA kernel would:

1. Read `grid_ffts[grid_idx[i]]` and `part_ffts[rot_idx[i]]` directly from their source locations
2. Multiply in-register (complex mul = 4 flops per element)
3. Write the product once

This eliminates 2 intermediate tensor allocations, reduces memory traffic from 6 passes to 3, and removes 2 of the 3 kernel launches per chunk. The complex `index_select` alone is 241ms/seed — even a 50% reduction saves ~120ms.

**Kernel signature:**
```cuda
__global__ void fused_gather_multiply(
    const c10::complex<float>* grid_ffts,    // (max_total_bins, H, W/2+1)
    const c10::complex<float>* part_ffts,    // (n_rot_total, H, W/2+1)
    const int64_t* grid_idx,                 // (chunk_n,)
    const int64_t* rot_idx,                  // (chunk_n,)
    c10::complex<float>* out,                // (chunk_n, H, W/2+1)
    int chunk_n, int fft_size                // fft_size = H * (W/2+1)
);
```

---

### 3. ~~Merge p1 and p2 FFT batches~~ — SKIPPED

**Expected savings (original estimate):** ~100ms/seed (5.1%)
**Effort:** Medium (restructure `process_wave`)
**Risk:** Low (slight increase in total tests evaluated)

Each wave calls `batch_fft_all_tests` **twice**: once for p1 (first valid bin per context) and once for p2 (remaining bins, only for contexts that had no p1 hit). This means all per-call overhead is paid twice per wave:

- 2× workspace loading (4 CPU→GPU transfers each)
- 2× index_select for grid/part FFTs
- 2× kernel launches for IFFT, mul, selector
- 2× GPU→CPU result transfers

**Why this idea is skipped:**

1. **p2 collection depends on p1 FFT results — true merge is impossible.** Phase 3 p2 runs only for contexts where p1 returned no valid placement. You cannot know which contexts those are without first running the p1 FFT batch and reading back results. Collecting all tests (p1 + p2) upfront would require dropping the first-valid-bin early exit and running Phase 3 p2 for every context unconditionally.

2. **Dropping the early exit multiplies Phase 3 CPU work 4–5×.** In Phase 3, p2 collects all remaining bin×rotation pairs for every context, not just the first valid bin. The p1 early exit exists precisely to avoid this. Removing it — or performing p2 vacancy checks for all contexts speculatively — would increase the CPU Phase 3 cost by 4–5× per wave. Since Phase 3 CPU is already the largest CPU cost (~200–400ms/seed estimated), this would cause a net regression.

3. **CPU/GPU already overlap, so GPU overhead savings are partially hidden.** Wall time (1.753s/seed) < CUDA time (2.496s/seed), meaning CPU Phase 3 vacancy checks already run concurrently with GPU Phase 4 IFFT work. Eliminating the overhead of the p2 `batch_fft_all_tests` call reduces CUDA time but not necessarily wall time, since the CPU work (Phase 3 p2 collection) was already running in parallel.

4. **p2 is nearly empty in practice.** >99% of contexts find a valid placement in p1. The p2 call processes very few tests but still pays full per-call overhead (workspace load, kernel launch, GPU→CPU transfer). The overhead to eliminate is real but modest — much smaller than the CPU work the early exit avoids.

**Bottom line:** not implementable without a regression in Phase 3. The two-pass structure is load-bearing. The per-call overhead of the p2 call is a real cost (addressed by idea #4 instead), but the early-exit correctness depends on the two-pass design.

---

### 4. ~~Pack CPU→GPU transfers~~ — TESTED, NO BENEFIT

**Result: P50M2-0 1.755s (baseline 1.753s), P75M2-0 3.225s (baseline 3.211s) — within noise (sequential runs, clean GPU)**
**Expected savings:** ~50–80ms/seed (2.5–4.1%)
**Effort:** Medium
**Risk:** Very Low

2289 `copy_` calls per seed totalling 147ms. The main sources:

- `batch_fft_all_tests`: 4 transfers per call (grid_idx, rot_idx, h, w) × 2 calls/wave = 8/wave
- `apply_gpu_updates`: 6 transfers per call (cell_offsets, grid_idxs, y_starts, x_starts, part_widths, part_offsets) × up to 2 calls/wave = 12/wave
- Phase 2 grid FFT recompute: 1 transfer/wave
- Phase 6 grid zeroing: 1 transfer/wave

**Fix:** for `batch_fft_all_tests`, interleave grid_idx, rot_idx, h, w into a single `int64_t` buffer (4 × n_tests), transfer once, then slice on GPU:

```cpp
// Pack on CPU side:
auto packed = ensure_cpu_pinned_long(ws_cpu_packed_, 4 * n_tests);
int64_t* p = packed.data_ptr<int64_t>();
for (int i = 0; i < n_tests; i++) {
    p[0*n_tests + i] = test_grid_indices[i];
    p[1*n_tests + i] = (int64_t)test_rot_global[i];
    p[2*n_tests + i] = (int64_t)test_heights[i];
    p[3*n_tests + i] = (int64_t)test_widths[i];
}
auto gpu_packed = ensure_workspace_long(ws_packed_long_, 4 * n_tests);
gpu_packed.copy_(packed, /*non_blocking=*/true);
auto all_grid_idx_t = gpu_packed.narrow(0, 0*n_tests, n_tests);
auto all_rot_idx_t  = gpu_packed.narrow(0, 1*n_tests, n_tests);
// ...
```

Same approach for `apply_gpu_updates` (6 → 1 transfer). Net effect: ~22 transfers/wave → ~4 transfers/wave.

**Why it didn't work:** Each individual transfer is only ~few KB. The PCIe DMA time for ~10 KB at 16 GB/s is ~0.6µs — the same total bytes move regardless of how many `copy_` calls they're split into. The per-kernel-launch overhead is ~5–10µs, and reducing 22 launches to 4 saves ~18 × 8µs ≈ 0.14ms/wave. With ~300 waves/seed that's ~42ms/seed in theory, but in practice the GPU is already processing other work between waves so the launch overhead was already hidden.

Note: a first run of this test produced inconclusive results because P50 and P75 benchmarks were launched concurrently, causing GPU contention (one process OOM'd). Re-run sequentially confirmed the null result: P50 1.755s vs 1.753s baseline, P75 3.225s vs 3.211s baseline — both within noise on a clean, uncontested GPU.

---

### 5. ~~GPU-accelerated vacancy check~~ — DONE

**Result: P50M2-0 1.445s (−0.5% vs 1.438s), P75M2-0 2.650s (−1.6% vs 2.692s). Correctness: exact match.**
**Expected savings:** ~200–300ms/seed (10–15%)
**Effort:** High (new CUDA kernel + restructure Phase 3)
**Risk:** Medium (correctness; changes CPU/GPU work split)

**What was there before:** Phase 3 vacancy checking was entirely CPU-bound. For each (context, bin, rotation) triple, the code called `check_vacancy_fit_simple_cpp` individually — a sequential loop iterating over every candidate triple. The function slides a 1D density array across the bin's vacancy vector, checking each starting position using vectorized SIMD instructions (AVX512 when available, falling back to AVX2, then scalar): it returns `true` as soon as the first fitting window is found, short-circuiting the rest. In the original `wave_batch_evaluator.py` era this was a Python+Numba loop; in the native decoder it was a C++ loop calling `check_vacancy_fit_simple_cpp` serially on the CPU thread. No cross-triple parallelism, no batching — each triple was processed one at a time, even though the per-window inner loop was SIMD-vectorized.

**What was changed:** Refactored Phase 3 (p1 and p2) into a three-pass GPU approach:

1. **Pass A (CPU, as before):** collect all (context, bin, rotation) candidate triples via the existing C++ vacancy-dimension check and first-valid-bin p1 logic — without running the sliding-window check yet. This produces lists of `(vac_row, den_off, den_len)` triples.

2. **Pass B (GPU):** upload all vacancy vectors to a GPU tensor (dirty-row incremental upload: only rows changed since last wave are re-uploaded via `index_copy_`). Upload the static density-flat tensor once at construction. Launch a CUDA kernel with one thread block per (bin, rotation) pair: 64 threads each check a subset of starting positions in the `H`-element vacancy vector, then reduce via shared memory to produce a single pass/fail bit per pair.

3. **Pass C (CPU):** read back the `int8` pass/fail array, filter the candidate list to only those that passed, and proceed to build the FFT test arrays exactly as before (preserving the first-valid-bin semantics).

The CUDA kernel (`_batch_vacancy_check_kernel`) uses 64 threads/block, shared-memory OR-reduction, and reads both `vacancy_flat` (GPU tensor, updated incrementally) and `density_flat_gpu_` (uploaded once at construction).

**Why gains were marginal on P50/P75:** Phase 3 CPU work for these small instances was already running concurrently with Phase 4 GPU work (wall time < CUDA time). The new GPU kernel replaces CPU work that was already hidden behind GPU execution. The kernel's GPU→CPU result transfer (`non_blocking=false`) adds a small sync point that partially negates the savings. On larger instances (P100M4+) where Phase 3 CPU time exceeds Phase 4 GPU time, this restructuring is expected to show larger wall-time reductions.

Phase 3 vacancy checking is the single largest CPU cost (~200–400ms/seed estimated). For each (context, bin, rotation), it slides the part's density array across the bin's vacancy vector to check if the part physically fits. This is called hundreds of times per wave.

**Caveat:** the vacancy arrays live on CPU and change every wave (updated when parts are placed). Transferring them to GPU for the check and back would add latency. The win depends on whether GPU batch parallelism outweighs the transfer cost. For large instances (P200M4+) with many bins and rotations, this should be strongly positive. For P50M2 it may be marginal.

---

### ~~6. Incremental FFT update~~ — REJECTED

**Expected savings (estimated):** ~30–40ms/seed (1.5–2%)
**Effort:** Medium-High
**Status:** Fully implemented and tested — causes correctness failures. Reverted.

**The idea:** When a part is placed on a grid, the grid's FFT is invalidated and recomputed with `rfft2(grid_state)` next wave. ~70 rfft2 calls per seed at ~1ms each = ~70ms. Since FFT is linear:

```
FFT(grid + delta) = FFT(grid) + FFT(delta)
```

For `delta` = placed part at position (y, x), using the shift theorem:

```
FFT(delta)[k,l] = rfft2(part_padded)[k,l] * exp(-2πi*(k*y/H + l*x/W))
```

This replaces one full rfft2 (O(HW log(HW))) with a pointwise complex multiply (O(HW)).

**Why it fails — the flip issue:** `collision_backend.py` stores `rfft2(flip(part))` in `machine_ffts_dense_`, not `rfft2(part)`. The flip is a linear flip (`torch.flip`), not a circular one, so it adds an extra phase factor that cannot be removed by simple conjugation. The solution — precomputing separate `machine_fft_dense_unflipped` tensors using `rfft2(part_padded)` directly — was implemented and verified: 0/500 mismatches on seeds 123/321 after the fix.

**Why it still fails — arithmetic divergence:** Even with the correct unflipped part FFTs, seed 777 produced 2/500 mismatches (makespan differences of 50,078 and 895). Root cause: `sincosf` in CUDA uses `~1.5 ULP` error, while cuFFT's butterfly algorithm produces bit-exact results. At boundary conditions where the IFFT output rounds to exactly 0, the incremental phase arithmetic diverges from cuFFT's result and tips placement decisions the wrong way. These wrong placements cascade into dramatically different makespans.

**Why periodic resync makes it worse:** Increasing resync frequency from period=30 to period=5 caused *more* mismatches (5 total vs 2), not fewer. Resyncs "lock in" wrong placements made during the incremental phase. With period=30, errors sometimes self-cancelled across waves. The paradox confirms the problem is not drift accumulation but deterministic divergence in single-wave boundary-condition computations.

**Decisive test:** Disabling the incremental kernel entirely (no phase arithmetic, no grid_fft update) → exact 0/500 match on all seeds. The correct approach would require using cuFFT-compatible phase computation, which is not exposed as a simple CUDA intrinsic.

**Alternative (rfft2 in Phase 5):** Computing rfft2 immediately in Phase 5 instead of deferring to Phase 2 processes the exact same set of grids and provides no speedup.

**Conclusion:** The mathematical derivation is correct, but the implementation is fundamentally incompatible with exact reproduction of cuFFT results due to `sincosf` arithmetic divergence. The small arithmetic error cascades into wrong placement decisions. Do not attempt again without a cuFFT-compatible phase computation approach.

---

## Summary table

| # | Idea | Savings/seed | % of wall | Effort | Risk | Cumulative |
|---|------|-------------|-----------|--------|------|-----------|
| 1 | Eliminate normalization multiply | −8.1% P50 / −14.9% P75 | 7.4% | Very Low | Very Low | **DONE** |
| 2 | Fused gather-multiply kernel | −18% P50 / −16.2% P75 | 16–18% | High | Medium | **DONE** |
| 3 | ~~Merge p1+p2 batches~~ | — | — | — | — | SKIPPED (see §3) |
| 4 | ~~Pack CPU→GPU transfers~~ | 0 | 0% | Medium | Very Low | TESTED, NO BENEFIT |
| 5 | GPU vacancy check | −1.6% P75 / ~0% P50 | ~1.6% | High | Medium | **DONE** (marginal win) |
| 6 | ~~Incremental FFT update~~ | — | — | — | — | REJECTED (sincosf arithmetic diverges from cuFFT; cascades into wrong placements) |

**Achieved so far:** #1 + #2 + #5 → −24.3% P50, −29.8% P75. **25%/30% targets met.**

| Instance | Original native | After #1 | After #2 | After #5 (current) | Cumulative |
|----------|----------------|----------|----------|--------------------|-----------|
| P50M2-0 | 1.907s | 1.753s | 1.438s | **1.445s** (std 0.015s) | **−24.3%** |
| P75M2-0 | 3.776s | 3.211s | 2.692s | **2.650s** (std 0.018s) | **−29.8%** |

Note: #5 (GPU vacancy check) produced only marginal gains (~0% P50, ~1.6% P75) on these small instances. The expected larger gains require bigger instances with many more open bins and rotations per wave (P100M4+), where Phase 3 CPU cost grows relative to Phase 4 GPU cost.

---

## Recommended order (updated)

1. ~~**#1 (normalization multiply)**~~ — **DONE** (−8.1% P50, −14.9% P75)
2. ~~**#3 (merge p1+p2)**~~ — **SKIPPED** (infeasible — p2 depends on p1 FFT results; merging would break early-exit and increase CPU work 4-5×)
3. ~~**#4 (pack transfers)**~~ — **TESTED, NO BENEFIT** (transfers too small; per-launch overhead already hidden behind GPU work)
4. ~~**#2 (fused gather-multiply)**~~ — **DONE** (−18.0% P50, −16.2% P75). Custom CUDA kernel fuses two `index_select` + complex multiply into one pass
5. ~~**#5 (GPU vacancy)**~~ — **DONE** (−0.5% P50, −1.6% P75). GPU kernel with dirty-row incremental upload; marginal on small instances where CPU was already hidden behind GPU
6. ~~**#6 (incremental FFT)**~~ — **REJECTED** (sincosf arithmetic diverges from cuFFT butterfly; small errors cascade into wrong placement decisions; periodic resync makes it worse not better)
