# Profiling Analysis — FullNativeDecoderEvaluator

**Date:** 2026-04-10 (initial), 2026-04-13 (Round 2+3 updates)  
**Original baseline after CPU optimizations:** P50M2-0 1.105s/gen, P75M2-0 2.010s/gen  
**Current (after Round 3):** P50M2-0 **0.974s/gen**, P75M2-0 **1.742s/gen**  
**Total speedup vs original:** P50 −15.7%, P75 −21.7%  

All data collected with `profile_cpu_hotspots.py` (10 reps, seed 123) and `profile_phases_native.py` (3 seeds: 123, 321, 777).

---

## CPU Hotspot Breakdown

### P50M2-0 (500 individuals, pop = 10 × nb_parts)

Wall clock: **1.105s/gen** (std 0.014s)

| Hot spot | ms/gen | % wall |
|---|---|---|
| `add_part_to_bin_cpu` (total, incl. UVR) | 140.0 | 12.7% |
| — `update_vacancy_rows` (AVX2 SIMD) | 90.4 | 8.2% |
| — non-UVR grid write (memcpy/AVX2-add) | ~50 | 4.5% |
| Phase 3 Pass B wait (vacancy check GPU) | 107.2 | 9.7% |
| Vacancy upload sync | 101.7 | 9.2% |
| Phase 6 new-bin loop (total) | 37.4 | 3.4% |
| — BinState ctor (pool reuse + memset) | 21.7 | 2.0% |
| Vacancy readback sync | 7.1 | 0.6% |
| Phase 3 Pass A + C | 6.1 | 0.6% |

Event counts (per gen): 2,170 new bins, 21,697 aptb calls, 1,650k vacancy rows, 135 vacancy-check invocations.

---

### P75M2-0 (750 individuals, pop = 10 × nb_parts)

Wall clock: **2.010s/gen** (std 0.017s)

| Hot spot | ms/gen | % wall |
|---|---|---|
| `add_part_to_bin_cpu` (total, incl. UVR) | 239.2 | 11.9% |
| — `update_vacancy_rows` (AVX2 SIMD) | 146.0 | 7.3% |
| — non-UVR grid write | ~93 | 4.6% |
| Phase 3 Pass B wait | 210.0 | 10.4% |
| Vacancy upload sync | 200.9 | 10.0% |
| Phase 6 new-bin loop (total) | 59.3 | 3.0% |
| — BinState ctor | 36.7 | 1.8% |
| Vacancy readback sync | 11.3 | 0.6% |
| Phase 3 Pass A + C | 13.6 | 0.7% |

Event counts (per gen): 4,157 new bins, 38,430 aptb calls, 2,465k vacancy rows, 187 vacancy-check invocations.

---

## GPU Phase Breakdown (P75M2-0)

Collected via `torch.profiler`. Wall time: **2.103s/gen**, CUDA time: **2.728s/gen**.  
CUDA time > wall time confirms GPU is fully occupied with no idle gaps.

### Attributed GPU ops

| GPU Phase | CUDA ms/gen | % of CUDA |
|---|---|---|
| irfft2 — Phase 4 inverse FFT | 541.6 | 19.8% |
| DtoD memcpy — FFT internal copies | 273.4 | 10.0% |
| CPU→GPU copy_ — vacancy pairs + grid uploads | 277.0 | 10.1% |
| rfft2 — Phase 2 grid FFT recompute | 126.3 | 4.6% |
| CUDA selector kernel — Phase 5 | 124.3 | 4.6% |
| index_copy_ — Phase 2 grid FFT writeback | 64.2 | 2.4% |
| fill_ — Phase 6 grid_states.index_fill_ | 40.2 | 1.5% |
| batch_grid_update kernel — Phase 5/6 | 8.6 | 0.3% |
| **Subtotal attributed** | **1455.6** | **55.9%** |
| **Unaccounted** | **1272.4** | **44.1%** |
| **TOTAL CUDA** | **2728.0** | — |

### P50M2-0 for comparison

Wall time: 1.163s/gen, CUDA time: 1.459s/gen.

| GPU Phase | CUDA ms/gen | % CUDA |
|---|---|---|
| Phase 4 (IFFT+mul+selector+DtoD) | 500.3 | 34.3% |
| Phase 2 (rfft2+index_copy_) | 100.8 | 6.9% |
| CPU→GPU transfers | 147.9 | 10.1% |
| **Unaccounted** | **645.5** | **44.2%** |
| **TOTAL CUDA** | **1459.0** | — |

The unaccounted fraction is consistently **~44%** across both instances. It does not scale differently with problem size — suggesting it is structural overhead (kernel launch latency, cuFFT internal ops not captured by the profiler, stream synchronization bookkeeping) rather than a single identifiable kernel.

---

## Call Count Analysis (P75M2-0)

| Metric | Per gen (3-seed avg) |
|---|---|
| irfft2 calls | 331 |
| rfft2 calls | 94 |
| copy_ calls | 3,894 |
| Estimated waves/gen | ~68 |
| irfft2 calls/wave | ~4.9 |
| FFT tests/wave (est. 750-test chunks) | ~3,675 |
| copy_ calls/wave | ~57 |

**Key ratio:** ~3,675 FFT tests/wave for 750 solutions × ~2 rotations ≈ 1,500 base candidates. The 2.4× multiplier means on average each solution tests ~2.4 `(grid, rotation)` pairs per wave — a mix of p1 (existing bins) and p2 (fallback after p1 miss). No obvious waste signal here.

**The 3,894 copy_ calls/gen** dominate the CPU profiler (`cudaMemcpyAsync` = 61% of CPU trace time on P75). Each is a small H2D or D2H transfer for vacancy pair data, readback, or workspace setup. These generate stream serialization overhead far above the raw bytes transferred.

---

## CPU Profiler Breakdown (P75M2-0, torch.profiler)

| Component | CPU ms (3 seeds) | % |
|---|---|---|
| cudaMemcpyAsync | 3,228 | 61.1% |
| aten::_fft_c2r (irfft2 launch) | 731 | 13.8% |
| cudaStreamSynchronize | 614 | 11.6% |
| aten::copy_ | 401 | 7.6% |
| aten::_fft_r2c (rfft2 launch) | 288 | 5.5% |
| aten::empty (workspace alloc) | 102 | 1.9% |
| misc (narrow, slice, to, etc.) | ~110 | 2.1% |

`cudaMemcpyAsync` at 61% of CPU time = 1,076 ms/gen. This is CPU-side time issuing async transfer commands to the driver — the actual bytes are being moved concurrently, but the *issuing* cost itself is 1 second per gen. This is consistent with 3,894 copy_ calls/gen — each incurring ~276 µs of driver-call overhead.

---

## Key Findings and Open Questions

### Finding 1 — Unaccounted 44% GPU time
Consistently ~44% of CUDA time is unaccounted. The profiler attributes known kernels (irfft2, rfft2, copy_, selector) but misses a significant chunk. Hypotheses:

- **cuFFT internal butterfly passes**: cuFFT launches multiple sub-kernels per irfft2 call; the profiler may only capture the outermost launch. For 331 irfft2 calls/gen × ~5 internal passes each = 1,655 uncaptured sub-kernels.
- **Stream synchronization overhead**: Each of the 187 vacancy-check `copy_(non_blocking=False)` calls issues a host-visible sync event, causing the CUDA driver to insert implicit stream barriers.
- **Memory allocation overhead**: `ensure_workspace_*` calls resize GPU tensors frequently; PyTorch's caching allocator may be issuing `cudaMalloc`/`cudaFree` on cold paths.
- **Kernel launch latency accumulation**: 3,894 copy_ calls × ~10µs driver overhead each ≈ 39ms — non-trivial at scale.

**Next step:** Use `nsys profile` (Nsight Systems) instead of `torch.profiler` to see all CUDA kernels and streams, including cuFFT sub-kernels. This will definitively identify what the 44% is.

### Finding 2 — copy_ call count is the structural bottleneck
3,894 copy_ calls/gen (P75) = 57 per wave. This comes from:
- 2 vacancy check uploads/wave (p1+p2) × ~68 waves = 136 uploads
- 2 vacancy check readbacks/wave = 136 readbacks
- ~14 workspace setup copies/wave (grid index tensors, pair buffers, etc.)
- Phase 2 + Phase 5/6 GPU update copies

Each copy_ has fixed driver overhead (~276 µs CPU issue time). Halving the call count would save ~540 ms/gen in CPU driver time — though the actual wall-clock gain depends on how many are on the critical path.

### Finding 3 — Phase 4 IFFT is still the largest single identified GPU cost
541.6 ms/gen on P75 = 19.8% of CUDA time, but the real cost including cuFFT sub-kernels is likely 2–3× higher (part of the 44% unaccounted). With 331 irfft2 calls/gen, the average batch size is ~11 tests/call (3,675 tests ÷ 331 calls). 

The CHUNK_SIZE=750 cap means most calls are well below capacity, especially in early waves when few solutions have open bins. There may be room to reduce irfft2 calls by batching more aggressively or by filtering out tests earlier (vacancy pre-check before IFFT).

### Finding 4 — GPU is genuinely memory-bandwidth bound on Phase 4
CUDA time (2.728s) > wall time (2.103s) with no idle gaps means the GPU is the bottleneck — adding CPU work wouldn't hurt as long as it completes within GPU execution time. DtoD memcpy at 273 ms/gen suggests significant data movement internal to cuFFT (staging between L2 and HBM for large batches).

---

## Nsight Systems (nsys) Deep Dive — P75M2-0

Collected with `nsys profile --trace=cuda,osrt` on 2 warmup reps of `evaluate_batch`.

### GPU Kernel Summary (`gpukernsum`)

| Kernel | Instances | Avg (ms) | Total/2reps (ms) | Per rep (ms) | % wall |
|---|---|---|---|---|---|
| `regular_fft<300>` — cuFFT column pass (irfft2) | 288 | 1.384 | 398.7 | **199.3** | 9.5% |
| `_fused_gather_multiply_kernel` (batch=750) | 244 | 1.571 | 383.2 | **191.6** | 9.1% |
| `regular_fft_c2r<300>` — cuFFT row pass (irfft2) | 244 | 1.387 | 338.4 | **169.2** | 8.0% |
| `_native_select_best_positions_kernel` — Phase 5 | 300 | 0.555 | 166.6 | **83.3** | 4.0% |
| `indexSelectLargeIndex` — gather (Phase 4 index_select) | 166 | 0.839 | 139.3 | **69.6** | 3.3% |
| `regular_fft_r2c<300>` + size-250 pairs (rfft2, Phase 2) | 44+112 | ~1.0 | 168.5 | **84.3** | 4.0% |
| `_fused_gather_multiply_kernel` (batch=124) | 56 | 1.088 | 60.9 | **30.5** | 1.5% |
| `index_elementwise_kernel` (index_copy_ + fill_) | 48+8 | ~1.4 | 107.3 | **53.7** | 2.6% |
| `vectorized_elementwise_kernel` (torch.zeros init) | 8 | ~5.0 | 40.3 | **20.1** | 1.0% |
| **Total attributed GPU kernel time** | | | **~1803** | **~901** | **42.9%** |

**Wall time: 2103ms/rep. Actual GPU kernel execution: ~901ms/rep. GPU utilization: ~43%.**

### The "44% Unaccounted" Mystery — Resolved

The torch.profiler reported "CUDA time = 2728ms/gen > wall time = 2103ms/gen", suggesting the GPU was fully occupied. This was **misleading**. torch.profiler's "CUDA time" measures *API wall-clock duration* from the CPU's perspective — it includes:

- Time the CPU blocks in `cudaStreamSynchronize` waiting for GPU results (measured as "CUDA time")
- Time the CPU blocks in blocking `copy_` calls
- Actual GPU kernel execution time

These get summed in a way that can exceed wall time (because overlapping CPU/GPU work is double-counted). The nsys kernel summary shows the **actual GPU execution time is only ~901ms/rep** — the GPU is idle for the other 57% of wall time.

There are no mystery kernels. The "44% unaccounted" was CPU-blocked time being mis-attributed by torch.profiler.

### CUDA API Summary — Critical Finding

| API call | Calls/2reps | Total time | Per rep | Median duration |
|---|---|---|---|---|
| `cudaMemcpyAsync` | 8,342 | 2,199ms | **1,099ms** | 6µs (avg **263µs**) |
| `cudaStreamSynchronize` | 3,321 | 420ms | **210ms** | (avg 126µs) |
| `cuModuleUnload` | 896 | 400ms | **200ms** | — |
| `cuModuleLoadData` | 960 | 150ms | **75ms** | — |
| `cudaLaunchKernel` | 4,429 | 144ms | **72ms** | — |

**`cudaMemcpyAsync` takes 1,099ms/rep of CPU time.** Median call is 6µs but avg is 263µs — heavily bimodal: most calls return instantly (async), but a significant fraction block implicitly when the CUDA command queue is full or when a sync fence is encountered. 8,342 calls / 2 = 4,171/rep (matches the 3,894 copy_ calls from torch.profiler + overhead).

**`cuModuleUnload`/`cuModuleLoadData` — 896 unloads + 960 loads per 2 reps = ~450/rep each.** This is cuFFT loading and unloading compiled PTX modules for different batch sizes. cuFFT plans are compiled per (H, W, batch_size) configuration. With wave-to-wave variable batch sizes (1 to 750 tests), cuFFT may be re-planning for each unique batch size. This costs ~275ms/rep of CUDA API time.

### Revised Phase Accounting (P75M2-0, per rep)

| Category | Time (ms/rep) | % wall | Source |
|---|---|---|---|
| irfft2 GPU kernels (2 sub-kernels each) | 368 | 17.5% | nsys gpukernsum |
| _fused_gather_multiply (Phase 4 pointwise mul) | 222 | 10.6% | nsys gpukernsum |
| _native_select_best_positions (Phase 5) | 83 | 3.9% | nsys gpukernsum |
| indexSelectLargeIndex (Phase 4 gather) | 70 | 3.3% | nsys gpukernsum |
| rfft2 GPU kernels (Phase 2) | 84 | 4.0% | nsys gpukernsum |
| index_copy_ + fill_ (Phase 2 writeback + Phase 6 zero) | 54 | 2.6% | nsys gpukernsum |
| torch.zeros init (grid_states + grid_ffts, 2 machines) | 20 | 1.0% | nsys gpukernsum |
| CPU code (UVR, grid writes, bin ctor, pass A/C) | ~300 | 14.3% | cpu hotspots |
| cudaStreamSynchronize waits | ~105 | 5.0% | nsys cudaapisum |
| cuModuleUnload/Load (cuFFT re-planning) | ~138 | 6.6% | nsys cudaapisum |
| GPU idle (waiting for CPU to submit next wave) | ~655 | 31.2% | wall − kernels |
| **Total** | **2103** | **100%** | |

### GPU Utilization: 43%

The GPU is idle for **57% of wall time**. The CPU is the pacemaker. Per-wave breakdown (68 waves/gen):
- GPU kernel time/wave: ~13.3ms (901ms ÷ 68)
- CPU blocking time/wave: ~6ms (sync waits + module loads)
- CPU compute time/wave: ~4.4ms (UVR + grid writes + bin ctor)
- GPU idle/wave: ~7.5ms (waiting for CPU to issue next wave's work)

The wave execution is fundamentally sequential: Phase 2 fires → CPU does Pass A → sync for vacancy check → CPU does Pass C → Phase 4 fires → CPU blocks until results back → Phase 5/6. There is no inter-wave pipelining.

---

## What to Investigate Next (Ranked by Expected Impact)

### 1. cuFFT Re-planning (~138ms/rep, 6.6% wall) — HIGH PRIORITY
450 `cuModuleUnload` + 450 `cuModuleLoadData` calls per rep = cuFFT recompiling a plan for each unique batch size encountered. Each wave has a different number of FFT tests (1 to 750), so cuFFT sees a different `batch` parameter each time.

**Fix:** Force cuFFT to always use the same plan by padding each batch to `CHUNK_SIZE=750`. Currently the last chunk in `batch_fft_all_tests` may be smaller — change it to pad with dummy (zero) entries. All 331 irfft2 calls/rep then use batch=750, eliminating re-planning. Expected saving: ~138ms/rep (6.6%).

**Risk:** Padding adds dummy FFT work for small batches. For early waves where there are few tests (e.g. 10 tests), padding to 750 is wasteful. Need to measure whether the plan-reuse saving outweighs the extra computation for small batches. Could use a tiered approach (pad to nearest power of 2 or to a small set of fixed sizes: 32, 64, 128, 256, 512, 750).

### 2. Wave Pipelining (~655ms idle GPU/rep, 31%) — HIGH IMPACT, HIGH EFFORT
GPU is idle 57% of the time waiting for the CPU to prepare the next wave. The wave loop is fundamentally sequential:
```
Phase 2 → Pass A → Vacancy check (sync) → Pass C → Phase 4 (sync) → Phase 5/6
```

Pipelining wave N+1 behind wave N would require starting wave N+1's Phase 2 rfft2 *while* wave N's Phase 4 irfft2 is executing. This requires decoupling contexts across waves and is a major restructure.

**Partial fix (lower effort):** Overlap Phase 6 (new-bin CPU work) with Phase 2 rfft2 of the *same* wave. Currently Phase 2 fires and then CPU does Pass A immediately — but Phase 6 from the previous wave hasn't happened yet (Phase 6 happens at the end of the wave). Restructuring to fire Phase 2 *before* Phase 6 cleanup could hide Phase 6's CPU work behind Phase 2's GPU work.

### 3. cudaMemcpyAsync Overhead (~1,099ms CPU/rep from API calls) — MEDIUM
The 4,171 async H2D/D2H copies/rep each cost ~263µs of CPU driver time on average (bimodal: most are 6µs but some block implicitly). The dominant ones are the vacancy pair uploads (3 int32 arrays × 187 calls × 2 machines × 2 passes). Reducing from 8342 to fewer, larger copies would reduce driver overhead. 

Practically: merging the p1 and p2 vacancy checks into a single GPU call per wave halves the vacancy check round-trips (187 → 93 calls/rep) and saves both sync time and copy_ count. The algorithmic challenge is that p2 pairs are only known after p1 FFT results come back.

### 4. torch.zeros Per-Batch Cost (20ms/rep, 1%) — LOW
`grid_states` and `grid_ffts` are re-allocated and zero-initialized at the start of each `evaluate_batch` call. For P75M2-0 these are ~3.8 GB each = 7.7 GB total fill. At 640 GB/s HBM bandwidth that's ~12ms. Could persist these tensors across calls (already the VRAM was calculated once). Expected saving: ~10-15ms/rep.

### 5. CHUNK_SIZE Tuning (irfft2 batch efficiency) — LOW, EASY EXPERIMENT
Current `CHUNK_SIZE=750` means the fused_gather_multiply and irfft2 are always done in chunks ≤750 tests. Increasing to 1024 or 2048 would reduce irfft2 call count but increase per-call time. With cuFFT re-planning fixed (item 1), larger chunks become more attractive. Profile with CHUNK_SIZE=1024 and CHUNK_SIZE=2048 to see if fewer, larger calls are faster end-to-end.

---

## Optimization Attempt: cuFFT Re-Planning Fix — ABANDONED

**Date:** 2026-04-10  
**Target:** `cuModuleUnload`/`cuModuleLoadData` overhead (138ms/rep, ~6.6% of wall, from nsys CUDA API summary)

### What was attempted

The nsys CUDA API summary showed 448 `cuModuleUnload` + 480 `cuModuleLoadData` calls per rep — cuFFT JIT-compiling new PTX modules for each unique (H, W, batch_size) combination.  The hypothesis was that variable batch sizes in `batch_fft_all_tests` were preventing cuFFT plan reuse.

Two versions were tried, both in the same `batch_fft_all_tests` chunk loop:

**Version 1 (incorrect):** Always allocate workspace for CHUNK_SIZE=750 and pass the full 750-row tensor to `irfft2`, regardless of actual `chunk_n`. Only the first `chunk_n` output rows are read. This ensures `irfft2` always sees batch=750.

**Version 2 (tiered canonical):** Add a `canonical_batch(n)` helper that rounds up to the nearest power-of-2 (16/32/64/128/256/512/750). The workspace is sized for `chunk_canonical` rows; `irfft2` is called on the padded tensor; output is narrowed back to `chunk_n` rows. Also set `torch.backends.cuda.cufft_plan_cache.max_size = 32` at evaluator init to ensure all canonical plans fit in the cache simultaneously.

### What went wrong: wrong eval mode discovered

During testing, the `full_native` argument was mistakenly used instead of the correct `native_full` argument to `BRKGA_alg3.py`. This silently fell through to the **serial single-individual evaluation path** (500 evaluations, one at a time) rather than the `FullNativeDecoderEvaluator` batch path. Serial mode runs at ~13s/gen for P50 regardless of any C++ changes, since it calls `evaluate_solution()` in a Python loop. All results during initial testing were invalid.

The correct invocation is:
```bash
python BRKGA_alg3.py 50 2 0 torch_gpu native_full 1 1 N
```

### Actual results (correct native_full mode, BRKGA_alg3.py)

**P50M2-0 baselines (before fix), gens 1–4:**

| Gen | Time (s) |
|---|---|
| 1 | 1.14 |
| 2 | 1.16 |
| 3 | 1.14 |
| 4 | 1.18 |
| **Avg** | **1.155** |

**P50M2-0 with tiered canonical fix, gens 1–9:**

| Gen | Time (s) |
|---|---|
| 1 | 1.10 |
| 2 | 1.10 |
| 3 | 1.10 |
| 4 | 1.19 |
| 5 | 1.37 (outlier) |
| 6 | 1.13 |
| 7 | 1.09 |
| 8 | 1.10 |
| 9 | 1.09 |
| **Avg (excl. gen5)** | **1.11** |

**P75M2-0 baseline (before fix), gens 1–9:**

| Gen | Time (s) |
|---|---|
| 1 | 2.20 |
| 2 | 2.27 |
| 3 | 2.33 |
| 4 | 2.46 |
| 5 | 2.30 |
| 6 | 2.36 |
| 7 | 2.35 |
| 8 | 2.38 |
| 9 | 2.38 |
| **Avg** | **2.226** |

**P75M2-0 with tiered canonical fix, gens 1–9:**

| Gen | Time (s) |
|---|---|
| 1 | 2.42 |
| 2 | 2.35 |
| 3 | 2.40 |
| 4 | 2.37 |
| 5 | 2.38 |
| 6 | 2.38 |
| 7 | 2.38 |
| 8 | 2.40 |
| 9 | 2.36 |
| **Avg** | **2.382** |

### Summary

| Instance | Baseline | With fix | Delta |
|---|---|---|---|
| P50M2-0 | 1.155s | 1.11s | **−45ms (−4%)** |
| P75M2-0 | 2.226s | 2.382s | **+156ms (+7%)** |

**Net result: negative.** P75 regresses more than P50 improves.

### Why it failed

The nsys data reported avg irfft2 batch size ~11 for P75M2-0 (3,675 total tests ÷ 331 irfft2 calls/gen). Rounding up to canonical=16 adds 45% more FFT computation per call.  Budget analysis:

| Factor | Value |
|---|---|
| Plan-loading savings (target) | ~138ms/rep |
| FFT overhead from padding (16/11 ≈ 1.45×) | ~165ms/rep (368ms × 0.45) |
| **Net** | **−27ms/rep (regression)** |

The empirical −156ms regression is larger than the predicted −27ms, likely because the actual batch-size distribution is more skewed (many batches of size 1–5, not just 11), making the padding overhead larger than the mean would suggest.

**P50 showed a small improvement** because in P50 the irfft2 accounts for a smaller fraction of wall time (fewer bins × parts), so the plan-loading savings slightly outweigh the padding overhead.

### What this tells us about the batch size distribution

The avg batch of ~11 with a long tail of very small batches (size 1–5) means:
- Padding to even 16 incurs 200–1500% FFT overhead for the smallest calls
- The fixed plan-loading cost per call is only ~275µs; small-batch irfft2 is even cheaper, so the overhead is never justified
- Any padding-based approach to limit distinct cuFFT plans will hurt more than it helps unless the padding is restricted to batches already close to CHUNK_SIZE

### Alternative approaches NOT tried

1. **Pre-warm all plans at init**: Call `irfft2` once with each batch size 1–750 during `FullNativeDecoderEvaluator.__init__` and set `cufft_plan_cache.max_size=750`. Zero runtime overhead. One-time cost ~1–2s per machine per (H, W). Would fully eliminate re-planning without any padding overhead. Not tried due to time constraints.
2. **Increase cufft_plan_cache.max_size alone**: Without padding, if all 750 possible batch sizes fit in the cache simultaneously, plans are compiled on first use and reused forever. Requires `max_size ≥ 750 × n_machines × n_fft_dims`. Not tried.

### Status: REVERTED (batch-padding approach)

The batch-padding approach was reverted. However, the alternative approach #2 (increasing the plan cache) was later implemented successfully — see the next section.

---

## Optimization Round 2 — Applied Successfully

**Date:** 2026-04-13  
**Starting baseline:** P50M2-0 ~1.155s/gen, P75M2-0 ~2.226s/gen

Four changes were applied. Two more were tried and reverted.

### Change 1: cuFFT plan cache size = 4096

Set `torch.backends.cuda.cufft_plan_cache[0].max_size = 4096` in `FullNativeDecoderEvaluator.__init__`. This was alternative #2 from the failed batch-padding attempt — instead of padding batches to fixed sizes, simply make the cache large enough to hold all plans simultaneously.

cuFFT plans are keyed by `(H, W, batch_size, direction)`. With variable batch sizes across waves (1–750), the default cache (typically 16–32 entries) evicts plans constantly, triggering `cuModuleLoad/Unload` cycles (~275ms/rep on P75). A cache of 4096 comfortably holds all plans for all machines.

**Result:** P50 improved by ~97ms (8.4%). P75 was neutral — P75 has fewer distinct batch sizes per wave (more tests/wave, values cluster near CHUNK_SIZE), so the default cache was already sufficient.

**Scaling to larger instances:** Multi-machine instances (P200M4, P300M10) should benefit more than P50/P75, because machines have different (H, W) dimensions. Without a large cache, switching machines causes plan eviction and recompilation. With 4096 entries, all plans for all machines coexist.

### Change 2: Vacancy check input copy non-blocking

Changed `gpu_pairs.copy_(cpu_pairs, /*non_blocking=*/false)` to `non_blocking=true` in `run_gpu_vacancy_check`. The original blocking copy was unnecessary — pinned→GPU copies on the default stream are already ordered before any subsequent kernel launch on that stream.

**Result:** No measurable difference in isolation. Kept because it's correct and removes a pointless sync.

### Change 3: apply_gpu_updates — 6 copy_ calls → 1 packed copy

The `apply_gpu_updates` function uploaded 6 separate int32 arrays (cell_offsets, grid_idxs, y_starts, x_starts, part_widths, part_offsets) via 6 individual `load_workspace_i32_from_i32` calls, each doing a pinned→GPU `copy_`. Replaced with a single packed buffer: all 6 arrays are packed contiguously into one pinned int32 tensor, uploaded in one `copy_` call, then split via `narrow()` on the GPU side.

This function is called twice per wave (Phase 5 placements + Phase 6 new bins). Saving 5 copy_ calls × 2 invocations/wave = 10 fewer copy_ calls per wave.

**Result (combined with change 4):** P50 improved by ~20ms on top of change 1. P75 improved by ~30ms.

**Scaling:** Savings scale linearly with wave count. P200M4 (~200+ waves/gen) would save ~2000+ copy_ calls/gen.

### Change 4: batch_fft_all_tests — 4 copy_ calls → 1 packed copy

The 4 index arrays uploaded at the start of `batch_fft_all_tests` (grid_idx int64, rot_idx int32→int64, heights int32→int64, widths int32→int64) were each uploaded separately. Replaced with a single packed int64 buffer with one `copy_` call. The int32→int64 widening is done during the packing memcpy on the CPU side.

Called twice per wave (p1 and p2). Saving 3 copy_ calls × 2 = 6 fewer calls/wave.

**Scaling:** Same linear scaling with wave count as change 3.

### Results — All 4 changes combined

| Instance | Previous baseline | After changes | Delta |
|---|---|---|---|
| P50M2-0 | 1.155s/gen | **~1.04s/gen** | **−115ms (−10.0%)** |
| P75M2-0 | 2.226s/gen | **~2.20s/gen** | **~−26ms (~−1.2%)** |

P50 sees a clear 10% improvement. P75 improvement is modest and within noise on individual runs, but consistently trends ~20-30ms better across longer runs (gens 10+ trend to ~2.15s).

---

## Optimization Attempts — Tried and Reverted

### Attempt: Eager rfft2 at end of wave

**Idea:** After Phase 5/6 GPU updates, immediately fire rfft2 for all invalidated bins at the end of `process_wave`. The next wave's Phase 2 would find all bins already valid and skip. The rfft2 GPU work would overlap with the CPU's start-of-next-wave bookkeeping.

**Result:** P50 regressed from ~1.04s to ~1.07s. P75 was neutral (~2.24s).

**Why it failed:** Everything runs on the default CUDA stream. The rfft2 queued at end of wave N serializes with the vacancy upload and vacancy check at the start of wave N+1 — the GPU must finish rfft2 before the vacancy kernel can run, even though the vacancy kernel doesn't need rfft2 results. The rfft2 just adds to the serial work on the default stream instead of overlapping with CPU work.

### Attempt: Multi-stream Phase 2 rfft2

**Idea:** Create a secondary CUDA stream (`fft_stream_`). Run Phase 2 rfft2 on `fft_stream_` while vacancy check runs on the default stream. Use `CUDAEvent` for synchronization: record on default stream → block `fft_stream_` (so it waits for grid_states writes) → run rfft2 on `fft_stream_` → record completion → default stream blocks before Phase 4.

**Result:** `CUDA error: device-side assert triggered` at runtime. Works with `CUDA_LAUNCH_BLOCKING=1` (which serializes everything, defeating the purpose).

**Why it failed:** PyTorch's caching CUDA allocator tracks which stream owns each memory allocation. When `fft_stream_` accesses `grid_states` and `grid_ffts` (allocated on the default stream), the allocator may recycle that memory for a default-stream allocation before `fft_stream_` is done with it. Fixing this requires calling `c10::cuda::CUDACachingAllocator::recordStream()` on every tensor shared between streams — a fragile pattern that must be maintained for all future code changes.

---

## Assessment: Is Multi-Stream Pipelining Worth Pursuing?

**Short answer: No.** The cost-benefit ratio is poor.

### What multi-stream Phase 2 would actually save

Phase 2 rfft2 GPU time: **84ms/gen on P75** (4.0% of wall time). This is the maximum saving from overlapping it with CPU work. On larger instances the absolute ms grows proportionally, but as a fraction of wall time it stays ~4%.

### Why the other 27% of GPU idle time can't be reclaimed

Total GPU idle: **655ms/gen (31%)**. Multi-stream Phase 2 reclaims only 84ms (13% of idle time). The remaining 571ms comes from structural sync points that can't be pipelined:

1. **Vacancy check readback sync** (~210ms/gen): CPU must read vacancy results before Pass C. Data dependency — can't overlap.
2. **Phase 4→5 readback sync**: CPU must read selector results before `add_part_to_bin`. Data dependency — can't overlap.
3. **Inter-phase CPU work** (Pass A, Pass C, active indices): CPU is busy, nothing to queue on GPU.

The fundamental constraint is that each wave has a **serial dependency chain**:
```
Phase 2 → vacancy check → Phase 4 → Phase 5 → Phase 6 → (next wave)
```
You can't pipeline wave N+1's Phase 4 with wave N's because Phase 4 of N+1 depends on grid_ffts updated by Phase 5/6 of N.

### Implementation cost

Multi-stream requires:
- `recordStream()` on every tensor shared across streams (`grid_states`, `grid_ffts`, index tensors)
- Careful event management (4 synchronization points per wave)
- Every future code change touching shared GPU tensors must be stream-aware

This is fragile infrastructure for a ~4% gain.

---

## What Would Actually Move the Needle for Large Instances

Ranked by expected impact and feasibility:

### ~~1. Merge p1/p2 vacancy checks — eliminate one sync round-trip per wave~~ — NOT VIABLE

> **Cross-reference:** Already analyzed in `NATIVE_OPTIMIZATION_IDEAS.md` Idea #3 ("Merge p1 and p2 FFT batches") and `CPU_OPTIMIZATION_CANDIDATES.md` #4.C. Conclusion: p2 depends on p1 FFT results (chicken-and-egg problem), true merge is impossible, and dropping the early exit would multiply Phase 3 CPU work 4-5×. **SKIPPED with good reason.**

~~Currently each wave does two vacancy check round-trips...~~

**Status: Removed from consideration.** The dependency chain between p1 and p2 makes this fundamentally infeasible without restructuring the entire decoder loop, which would likely be net-negative.

### 2. Move update_vacancy_rows to GPU — GENUINELY NEW

> **Cross-reference:** NOT previously tried. The closest attempt was `OPTIMIZATION_IDEAS.md` Idea #6 "Eliminate CPU grid mirror," which tried the *inverse* — moving grid reads to GPU for vacancy — and failed catastrophically (+137%) because each `.cpu()` call forced a GPU stream flush. This proposal is fundamentally different: it does the vacancy row scan *on* the GPU where the grid already lives, avoiding CPU↔GPU sync entirely.

`add_part_to_bin` → `update_vacancy_rows` is the single largest CPU hotspot (146ms on P75, 7.3%). It's a per-row max-consecutive-zeros scan over the grid — embarrassingly parallel across rows.

A GPU kernel doing this would eliminate the CPU bottleneck that causes GPU idle time. The grid data is already on the GPU (`grid_states`). The vacancy result could stay on the GPU (`ws_vacancy_gpu_`), eliminating the entire vacancy upload path (`flush_dirty_vacancies` + its copy_ calls).

**Expected saving:** 146ms CPU + 200ms vacancy upload sync = **~346ms/gen on P75 (17%)**. This is the single biggest opportunity.

**Feasibility:** High complexity. Requires a new CUDA kernel for max-consecutive-zeros, plus restructuring `add_part_to_bin` so the CPU grid write and GPU vacancy compute can coexist. The CPU still needs the grid for `add_part_to_bin` (memcpy/AVX2-add), so the grid must exist in both CPU and GPU memory, with the GPU vacancy kernel running after the GPU grid update.

### ~~3. Reduce wave count (algorithmic)~~ — NOT VIABLE

> **Reassessment (2026-04-13):** On closer analysis, this doesn't actually save meaningful time.
>
> **Loop restructuring (place 2 parts sequentially per wave):** This preserves algorithm correctness — place part N, commit it, then place part N+1 against the updated grid, all inside one `process_wave` call. However, the expensive per-part work (vacancy check upload+readback, FFT batch, grid update, vacancy recompute) still runs once per part regardless of nesting. The only saving is the trivial outer-loop bookkeeping (scanning `scratch_active_`, checking `is_done`) — microseconds, not milliseconds.
>
> **Speculative placement (evaluate N+1 before committing N):** This would actually halve the expensive pipeline calls, but it's a major algorithmic change — placement quality would differ because part N+1 would be evaluated against a stale grid. Ruled out.
>
> **Conclusion:** The original framing was misleading. "Reduce wave count" only helps if you reduce the number of times the expensive pipeline runs, which requires speculative placement. Simply restructuring the loop around sequential placements per wave gives the same total work.

### Summary

| Opportunity | Expected saving (P75) | Effort | Scalability | Status |
|---|---|---|---|---|
| ~~Merge p1/p2 vacancy checks~~ | ~~100-200ms (5-10%)~~ | ~~Medium~~ | ~~Linear with waves~~ | **Not viable** — p2 depends on p1 results |
| ~~GPU update_vacancy_rows~~ | ~~~346ms (17%)~~ | ~~High~~ | ~~Linear with bins×parts~~ | **DONE — see Round 3 below** |
| ~~Reduce wave count~~ | ~~500ms (25%)~~ | ~~Very high~~ | ~~Halves all overhead~~ | **Not viable** — loop restructuring saves nothing, speculative placement is algorithmic change |
| Multi-stream Phase 2 | ~84ms (4%) | Medium-high, fragile | Constant fraction | Tried and reverted |

---

## Optimization Round 3 — GPU Vacancy Recompute

**Date:** 2026-04-13  
**Starting baseline (after Round 2):** P50M2-0 ~1.04s/gen, P75M2-0 ~2.20s/gen

### What was done

Moved `update_vacancy_rows` (max-consecutive-zeros per grid row) from CPU to GPU. Previously the vacancy data lived on CPU, was computed by AVX2 SIMD code in `add_part_to_bin_impl`, tracked with dirty flags, and bulk-uploaded to GPU before each Phase 3 vacancy check via `flush_dirty_vacancies`.

The new approach:
1. **New CUDA kernel `_compute_vacancy_rows_kernel`**: One thread per affected row. Each thread scans `grid_states[grid_idx][row][0..W-1]` and computes max consecutive zeros, writing directly to `ws_vacancy_gpu_[grid_idx][row]`.
2. **`recompute_vacancy_gpu` method**: Called after `apply_gpu_updates` in both Phase 5 (existing bin placements) and Phase 6 (new bin placements). Packs affected `(grid_idx, row_idx)` pairs into a pinned buffer, uploads once, launches the kernel.
3. **Phase 6 new-bin init**: Before placing the first part, `ws_vacancy_gpu_.index_fill_(0, idx_t, W)` sets all rows to full width (empty grid).
4. **Removed**: `update_vacancy_rows_cpp` call from `add_part_to_bin_impl`, `flush_dirty_vacancies` call from `process_wave`, `vacancy_gpu_dirty` flag usage.

The CPU `vacancy` vector in `BinStateNative` is no longer maintained. Vacancy data lives exclusively on GPU.

### Results

Profiled with `profile_cpu_hotspots.py` (5 reps, seed 123):

| Instance | Round 2 baseline | After GPU vacancy | Delta |
|---|---|---|---|
| **P50M2-0** | 1.04s/gen | **0.974s/gen** | **−66ms (−6.3%)** |
| **P75M2-0** | 2.20s/gen | **1.742s/gen** | **−458ms (−20.8%)** |

P75 improvement is dramatically larger than predicted (−458ms actual vs −346ms predicted). The full profiler run also shows better numbers than the initial `BRKGA_alg3.py` test (~2.05s), likely because the profiler uses a stable single-seed population.

### New CPU Hotspot Breakdown

#### P50M2-0 — 0.974s/gen

| Hot spot | ms/gen | % wall |
|---|---|---|
| `add_part_to_bin_cpu` (no UVR) | 56.6 | 5.8% |
| Phase 3 Pass B wait (vacancy check GPU) | 130.5 | 13.4% |
| Vacancy readback sync | 130.8 | 13.4% |
| Phase 6 new-bin loop (total) | 32.5 | 3.3% |
| — BinState ctor | 27.0 | 2.8% |
| Vacancy upload sync | 1.0 | 0.1% |
| Phase 3 Pass A + C | 8.2 | 0.8% |
| `update_vacancy_rows` | **0** | **0%** |

Event counts (per gen): 2,942 new bins, 21,614 aptb calls, 136 vacancy-check invocations.

#### P75M2-0 — 1.742s/gen

| Hot spot | ms/gen | % wall |
|---|---|---|
| `add_part_to_bin_cpu` (no UVR) | 70.1 | 4.0% |
| Phase 3 Pass B wait (vacancy check GPU) | 257.0 | 14.8% |
| Vacancy readback sync | 258.1 | 14.8% |
| Phase 6 new-bin loop (total) | 39.9 | 2.3% |
| — BinState ctor | 32.7 | 1.9% |
| Vacancy upload sync | 1.2 | 0.1% |
| Phase 3 Pass A + C | 10.4 | 0.6% |
| `update_vacancy_rows` | **0** | **0%** |

Event counts (per gen): 4,178 new bins, 38,558 aptb calls, 187 vacancy-check invocations.

### Analysis — What changed

**Eliminated entirely:**
- CPU `update_vacancy_rows` (was 90ms P50 / 146ms P75) — now 0ms
- CPU→GPU vacancy upload path (`flush_dirty_vacancies` + dirty flags) — vacancy upload sync dropped from ~102ms to 1ms (P50), ~201ms to 1ms (P75)

**Shifted profile:**
- `add_part_to_bin_cpu` dropped from 140ms to 57ms (P50) and 239ms to 70ms (P75) — the UVR component is gone
- Phase 3 Pass B wait and vacancy readback sync are now the dominant costs, both at ~13-15% each. These represent the GPU vacancy check kernel execution + sync, which was always there but previously masked by the larger CPU+upload overhead

**New bottleneck structure (P75):**
- ~30% of wall time is spent in Phase 3 vacancy check round-trips (Pass B wait + readback sync)
- ~4% is CPU grid writes (`add_part_to_bin_cpu` without UVR)
- ~2.3% is Phase 6 new-bin construction
- The remaining ~64% is GPU-side work (Phase 2 FFT, Phase 4 IFFT, Phase 5 selector) + structural sync overhead

### Cumulative optimization summary

| Change | P50M2-0 | P75M2-0 |
|---|---|---|
| Original baseline | 1.155s | 2.226s |
| + Round 2 (cuFFT cache, packed copies, non-blocking) | 1.04s (−10%) | 2.20s (−1.2%) |
| + Round 3 (GPU vacancy recompute) | **0.974s (−15.7%)** | **1.742s (−21.7%)** |

### What was investigated next — and why we stopped

Three additional optimizations were attempted after Round 3. None produced measurable improvement.

---

## Optimization Round 4 — Copy_ Consolidation — NEUTRAL

**Date:** 2026-04-14

### What was done

Full audit of every `copy_` call per wave identified 17 calls/wave (9 non-blocking CPU→GPU, 8 blocking GPU→CPU). Two consolidation changes were implemented:

**Candidate A — Merge 3 selector readbacks into 1:** The CUDA selector kernel writes has/row/col to 3 separate GPU tensors, each read back with `.to(kCPU)` (3 blocking syncs per chunk). Changed to allocate all 3 as views into a single contiguous GPU buffer, then read back with one blocking `copy_`.

**Candidate B — Merge apply_gpu_updates + recompute_vacancy_gpu uploads:** These two methods each did a separate packed CPU→GPU `copy_`. Combined into a single method `apply_gpu_updates_and_vacancy` with one packed buffer containing both grid-update params and vacancy-recompute params.

### Results

| Instance | Round 3 baseline | After consolidation | Delta |
|---|---|---|---|
| P50M2-0 | 0.974s | 0.96s (best reps) / 1.00s (mean) | Within noise |
| P75M2-0 | 1.742s | 1.74s (best reps) / 1.80s (mean) | Within noise |

### Why it didn't help

The 263µs average copy_ overhead from nsys was **bimodal**: most calls complete in ~6µs (median), with a long tail of implicit blocking when the CUDA command queue is full. Saving 2-3 calls per wave at 6µs each ≈ ~1ms/gen total. The blocking readback syncs (Candidate A) were similarly cheap because by the time the first `.to(kCPU)` fires, the GPU selector kernel has already finished — the "sync" is just a fast memcpy check.

### Status: REVERTED

Changes were clean but negligible. Reverted to keep codebase simpler.

---

## Optimization Attempt — Eager rfft2 After Phase 5 — NEUTRAL

**Date:** 2026-04-14

### What was done

After Phase 5's `apply_gpu_updates` + `recompute_vacancy_gpu`, immediately fire `rfft2` for all bins invalidated by Phase 5 placements (async on default stream). The intent was to overlap this GPU rfft2 work with Phase 6's CPU work (BinState ctor, memset, add_part_to_bin_cpu — ~40ms/gen on P75).

Implementation: collect grid indices of placed bins, deduplicate, fire `rfft2` + `index_copy_` into `grid_ffts`, mark bins as `grid_fft_valid = true`. The next wave's Phase 2 then only needs to handle Phase-6-new bins (smaller batch).

### Results

| Instance | Round 3 baseline | With eager rfft2 | Delta |
|---|---|---|---|
| P50M2-0 | 0.974s | ~0.96s (best) / 1.02s (mean) | Within noise |
| P75M2-0 | 1.742s | ~1.78s (best) / 1.83s (mean) | Slightly worse |

### Why it didn't help

Same root cause as the earlier "Eager rfft2 at end of wave" attempt (see above): on a single CUDA stream, all GPU work is serial. The rfft2 is queued after `apply_gpu_updates` and `recompute_vacancy_gpu` — it must wait for those to finish. Meanwhile, Phase 6 CPU work was **already overlapping** with the async GPU kernels from Phase 5. Adding rfft2 just extends the GPU queue without changing when the CPU is free to work. On P75, the extra GPU work slightly delays the next wave's vacancy check.

This is the **third** time eager/early rfft2 has been tried and failed (two from this round, one from Round 2). The fundamental limitation is single-stream serialization: you cannot overlap independent GPU operations without multiple CUDA streams, and multi-stream was already tried and rejected as too fragile.

### Status: REVERTED

---

## Performance Floor Assessment

**Date:** 2026-04-14

After Round 3 (GPU vacancy recompute) and two additional failed optimization attempts, we have high confidence that the current performance represents the **practical floor for single-stream optimization** on this architecture.

### Final performance

| Instance | Original baseline | Current (Round 3) | Total improvement |
|---|---|---|---|
| P50M2-0 | 1.155s/gen | **0.974s/gen** | **−15.7%** |
| P75M2-0 | 2.226s/gen | **1.742s/gen** | **−21.7%** |

### Why further single-stream optimization is unlikely to help

1. **GPU idle time is structural, not fixable:** The wave pipeline has unavoidable serial dependencies (vacancy readback → Pass C → Phase 4 → selector readback → Phase 5). At each sync point, CPU waits for GPU results before issuing next GPU work. This creates idle gaps that cannot be filled without multi-stream pipelining.

2. **CPU hotspots are eliminated:** `update_vacancy_rows` (was ~8% of wall time) is now 0ms. The remaining CPU work (`add_part_to_bin_cpu` at ~4%, Phase 6 at ~2.3%) already overlaps with async GPU kernels.

3. **Copy_ consolidation has diminishing returns:** The remaining per-wave copies are either (a) blocking readbacks that are inherently sequential, or (b) non-blocking uploads completing in ~6µs median. Consolidating these saves single-digit milliseconds per generation.

4. **Eager rfft2 fails on single stream:** Tried 3 times with different approaches. All fail because GPU work serializes on the default stream — you can't overlap independent GPU operations.

### What would be needed for further improvement

- **Multi-stream pipelining** (~4% gain, tried and reverted due to PyTorch allocator fragility with `recordStream()`)
- **Larger instances** (P200M4+) where GPU utilization is naturally higher and per-wave overhead is a smaller fraction
- **CHUNK_SIZE tuning** (instance/GPU-dependent, deferred)
- **Persist grid_states/grid_ffts across evaluate_batch calls** (~10-15ms/gen, trivial but small)

The last item (tensor persistence) is the only untried change likely to help across all instances, but its ~10-15ms saving is marginal compared to the ~180-480ms already gained.
