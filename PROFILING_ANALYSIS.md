# Profiling Analysis — FullNativeDecoderEvaluator

**Date:** 2026-04-10  
**Baseline after CPU optimizations:** P50M2-0 1.105s/gen, P75M2-0 2.010s/gen  
**vs wave_batch (P75M2-0):** 2.15×  

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
