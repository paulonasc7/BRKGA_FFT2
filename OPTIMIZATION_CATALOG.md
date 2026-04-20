# Optimization Catalog — BRKGA_FFT2

**Compiled:** 2026-04-14
**Purpose:** Single reference for all optimization ideas across all markdown files, categorized by outcome.
**Current performance:** P50M2-0 **0.974s/gen**, P75M2-0 **1.742s/gen** (full native decoder, RTX A4000)
**Original baseline (pre-all-optimizations):** ~40s/gen (serial Python), ~13s/gen (early wave_batch)

---

## Legend

- **Tested, Positive** — Implemented and kept; measurable improvement
- **Tested, Negative/Neutral** — Implemented and reverted or kept with no benefit
- **Untested, Worth Testing** — Not yet attempted; expected positive ROI
- **Untested, Not Worth Testing** — Not attempted; analysis shows poor ROI or infeasibility

---

## 1. Tested, Positive Results

These optimizations are currently live in the codebase and contributed measurable speedups.

### 1.1 Early Python-era optimizations (IMPLEMENTATION_PHASES.md, OPTIMIZATION_RECOMMENDATIONS.md)

All 17 items from Phase 1–6 of the original recommendations were implemented:

| # | Change | Files | Impact |
|---|--------|-------|--------|
| 1 | Replace string-key dict lookups with dataclasses + integer indexing | All files | 5–15% CPU reduction in hot loops |
| 2 | Separate part loading from machine-specific FFT (load parts once, not per machine) | BRKGA_alg3.py | 10–30% init time reduction |
| 3 | Create ThreadPoolExecutor once at init (not per generation) | BRKGA_alg3.py | 2–5% runtime |
| 4 | Improve initial solution construction (cache makespan, early exit) | BRKGA_alg3.py | Init phase speedup |
| 5 | Vectorize density calculation (replace itertools.groupby with NumPy) | BRKGA_alg3.py | Minor |
| 6 | Pre-allocate buffers for vacancy vector | binClassNew.py | Hot path speedup |
| 7 | Track enclosure box bounds incrementally (O(1) vs O(n×m)) | binClassNew.py | Algorithmic improvement |
| 8 | Remove unused grid2 array | binClassNew.py | Memory savings |
| 9 | Use uint8 dtype for grid (8× memory reduction) | binClassNew.py | Memory + cache improvement |
| 10 | Make arrays contiguous at load time | BRKGA_alg3.py | Minor |
| 11 | Pre-compute best_rotation | BRKGA_alg3.py, placement.py | Minor |
| 12 | Machine-level parallelism (ThreadPoolExecutor in placement.py) | placement.py | Multi-machine speedup |
| 13 | GPU batching evaluation | collision_backend.py | Already efficient |
| 14 | Dataclasses for part/machine data | data_structures.py | Cleaner + faster access |
| 15 | Sort parts once and reuse order | BRKGA_alg3.py | Minor |
| 16 | Optimize sliding window / dictionary access | binClassNew.py | Hot path speedup |
| 17 | Cache grid FFT between calls | collision_backend.py | Limited benefit (changes each insert) |

### 1.2 Wave-batch evaluator improvements (PERFORMANCE_IMPROVEMENTS.md)

| ID | Change | Savings | Source |
|----|--------|---------|--------|
| IMP-1 | Use pre-computed GPU tensors in `_place_part_in_bin` | −1.0s (−13%) | wave_batch_evaluator.py |
| IMP-4 | Replace list-of-dicts with parallel arrays in Phase 3 | −0.09s | wave_batch_evaluator.py |
| IMP-5 | Remove redundant `.astype(np.int32)` on density arrays | −0.05s | wave_batch_evaluator.py |
| IMP-6 | Increase CHUNK_SIZE from 500 to 750 | −0.04s | wave_batch_evaluator.py |
| IMP-7 | Single `torch.max` instead of separate `argmax` + `max` | −0.17s | wave_batch_evaluator.py |
| IMP-8 | Pre-store `uint8` rotations, remove per-insert `.astype(np.uint8)` | −0.05s | data_structures.py, all insert sites |
| IMP-9 | Use Numba JIT in `binClassInitialSol.py` for vacancy update | Init phase only | binClassInitialSol.py |
| IMP-11 | Pre-build integer index tensors before chunk loop | −0.12s | wave_batch_evaluator.py |

### 1.3 Wave-batch algorithmic improvements (OPTIMIZATION_IDEAS.md)

| ID | Change | Savings | Source |
|----|--------|---------|--------|
| Idea 2 | First-valid-bin early exit (split Phase 3+4 into two passes) | −1.03s (−31%) | wave_batch_evaluator.py |
| Idea 5 | `torch.compile` on post-IFFT score computation | −0.26s (−7%) | wave_batch_evaluator.py |
| Idea 8 | Batch Phase 6 new-bin creation | −0.22s (−6%) | wave_batch_evaluator.py |

### 1.4 FFT optimizations (FFT_OPTIMIZATION_OPTIONS.md, OPTIMIZATION_IDEAS.md)

| Change | Savings | Source |
|--------|---------|--------|
| `rfft2`/`irfft2` instead of `fft2`/`ifft2` (real-valued FFT) | 5.74s → 4.53s | wave_batch_evaluator.py, collision_backend.py |
| NumPy-side composite scoring (density calc out of Python loop) | 4.53s → 4.23s | wave_batch_evaluator.py |
| Custom CUDA kernel for batched GPU grid updates | 4.23s → 3.80s | cuda_batch_update.py |

### 1.5 Full native decoder — C++ migration (OPTIMIZATION_ROADMAP.md, PHASE345_NATIVE_FUSION_PLAN.md)

Moving the entire decoder (Phases 1–6) into a single C++/pybind11 extension:

| Change | Savings | Source |
|--------|---------|--------|
| Full native C++ decoder (all phases) | wave_batch 2.61s → native 1.91s (P50), 4.89s → 3.78s (P75) | full_native_decoder.py |
| CUDA selector kernel (GPU-side best position) | −1.4% P50 | full_native_decoder.py |
| Lazy valid_zeros (skip 6 GPU ops when selector kernel handles it) | −17.4% P50, −15.7% P75 | full_native_decoder.py |
| Selector output pre-allocation (reuse scratch tensors) | Negligible but cleaner | full_native_decoder.py |

### 1.6 Native decoder optimizations (NATIVE_OPTIMIZATION_IDEAS.md)

| # | Change | Savings | Source |
|---|--------|---------|--------|
| 1 | Eliminate irfft2 normalization multiply (use `"forward"` norm, pre-divide part FFTs) | −8.1% P50, −14.9% P75 | full_native_decoder.py |
| 2 | Fused gather-multiply CUDA kernel (3 ops → 1 pass) | −18.0% P50, −16.2% P75 | full_native_decoder.py |
| 5 | GPU-accelerated vacancy check (3-pass GPU approach) | −0.5% P50, −1.6% P75 | full_native_decoder.py |

### 1.7 CPU-side optimizations (CPU_OPTIMIZATION_CANDIDATES.md)

| # | Change | Savings | Source |
|---|--------|---------|--------|
| 1 | SIMD `update_vacancy_rows_cpp` (AVX2) | −287ms/gen (4.84× function speedup) | full_native_decoder.py |
| 3 | Template + memcpy/AVX2-add grid write in `add_part_to_bin_cpu` | −93ms/gen (−7.7%) | full_native_decoder.py |
| 4 | Phase 6 bin pool (keyed by (H,W), warm memory reuse) | −16ms wall (P50), noise (P75) | full_native_decoder.py |

### 1.8 Profiling-era optimizations (PROFILING_ANALYSIS.md)

| Change | Savings | Source |
|--------|---------|--------|
| cuFFT plan cache size = 4096 | −97ms P50 (−8.4%), neutral P75 | full_native_decoder.py |
| Vacancy check input copy `non_blocking=true` | Negligible, but correct | full_native_decoder.py |
| `apply_gpu_updates` — pack 6 copy_ calls into 1 | ~−20ms P50, ~−30ms P75 | full_native_decoder.py |
| `batch_fft_all_tests` — pack 4 copy_ calls into 1 | (combined with above) | full_native_decoder.py |
| GPU vacancy recompute (Round 3 — CUDA kernel for max-consecutive-zeros) | −66ms P50 (−6.3%), −458ms P75 (−20.8%) | full_native_decoder.py |
| VRAM cap in C++ evaluator (safety, prevents OOM on large instances) | Safety only | full_native_decoder.py |
| Nsight Systems profiling (`nsys profile --trace=cuda,osrt`) | Resolved "44% unaccounted CUDA time" mystery — torch.profiler mis-attributed CPU-blocked time; actual GPU util is ~43% | PROFILING_ANALYSIS.md |

### 1.9 Phase 2 custom gather/scatter kernels (2026-04-19)

**Source:** Deep CUDA-activity profiling (torch.profiler + CUDA events) identifying transfer bucket as 26% of GPU time.

Replaced two PyTorch ops in Phase 2 (batch rfft2 pipeline) with custom CUDA kernels:
- `grid_states.index_select(0, idx_t)` → `_native_gather_grids_kernel` (gather uint8 grids into batched float32 input for rfft2)
- `grid_ffts.index_copy_(0, idx_t, batch_ffts)` → `_native_scatter_ffts_kernel` (scatter rfft2 output back into the per-bin FFT tensor)

| Metric | P50M2-0 | P75M2-0 |
|--------|---------|---------|
| Wall clock | 870 → 875ms (flat, within ±35ms noise) | 1639 → 1634ms (flat) |
| GPU kernel time | 1470 → 1354ms (**−116ms**) | — |
| Transfer GPU time | 389 → 313ms (**−75ms**) | — |
| Custom kernel time | 254 → 284ms (+30ms) | — |
| Phase 2 wall time | 176 → 156ms (−20ms) | flat |

**Correctness:** fingerprints match baseline exactly (P50: `281426.499026`; P75: all `10000000000000000`).

**Nuance — kept despite flat wall clock:** The GPU is ~100% saturated on the critical path (rfft2 → vacancy readback → irfft2). The gather/scatter ops ran concurrently with neighboring GPU work, so they weren't on the serial dependency chain. Removing them freed GPU cycles without shortening the chain. Retained because it (a) reduces GPU pressure / thermal load, and (b) provides headroom for any future optimization that introduces concurrent streams or overlaps work across machines (§3.1).

### 1.10 Bug fixes that affected performance (BUG_FIXES_LOG.md)

| Bug | Impact | Files |
|-----|--------|-------|
| Float32 sentinel comparison (Pass 2 never executing) | +20% gen time (correct behavior now runs Pass 2) | wave_batch_evaluator.py |
| Float32 score precision for column tie-breaking | Correctness fix | wave_batch_evaluator.py |
| Composite score formula doesn't match lexicographic hierarchy | Correctness fix, switched to tuple comparison | wave_batch_evaluator.py |

---

## 2. Tested, Negative/Neutral Results

These were implemented, measured, and either reverted or shown to have no benefit.

### 2.1 FFT backend alternatives (FFT_OPTIMIZATION_OPTIONS.md)

| Idea | Result | Reason |
|------|--------|--------|
| TF32 (Tensor Float 32) | 0% improvement | cuFFT does not use TF32; only affects matmul/convolution |
| CUDA streams (2/4/8 streams within a machine) | 23–29% **slower** | GPU already saturated by FFT; streams add overhead |
| CuPy backend | 4.7× **slower** than PyTorch | PyTorch has better cuFFT plan caching |
| FP16 FFT | 0% improvement | cuFFT requires power-of-2 padding for FP16; padding cancels gains |

### 2.2 Wave-batch neutral changes (PERFORMANCE_IMPROVEMENTS.md, OPTIMIZATION_IDEAS.md)

| ID | Change | Result | Reason |
|----|--------|--------|--------|
| IMP-2 | Cache `torch.arange` and scalar tensors | ~0s (noise) | Tensor creation is ~11ms total, below noise floor |
| IMP-3 | GPU vacancy (dual grid elimination in wave_batch) | +12% **regression** | Broke CPU-GPU overlap; serialized Phase 3 GPU check |
| IMP-10 | Fitness cache for duplicate chromosomes | 0% hit rate | Continuous random-key chromosomes are never duplicates |
| Idea 3 | Pre-allocate chunk tensors | ~0s (noise) | PyTorch's CUDA caching allocator already reuses blocks |
| Idea 6 | Eliminate CPU grid mirror (read from GPU for vacancy) | +137% Phase 5 **regression** | Each `.cpu()` call forces GPU stream flush; destroys CPU-GPU overlap |
| Idea 9 | Pre-allocate grid_states/grid_ffts across generations | ~0s (noise) | PyTorch caching allocator handles it; just tested and confirmed in current session |

### 2.3 Native decoder neutral/failed attempts (NATIVE_OPTIMIZATION_IDEAS.md)

| # | Change | Result | Reason |
|---|--------|--------|--------|
| 4 | Pack CPU→GPU transfers (4 → 1 per batch_fft call) | 0% improvement | Individual transfers are ~few KB; PCIe DMA at 16 GB/s makes each ~0.6µs; launch overhead already hidden |
| 6 | Incremental FFT update (shift theorem) | **Rejected** — correctness failures | `sincosf` CUDA arithmetic diverges from cuFFT butterfly results; small errors cascade into wrong placements; periodic resync makes it *worse* |

### 2.4 Full native decoder failed attempts (OPTIMIZATION_ROADMAP.md)

| Change | Result | Reason |
|--------|--------|--------|
| CHUNK_SIZE = 1500 | P50 noise, P75 OOM | Doubles overlap_batch VRAM; not viable without sparse allocation |
| Pinned memory for CPU→GPU transfers | 0% improvement | Transfers too small (~10 KB) for staging overhead to matter |

### 2.5 Profiling-era failures (PROFILING_ANALYSIS.md)

| Change | Result | Reason |
|--------|--------|--------|
| cuFFT batch-padding (pad to canonical sizes) | P50 −4%, P75 **+7% regression** | Padding small batches (1–5 tests) adds 200–1500% FFT overhead; outweighs plan-reuse savings |
| Eager rfft2 at end of wave | P50 regression, P75 neutral | Single CUDA stream serializes; rfft2 adds to serial queue instead of overlapping |
| Multi-stream Phase 2 rfft2 | CUDA device-side assert failure | PyTorch caching allocator tracks stream ownership; cross-stream tensor access causes memory corruption without fragile `recordStream()` |
| Copy_ consolidation (Round 4 — merge selector readbacks + vacancy uploads) | Neutral | Non-blocking copies have 6µs median overhead; blocking readbacks already fast because GPU finished |
| Eager rfft2 after Phase 5 | Neutral | Third attempt at early rfft2; same single-stream serialization problem |
| Persist grid_states/grid_ffts across evaluate_batch calls | Neutral | PyTorch caching allocator already pools freed CUDA memory; tested today |

### 2.6 Transfer-bucket attempts (2026-04-19)

Two attempts targeted the 26%-of-GPU-time transfer bucket. Both reverted.

| Change | Result | Reason |
|--------|--------|--------|
| Stride-aware selector kernel (eliminate `.contiguous()` DtoD in Phase 5) | 0% improvement; DtoD count unchanged (198 calls) | The 146ms DtoD is **inside** `aten::_fft_c2r` — PyTorch normalizes cuFFT output via an internal DtoD-style copy. Our `.contiguous()` was not the source. Removing it had no effect. Reverted. |
| Manual cuFFT (bypass PyTorch `_fft_c2r` to skip internal DtoD) | `CUFFT_INVALID_VALUE` (code 5) on plan creation | Tried `cufftPlanMany` (explicit inembed/onembed), `cufftMakePlanMany64` (explicit + NULL embed), `cufftPlanMany` 32-bit w/ NULL — all fail at batched C2R plan creation. Unbatched `cufftPlan2d(300,300,C2R)` succeeds. Suspected conflict with PyTorch's cuFFT plan cache for batched C2R plans. Not debuggable remotely in reasonable time. Reverted. |

**Lesson:** The largest DtoD in the native decoder lives inside PyTorch's FFT wrapper, not in our code. Reducing it requires either bypassing `torch::fft::irfft2` (blocked by plan-creation conflict above) or accepting it as structural.

### 2.7 OPTIMIZATION_ANALYSIS.md — Phase 3/6 native experiments

| Change | Result | Reason |
|--------|--------|--------|
| Phase 6 fully fused GPU placement + vacancy metadata | Neutral-to-worse | No consistent end-to-end win vs structured baseline |
| Phase 6 batched CPU-side bin/vacancy updates via C++ extension | Neutral-to-worse | Same as above |
| Phase 3 native collector (`ABRKGA_PHASE3_CPP=1`) | −2% vs reverted Phase 3 | Indirect slowdowns from BinState object size growth, module imports, cache locality |

---

## 3. Untested, Worth Testing

These ideas have not been implemented but analysis suggests they could provide meaningful improvement.

### 3.1 Parallel machine processing — HIGHEST PRIORITY for multi-machine instances

**Source:** OPTIMIZATION_ANALYSIS.md §1, OPTIMIZATION_ROADMAP.md §1
**Expected:** 1.5–2.5× on multi-machine instances (P100M4: 12.5s → ~5.8s)
**Effort:** High
**Prerequisites:** MachineWorkspace refactor (§1a), sparse grid allocation for large instances

Machines are processed sequentially but are fully independent. Two threads with two CUDA streams would let Machine 0's GPU Phase 4 overlap with Machine 1's CPU Phase 3. Different machine dimensions (250×250 vs 400×400) are handled naturally since each thread owns its own tensors.

**For P50M2/P75M2:** Both machines' tensors fit in VRAM simultaneously — can implement without sparse allocation.

### 3.2 Sparse grid allocation — Required for P200M4+

**Source:** SCALABILITY_ANALYSIS.md, OPTIMIZATION_ANALYSIS.md §4, OPTIMIZATION_ROADMAP.md §2
**Expected:** Enables instances that currently OOM (P200M4: 21 GB grid_states for one machine)
**Effort:** High
**Risk:** Medium (correctness)

Replace pre-allocated `(max_total_bins, H, W)` tensor with a pool allocator that hands out grid slots on demand. Most solutions use ≤5 bins but current code pre-allocates 16–33 per solution. Peak VRAM reduction: 3–5×.

Also needs: sub-batching loop in `process_machine_batch` to handle cases where even sparse allocation can't fit all solutions at once.

### 3.3 Adaptive FFT size based on grid occupancy

**Source:** OPTIMIZATION_ANALYSIS.md §6a, OPTIMIZATION_ROADMAP.md §3b
**Expected:** Up to 25% Phase 4 reduction (Phase 4 is ~52% of total)
**Effort:** Medium
**Risk:** Low

Early in placement, grids are mostly empty. Using `irfft2((max_occupied_row + part_height) × W)` instead of full `irfft2(H × W)` could be 2× smaller per call. Challenge: tests in one batched irfft2 must share dimensions, so tests need grouping by fill level.

### 3.4 Pre-filter trivially-failing IFFT tests

**Source:** OPTIMIZATION_ANALYSIS.md §6b, OPTIMIZATION_ROADMAP.md §3c
**Expected:** 10–15% Phase 4 reduction
**Effort:** Medium
**Risk:** Low

After Phase 3 vacancy filtering, some tests still fail geometrically (has_result=0). Pre-filtering via area fill ratio or bounding-box overlap before IFFT could skip these.

### 3.5 Early pruning across machines

**Source:** OPTIMIZATION_ANALYSIS.md §2
**Expected:** 10–30% on multi-machine instances
**Effort:** Low-Medium
**Risk:** Low

After evaluating Machine 0, solutions whose makespan already exceeds the worst elite fitness cannot become elites. Skip remaining machines for those solutions.

**Note:** This is safe — the assigned fitness (Machine 0 makespan) is a lower bound on the true fitness, so the solution ranks even worse than estimated.

### 3.6 Vectorize BRKGA mating/crossover

**Source:** REMAINING_OPTIMIZATIONS.md §1
**Expected:** Medium-High for large populations (500+)
**Effort:** Low
**Risk:** None

Current Python loop creates offspring one at a time. Generate all parent indices and crossover masks in bulk NumPy operations.

### 3.7 Use float32 for chromosomes/populations

**Source:** REMAINING_OPTIMIZATIONS.md §2
**Expected:** Medium — halves memory bandwidth, better cache utilization
**Effort:** Very Low
**Risk:** Low (may affect hash stability if fitness caching is re-enabled)

### 3.8 Elite selection: argpartition instead of full argsort

**Source:** REMAINING_OPTIMIZATIONS.md §4
**Expected:** Low-Medium for large populations
**Effort:** Very Low
**Risk:** None

`np.argpartition` is O(n) average vs O(n log n) for argsort.

### 3.9 Cheap prefilters before FFT collision test

**Source:** REMAINING_OPTIMIZATIONS.md §7
**Expected:** Medium-High (instance-dependent)
**Effort:** Medium
**Risk:** Low

Add bounding box check and row-range vacancy pre-check before committing to FFT.

### ~~3.10 Phase 4 early-exit by bin layer~~ → Already implemented (first-valid-bin)

**Source:** PHASE5_OPTIMIZATION.md Idea 3
**Status:** Redundant — the existing first-valid-bin two-pass (p1/p2) design already achieves this. Each context's first vacancy-passing bin (regardless of bin index) is tested in p1. Only contexts that fail the FFT collision check go to p2 for remaining bins. Splitting by bin layer would add kernel launch overhead without reducing IFFT work, since each context's first-valid bin is independent of other contexts' results.

### 3.11 CHUNK_SIZE tuning

**Source:** PROFILING_ANALYSIS.md §5, OPTIMIZATION_ROADMAP.md §3a
**Expected:** Low — previously neutral at 750 vs 1000/1500, but worth retesting after cuFFT cache fix
**Effort:** Very Low
**Risk:** None (easy experiment)

Test CHUNK_SIZE=1024 or 1500 now that cuFFT plan cache is 4096.

### ~~3.12 Nsight Systems profiling for unaccounted GPU time~~ → MOVED to §1 (Tested, Positive)

**Completed.** Nsight Systems profiling was run (`nsys profile --trace=cuda,osrt`) and fully resolved the "44% unaccounted CUDA time" mystery. See PROFILING_ANALYSIS.md §"Nsight Systems (nsys) Deep Dive". Result: torch.profiler was mis-attributing CPU-blocked time as CUDA time; actual GPU utilization is ~43%, not 100%. No mystery kernels exist.

### 3.13 Profile the BRKGA outer loop

**Source:** (Not in any existing markdown — identified in current conversation)
**Expected:** Unknown — crossover, selection, and population management have never been profiled
**Effort:** Low (profiling only)
**Risk:** None

The decoder is at its floor. The BRKGA loop in `BRKGA_alg3.py` may have low-hanging fruit.

---

## 4. Untested, Not Worth Testing

These ideas were analyzed and determined to have poor ROI, fundamental infeasibility, or to be superseded by other work.

### 4.1 Merge p1/p2 vacancy checks

**Source:** NATIVE_OPTIMIZATION_IDEAS.md §3, PROFILING_ANALYSIS.md, CPU_OPTIMIZATION_CANDIDATES.md §4.C
**Why not:** p2 depends on p1 FFT results — chicken-and-egg problem. Dropping the early exit would multiply Phase 3 CPU work 4–5×. Analyzed multiple times, consistently rejected.

### 4.2 Reduce wave count (algorithmic)

**Source:** PROFILING_ANALYSIS.md §3
**Why not:** Loop restructuring (place 2 parts per wave) saves nothing — expensive per-part work runs once per part regardless. Speculative placement (evaluate N+1 before committing N) would actually halve pipeline calls but is a major algorithmic change that affects placement quality.

### 4.3 Multi-stream pipelining within a machine

**Source:** FFT_OPTIMIZATION_OPTIONS.md §4, PROFILING_ANALYSIS.md, OPTIMIZATION_ROADMAP.md §10
**Why not:** Previously tested (23–29% slower due to GPU saturation). Multi-stream Phase 2 rfft2 attempted — CUDA device-side assert from allocator conflicts. Fragile `recordStream()` pattern required for all shared tensors. Maximum theoretical gain is only ~4% (84ms/gen on P75).

### 4.4 CuPy backend

**Source:** FFT_OPTIMIZATION_OPTIONS.md §6, PERFORMANCE_OPTIMIZATION_GUIDE.md §O2
**Why not:** Benchmarked at 4.7× slower than PyTorch for FFT. PyTorch has superior cuFFT plan caching.

### 4.5 FP16 FFT

**Source:** FFT_OPTIMIZATION_OPTIONS.md §7, OPTIMIZATION_IDEAS.md Idea 1
**Why not:** cuFFT requires power-of-2 dimensions for FP16. Padding 300×250 → 512×256 cancels all gains. Benchmarked: 0.99× (no improvement). Only revisit if cuFFT lifts the power-of-2 restriction.

### 4.6 Dynamic free-list for grid allocation

**Source:** SCALABILITY_ANALYSIS.md
**Why not:** Peak slot count is still close to `num_solutions × max_bins` (bins don't close during wave-based placement). Complicates Phase 2/4 index lookups. Sub-batching is the correct primary fix; free-list is at best a secondary micro-optimization.

### 4.7 Early termination heuristic (stop checking rotations)

**Source:** FFT_OPTIMIZATION_OPTIONS.md §5
**Why not:** Changes algorithm semantics — may find different (worse) solutions. Not worth the quality risk.

### 4.8 Vacancy check async + double-buffer

**Source:** CPU_OPTIMIZATION_CANDIDATES.md §4
**Why not:** Investigated — the 0.71ms per-call wait is mostly real GPU execution time, not preventable stall. Making upload non_blocking + CUDA event deferred sync shifts the wait from upload to readback but doesn't change total wait. Net saving: ~0ms.

### 4.9 Phase 3 Pass A/C merge + SoA refactor

**Source:** CPU_OPTIMIZATION_CANDIDATES.md §3
**Why not:** Measured at only 8ms/gen combined. Dead target — not worth the complexity.

### 4.10 Incremental FFT update (shift theorem)

**Source:** NATIVE_OPTIMIZATION_IDEAS.md §6
**Why not:** Mathematically correct but `sincosf` CUDA arithmetic diverges from cuFFT butterfly results. Small errors cascade into wrong placements through boundary-condition sensitivity. Periodic resync paradoxically makes it worse. Cannot be fixed without cuFFT-compatible phase computation.

### 4.11 Fitness memoization

**Source:** REMAINING_OPTIMIZATIONS.md §9
**Why not:** With continuous random-key chromosomes (float32 uniform [0,1]), probability of duplicate quantized chromosomes is essentially zero. Confirmed 0% hit rate in profiling.

### 4.12 Hybrid collision method (direct bitwise for small parts)

**Source:** REMAINING_OPTIMIZATIONS.md §8
**Why not:** All collision detection is now batched through FFT in the native decoder. The overhead of switching between methods for different part sizes would add complexity for marginal benefit.

### 4.13 Phase 5 composite float64 score (replace tuple comparison)

**Source:** OPTIMIZATION_ANALYSIS.md §5
**Why not:** Saves ~0.17ms/wave × 355 waves = ~60ms/gen for P50M2. Marginal. Also, tuple comparison in the native decoder is already a C++ struct comparison (`lex_better`), so the Python overhead that motivated this is gone.

### 4.14 Persist grid_states/grid_ffts across evaluate_batch calls

**Source:** PROFILING_ANALYSIS.md §4, OPTIMIZATION_IDEAS.md Idea 9
**Why not:** Tested today (2026-04-14). PyTorch's caching CUDA allocator already pools freed GPU memory. `torch::zeros` on re-allocation reuses cached blocks without hitting `cudaMalloc`. For multi-machine instances, tensors must be freed between machines anyway (VRAM budget). Result: 0% improvement.

### 4.15 Reduce FFT size dynamically

**Source:** FFT_OPTIMIZATION_OPTIONS.md §2
**Why not:** Superseded by §3.3 (adaptive FFT size based on grid occupancy) which is the more refined version of this idea. The original proposal was too vague and didn't account for batching constraints.

### 4.16 Batch grid FFTs across multiple parts (speculative execution)

**Source:** FFT_OPTIMIZATION_OPTIONS.md §3
**Why not:** Complex, memory-heavy, and the greedy sequential placement algorithm means you can't speculate on future part placements without changing the algorithm.

### 4.17 torch.compile on Phase 4 inner loop (in native decoder context)

**Source:** OPTIMIZATION_IDEAS.md Idea 5
**Why not:** This was effective in the Python wave_batch era (−7%). In the native decoder, the equivalent work is already done by the fused gather-multiply CUDA kernel and the CUDA selector kernel — there's no Python-level computation left to compile.

### 4.18 Batch Phase 3 vacancy checks (Numba prange)

**Source:** OPTIMIZATION_IDEAS.md Idea 7
**Why not:** Superseded by GPU vacancy check (NATIVE_OPTIMIZATION_IDEAS.md §5) which moved vacancy checking to GPU entirely. The Numba prange approach would have marginal benefit since per-call overhead was small and `prange` thread-pool overhead for individually-fast calls may not help.

### 4.19 Vacancy readback sync optimization

**Source:** CPU_OPTIMIZATION_CANDIDATES.md §2
**Why not:** Measured at only 7ms/gen. Dead target.

### 4.20 Python ↔ C++ boundary optimization

**Source:** CPU_OPTIMIZATION_CANDIDATES.md §5
**Why not:** Confirmed negligible — single pybind11 call per evaluate_batch, ~microseconds.

---

## Summary — Performance Evolution

| Era | Best time (P50M2-0) | Key milestone |
|-----|---------------------|---------------|
| Original serial Python | ~40s/gen | Baseline |
| Early wave_batch | ~13s/gen | GPU FFT batching |
| wave_batch + all IMP changes | ~5.74s/gen | rfft2, composite scoring, CUDA grid updates |
| wave_batch + algorithmic opts | ~3.80s/gen | First-valid-bin early exit, torch.compile, batch Phase 6 |
| Full native decoder (initial) | ~1.91s/gen | C++ migration of all phases |
| Native + lazy valid_zeros | ~1.44s/gen | Skip unnecessary GPU ops |
| Native + fused gather-multiply | ~1.44s/gen | Custom CUDA kernel |
| Native + CPU SIMD + bin pool | ~1.10s/gen | AVX2 vacancy, template grid write |
| Native + profiling optimizations | **0.974s/gen** | cuFFT cache, packed copies, GPU vacancy recompute |

**Total speedup: ~41× from original baseline.**

---

## Highest-Impact Remaining Opportunities

1. **Parallel machine processing** (§3.1) — 1.5–2.5× on multi-machine, requires workspace refactor
2. **Sparse grid allocation** (§3.2) — enables P200M4+, required for parallel machines on large instances
3. **Adaptive FFT size** (§3.3) — up to 25% Phase 4 reduction
4. **Pre-filter failing IFFT tests** (§3.4) — 10–15% Phase 4 reduction
5. **Early pruning across machines** (§3.5) — 10–30% on multi-machine
6. **BRKGA outer loop profiling** (§3.13) — unknown, never examined
