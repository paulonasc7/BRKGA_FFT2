# Deep Optimization Analysis — BRKGA_FFT2

**Date:** 2026-04-01
**Baseline:** P50M2-0 = 2.76s/gen, P100M4-0 = 12.8s/gen
**Hardware:** RTX A4000 (16GB VRAM), Paperspace Gradient

---

## Update (2026-04-06): Experiments Run and Conclusions

This section summarizes what was actually tested after the initial analysis, using remote GPU runs with fixed seeds and explicit correctness checks.

### A. Correctness status (golden-reference validated)

`verify_correctness.py` was run on both available golden instances with the current local code:

- `P50M2-0`
- `P75M2-0`

And under both Phase-3 flag settings:

- `ABRKGA_PHASE3_CPP=1`
- `ABRKGA_PHASE3_CPP=0`

Result: **PASS in all cases** (exact makespan match and exact per-placement log match: `part_id, bin_idx, col, row, rot`).

Conclusion: current behavior is still aligned with golden reference for tested instances.

### B. Phase 3 rollback experiment (strict fair A/B)

Question tested: should we keep current Phase 3 or revert to the older Phase 3 logic from commit `5a70d85` ("Add golden reference creation and verification scripts")?

Method:

- Baseline: current pushed code (`c2c3827`)
- Variant: same code, but Phase 3 collector path reverted to old two-pass Python collection
- Same seeds: `123, 321, 777`
- Same command/flags in both runs:
  - `ABRKGA_PHASE3_CPP=1 ABRKGA_PHASE5_CPP=1 ABRKGA_PHASE56_CPP=1 ABRKGA_PHASE6_STRUCTURED=1 ABRKGA_PHASE6_GPU_FUSED=0 ABRKGA_PHASE6_CPU_CPP=0`
- Instance: `P75M2-0`, profiler: `profile_phases.py`, 5 generations

Aggregate results (mean across seeds):

| Metric | Baseline (`c2c3827`) | Phase3-reverted | Delta |
|---|---:|---:|---:|
| Mean gen time (s) | 5.281 | 5.174 | **-2.03%** |
| Total wave fn time (s) | 24.622 | 24.087 | **-2.17%** |
| Phase 3a time (s) | 3.016 | 2.303 | **-23.64%** |
| Phase 3b time (s) | 0.908 | 0.834 | **-8.19%** |
| Phase 5 time (s) | 4.257 | 4.414 | +3.69% |
| Phase 6 time (s) | 0.668 | 0.719 | +7.58% |

Conclusion:

- Reverting Phase 3 is currently the fastest tested variant overall.
- Phase 5/6 became slightly slower, but the Phase 3 gain is larger, so net runtime improves.

### B2. Phase 3 indirect slowdown investigation

Phase 3 code is **byte-for-byte identical** between current-best and 5a70d85. Yet current-best shows slower Phase 3 times (P75M2-0):

| Metric | Current-best | 5a70d85 | Delta |
|--------|---:|---:|---:|
| Phase 3a (ms/wave) | 4.943 | 4.590 | **+7.7%** |
| Phase 3b (ms/wave) | 1.797 | 1.657 | **+8.4%** |

Since Phase 3 is pure CPU with `torch.cuda.synchronize()` on both boundaries, this is a real code-level difference caused by indirect effects from other phase changes. Five hypotheses, ranked by likelihood:

**1. BinState object size (4 extra fields).**
Current-best's BinState has 20 fields vs 16 in 5a70d85 (`grid_materialized`, `pending_part_matrix`, `pending_y_start`, `pending_width`). Without `__slots__`, each instance uses a `__dict__` with a larger hash table (32-slot vs 24-slot). Phase 3 does ~9,000 attribute lookups per wave (`bin_state.area`, `.vacancy_vector`, `.enclosure_box_length`, `.grid_state_idx`). Per-lookup overhead is small (~1-2ns) but sums to ~10-20ms/gen.

**2. `_ensure_bin_grid_materialized` method call in Phase 5.**
Called per placement (always returns immediately with `GPU_FUSED=0`). Costs ~500ns per call. Charged to Phase 5 timer, but shifts Phase 5/6 timing which affects when Phase 3's BinState objects are created — potentially worsening memory locality for Phase 3's iteration.

**3. Phase 6 code path branching and GPU/CPU interleaving.**
Old Phase 6 was 3 lines calling `_start_new_bin` (individual GPU ops interleaved with CPU). Current Phase 6 is ~140 lines with multiple branches, batched GPU ops (`index_fill_` + `_cuda_batch_update`), then all CPU updates. This changes when BinState objects are finalized in memory, potentially affecting cache behavior when Phase 3 iterates over them in the next wave.

**4. Module-level C++ extension imports.**
Current-best imports 4 compiled `.so` modules at startup (`phase3_collector`, `phase5_selector`, `phase56_planner`, `phase6_cpu_update`). These consume process memory and may affect CPU instruction/data cache globally. The `phase3_collector` import is unused since Phase 3 was reverted.

**5. GC timing redistribution.**
Current Phase 5 returns numpy arrays from C++ functions; old Phase 5 used Python lists. Different allocation patterns shift when Python's garbage collector runs. GC cycles that previously fell in Phase 5/6 may now fall in Phase 3.

**Quick tests to quantify:**
- Add `__slots__` to BinState → eliminates `__dict__` overhead, tests hypothesis 1
- Remove the unused `from phase3_collector import ...` → tests hypothesis 4
- Both are low-risk, non-functional changes

### C. Phase 6 deep-native attempts (status)

Two higher-risk Phase 6 acceleration attempts were implemented/tested during this optimization cycle:

1. Fully fused GPU placement + vacancy metadata updates
2. Batched CPU-side bin/vacancy updates via C++ extension

Observed outcome: no consistent end-to-end win versus the structured baseline; in multiple runs these were neutral-to-worse.

Current practical recommendation:

- Keep `ABRKGA_PHASE6_STRUCTURED=1`
- Keep `ABRKGA_PHASE6_GPU_FUSED=0`
- Keep `ABRKGA_PHASE6_CPU_CPP=0`

### D. Flag overhead vs maintainability

Runtime overhead from checking env flags is negligible in practice (micro-benchmark on local machine: ~0.48 microseconds per `os.getenv(...).strip()` check). The real concern with flags is maintainability and testing matrix size, not raw speed.

Note: after the Phase 3 rollback, `ABRKGA_PHASE3_CPP` is effectively non-impacting in the main evaluator path unless Phase 3 native collection is reintroduced.

---

## Current Performance Profile

### P50M2-0 (500 individuals, 2 machines)

| Phase | % | ms/wave | Description |
|-------|---|---------|-------------|
| Phase 1 | 0.5% | 0.20 | Gather context info |
| Phase 2 | 7.0% | 2.55 | Batch grid FFTs |
| Phase 3a | 8.0% | 2.90 | Pass 1 vacancy + collect |
| **Phase 4a** | **41.5%** | **15.14** | **Pass 1 IFFT** |
| Phase 3b | 3.1% | 1.12 | Pass 2 vacancy + collect |
| Phase 4b | 10.3% | 3.77 | Pass 2 IFFT |
| Phase 5 | 19.6% | 7.17 | Best placements + grid updates |
| Phase 6 | 10.0% | 3.63 | Open new bins |

**Per-machine time:** M0 (250×250) = 1.04s, M1 (300×300) = 1.60s
**Tests/wave:** ~1255, **Waves:** ~355 across 5 gens

### P100M4-0 (1000 individuals, 4 machines)

**Gen time: 12.8s**
Per-machine: M0=1.67s, M1=3.09s, M2=3.49s, M3=4.29s (sum=12.5s)

Phase breakdown is similar — Phase 4 at ~51%, Phase 5 at ~19%.

### Key Scaling Factors

| Factor | P50M2 | P100M4 | Growth |
|--------|-------|--------|--------|
| Population | 500 | 1000 | 2× |
| Machines | 2 | 4 | 2× |
| Parts/machine | ~25 | ~25 | 1× |
| Waves/machine | ~35 | ~40 | ~1.1× |
| Gen time | 2.76s | 12.8s | **4.6×** |

The 4.6× growth from 2× population × 2× machines × larger grids (400×400 vs 300×300).

---

## Optimization Opportunities (Ranked by Expected Impact)

### 1. PARALLEL MACHINE PROCESSING — Expected: 2-3× speedup on multi-machine instances

**Priority: HIGHEST**
**Effort: Medium**
**Risk: Low**

**The Problem:**
`evaluate_batch()` processes machines **sequentially** in a `for machine_idx in range(self.nbMachines)` loop. For P100M4, this means 4 × ~3s = 12s, when the machines are completely independent — each machine's placement depends only on that machine's assigned parts.

**The Opportunity:**
Machine placements are independent. The only shared resource is VRAM. Two approaches:

**Approach A: CPU threading (overlap CPU work with GPU).**
Each machine needs its own `grid_states` and `grid_ffts` tensors. For P100M4, Machine 3 alone needs 33K × 400 × 400 × 4 bytes = ~21GB for grid_states — way too much to fit two machines simultaneously. So naively parallelizing on GPU won't work.

**Approach B: Interleaved pipeline.**
Process machines in a pipelined fashion. While the GPU runs Phase 4 IFFT for Machine 0, the CPU runs Phase 3 (vacancy collection) for Machine 1, and Phase 5 (placements) for the previous wave of Machine 0. This requires restructuring the wave loop but could overlap CPU and GPU across machines.

**Approach C: Reduce per-machine VRAM, then parallel.**
The `grid_states` tensor is pre-allocated for `max_bins_per_sol × num_solutions` bins. For P100M4-M3: 33 × 1000 = 33,000 grids of 400×400 = 21GB. But most solutions only use 2-5 bins. A **sparse allocation** scheme that only allocates GPU grids when bins are opened (not pre-allocated) would dramatically reduce VRAM, enabling parallel machines. This also fixes the P200M4 OOM.

**Approach D: Sequential machines, but evaluate only the non-elites.**
Currently `cal_fitness(offspring)` passes all 900 non-elites (mutants + offspring) at once. For P100M4 that's 900 solutions. Consider: do we need to evaluate all 900 on *every* machine? The final fitness is `max(makespan across machines)`. If we could identify early that a solution is already worse than the current best on Machine 0, we could skip Machine 1-3 entirely for that solution. This is a **pruning** strategy.

**Handling Different Machine Dimensions (e.g., 250×250 vs 400×400):**

Different grid sizes are **not an obstacle**. Each machine gets its own tensors on its own CUDA stream:

```
Time →
Machine 0:  [Phase3-CPU] [Phase4-GPU·stream0] [Phase5-CPU+GPU]  [Phase3] [Phase4] ...
Machine 1:              [Phase3-CPU]          [Phase4-GPU·stream1] [Phase5] ...
```

- While Machine 0's IFFT runs on the GPU (stream 0), the CPU is idle — use it for Machine 1's Phase 3 vacancy collection.
- CUDA streams handle different-sized kernels independently — stream 0 does `irfft2(250×250)` while stream 1 does `irfft2(400×400)`, no problem.
- Python's GIL releases during Numba JIT calls, PyTorch GPU ops, and NumPy operations — real concurrency happens.

**Practical implementation:**
1. Use `threading.Thread` or `concurrent.futures` with 2 workers
2. Each worker owns a CUDA stream (`torch.cuda.Stream()`)
3. Process machines in pairs: worker 0 handles M0 + M2, worker 1 handles M1 + M3
4. Different dimensions handled naturally since each worker has its own tensors

**Expected speedup for P100M4:** Instead of `1.67 + 3.09 + 3.49 + 4.29 = 12.54s`, you'd get roughly `max(1.67+3.49, 3.09+4.29) ≈ 7.38s` with 2-worker pipelining — a **1.7× speedup**. With better CPU/GPU interleaving, closer to **2×**.

**Main prerequisite:** Sparse grid allocation (idea #4) to fit multiple machines' tensors in VRAM simultaneously.

### 2. EVALUATE ONLY NON-ELITES PER MACHINE (Pruning) — Expected: 10-30% speedup

**Priority: HIGH**
**Effort: Low-Medium**
**Risk: Low**

**Observation:** After Machine 0, we know each solution's Machine-0 makespan. If `makespan_m0[i]` already exceeds the population's current best fitness, that solution cannot be the new best, so evaluating it on M1/M2/M3 won't change the population ranking. However, we still need its fitness for the next generation's partition step.

**Refinement:** We don't need exact fitness, just the ranking. After evaluating Machine 0, compute a lower bound on total makespan per solution (= Machine-0 makespan, since that's a lower bound on `max(all machines)`). Solutions that are clearly in the bottom 50% (non-elite) can be assigned a "lazy" fitness upper bound and skipped on remaining machines. This is speculative but could be valuable.

**Simpler version:** If a solution's makespan on Machine 0 alone exceeds the worst elite fitness, we know it won't become an elite. Assign it `makespan_m0` as a lower-bound fitness and skip remaining machines. This is **safe** — we're just assigning a value that's ≤ the true fitness (the true fitness is `max(all machines) ≥ makespan_m0`), so the solution will rank even worse than we estimate, which is fine since it wasn't going to be an elite anyway.

### 3. VECTORIZE _decode_sequences — Expected: 0.1-0.3s/gen

**Priority: MEDIUM**
**Effort: Low**
**Risk: None**

`_decode_sequences` iterates over every chromosome in a Python loop:
```python
for sol_idx in range(len(chromosomes)):
    chrom = chromosomes[sol_idx]
    SV, MV = chrom[:self.nbParts], chrom[self.nbParts:]
    mask = MV <= self.thresholds[0]
    ...
    sorted_sequence = actual_parts[np.argsort(values)]
```

This takes ~5ms per machine for P100M4 (1000 solutions), so ~20ms total for 4 machines — negligible. But as population grows, this Python loop will eventually dominate. The masking and sorting can be fully vectorized:

```python
# All solutions at once
SV = chromosomes[:, :nbParts]         # (N_sol, N_parts)
MV = chromosomes[:, nbParts:]         # (N_sol, N_parts)
mask = MV <= thresholds[0]            # (N_sol, N_parts) — vectorized
# Sort SV within each solution's masked subset
```

The sorting step is harder to vectorize because each solution has a different-sized subset. But `np.argsort` on the full SV array followed by masked selection would help.

### 4. SPARSE GRID ALLOCATION — Expected: enables larger instances + parallel machines

**Priority: HIGH**
**Effort: High**
**Risk: Medium (correctness)**

**The Problem:**
Grid states are pre-allocated as a single tensor `(max_total_bins, H, W)`. For P100M4:
- max_bins_per_sol = 33, num_solutions = 1000
- Machine 3 (400×400): 33 × 1000 = 33K grids, float32 → **21 GB** — OOM!
- Even with VRAM cap limiting to 5 bins/sol: 5 × 1000 = 5000 × 400 × 400 × 4 = **3.2 GB**

This is why P200M4 OOMs. It also prevents parallel machine processing.

**Solution:** Lazy allocation. Only allocate GPU grid slots when a bin is actually opened (Phase 6). Use a pool allocator:
- Maintain a free-list of GPU grid indices
- When a bin opens, pop from free-list
- Never need all `max_total_bins` simultaneously because most solutions use ≤5 bins

This reduces peak VRAM from `33K × H × W × 4` to `(avg_bins × num_solutions) × H × W × 4`, typically 5-10× less.

### 5. PHASE 5 TUPLE COMPARISON → COMPOSITE FLOAT64 SCORE — Expected: 0.1-0.3s/gen

**Priority: MEDIUM-LOW**
**Effort: Low**
**Risk: Must verify correctness**

The lexicographic tuple comparison in Phase 5 (`key = (-bin_idx, density, row, -col)`) creates Python tuples for every test and compares them. The profiler shows this takes 0.66ms/wave vs 0.49ms/wave for a composite float64 score — a 1.37× speedup.

**Key insight:** The reason we switched from composite to tuple was that `density * 1e6 + row * 1e3` could lose the hierarchy when density differences < `row_diff * 1e-3`. But with **float64** and careful multiplier choice:
- `bin_idx` ∈ [0, 30]: multiplier 1e12 → contribution up to 3e13
- `density` ∈ [0, 1]: multiplier 1e9 → contribution up to 1e9
- `row` ∈ [0, 400]: multiplier 1e3 → contribution up to 4e5
- `col` ∈ [0, 400]: multiplier 1 → contribution up to 400

For density to dominate row: need `density_diff * 1e9 > row_diff * 1e3`, i.e., `density_diff > row_diff * 1e-6`. With row_diff ≤ 400, we need `density_diff > 4e-4`.

**Is this always true?** Density = `(bin_area + part_area) / (new_length * W)`. Two rotations of the same part in the same bin: the only difference is `new_length`. If new_length differs by even 1 pixel (out of 250), density differs by `~(area) / (L² * W)`. For a bin with area 10000 and L=200, W=250: `10000 / (200 * 201 * 250) ≈ 1e-3`. So density differences are typically ≥ 1e-3 >> 4e-4.

However, edge cases exist where two rotations produce identical `new_length` but different rows. In that case, both density and row differ, but the density difference is 0 — so the formula correctly falls through to row comparison.

**Verdict:** A composite float64 score with multipliers `1e12, 1e9, 1e3, 1` would work correctly for all practical cases, and avoids Python tuple creation overhead. But the savings are only ~0.17ms/wave × 355 waves = ~60ms/gen for P50M2. Marginal.

**However**, for larger instances with more tests/wave, this could be more significant. And it's easy to implement with a correctness guard.

### 6. REDUCE PHASE 4 IFFT WORK — Expected: up to 30% reduction in Phase 4

**Priority: HIGH**
**Effort: Medium**
**Risk: Low**

Phase 4 (IFFT) is 52% of total time. Ideas to reduce it:

**6a. Dynamic FFT size based on grid occupancy.**
Currently, every IFFT is `irfft2(H × W)` even when most of the grid is empty. Early in the placement process, only the bottom rows are occupied. The FFT-based collision only needs to cover the region where either the grid OR the part can be. If we could detect that the top 150 rows of a 300-high grid are all zeros, we could use `irfft2(150 × W)` — 2× fewer elements, ~2× faster.

**Caveat:** All tests in a chunk must use the same FFT size (batched). Bins at different fill levels have different "effective heights". This would require grouping tests by grid fill level — adding Phase 3 complexity to save Phase 4 time. May not be worth it.

**6b. Skip IFFT for trivially-failing tests.**
After vacancy filtering, some tests still fail geometrically (no zero in the overlap matrix). Pre-filtering based on simpler checks (e.g., area fill ratio > threshold, or bounding-box overlap) could eliminate some tests before the expensive IFFT.

**6c. Increase CHUNK_SIZE.**
Currently 750. For P50M2 with 1255 tests/wave, that's 2 chunks. Increasing to 1500 would mean 1 chunk, eliminating one GPU sync point. Previously benchmarked as within noise (750 vs 1500), but worth retesting after the bug fixes since the test distribution has changed.

### 7. REDUCE PHASE 5 CPU WORK — Expected: 0.2-0.5s/gen

**Priority: MEDIUM**
**Effort: Medium**
**Risk: Low**

Phase 5 is 19.6% = ~0.54s/gen for P50M2. It includes:
1. Lexicographic comparison loop (0.66ms/wave)
2. CUDA kernel launch for GPU grid updates
3. CPU numpy grid updates + Numba vacancy updates

Items 2-3 should overlap (async GPU + CPU), but the `torch.cuda.synchronize()` in the profiler may distort this. The CPU work (numpy grid `+=` and Numba `update_vacancy_vector_rows`) is ~3-5ms/wave.

**Idea:** The numpy grid copy exists only for vacancy vector updates. Could we compute the vacancy vector differently? For example, maintain a running vacancy vector that's updated incrementally based on the part's shape and position, without needing the full grid? Each placement adds a rectangular region of 1s — the vacancy update only needs to recompute max-consecutive-zeros for the affected rows, which `update_vacancy_vector_rows` already does. The issue is it reads `bin_state.grid[y_start:y+1, :]` — the full row including the new part.

If we maintained the vacancy vector as "max consecutive zeros" and updated it analytically based on the placed part's position and the previous row contents, we could avoid the grid read. But this is complex — the placed part may split an existing run of zeros into two runs.

### 8. REDUCE PHASE 6 NEW BIN CREATION — Expected: 0.1-0.2s/gen

**Priority: LOW**
**Effort: Low**
**Risk: None**

Phase 6 (10%) creates new bins when parts don't fit. This involves:
- Python object creation (`BinState` dataclass)
- `np.zeros((H, W), dtype=np.uint8)` for the numpy grid
- GPU grid zeroing (`index_fill_`)
- First part placement (CUDA kernel + numpy + Numba)

The numpy grid allocation (`np.zeros(250×250)` = 62.5KB) is fast but happens per new bin. Pre-allocating a pool of numpy grids at the start of each machine batch would avoid repeated allocation.

### 9. NATIVE C/C++ EXTENSION FOR CPU-BOUND PHASES — Expected: 0.5-1.5s/gen

**Priority: MEDIUM-HIGH**
**Effort: High**
**Risk: Medium**

**Which phases benefit from native code?**

All phases with Python loops over contexts/bins/rotations benefit. Here's the full breakdown of CPU-bound work per wave:

| Phase | CPU Work | Cost/Wave | What Runs in Python |
|-------|----------|-----------|-------------------|
| **1** | Context filtering | ~1-2 ms | Loop over ~500 contexts, attribute lookups, list append |
| **2** | Collect invalid grids | ~2-3 ms | Nested loop: 500 contexts × 3-4 bins, attribute checks |
| **3a** | Pass 1 vacancy + collect | ~15 ms | 500 ctx × 3 bins × 2 rots = ~3000 iterations, Numba calls, 12× list appends |
| **3b** | Pass 2 vacancy + collect | ~12 ms | Same as 3a but fewer contexts |
| **5** | Tuple comparison + placement collection | ~7 ms | Tuple construction per test, validity checks, list appends |
| **6** | New bin creation | **~22 ms** | **np.zeros(H,W) per new bin, BinState dataclass creation, Numba vacancy updates** |
| **Total** | | **~60 ms/wave** | |

Over ~35 waves/machine × 2 machines = 70 waves/gen for P50M2: **~4.2s/gen of CPU overhead**.

**Phase 6 is surprisingly expensive (10% of total time)** because:
1. Each new bin allocates `np.zeros((H, W), dtype=uint8)` — 62-160 KB per bin depending on machine
2. ~50-100 new bins created per wave → 3-16 MB of allocation + zero-fill per wave
3. BinState dataclass creation with 11 field assignments per bin
4. Numba `update_vacancy_vector_rows` calls per new bin

**What a native extension would cover:**

A single C/C++ extension could replace the CPU work in Phases 1, 2, 3, 5, and 6 — essentially everything between GPU calls. The extension would:
- Replace Python attribute lookups with direct struct field access (~100ns → ~1ns)
- Replace Python list appends with C array writes
- Replace Python tuple comparison with C struct comparison
- Replace per-bin `np.zeros()` with a pre-allocated memory pool
- Inline the vacancy check (currently Numba) directly in C
- Return results as NumPy arrays for the GPU phases to consume

**Estimated savings:** If native code reduces CPU overhead by 60-80%, that's **0.5-1.5s/gen** depending on instance size. For P100M4 where waves/gen is higher, savings scale proportionally.

**Implementation approach — pybind11 (C++) recommended over Cython:**

| Criterion | Cython | C++ with pybind11 |
|-----------|--------|-------------------|
| **Runtime speed** | Equal — both compile to native code | Equal |
| **Build complexity** | Needs `.pyx` files + `setup.py`, separate build step | Header-only pybind11, `load_inline` or `setup.py` |
| **Debugging** | Hard — generated C is unreadable | Standard C++ debugging (gdb, sanitizers) |
| **IDE support** | Poor — `.pyx` syntax not well supported | Full C++ IDE support |
| **NumPy interop** | Good via typed memoryviews | Excellent via `py::array_t<>` with zero-copy |
| **Type safety** | Partial — still Python semantics in places | Full C++ type safety |
| **Maintenance** | Cython-specific syntax to learn | Standard C++ |
| **Existing pattern** | None in this project | **Already used**: `cuda_batch_update.py` uses `load_inline` |

**Recommendation: C++ with pybind11** because:
1. This project already uses `torch.utils.cpp_extension.load_inline` for the CUDA kernel — same build system
2. pybind11 gives zero-copy NumPy array access, which is critical for passing grid data
3. Standard C++ is easier to maintain and debug than Cython's hybrid syntax
4. Can later add CUDA kernels in the same extension if needed

**Note on ctypes:** ctypes has essentially zero overhead for the call itself (~50ns vs pybind11's ~200ns), but requires manual memory management and no automatic NumPy integration. For our use case where each call does significant work (milliseconds, not microseconds), the per-call overhead is irrelevant — pybind11's ergonomics win.

**Architecture sketch:**
```cpp
// wave_cpu_ext.cpp — compiled via load_inline (same pattern as cuda_batch_update.py)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

struct BinStateC {
    int grid_state_idx;
    float area;
    int enclosure_box_length;
    int* vacancy_vector;    // direct pointer to numpy array data
    uint8_t* grid;          // direct pointer to numpy grid
    // ...
};

// Phase 3: collect valid (ctx, bin, rotation) triples
py::dict collect_tests(/* context arrays, bin arrays, part arrays */) {
    // C loops over contexts × bins × rotations
    // Inline vacancy check (replaces Numba)
    // Write results directly to pre-allocated numpy arrays
}

// Phase 5: lexicographic best-per-context
py::dict find_best_placements(/* score arrays, ctx indices */) {
    // C struct comparison instead of Python tuples
}

// Phase 6: create new bins from pre-allocated pool
py::list create_new_bins(/* pool, contexts_needing_new_bin */) {
    // Reuse memory from pool instead of np.zeros() per bin
}
```

### 10. PROCESS MACHINES ON SEPARATE CUDA STREAMS — Expected: unclear, previously tested and rejected

**Priority: LOW**
**Effort: Medium**
**Risk: Previously showed 23-29% regression**

CUDA streams were previously tested and showed 23-29% slowdown due to GPU saturation. This was for within-machine parallelism. For cross-machine parallelism (Idea 1), streams might work differently — Machine 0's Phase 5 CPU work could overlap with Machine 1's Phase 4 GPU work on a separate stream. But this requires VRAM for two machines simultaneously.

---

## Summary: Recommended Implementation Order

| # | Idea | Expected Impact | Effort | Instance Benefit |
|---|------|----------------|--------|-----------------|
| 1 | Parallel machines (Approach C: sparse alloc) | **2-3× on P100M4** | High | All multi-machine |
| 2 | Early pruning across machines | 10-30% | Low-Med | Multi-machine |
| 3 | Vectorize _decode_sequences | Marginal now | Low | Large pop |
| 4 | Sparse grid allocation | Enables P200M4+ | High | Large instances |
| 5 | Composite float64 score | ~60ms/gen | Low | All |
| 6 | Reduce Phase 4 (chunk size, prefilters) | Up to 30% Phase 4 | Med | All |
| 7 | Reduce Phase 5 CPU work | 0.2-0.5s/gen | Med | All |
| 8 | Reduce Phase 6 allocation | ~0.1s/gen | Low | All |
| 9 | C++/pybind11 for all CPU phases (1,2,3,5,6) | 0.5-1.5s/gen | High | All (scales well) |
| 10 | CUDA streams cross-machine | Unclear | Med | Multi-machine |

### For P50M2 (your primary benchmark):
Focus on **6** (Phase 4 reduction) and **9** (C++/pybind11 hot loops). These are the highest-impact optimizations for 2-machine instances where parallel machines doesn't apply.

### For P100M4 and larger:
Focus on **1** (parallel machines) and **4** (sparse grid allocation). These are transformative for multi-machine instances — potentially cutting gen time from 12.8s to 4-5s by running machines concurrently.

### For very large instances (P200+, future):
**4** (sparse grid) is mandatory to even fit in VRAM. **1** (parallel machines) + **9** (C++/pybind11) become critical at this scale.

---

## Critical Bottleneck: The Sequential Machine Loop

The single biggest structural inefficiency is that machines are processed sequentially. For P100M4:

```
Machine 0: 1.67s  ████
Machine 1: 3.09s  ████████
Machine 2: 3.49s  █████████
Machine 3: 4.29s  ███████████
Total:    12.54s  ████████████████████████████████
```

If machines were processed in parallel (with GPU time-sharing or pipelining):
```
Machine 3: 4.29s  ███████████  (longest machine determines time)
Overhead:  ~1.5s  ████
Total:     ~5.8s  ███████████████
```

This is a **2.2× speedup** for P100M4 from parallelism alone. Combined with the other optimizations, 3-4× total improvement is realistic.

---

## Language Considerations

**Should you rewrite in C++/Rust?**

Not the full project, but targeted hot paths:

| Component | Current | Recommendation |
|-----------|---------|---------------|
| FFT/IFFT (Phase 4) | PyTorch (already C++/CUDA internally) | Keep as-is |
| CUDA grid updates (Phase 5) | Custom CUDA kernel | Keep as-is |
| Phase 1 context filtering | Python | **C++ pybind11 extension** |
| Phase 2 grid collection loop | Python | **C++ pybind11 extension** |
| Phase 3 collection loop | Python | **C++ pybind11 extension** |
| Phase 5 comparison loop | Python | **C++ pybind11 extension** |
| Phase 6 bin creation | Python + np.zeros | **C++ pybind11 + memory pool** |
| Vacancy checks | Numba JIT (already compiled) | Keep as-is |
| _decode_sequences | Python/NumPy | Vectorize with NumPy |
| BRKGA outer loop | Python/NumPy | Keep as-is (fast) |

The GPU-side code is already effectively C++/CUDA. The bottleneck is the **Python glue** between GPU calls — Phases 1, 2, 3, 5, and 6 all run Python loops with attribute lookups, list operations, and object creation. These are ideal for a C++ pybind11 extension because:
- The project already uses `load_inline` for CUDA (same build system)
- pybind11 gives zero-copy NumPy array access
- Standard C++ is easier to maintain than Cython's hybrid syntax
- A single extension can cover all CPU phases

The wave loop runs ~35 times per machine per generation, with ~60ms of CPU work per wave. A C++ extension reducing this by 60-80% would save **~1.5-3.4s across 70 waves/gen** for P50M2.

**Verdict:** A single C++ pybind11 extension covering all CPU phases gives the best effort-to-impact ratio. It follows the existing pattern (`cuda_batch_update.py`) and avoids introducing a second build system (Cython).
