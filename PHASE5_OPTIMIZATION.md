# Phase 5 Optimization Analysis

## Current state

Phase 5 accounts for **27.2% of total wave time** (~6.2s across 5 generations, ~17ms/wave, ~365 waves).

It runs **after** Phase 4 has returned `placement_results` — a flat list of `(col, row)` or `None` for every (context, bin, rotation) test that was submitted to the batched IFFT.

Phase 5 has two distinct sub-parts:

### 5a. Selection loop (lines 272-325)

For each of ~500 contexts per wave, iterates over its test results (typically 2-12 per context), computes density, and applies first-fit + density + bottom-left tie-breaking to select the single best placement. Pure Python arithmetic and comparisons — no GPU, no Numba.

**Scale per wave:** ~500 contexts x ~4 tests/context = ~2000 iterations of the inner loop. Each iteration does a few arithmetic ops, list indexing, and comparisons. This is lightweight per-iteration, but adds up across 365 waves.

### 5b. `_place_part_in_bin` (lines 422-441)

Called once per context that found a valid placement (~400-490 of the 500 contexts per wave). Each call does:

1. **CPU grid update:** `bin_state.grid[y_start:y_end, x:x+shape[1]] += part_matrix` — NumPy slice-add on a uint8 array. Small region (part-sized), fast.
2. **GPU grid update:** `grid_states[bin_state.grid_state_idx, y_start:y_end, x:x+shape[1]] += part_gpu_tensor` — a CUDA kernel launch for a small 2D slice. **Each call is a separate CUDA kernel launch.**
3. **Numba vacancy update:** `update_vacancy_vector_rows(vacancy_vector, grid_rows, y_start)` — JIT-compiled, scans modified rows for max consecutive zeros. Fast per call, but called ~450 times per wave.
4. **Scalar bookkeeping:** area, min/max row, enclosure box, proc_time updates.

**The critical bottleneck in 5b is the sequential CUDA kernel launches.** With ~450 placements per wave and ~365 waves, that's ~164,000 individual CUDA kernel launches for tiny slice updates. Each launch has fixed overhead (~5-10 us on A4000), so this alone accounts for ~1-2s.

---

## Optimization ideas

### Idea 1: Batch the GPU grid updates (5b)

**What:** Instead of calling `grid_states[idx, y:y+h, x:x+w] += tensor` 450 times per wave (450 CUDA kernel launches), collect all placement coordinates first, then apply them in a single batched operation.

**How:**

Phase 5a runs as-is (pure Python selection). But instead of calling `_place_part_in_bin` immediately for each context, collect the placement decisions into arrays:

```python
# After selection loop, we have lists:
place_grid_idxs = []    # which grid_state to update
place_y_starts  = []    # y start
place_x_starts  = []    # x start
place_rotations = []    # which rotation was chosen
place_part_ids  = []    # which part
place_bin_states = []   # BinState objects (for CPU-side updates)
```

Then do the CPU-side updates (NumPy grid, vacancy, bookkeeping) in a loop (these are unavoidably sequential because they mutate shared state). But batch ALL the GPU slice updates into a single custom CUDA operation.

**Challenge:** The GPU updates are scatter-adds into different regions of different grids with different-sized parts. There's no built-in PyTorch operation for this. Options:

- **Option A: Write a custom CUDA kernel** using `torch.utils.cpp_extension` that takes all placements and applies them in one kernel. Maximum performance, significant implementation effort.
- **Option B: Use `torch.scatter_add_` on a flattened representation.** Possible but awkward — need to convert 2D slice-adds into flat index scatter operations.
- **Option C: Accept the sequential launches but defer them.** Collect all placements, then do all GPU updates in a tight loop with no Python overhead between them (no vacancy, no bookkeeping — just the GPU ops). The CUDA command queue can then pipeline them more efficiently. Then do the CPU updates in a second pass. This is simpler and may capture most of the benefit.

**Expected impact:** Moderate. The 450 kernel launches per wave cost ~1-2s total. Option C could recover some of this by allowing GPU pipelining; Option A could eliminate it but is complex. Savings: **0.5-1.5s over 5 generations** (rough estimate).

**Risk:** Low for Option C (no correctness risk, just reordering). Medium for A/B (custom CUDA code).

---

### Idea 2: Move the selection loop (5a) to GPU

**What:** Instead of iterating over `placement_results` in Python to find the best placement per context, compute the best placement per context entirely on the GPU as part of Phase 4.

**How:**

Phase 4 already has all the information needed: for each test, it computes `(col, row)` or determines infeasibility. The tie-breaking rule (first-fit by `bin_idx`, then density, then bottom-row, then left-col) can be encoded as a sorting key:

```python
# For each test, compute a score:
#   primary:   -bin_idx        (lower bin wins)
#   secondary:  density        (higher density wins)
#   tertiary:   row            (higher row wins)
#   quaternary: -col           (lower col wins)
# Encode into a single float: score = -bin_idx * 1e12 + density * 1e6 + row * 1e3 - col
```

But there's a subtlety: **density depends on `bin_state.enclosure_box_length` and `bin_state.area`**, which are CPU-side state per bin. These would need to be uploaded to the GPU per wave.

**Detailed approach:**

1. In Phase 3, also collect `bin_state.enclosure_box_length` and `bin_state.area` per test (into arrays like `test_enclosure_lengths`, `test_bin_areas`).
2. Upload these to GPU as tensors alongside the other test metadata.
3. In Phase 4, after computing `(col, row)` per test, compute density on the GPU:
   ```python
   y_start = best_row - part_heights + 1
   new_length = torch.max(test_enclosure_lengths, H - y_start)
   density = (test_bin_areas + test_part_areas) / (new_length * W)
   ```
4. Compute a composite score per test:
   ```python
   score = (-test_bin_indices.float() * 1e12
            + density * 1e6
            + best_row.float() * 1e3
            - smallest_cols.float())
   ```
5. Group tests by context (using a `ctx_idx` array). Use `scatter_max` or segment-reduce to find the best test per context.
6. Return only the winning test index per context to CPU.

**Expected impact:** High. This would eliminate the entire Python selection loop (~500 contexts x ~4 tests x 365 waves = ~730K Python iterations). It would also reduce the CPU<->GPU transfer from `n_tests x 3` values to `n_contexts x ~5` values. Savings: **1-3s over 5 generations** (rough estimate).

**Risk:** Medium. The composite-score approach needs care with floating point precision to avoid tie-breaking errors. The `scatter_max` for per-context reduction is well-supported in PyTorch (`scatter_reduce` with "amax"). The density calculation on GPU needs `enclosure_box_length` and `area` data uploaded per wave — a small overhead.

**Note on data flow:** Currently Phase 4 returns `all_results` as a Python list of `(col, row)` or `None`. This idea would restructure Phase 4 to also do the selection, returning `(best_test_idx_per_context, ...)` instead. The boundary between Phase 4 and Phase 5 would shift.

---

### Idea 3: Batch the CPU-side updates in `_place_part_in_bin` (5b)

**What:** The CPU updates (NumPy grid, vacancy vector, bookkeeping) are inherently sequential per bin because later placements in the same bin depend on earlier ones (vacancy changes, area changes). But across different contexts, they're independent. Use Numba `prange` or similar to parallelize across contexts.

**How:**

After the selection loop produces all placement decisions, group them by solution (they can't share bins across solutions). Apply all placements for different solutions in parallel using threading.

**Expected impact:** Low to moderate. The CPU updates (NumPy slice-add on ~50x30 region + Numba vacancy scan on ~5 rows x 200 cols) are individually fast (~5-10 us each). With ~450 placements/wave, that's ~2-5ms of CPU work per wave, or ~0.7-1.8s over 365 waves. Parallelizing over 4-8 threads could save ~0.5-1.5s.

**Risk:** Medium. Thread safety with Numba + NumPy needs care. Different contexts may share the same `grid_states` GPU tensor (though at non-overlapping indices).

---

### Idea 4: Overlap Phase 5 CPU work with Phase 4 GPU work of the next chunk

**What:** Phase 4 processes tests in chunks of 750. While one chunk's GPU IFFT is running, process Phase 5 results from the *previous* chunk on CPU.

**How:**

Currently Phase 4 collects all results, then Phase 5 processes them. Instead, interleave:

```
Chunk 0: launch GPU IFFT
Chunk 0: wait for GPU, read results
Chunk 1: launch GPU IFFT
         while GPU runs Chunk 1: process Chunk 0 results (selection + placement for those contexts)
Chunk 1: wait for GPU, read results
Chunk 2: launch GPU IFFT
         while GPU runs Chunk 2: process Chunk 1 results
...
```

**Challenge:** This is architecturally complex. The chunks contain tests from ALL contexts interleaved, so you can't fully resolve a context's best placement until all chunks are processed (a test in chunk 2 might be better than the best from chunk 0). You'd need a partial-best tracking scheme.

**Expected impact:** Low to moderate. The benefit only materializes if Phase 5 CPU work and Phase 4 GPU work are roughly balanced per chunk, and the interleaving overhead doesn't negate the overlap. Given Phase 4 is ~55% GPU-bound and Phase 5 is ~27% CPU-bound, there's theoretical room for overlap.

**Risk:** High complexity. The partial-result tracking and the fact that contexts span multiple chunks makes this very tricky.

**Verdict:** Not recommended — complexity far outweighs likely benefit.

---

### Idea 5: Early-exit in Phase 4 when bin 0 has a valid result

**What:** Phase 4 currently computes IFFT for ALL tests (all bins, all rotations). But first-fit means: if bin 0, any rotation, has a valid placement, we don't care about bins 1, 2, etc. for that context. Skipping those IFFTs saves GPU compute.

**How:**

This requires changing the Phase 4 processing order: process bin 0 tests first for all contexts, check which contexts found placements, then only submit bin 1+ tests for contexts that had no bin 0 result.

**Challenge:** Requires restructuring Phase 3/4 to process bins in layers rather than all at once. Each layer needs its own batch IFFT call. If most contexts find placement in bin 0 (likely, since bins are opened on demand), the second-layer batch is much smaller.

**Expected impact:** Potentially high. If 80% of contexts succeed on bin 0, this eliminates ~60-70% of Phase 4 tests for bins 1+. But it adds extra kernel launches (one per bin layer instead of one total). Net effect depends on the distribution:
- If most contexts test 1 bin: big win (fewer total IFFTs)
- If most contexts test 3+ bins: modest win

**Risk:** Medium. Restructuring Phase 3/4 is non-trivial but conceptually clean. Need to measure the actual distribution of how many bins are tested per context.

**Data needed before implementing:** Count how many tests are for bin 0 vs bin 1 vs bin 2+ per wave on average.

---

## Recommendations

| Priority | Idea | Expected savings | Complexity | Risk |
|----------|------|-----------------|------------|------|
| 1 | **Idea 2: GPU-side selection** | 1-3s | Medium | Medium |
| 2 | **Idea 5: Early-exit by bin layer** | 1-4s (needs data) | Medium | Medium |
| 3 | **Idea 1C: Defer GPU updates** | 0.5-1.5s | Low | Low |
| 4 | **Idea 3: Parallel CPU updates** | 0.5-1.5s | Medium | Medium |
| — | Idea 4: Overlap chunks | 0.5-1s | High | High |

**Recommended order of implementation:**

1. First, **gather data** for Idea 5: instrument Phase 3 to count how many tests belong to bin 0, bin 1, bin 2+ per wave. This informs whether the early-exit approach is worthwhile (it could also reduce Phase 4 time, not just Phase 5).

2. **Idea 2 (GPU selection)** is the highest-impact single change for Phase 5 pure Python time. It eliminates the per-context Python iteration entirely.

3. **Idea 1C (defer GPU updates)** is the lowest-risk change and addresses the CUDA launch overhead in `_place_part_in_bin`. Can be done independently of Ideas 2 and 5.
