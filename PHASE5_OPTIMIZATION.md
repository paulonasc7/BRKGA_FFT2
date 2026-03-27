# Phase 5 Optimization Analysis

**Last updated:** 2026-03-26

## Current state (after optimizations)

Phase 5 now accounts for **14.0% of total wave time** (~2.63s across 5 generations, ~7.3ms/wave, 360 waves). Down from 27.2% before optimizations.

Both Idea 1 (NumPy scoring) and Idea 2 (CUDA batch kernel) have been implemented. The remaining Phase 5 time is dominated by CPU updates (NumPy grid writes + Numba vacancy), not GPU launches.

### What Phase 5 does now (post-optimization)

1. **Best-per-context selection** — Uses pre-computed `all_scores` (numpy composite scores from `_batch_fft_all_tests`) to find the best test per context via simple numpy operations. No per-test Python density loop.
2. **GPU grid updates** — Single custom CUDA kernel launch via `cuda_batch_update.py` processes all ~450 placements per wave in parallel.
3. **CPU grid updates** — NumPy grid writes + Numba vacancy updates (overlaps with async GPU execution).

### Original analysis (pre-optimization, for reference)

Phase 5 had two distinct sub-parts:

#### 5a. Selection loop (eliminated by Idea 1)

For each of ~500 contexts per wave, iterated over test results (typically 2-12 per context), computed density, and applied tie-breaking. ~2000 Python iterations per wave with density arithmetic.

#### 5b. `_place_part_in_bin` (restructured by Idea 2)

Each call was a separate CUDA kernel launch. With ~450 placements per wave and ~365 waves, that was ~164,000 individual CUDA kernel launches for tiny slice updates.

---

## Optimization ideas

### Idea 1: Move scoring out of Python loop → NumPy composite scores

**Status:** IMPLEMENTED (2026-03-25). Saved ~0.30s/gen (4.53s → 4.23s).

**What:** Instead of iterating over `placement_results` in Python to find the best placement per context, compute the best placement per context entirely on the GPU as part of Phase 4.

**How:**

Phase 4 already has all the information needed: for each test, it computes `(col, row)` or determines infeasibility. The tie-breaking rule (first-fit by `bin_idx`, then density, then bottom-row, then left-col) can be encoded as a composite scoring key.

**Detailed implementation plan:**

#### Step 1: Collect extra metadata in Phase 3

In Phase 3, alongside the existing parallel arrays, also collect per-test:

```python
test_enclosure_lengths = []  # bin_state.enclosure_box_length (int)
test_bin_areas         = []  # bin_state.area (float)
test_part_areas        = []  # part_data.area (float)
test_ctx_indices       = []  # ctx_idx (int) — which context owns this test
```

These are appended in the same inner loop where `test_grid_indices`, `test_part_ffts`, etc. are already appended. Trivial change.

#### Step 2: Upload extra metadata to GPU in `_batch_fft_all_tests`

In the pre-loop section of `_batch_fft_all_tests` (where `all_grid_indices`, `all_heights`, `all_widths` are already built), also build:

```python
all_enclosure_lengths = torch.tensor(test_enclosure_lengths, device=self.device, dtype=torch.float32)
all_bin_areas         = torch.tensor(test_bin_areas,         device=self.device, dtype=torch.float32)
all_part_areas        = torch.tensor(test_part_areas,        device=self.device, dtype=torch.float32)
all_bin_indices       = torch.tensor(test_bin_indices,       device=self.device, dtype=torch.long)
all_ctx_indices       = torch.tensor(test_ctx_indices,       device=self.device, dtype=torch.long)
```

#### Step 3: Compute density and composite score on GPU (inside chunk loop)

After computing `has_valid`, `smallest_cols`, `best_row` for a chunk (the existing Phase 4 logic), add:

```python
# Compute density for valid results
chunk_enclosure = all_enclosure_lengths[chunk_start:chunk_end]
chunk_bin_areas = all_bin_areas[chunk_start:chunk_end]
chunk_part_areas = all_part_areas[chunk_start:chunk_end]
chunk_bin_idx = all_bin_indices[chunk_start:chunk_end]

y_start = best_row - part_heights + 1
new_length = torch.maximum(chunk_enclosure, (H - y_start).float())
density = (chunk_bin_areas + chunk_part_areas) / (new_length * W)

# Composite score encoding the tie-breaking hierarchy:
#   1. Lower bin_idx wins  (most significant)
#   2. Higher density wins
#   3. Higher row wins
#   4. Lower col wins     (least significant)
#
# Use large multipliers to keep the hierarchy strict.
# bin_idx is small (0-10), density is 0-1, row is 0-H (~300), col is 0-W (~200).
score = (
    -chunk_bin_idx.float() * 1e9
    + density * 1e6
    + best_row.float() * 1e3
    - smallest_cols.float()
)

# Invalidate tests that had no valid placement
score = torch.where(has_valid, score, torch.tensor(-float('inf'), device=self.device))
```

#### Step 4: Reduce per-context across all chunks

After all chunks are processed, we have per-test scores on GPU. Use `scatter_reduce` to find the best test index per context:

```python
n_contexts = len(context_info)

# Concatenate per-chunk scores into a single tensor (or accumulate across chunks)
# Then find best test per context:
best_score_per_ctx = torch.full((n_contexts,), -float('inf'), device=self.device)
best_test_per_ctx  = torch.full((n_contexts,), -1, dtype=torch.long, device=self.device)

# scatter_reduce to find max score per context
# PyTorch >= 2.0: scatter_reduce with "amax"
best_score_per_ctx.scatter_reduce_(0, all_ctx_indices, all_scores, reduce="amax")

# To get the actual test index (not just the max score), use:
# For each context, find which test achieved the max score
mask = (all_scores == best_score_per_ctx[all_ctx_indices]) & (all_scores > -float('inf'))
# Use argmax per context — or simpler: scatter with the test index where score equals best
test_indices_arange = torch.arange(n_tests, device=self.device)
best_test_per_ctx.scatter_reduce_(0, all_ctx_indices, test_indices_arange, reduce="amin",
                                   include_self=False)
# (amin of indices where score == best gives first-occurring best test)
```

**Alternative simpler approach for Step 4:** Instead of `scatter_reduce`, since each context has only ~4 tests on average, it may be simpler to:
- Transfer `all_scores` (n_tests floats) to CPU in one `.cpu().numpy()` call
- Do the per-context argmax in Python/NumPy (still fast — just ~2000 comparisons per wave, but now on pre-computed scores rather than doing density math)

This hybrid approach avoids the `scatter_reduce` complexity while still moving the expensive density computation to GPU.

#### Step 5: Update Phase 5 to use pre-computed winners

Phase 5 becomes much simpler — instead of the inner loop over test_indices with density computation, it just reads the winning test index per context and calls `_place_part_in_bin`.

#### What changes in the function signature

`_batch_fft_all_tests` currently returns `List[Optional[Tuple[int,int]]]` — a list of `(col, row)` or `None` per test. After this change, it should additionally return:
- `all_scores` (numpy array, n_tests) — composite scores per test
- Or: `best_test_per_ctx` (numpy array, n_contexts) — the winning test index per context

The latter is cleaner but requires passing `test_ctx_indices` into the function.

**Expected impact:** High. Eliminates the entire per-test Python iteration in Phase 5a (~500 contexts x ~4 tests x 365 waves = ~730K Python iterations with density arithmetic). Also reduces CPU<->GPU transfer: currently transfers `n_tests x 3` int values to CPU; after this change, the density/score computation stays on GPU and only the winning index per context needs to come back.

Savings estimate: **1-3s over 5 generations**.

**Risk:** Medium. The composite-score approach needs care with floating point precision — the multipliers (1e9, 1e6, 1e3) must keep the hierarchy strict for the actual value ranges. Since bin_idx < 10, density in [0,1], row < 300, col < 200, the chosen multipliers provide wide separation. The `scatter_reduce` path needs PyTorch >= 2.0; the hybrid path (GPU scores, CPU argmax) works everywhere.

---

### Idea 2: Batch the GPU grid updates (5b) → Custom CUDA kernel

**Status:** IMPLEMENTED (2026-03-26). Saved ~0.43s/gen (4.23s → 3.80s). Phase 5 dropped from ~22.7% → 14.0%.

**What was implemented:** Option A (custom CUDA kernel) instead of the originally-recommended Option C (tight GPU loop). A single CUDA kernel launch processes all ~450 placements per wave in parallel, using 1D thread indexing with binary search over prefix-sum offsets.

**Key files:**
- `cuda_batch_update.py` — CUDA kernel source, JIT compilation via `load_inline`, Python wrapper
- `wave_batch_evaluator.py` — `flat_parts_gpu` pre-computation in `__init__`, two-pass Phase 5 restructuring

**Implementation details:**
- All part rotation matrices pre-concatenated into `flat_parts_gpu` (single GPU tensor) at init time
- `part_update_meta` dict maps `(part_id, rot) → (flat_offset, h, w)` for indexing
- Falls back to sequential GPU slice updates if CUDA kernel compilation fails
- First compilation takes 2-3 min; cached `.so` used for subsequent runs

**Bugs encountered during implementation:**
1. GPU pointer dereference from host C++ code → segfault. Fixed by passing `total_cells` as explicit int parameter.
2. `load_inline(name='cuda_batch_update')` overrode `sys.modules['cuda_batch_update']` → TypeError when importing Python wrapper. Fixed by using `name='_cuda_batch_update_ext'`.

**Option C (tight GPU loop) was also implemented as fallback** — used when `flat_parts_gpu` is None.

---

### Idea 3: Early-exit in Phase 4 by bin layer

**Status:** Deferred — might be worth trying later.

**What:** Phase 4 currently computes IFFT for ALL tests (all bins, all rotations) in one batch. But first-fit means: if bin 0 has a valid placement for a context, bins 1, 2, etc. are irrelevant. This idea processes bins in layers — bin 0 first, then bin 1 only for contexts that failed bin 0, etc.

**How:** Restructure Phase 3/4 to submit bin 0 tests first, get results, identify which contexts still need testing, then submit bin 1 tests only for those. Each layer is a separate batch IFFT call.

**Why deferred:** The benefit is instance-dependent. Instances where bins fill quickly (large/irregular parts) would see contexts frequently needing bin 1+, yielding little savings while still paying extra kernel launch overhead per layer. The current all-at-once approach is instance-agnostic.

**When to revisit:** If profiling on the user's target instances shows that a large majority of tests (>70%) are for bin 1+ and most bin 0 tests succeed, this becomes worthwhile. Could also be exposed as a toggle.

---

### Idea 4: Parallel CPU updates across contexts

**Status:** Deferred — lower priority.

**What:** The CPU updates in `_place_part_in_bin` (NumPy grid, vacancy vector, bookkeeping) are independent across different contexts (each context owns its own bins). Parallelize them using threading/Numba `prange`.

**Why deferred:** The per-placement CPU work is individually very fast (~5-10 us). Even with ~450 placements/wave over 365 waves, the total CPU update time is estimated at ~0.7-1.8s. Threading overhead and GIL contention with NumPy/Numba would reduce the effective speedup. Better to pursue Ideas 1 and 2 first, then reassess whether CPU updates are still a bottleneck.

---

### Idea 5: Overlap Phase 5 CPU with Phase 4 GPU (chunk interleaving)

**Status:** Not recommended.

**What:** While one Phase 4 chunk's GPU IFFT is running, process Phase 5 results from the previous chunk on CPU.

**Why rejected:** Architecturally complex. Chunks contain tests from ALL contexts interleaved, so a context's best placement can't be resolved until all chunks are processed. Partial-best tracking adds significant complexity for modest benefit. The Phase 4 chunks are also not independently useful for Phase 5 decisions.

---

## Implementation status

1. **Idea 1 (NumPy scoring)** — DONE. Eliminated per-test Python density loop. Saved ~0.30s/gen.
2. **Idea 2 (CUDA batch kernel)** — DONE. Eliminated sequential CUDA launches. Saved ~0.43s/gen.
3. **Ideas 3, 4, 5** — Deferred. Phase 5 is now 14% of total time. The dominant bottleneck is now Phase 4 (66%), so further Phase 5 optimization has diminishing returns. Future optimization effort should target Phase 4 (chunk strategy, VRAM management) or Phase 3 (vacancy checks).
