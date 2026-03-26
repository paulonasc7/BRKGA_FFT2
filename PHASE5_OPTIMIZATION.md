# Phase 5 Optimization Analysis

## Current state

Phase 5 accounts for **27.2% of total wave time** (~6.2s across 5 generations, ~17ms/wave, ~365 waves).

It runs **after** Phase 4 has returned `placement_results` — a flat list of `(col, row)` or `None` for every (context, bin, rotation) test that was submitted to the batched IFFT.

Phase 5 has two distinct sub-parts:

### 5a. Selection loop (lines 272-325 in wave_batch_evaluator.py)

For each of ~500 contexts per wave, iterates over its test results (typically 2-12 per context), computes density, and applies first-fit + density + bottom-left tie-breaking to select the single best placement. Pure Python arithmetic and comparisons — no GPU, no Numba.

**Scale per wave:** ~500 contexts x ~4 tests/context = ~2000 iterations of the inner loop. Each iteration does a few arithmetic ops, list indexing, and comparisons. This is lightweight per-iteration, but adds up across 365 waves.

### 5b. `_place_part_in_bin` (lines 422-441 in wave_batch_evaluator.py)

Called once per context that found a valid placement (~400-490 of the 500 contexts per wave). Each call does:

1. **CPU grid update:** `bin_state.grid[y_start:y_end, x:x+shape[1]] += part_matrix` — NumPy slice-add on a uint8 array. Small region (part-sized), fast.
2. **GPU grid update:** `grid_states[bin_state.grid_state_idx, y_start:y_end, x:x+shape[1]] += part_gpu_tensor` — a CUDA kernel launch for a small 2D slice. **Each call is a separate CUDA kernel launch.**
3. **Numba vacancy update:** `update_vacancy_vector_rows(vacancy_vector, grid_rows, y_start)` — JIT-compiled, scans modified rows for max consecutive zeros. Fast per call, but called ~450 times per wave.
4. **Scalar bookkeeping:** area, min/max row, enclosure box, proc_time updates.

**The critical bottleneck in 5b is the sequential CUDA kernel launches.** With ~450 placements per wave and ~365 waves, that's ~164,000 individual CUDA kernel launches for tiny slice updates. Each launch has fixed overhead (~5-10 us on A4000), so this alone accounts for ~1-2s.

---

## Optimization ideas

### Idea 1: Move the selection loop (5a) to GPU

**Status:** Recommended as first implementation.

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

### Idea 2: Batch the GPU grid updates (5b)

**Status:** Recommended as second implementation.

**What:** Instead of calling `grid_states[idx, y:y+h, x:x+w] += tensor` ~450 times per wave (450 separate CUDA kernel launches), restructure `_place_part_in_bin` to separate GPU and CPU work, allowing better CUDA pipelining.

**How (Option C — recommended, lowest risk):**

Currently, for each context that found a placement, we call `_place_part_in_bin` which interleaves:
1. CPU grid update
2. GPU grid update (CUDA kernel launch)
3. Numba vacancy update
4. Scalar bookkeeping

The CPU work between GPU launches forces the CUDA command queue to stall — each launch must wait for the Python interpreter to finish the CPU work before issuing the next one.

**Restructure into two passes:**

```python
# Pass 1: Collect all placement decisions and do ALL GPU updates back-to-back
gpu_updates = []  # list of (grid_state_idx, y_start, y_end, x, x_end, part_gpu_tensor)
cpu_updates = []  # list of (bin_state, y_start, y_end, x, shape, part_matrix_uint8, area, mpd)

for ctx_idx, (ctx, part_data, mach_part_data) in enumerate(context_info):
    # ... existing selection logic to find best_result ...
    if best_result is not None:
        bs, x, y, rot, shape, pd, mpd = best_result
        y_start = y - shape[0] + 1
        y_end = y + 1
        # Queue GPU update
        gpu_updates.append((bs.grid_state_idx, y_start, y_end, x, x + shape[1], pd.rotations_gpu[rot]))
        # Queue CPU update
        cpu_updates.append((bs, y_start, y_end, x, shape, pd.rotations_uint8[rot], pd.area, mpd))
        bs.grid_fft_valid = False
        ctx.current_part_idx += 1
    else:
        contexts_needing_new_bin.append(...)

# Pass 2a: All GPU updates in a tight loop (CUDA can pipeline these)
for grid_idx, y_start, y_end, x, x_end, part_gpu in gpu_updates:
    grid_states[grid_idx, y_start:y_end, x:x_end] += part_gpu

# Pass 2b: All CPU updates (NumPy grid, vacancy, bookkeeping)
for bs, y_start, y_end, x, shape, part_uint8, area, mpd in cpu_updates:
    bs.grid[y_start:y_end, x:x+shape[1]] += part_uint8
    update_vacancy_vector_rows(bs.vacancy_vector, bs.grid[y_start:y_end, :], y_start)
    bs.area += area
    bs.min_occupied_row = min(bs.min_occupied_row, y_start)
    bs.max_occupied_row = max(bs.max_occupied_row, y_end - 1)
    bs.enclosure_box_length = bs.bin_length - bs.min_occupied_row
    bs.proc_time += mpd.proc_time
    bs.proc_time_height = max(bs.proc_time_height, mpd.proc_time_height)
```

**Why this helps:** Pass 2a issues ~450 CUDA kernel launches with no Python computation between them. The CUDA driver can queue them all up and execute them with minimal stall. Pass 2b runs on CPU and can overlap with GPU execution of the queued kernels.

**Correctness note:** This is safe because within a single wave, no two contexts share a bin — each context places one part into one of *its own* bins. So the GPU grid updates are all to non-overlapping regions, and the order doesn't matter.

**Expected impact:** Moderate. Savings: **0.5-1.5s over 5 generations** (rough estimate). The benefit comes from CUDA pipelining — instead of launch-wait-CPU-launch-wait-CPU, it's launch-launch-launch-...-then-CPU.

**Risk:** Low. No correctness risk — just reordering operations that are already independent. The only constraint is that `grid_fft_valid` must be set to False before the next wave's Phase 2, which already happens.

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

## Recommended implementation order

1. **Idea 1 (GPU-side selection)** — highest impact on Phase 5 Python time. Eliminates the per-context selection loop entirely.

2. **Idea 2 (defer GPU updates)** — low risk, addresses CUDA launch overhead in `_place_part_in_bin`. Independent of Idea 1.

3. Reassess after implementing 1 and 2. If Phase 5 is still significant, revisit Ideas 3 and 4.
