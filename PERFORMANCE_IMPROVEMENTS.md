# Performance Improvements: CPU ↔ GPU Transition Elimination

## Profiling Context

Profiled with: `python BRKGA_alg3.py 50 2 0 torch_gpu wave_batch 1 1 3`
Hardware: NVIDIA RTX A4000 (16 GB), CUDA
Benchmark: 500 individuals, 3 generations

| Mode | Time/gen | Total (3 gen) |
|------|----------|---------------|
| serial | 18.2s | 54.7s |
| wave_batch | 10.0s | 29.9s |

**Current speedup: 1.9x.** The FFT computation itself is only **3.5%** of total `wave_batch` time. The dominant cost is CPU↔GPU data transfer and Python overhead.

### Where time is spent in `wave_batch` (500 solutions, single batch eval = 9.1s):

| Component | Time | % | Calls |
|-----------|------|---|-------|
| `torch.tensor()` (GPU sync stalls) | 3.07s | 33.7% | 1,296 |
| `torch.as_tensor()` (CPU→GPU) | 1.46s | 16.0% | 21,798 |
| `_process_wave_true_batch` Python overhead | 1.25s | 13.6% | 72 |
| `_place_part_in_bin` logic overhead | 1.11s | 12.1% | 21,798 |
| `update_vacancy_vector_rows` (Numba) | 0.44s | 4.9% | 21,798 |
| `.cpu()` transfer | 0.30s | 3.3% | 307 |
| `torch.fft.ifft2` | 0.22s | 2.5% | 307 |
| `torch.fft.fft2` | 0.09s | 1.0% | 68 |

---

## IMP-1: Use pre-computed GPU tensors in `_place_part_in_bin`

**File:** `wave_batch_evaluator.py`, lines 397-414
**Also affects:** lines 314 and 436 (the two call sites)

### Issue

Every part placement calls `torch.as_tensor(part_matrix, dtype=torch.float32, device=self.device)` to convert a numpy array to a GPU tensor. This is called **21,798 times** per batch evaluation (500 solutions) and accounts for **~1.5s (16%)** of total time.

However, `PartData.rotations_gpu` already stores **pre-computed GPU tensors** for every rotation of every part. These are created once at startup in `BRKGA_alg3.py` line 344 via `collision_backend.prepare_rotation_tensor()`. The wave_batch evaluator simply never uses them.

### Current code

```python
# Line 314: caller passes numpy array
self._place_part_in_bin(bin_state, x, y, pd.rotations[rot],
                        shape, pd.area, mpd, grid_states)

# Line 436: caller passes numpy array
self._place_part_in_bin(new_bin, 0, ctx.bin_length - 1, part_data.rotations[best_rot],
                        shape, part_data.area, mach_part_data, grid_states)

# Line 403: method converts numpy→GPU every time
part_tensor = torch.as_tensor(part_matrix, dtype=torch.float32, device=self.device)
grid_states[bin_state.grid_state_idx, y_start:y_end, x:x+shape[1]] += part_tensor
```

### Suggested change

Pass the GPU tensor directly alongside the numpy matrix (needed for the CPU grid update):

```python
# Updated signature:
def _place_part_in_bin(self, bin_state, x, y, part_matrix, shape, area,
                       mach_part_data, grid_states, part_gpu_tensor=None):
    y_start = y - shape[0] + 1
    y_end = y + 1

    bin_state.grid[y_start:y_end, x:x+shape[1]] += part_matrix.astype(np.uint8)

    # Use pre-computed GPU tensor instead of converting every time
    if part_gpu_tensor is None:
        part_gpu_tensor = torch.as_tensor(part_matrix, dtype=torch.float32, device=self.device)
    grid_states[bin_state.grid_state_idx, y_start:y_end, x:x+shape[1]] += part_gpu_tensor
    ...

# Updated call sites:
# Line 314:
self._place_part_in_bin(bin_state, x, y, pd.rotations[rot],
                        shape, pd.area, mpd, grid_states,
                        part_gpu_tensor=pd.rotations_gpu[rot])

# Line 436:
self._place_part_in_bin(new_bin, 0, ctx.bin_length - 1, part_data.rotations[best_rot],
                        shape, part_data.area, mach_part_data, grid_states,
                        part_gpu_tensor=part_data.rotations_gpu[best_rot])
```

### Expected impact

**~1.5s savings per batch eval (~16% of total).** Eliminates 21,798 CPU→GPU transfers per evaluation cycle. The GPU tensors are already allocated and resident in VRAM.

### Implementation result

**Status: IMPLEMENTED** — `wave_batch_evaluator.py`

Changes made:
- `_place_part_in_bin` signature extended with `part_gpu_tensor=None` parameter
- Call site in Phase 5 (line ~314) now passes `pd.rotations_gpu[rot]`
- Call site in `_start_new_bin` (line ~436) now passes `part_data.rotations_gpu[best_rot]`

Measured result (500-solution batch):

| | Time | ms/sol |
|--|------|--------|
| Before IMP-1 | 7.30s | 14.60ms |
| After IMP-1 | 6.33s | 12.65ms |
| **Saving** | **−0.97s** | **−1.95ms** |

Note: the profiler attributed ~1.46s to `torch.as_tensor`, but part of that was GPU sync stall time that shifted elsewhere. Net measured saving is ~1.0s (~13%).

---

## IMP-2: Cache `torch.arange` and scalar tensors in `_batch_fft_all_tests`

**File:** `wave_batch_evaluator.py`, lines 363-375

### Issue

Inside the chunk loop (307 iterations), the following tensors are recreated on every iteration:

```python
row_idx = torch.arange(H, device=self.device).view(1, H, 1)    # line 363
col_idx = torch.arange(W, device=self.device).view(1, 1, W)    # line 364
torch.tensor(-1e9, device=self.device)                          # line 375
```

Each `torch.arange` and `torch.tensor` call has kernel launch and allocation overhead. With 307 chunks, this adds up.

### Suggested change

Cache these tensors once per machine in `_process_machine_batch` (or in `__init__`), and pass them into `_batch_fft_all_tests`:

```python
# In _process_machine_batch, before the wave loop:
row_idx = torch.arange(H, device=self.device).view(1, H, 1)
col_idx = torch.arange(W, device=self.device).view(1, 1, W)
neg_inf_scalar = torch.tensor(-1e9, device=self.device)

# Pass to _batch_fft_all_tests and reuse
```

### Expected impact

**~0.2–0.3s savings.** Eliminates ~920 redundant small tensor allocations (307 chunks × 3 tensors). Minor individually, but cumulative.

### Implementation result

**Status: IMPLEMENTED — NEUTRAL** — `wave_batch_evaluator.py`

Changes made:
- `row_idx`, `col_idx`, `neg_inf` tensors created once in `_process_machine_batch` (before the wave loop)
- Passed through `_process_wave_true_batch` → `_batch_fft_all_tests` as parameters
- Removed the three per-chunk tensor creations inside the `_batch_fft_all_tests` loop

Measured result (500-solution batch, 10-run average):

| | Mean | Std |
|--|------|-----|
| After IMP-1 (before IMP-2) | ~6.33s (single run) | — |
| After IMP-2 | **6.26s** | ±0.07s |

Single-run measurements initially showed 6.33s → 6.44s (apparently worse), but with 10 runs the mean settles at 6.26s — **no meaningful difference**. The actual cost of creating these three tensors is only ~11ms total for 307 chunks, far below the ~70ms run-to-run noise floor. The profiler's 3.07s attribution to `torch.tensor` was GPU sync stall time, not tensor allocation cost itself.

Change is kept since it removes redundant allocations and is cleaner code, but it does not contribute measurable speedup.

---

## IMP-3: Eliminate dual grid maintenance (numpy + GPU) in wave_batch

**File:** `wave_batch_evaluator.py`, lines 401-404 and 420-423
**Related:** `numba_utils.py` lines 160-188

### Issue

The wave_batch evaluator maintains **two copies** of every bin grid:

1. `bin_state.grid` — a **numpy** `uint8` array on CPU (line 422)
2. `grid_states[grid_state_idx]` — a **torch** `float32` tensor on GPU (line 94)

Every placement writes to **both**:
- Line 401: `bin_state.grid[...] += part_matrix.astype(np.uint8)` (CPU)
- Line 404: `grid_states[...] += part_tensor` (GPU)

The CPU grid exists solely to support:
1. `update_vacancy_vector_rows()` — Numba JIT function that reads the numpy grid (line 407)
2. `check_vacancy_fit_simple()` — Numba JIT vacancy check that reads `bin_state.vacancy_vector` (line 233)

### Suggested change

**Option A (moderate effort, high impact):** Move the vacancy vector to GPU as a torch tensor. Replace the Numba `update_vacancy_vector_rows` with a GPU kernel that computes max-consecutive-zeros per row directly on the GPU grid. Similarly, replace `check_vacancy_fit_simple` with a GPU-based sliding-window check. This eliminates the CPU grid entirely.

**Option B (lower effort, medium impact):** Keep the vacancy vector on CPU but derive it from the GPU grid via a single `.cpu()` transfer of only the modified rows, rather than maintaining a full CPU grid copy. Remove `bin_state.grid` and compute vacancy from `grid_states[idx][y_start:y_end].cpu().numpy()` only when needed.

### Expected impact

**Option A:** Eliminates the numpy grid entirely (~0.4s from `update_vacancy_vector_rows` + ~0.2s from numpy grid writes = **~0.6s savings**). Also eliminates the `check_vacancy_fit_simple` calls (159K calls, 0.175s). However, this requires re-implementing vacancy logic as GPU kernels.

**Option B:** Saves the numpy grid write overhead (~0.2s) while keeping the Numba path for vacancy computation. Simpler to implement.

---

## IMP-4: Replace list-of-dicts with parallel arrays in `_process_wave_true_batch`

**File:** `wave_batch_evaluator.py`, lines 218-245 and 348-351

### Issue

Phase 3 builds `all_tests` as a **list of Python dicts** with string keys:

```python
test_entry = {
    'ctx_idx': ctx_idx,
    'bin_idx': bin_idx,
    'bin_state': bin_state,
    'rot': rot,
    'shape': shape,
    'part_fft': mach_part_data.ffts[rot],
    'part_data': part_data,
    'mach_part_data': mach_part_data
}
```

This list can grow to thousands of entries. Then in `_batch_fft_all_tests`, list comprehensions extract fields one by one:

```python
grid_indices = torch.tensor([t['bin_state'].grid_state_idx for t in chunk_tests], ...)
batch_part_ffts = torch.stack([t['part_fft'] for t in chunk_tests], dim=0)
part_heights = torch.tensor([t['shape'][0] for t in chunk_tests], ...)
part_widths = torch.tensor([t['shape'][1] for t in chunk_tests], ...)
```

Each list comprehension iterates over all tests with dict lookups and attribute access.

### Suggested change

Use parallel numpy arrays or simple lists built during Phase 3:

```python
# Build parallel arrays instead of list of dicts
test_grid_indices = []
test_part_ffts = []
test_heights = []
test_widths = []
test_ctx_indices = []
test_bin_indices = []
test_bin_states = []
test_rotations = []
test_part_data = []
test_mach_part_data = []

# In the loop:
test_grid_indices.append(bin_state.grid_state_idx)
test_part_ffts.append(mach_part_data.ffts[rot])
test_heights.append(shape[0])
test_widths.append(shape[1])
# ... etc

# In _batch_fft_all_tests, use pre-built arrays directly:
grid_indices = torch.tensor(test_grid_indices[chunk_start:chunk_end], device=self.device, dtype=torch.long)
batch_part_ffts = torch.stack(test_part_ffts[chunk_start:chunk_end], dim=0)
part_heights = torch.tensor(test_heights[chunk_start:chunk_end], device=self.device)
```

### Expected impact

**~0.3–0.5s savings.** Eliminates thousands of dict allocations, string-keyed lookups, and per-element attribute access in list comprehensions. The parallel arrays allow direct slicing.

### Implementation result

**Status: IMPLEMENTED** — `wave_batch_evaluator.py`

Changes made:
- Phase 3 builds 8 parallel lists (`test_grid_indices`, `test_part_ffts`, `test_heights`, `test_widths`, `test_bin_indices`, `test_shapes`, `test_bin_states`, `test_rotations`) instead of a list of dicts
- `_batch_fft_all_tests` signature updated to accept the parallel lists; list comprehensions replaced with direct slices (`test_grid_indices[chunk_start:chunk_end]`, etc.)
- Phase 5 indexes into parallel lists by `test_idx` instead of `all_tests[test_idx]['key']`

Measured result (500-solution batch, 10-run average):

| | Mean | Std |
|--|------|-----|
| Before IMP-4 (after IMP-1+2) | 6.26s | ±0.07s |
| After IMP-4 | **6.17s** | ±0.08s |
| **Saving** | **−0.09s** | |

~90ms improvement, at the lower end of the expected range. The dict overhead was smaller than estimated — most of the Python loop cost is in the vacancy checks and bin iteration, not the dict construction itself.

---

## IMP-5: Avoid repeated `.astype(np.int32)` on density arrays

**File:** `wave_batch_evaluator.py`, line 232

### Issue

Inside the triple-nested loop (contexts × bins × rotations), the code does:

```python
dens = part_data.densities[rot].astype(np.int32)
```

This creates a **new numpy array** on every call. Since parts are reused across contexts, the same density array is converted thousands of times. The `densities` list stores arrays that may already be int32 (they're created as int32 in `BRKGA_alg3.py` line 333).

### Suggested change

Ensure densities are stored as `int32` at creation time (they already are — see `BRKGA_alg3.py` line 333: `max_runs = np.zeros(rotated.shape[0], dtype=np.int32)`). Then remove the `.astype()` call:

```python
# Line 232: Remove the cast - densities are already int32
dens = part_data.densities[rot]
if check_vacancy_fit_simple(bin_state.vacancy_vector, dens):
```

Or if there's any doubt, pre-cast once in `prepare_jit_data()`.

### Expected impact

**~0.1s savings.** Eliminates ~159K unnecessary array allocations. Small individually but called in the tightest loop.

### Implementation result

**Status: IMPLEMENTED** — `wave_batch_evaluator.py`, line 248

Changes made:
- Removed `.astype(np.int32)` from `dens = part_data.densities[rot].astype(np.int32)`
- Densities are already `int32` at creation time (`BRKGA_alg3.py` line 333: `np.zeros(..., dtype=np.int32)`)

Measured result (500-solution batch, 10-run average):

| | Mean | Std |
|--|------|-----|
| Before IMP-5 | 6.17s | ±0.08s |
| After IMP-5 | **6.12s** | ±0.09s |
| **Saving** | **−0.05s** | |

Within noise given the ±0.09s std, but directionally consistent with the estimate. The saving is real but marginal — the ~159K allocations are individually tiny.

---

## IMP-6: Increase `CHUNK_SIZE` in `_batch_fft_all_tests`

**File:** `wave_batch_evaluator.py`, line 338

### Issue

`CHUNK_SIZE = 500` results in 307 chunks for 500 solutions. Each chunk incurs:
- 1 `torch.stack` call
- 1 `torch.fft.ifft2` kernel launch
- 1 `.cpu()` transfer
- Multiple small tensor creations

The RTX A4000 has 16 GB VRAM. Each chunk uses roughly `500 × 300 × 300 × 8 bytes = 360 MB` for complex64 tensors, well within memory.

### Suggested change

Increase `CHUNK_SIZE` to 1000–2000 and benchmark:

```python
CHUNK_SIZE = 1500  # Tune based on available VRAM and machine dimensions
```

### Expected impact

**~0.3–0.5s savings.** Halving the number of chunks halves the kernel launch overhead, tensor creation overhead, and CPU↔GPU sync points. The FFT itself scales well with batch size.

### Implementation result

**Status: IMPLEMENTED** — `wave_batch_evaluator.py`, `CHUNK_SIZE` constant

Approach: benchmarked all candidate values before choosing (8 runs each):

| CHUNK_SIZE | Mean | Std |
|-----------|------|-----|
| 250 | 6.051s | ±0.061s |
| **500** (old) | 5.897s | ±0.015s |
| **750** (new) | 5.877s | ±0.033s |
| 1000 | 5.886s | ±0.044s |
| 1500 | 5.882s | ±0.040s |
| 2000 | 5.970s | ±0.265s (memory pressure) |
| 3000+ | OOM | — |

750–1500 are all within noise of each other. 750 chosen as the sweet spot: best mean, safe VRAM headroom, no memory pressure.

Measured result vs prior baseline (10-run average, clean invocation):

| | Mean | Std |
|--|------|-----|
| Before IMP-6 | 6.12s | ±0.09s |
| After IMP-6 | **6.08s** | ±0.12s |
| **Saving** | **−0.04s** | |

The saving is smaller than expected. The original 3.07s attributed to `torch.tensor` in the profiler was mostly GPU sync stall time — reducing the number of chunks doesn't eliminate the stalls, it just merges them. The actual kernel launch and chunk overhead was already modest.

---

## IMP-7: Pre-compute `argmax` + `max` as single operation

**File:** `wave_batch_evaluator.py`, lines 379-380

### Issue

```python
best_flat_idx = flat_scores.argmax(dim=1)    # full pass over data
max_scores = flat_scores.max(dim=1).values   # second full pass over same data
```

Two separate reductions over the same large tensor (`chunk_n × H × W`).

### Suggested change

Use `torch.max` which returns both values and indices in a single pass:

```python
max_scores, best_flat_idx = flat_scores.max(dim=1)
```

### Expected impact

**~0.05–0.1s savings.** Eliminates one full-tensor reduction pass per chunk. Minor but free.

### Implementation result

**Status: IMPLEMENTED** — `wave_batch_evaluator.py`, line 397

Changes made:
- Replaced two separate calls (`flat_scores.argmax(dim=1)` + `flat_scores.max(dim=1).values`) with a single `flat_scores.max(dim=1)` which returns both values and indices in one GPU pass.

Measured result (500-solution batch, 10-run average):

| | Mean | Std |
|--|------|-----|
| Before IMP-7 | 6.08s | ±0.12s |
| After IMP-7 | **5.91s** | ±0.04s |
| **Saving** | **−0.17s** | |

Larger than expected — beat the 0.05–0.1s estimate by ~2x. Notably, the standard deviation also dropped from ±0.12s to ±0.04s, suggesting the redundant reduction was contributing to memory bandwidth contention and occasional stalls.

---

## IMP-8: Eliminate `partMatrix.astype(np.uint8)` on every insert

**File:** `binClassNew.py`, line 157
**Also:** `wave_batch_evaluator.py`, line 401

### Issue

```python
self.grid[y_start:y_end, x:x + shapes[1]] += partMatrix.astype(np.uint8)
```

`partMatrix` is `int32` (created in `BRKGA_alg3.py` line 310). The `.astype(np.uint8)` creates a new array on every insert. Since the grid is `uint8`, numpy will handle the cast implicitly during the `+=` operation.

### Suggested change

Either:
- Pre-store `uint8` versions of rotations in `PartData` (one-time cost at startup)
- Or rely on numpy's implicit casting: `self.grid[y_start:y_end, x:x + shapes[1]] += partMatrix`

```python
# Option A: Add to PartData
rotations_uint8: Optional[List[np.ndarray]] = None  # Pre-cast versions

# Option B: Just remove the .astype() — numpy handles uint8 += int32
self.grid[y_start:y_end, x:x + shapes[1]] += partMatrix
```

### Expected impact

**~0.05–0.1s savings.** Eliminates ~22K array allocations per batch eval. Small but cumulative.

### Implementation result

**Status: IMPLEMENTED (Option A)** — `data_structures.py`, `BRKGA_alg3.py`, `wave_batch_evaluator.py`, `binClassNew.py`, `binClassInitialSol.py`, `placement.py`

Option A chosen over Option B for maximum impact: pre-stored `uint8` rotations eliminate the cast entirely and are 4x smaller in memory than `int32`, improving cache utilisation.

Changes made:
- `PartData`: new `rotations_uint8` field
- `BRKGA_alg3.py`: `rotations_uint8 = [r.astype(np.uint8) for r in rotations]` computed once at startup
- All `insert` / `_place_part_in_bin` call sites updated to pass `rotations_uint8[rot]`
- `.astype(np.uint8)` removed from `_place_part_in_bin`, `binClassNew.insert`, `binClassInitialSol.insert`

Measured result (500-solution batch, 10-run average):

| | Mean | Std |
|--|------|-----|
| Before IMP-8 | 5.91s | ±0.04s |
| After IMP-8 | **5.86s** | ±0.09s |
| **Saving** | **−0.05s** | |

At the low end of the estimate. The per-call savings are real but individually tiny (~2μs each across ~22K calls).

---

## IMP-9: `binClassInitialSol.py` uses numpy diff instead of Numba for vacancy update

**File:** `binClassInitialSol.py`, lines 144-169

### Issue

The `binClassInitialSol.py` version of `insert()` uses numpy `diff/where/maximum.at` for vacancy vector updates (lines 154-166), while `binClassNew.py` uses the Numba JIT `update_vacancy_vector_rows()`. The initial solution builder uses the slower path.

### Current code (binClassInitialSol.py)

```python
padded = self._padded_buffer[:num_rows, :]
padded[:, 1:-1] = self.grid[y_start:y_end, :]
diffs = np.diff(padded.astype(np.int8), axis=1)
start_indices = np.where(diffs == -1)
end_indices = np.where(diffs == 1)
run_lengths = end_indices[1] - start_indices[1]
max_zeros = self._max_zeros_buffer[:num_rows]
max_zeros.fill(0)
np.maximum.at(max_zeros, start_indices[0], run_lengths)
self.vacancy_vector[y_start:y_end] = max_zeros
```

### Suggested change

Replace with the same Numba JIT call used in `binClassNew.py`:

```python
from numba_utils import update_vacancy_vector_rows
update_vacancy_vector_rows(self.vacancy_vector, self.grid[y_start:y_end, :], y_start)
```

### Expected impact

**~2-5x faster vacancy updates** during initial solution construction. The initial solution phase processes all 50 parts sequentially, so this adds up. Also simplifies the code by removing the pre-allocated buffer fields (`_padded_buffer`, `_max_zeros_buffer`).

### Implementation result

**Status: IMPLEMENTED** — `binClassInitialSol.py`

Changes made:
- Added `update_vacancy_vector_rows` to the import from `numba_utils`
- Replaced the 20-line numpy diff/where/maximum.at block with a single `update_vacancy_vector_rows(...)` call (matching `binClassNew.py`)
- Removed the now-unused `_padded_buffer` and `_max_zeros_buffer` pre-allocated fields from `__init__`

This change only affects the **initial solution construction** phase, not the BRKGA generation loop. As such it has no measurable effect on the 500-solution batch benchmark. Verified end-to-end: initial solution makespan unchanged at 246042.18, and all 3 generations complete correctly.

---

## IMP-10: Fitness cache provides 0% hit rate — remove or fix

**File:** `BRKGA_alg3.py`, lines 83-86, 101-148

### Issue

The fitness cache quantizes chromosomes to 4 decimal places and hashes them. With continuous random-key chromosomes (float32 uniform [0,1]), the probability of two identical quantized chromosomes is essentially zero. Profiling confirms: **0/1850 hits (0.0%)**.

The cache adds overhead per evaluation:
- `np.round(solution, 4)` on every chromosome
- `tuple()` conversion (allocates a 100-element tuple)
- Dict lookup

### Suggested change

**Option A:** Remove the cache entirely — it provides no benefit with continuous chromosomes.

**Option B:** If caching is desired for future use (e.g., with discrete keys), make it opt-in via a parameter.

### Expected impact

**~0.05s savings per generation** from eliminating useless hashing/lookup. More importantly, eliminates memory growth from storing all unique chromosomes in a dict that's never hit.

---

## Summary: Prioritized by Impact

| # | Improvement | Est. Savings | Actual Saving | Status | Effort |
|---|-------------|-------------|---------------|--------|--------|
| IMP-1 | Use pre-computed GPU tensors in `_place_part_in_bin` | ~1.5s (16%) | **−1.0s (−13%)** | DONE | Low |
| IMP-2 | Cache `torch.arange` / scalar tensors | ~0.2–0.3s (3%) | **~0s (noise)** | DONE | Low |
| IMP-3 | Eliminate dual numpy/GPU grid | ~0.6s (7%) | — | Pending | High |
| IMP-4 | Parallel arrays instead of list-of-dicts | ~0.3–0.5s (4%) | **−0.09s** | DONE | Medium |
| IMP-6 | Increase CHUNK_SIZE | ~0.3–0.5s (4%) | **−0.04s** | DONE | Low |
| IMP-5 | Remove redundant `.astype(np.int32)` | ~0.1s (1%) | **−0.05s** | DONE | Low |
| IMP-8 | Remove `.astype(np.uint8)` per insert | ~0.05–0.1s | **−0.05s** | DONE | Low |
| IMP-7 | Single `torch.max` instead of `argmax`+`max` | ~0.05–0.1s | **−0.17s** | DONE | Low |
| IMP-9 | Use Numba JIT in `binClassInitialSol.py` | N/A (init only) | N/A (init only) | DONE | Low |
| IMP-10 | Remove useless fitness cache | ~0.05s/gen | ~0s (early-return bypass already eliminated the per-gen overhead) | DONE | Low |

### Measured progress

| State | Mean time (500-sol batch) | Std |
|-------|--------------------------|-----|
| Original baseline | 7.30s (single run) | — |
| After IMP-1 | 6.33s (single run) | — |
| After IMP-1 + IMP-2 | 6.26s | ±0.07s |
| After IMP-1 + IMP-2 + IMP-4 | 6.17s | ±0.08s |
| After + IMP-5 | 6.12s | ±0.09s |
| After + IMP-6 | 6.08s | ±0.12s |
| After + IMP-7 | 5.91s | ±0.04s |
| After + IMP-8 | **5.86s** | ±0.09s |
