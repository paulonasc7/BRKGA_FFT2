# Optimization Ideas for Generation Speed

**Current baseline:** ~3.80s/gen (P50M2-0, 500 individuals, wave_batch, torch_gpu, RTX A4000)

**Phase breakdown (% of total time):**
- Phase 4 (Batch IFFT): **66.1%** — the dominant target
- Phase 5 (Best placements + grid updates): 14.0%
- Phase 3 (Vacancy check + collect tests): 7.6%
- Phase 6 (Open new bins): 7.2%
- Phase 2 (Batch grid FFTs): 4.8%
- Phase 1 (Gather context): 0.4%

---

## ~~Idea 1: Half-precision FFT (fp16)~~ — PREVIOUSLY TESTED, REJECTED

**Status:** Already tested and rejected. See `PROJECT_DEEP_ANALYSIS.md` line 478 and `FFT_OPTIMIZATION_OPTIONS.md` line 142.

**Why it doesn't work:** cuFFT only supports fp16 for power-of-2 dimensions. Our grids are 300x250, which would need padding to 512x256. The increased FFT size cancels out the bandwidth savings from halved precision. Benchmarked result: 0% improvement.

**Possible revisit:** Only if a future PyTorch version adds non-power-of-2 fp16 cuFFT support, or if we find a way to make power-of-2 dimensions work naturally (e.g., if the problem instances change to have power-of-2 bin dimensions). Otherwise, skip.

---

## Idea 2: Speculative bin-0-first early exit

**Target:** Phase 4 (66.1%)
**Expected savings:** 0.3-0.8s/gen (instance-dependent, likely on the optimistic end)
**Effort:** Medium
**Priority:** MEDIUM (deprioritized below Ideas 3 and 5 due to two-sync-point overhead risk)

**What:** Currently all (context x bin x rotation) tests are batched into one IFFT call. But the algorithm is first-fit: if bin 0 has a valid placement, bins 1+ are irrelevant for that context. By processing bin 0 first, we can skip IFFT work for contexts that succeed.

**How:** Split `_process_wave_true_batch` Phase 3+4 into two passes:

1. **Pass 1 — bin 0 only:** Collect tests only for `bin_idx == 0`. Run `_batch_fft_all_tests` on those. Identify which contexts got a valid placement.
2. **Pass 2 — remaining bins:** For contexts that failed bin 0, collect tests for bins 1+. Run `_batch_fft_all_tests` again.

Phase 5 then merges results from both passes.

**Where to change:**
- `wave_batch_evaluator.py` — `_process_wave_true_batch()`, Phase 3 (lines ~258-300) and Phase 4 (lines ~303-309). The Phase 3 collection loop already iterates `for bin_idx, bin_state in enumerate(ctx.open_bins)` — split this into two collection passes.

**Key detail:** The scoring already encodes `bin_idx` as the most significant component (`-bin_idx * 1e9`), so bin 0 always wins over bin 1+ when valid. This means pass 1 results are final for any context that got a hit — no need to re-evaluate.

**When this helps most:** When bins are relatively empty (early waves) and most parts fit in bin 0. As bins fill up, more contexts need bin 1+, and the savings decrease. The first ~50% of waves typically have high bin-0 success rates.

**When this hurts:** If almost all contexts need bin 1+ anyway (heavily packed instances), you pay the overhead of two `_batch_fft_all_tests` calls (two GPU syncs, two CPU transfers, two sets of tensor allocations) instead of one. Consider making this adaptive: if the previous wave had >80% bin-0 success, use two-pass; otherwise use single-pass.

**Important risk:** Late waves (where bins are fuller and Phase 4 time is highest) are exactly the waves where bin-0 success rate is lowest — so the savings are smallest when they'd matter most.

---

## Idea 3: Pre-allocate chunk tensors

**Target:** Phase 4 (66.1%)
**Expected savings:** 0.1-0.15s/gen
**Effort:** Low
**Priority:** HIGH (easy win)

**What:** Inside `_batch_fft_all_tests`, each chunk iteration allocates new GPU tensors via `torch.tensor(...)` and `torch.stack(...)`. These trigger CUDA memory allocation calls. Pre-allocating reusable buffers eliminates this overhead.

**Where to change:**
- `wave_batch_evaluator.py` — `_batch_fft_all_tests()`:
  - Line ~415-417: `torch.tensor(test_grid_indices, ...)`, `torch.tensor(test_heights, ...)`, `torch.tensor(test_widths, ...)` — these are rebuilt every call. Instead, pre-allocate `self._chunk_grid_indices`, `self._chunk_heights`, `self._chunk_widths` as `(CHUNK_SIZE,)` tensors in `__init__` and copy into them.
  - Line ~430: `torch.stack(test_part_ffts[chunk_start:chunk_end])` — pre-allocate a `(CHUNK_SIZE, H, W//2+1)` complex64 buffer and copy part FFTs into it. This is the biggest allocation per chunk.

**How:** In `__init__`, after `self.device = device`:
```python
CHUNK_SIZE = 750
H, W = mach_data.bin_length, mach_data.bin_width  # need per-machine
self._part_fft_buf = torch.zeros((CHUNK_SIZE, H, W//2+1), dtype=torch.complex64, device=device)
self._grid_idx_buf = torch.zeros(CHUNK_SIZE, dtype=torch.long, device=device)
# etc.
```

Then in the chunk loop, use `buf[:chunk_n].copy_(data)` instead of creating new tensors.

**Also:** The `results_cpu` transfer (line ~460-461: `.cpu().numpy()`) creates a new CPU tensor each time. Pre-allocate a pinned-memory CPU buffer and use `.copy_()` with `non_blocking=True`.

**Note on `torch.stack`:** Stacking a Python list of tensors is already somewhat optimized in PyTorch. The bigger win is likely the index tensors (`grid_indices`, `heights`, `widths`) which go through `torch.tensor(python_list)` — that's a CPU→GPU copy with implicit sync.

---

## Idea 4: Fused post-IFFT CUDA kernel

**Target:** Phase 4 (66.1%)
**Expected savings:** 0.15-0.3s/gen
**Effort:** High
**Priority:** MEDIUM

**What:** After `irfft2`, the code runs 5 separate GPU kernels: `round` → `zero_mask` → `valid_mask` → `score (where)` → `max`. Each reads/writes the full `(chunk_n, H, W)` tensor. A single custom CUDA kernel can do all of this in one pass: read the IFFT output, round, check validity, compute score, and reduce to find the best (row, col) per test.

**Where to change:**
- Create a new file (e.g., `cuda_post_ifft.py`) with a custom CUDA kernel, similar to `cuda_batch_update.py`.
- `wave_batch_evaluator.py` — `_batch_fft_all_tests()`, lines ~434-461. Replace the entire post-IFFT block with a single kernel call.

**Kernel signature:**
```
Input:  ifft_output (chunk_n, H, W) float32
        part_heights (chunk_n,) int
        part_widths  (chunk_n,) int
Output: has_valid    (chunk_n,) bool
        best_col     (chunk_n,) int
        best_row     (chunk_n,) int
```

**Kernel logic per test `i`:** For each `(row, col)` in the valid region (`row >= h-1, col >= w-1`), check if `round(ifft_output[i, row, col]) == 0`. If so, compute score = `row * (W+1) - col`. Use shared-memory reduction to find the max score and its (row, col) across all valid positions.

**Alternative (easier):** Use `torch.compile` (see Idea 5) which may auto-fuse these operations.

---

## Idea 5: `torch.compile` on Phase 4 inner loop

**Target:** Phase 4 (66.1%)
**Expected savings:** 0.1-0.3s/gen
**Effort:** Low
**Priority:** MEDIUM

**What:** PyTorch 2.x `torch.compile` with the Triton backend can automatically fuse elementwise + reduction operations. The post-IFFT operations (round, masking, scoring, argmax) are good candidates.

**Where to change:**
- `wave_batch_evaluator.py` — Extract the post-IFFT code (lines ~434-461) into a standalone function and decorate it:

```python
@torch.compile(mode="reduce-overhead")
def _post_ifft_score(overlap_batch, part_heights, part_widths, row_idx, col_idx, neg_inf, H, W):
    rounded_batch = torch.round(overlap_batch)
    zero_mask = (rounded_batch == 0)
    valid_row = row_idx >= (part_heights - 1).view(-1, 1, 1)
    valid_col = col_idx >= (part_widths - 1).view(-1, 1, 1)
    valid_zeros = zero_mask[:, :H, :W] & valid_row & valid_col
    score = torch.where(valid_zeros, row_idx.float() * (W + 1) - col_idx.float(), neg_inf)
    flat_scores = score.view(score.shape[0], -1)
    max_scores, best_flat_idx = flat_scores.max(dim=1)
    best_row = best_flat_idx // W
    best_col = best_flat_idx % W
    has_valid = max_scores > -1e8
    smallest_cols = best_col - (part_widths - 1)
    return has_valid, smallest_cols, best_row
```

**Caveat:** First call triggers compilation (~30s). Use `mode="reduce-overhead"` for CUDA graph capture. Test that results are identical. If `torch.compile` doesn't help, the manual CUDA kernel (Idea 4) is the fallback.

**Dynamic shapes:** The last chunk is often smaller than `CHUNK_SIZE`, which would trigger a recompilation. Pad the last chunk to full `CHUNK_SIZE` and mask out padding to ensure a single compiled graph.

**PyTorch version:** Update to PyTorch >= 2.4 on Paperspace for best `torch.compile` stability. 2.1.1 had rough edges.

---

## Idea 6: Eliminate CPU grid mirror

**Target:** Phase 5 (14.0%)
**Expected savings:** 0.05-0.1s/gen
**Effort:** Medium
**Priority:** LOW

**What:** Every `BinState` has both `bin_state.grid` (NumPy uint8) and a GPU `grid_states[idx]` (float32). After each placement, the part is written to both. The CPU grid exists only for vacancy vector updates.

**Alternative:** After the GPU grid update, transfer only the affected rows back to CPU for the vacancy update:
```python
# Instead of:
bin_state.grid[y_start:row+1, col:col+shape[1]] += pd_.rotations_uint8[rot]
update_vacancy_vector_rows(bin_state.vacancy_vector, bin_state.grid[y_start:row+1, :], y_start)

# Do:
rows_cpu = grid_states[bin_state.grid_state_idx, y_start:row+1, :].cpu().numpy().astype(np.uint8)
update_vacancy_vector_rows(bin_state.vacancy_vector, rows_cpu, y_start)
```

This eliminates the `bin_state.grid` array entirely. The GPU→CPU transfer of a few rows (~30-50 x 250 = ~10KB) is fast.

**Where to change:**
- `wave_batch_evaluator.py` — Phase 5 CPU updates (lines ~367-378) and `_place_part_in_bin` (lines ~491-510)
- `BinState` dataclass — remove `grid` field

**Risk:** The GPU grid is float32 and accumulates via addition. After many parts, values could be 2, 3, etc. The vacancy check only cares about zero vs nonzero, so `rows_cpu > 0` gives the correct binary grid for vacancy computation. However, `update_vacancy_vector_rows` currently counts consecutive zeros — it needs the grid values to be 0 where empty. As long as the GPU grid only ever gets part matrices added (which are 0/1), nonzero means occupied, which is correct.

**Sync hazard:** The CUDA batch kernel updates GPU grids asynchronously. If you immediately read `grid_states[idx, y_start:row+1, :].cpu()`, you'll race with the kernel. You need a `torch.cuda.synchronize()` before the CPU transfer, which partially negates the savings.

---

## Idea 7: Batch Phase 3 vacancy checks

**Target:** Phase 3 (7.6%)
**Expected savings:** 0.05-0.1s/gen
**Effort:** Medium
**Priority:** LOW

**What:** Phase 3 calls `check_vacancy_fit_simple` individually for each (context, bin, rotation) triple — ~3750 calls/wave, each a separate Numba function call with Python overhead.

**Alternative:** Collect all (vacancy_vector, density) pairs into contiguous arrays, then run a single Numba `prange`-parallelized function:

```python
@jit(nopython=True, cache=True, parallel=True)
def batch_vacancy_check(vacancies, densities, vac_offsets, den_offsets, n_checks):
    results = np.zeros(n_checks, dtype=np.bool_)
    for i in prange(n_checks):
        vac = vacancies[vac_offsets[i]:vac_offsets[i+1]]
        den = densities[den_offsets[i]:den_offsets[i+1]]
        results[i] = check_vacancy_fit_single(vac, den, len(den))
    return results
```

**Where to change:**
- `numba_utils.py` — add `batch_vacancy_check`
- `wave_batch_evaluator.py` — Phase 3 (lines ~275-300): collect all vacancy/density pairs first, call batch check, then build test arrays only for those that passed

**Caveat:** The vacancy vectors are different lengths per bin (all are `H=300` but that's the same for all bins on a machine). The density arrays vary by part and rotation. Pre-packing into contiguous arrays has overhead. This is worth it only if the Python-level per-call overhead of Numba is significant vs the actual computation.

**Math check:** Phase 3 total is 1.42s over 5 gens with ~3750 calls/wave x 360 waves = ~1.35M calls. That's ~1us/call. The Numba compute itself (scanning a 300-element array) is most of that time — Python call overhead is a small fraction. `prange` thread-pool overhead for individually-fast calls may not help. **Skip unless Phase 3 grows in relative share after other optimizations.**

---

## Idea 8: Batch Phase 6 new-bin creation

**Target:** Phase 6 (7.2%)
**Expected savings:** 0.05-0.1s/gen
**Effort:** Low
**Priority:** LOW

**What:** `_start_new_bin` calls `_place_part_in_bin` which does individual GPU operations. When many contexts need new bins in the same wave, batch the GPU work.

**Where to change:**
- `wave_batch_evaluator.py` — Phase 6 (lines ~380-382). Instead of calling `_start_new_bin` in a loop:
  1. Create all new `BinState` objects (CPU-only: numpy grid, vacancy vector)
  2. Batch all `grid_states[grid_idx].zero_()` into one operation
  3. Use the existing CUDA batch kernel to place all first parts at once
  4. Do CPU updates for all new bins

This reuses the same pattern as Phase 5's batched updates.

---

## Idea 9: Pre-allocate grid_states/grid_ffts across generations

**Target:** Overhead between waves
**Expected savings:** 0.01-0.02s/gen
**Effort:** Low
**Priority:** LOW

**What:** `_process_machine_batch` allocates `grid_states` (4500 x 300 x 250 float32 = ~1.25 GB) and `grid_ffts` (4500 x 300 x 126 complex64 = ~1.36 GB) fresh every call. These could be allocated once in `__init__` and reused.

**Where to change:**
- `wave_batch_evaluator.py` — Move lines ~128-131 from `_process_machine_batch` to `__init__`. Zero out at the start of each call instead of re-allocating.

**Reality check:** PyTorch's CUDA caching allocator reuses memory blocks after the first generation, so re-allocation after gen 1 is effectively just a cache lookup. Savings are marginal (~0.01s). Not harmful, but don't expect measurable improvements.

---

## Implementation Order

1. **Idea 3** (pre-allocate chunk tensors) — guaranteed win, low effort, do first
2. **Idea 5** (`torch.compile`) — low effort, update PyTorch to >= 2.4 first, test if it fuses post-IFFT ops
3. **Idea 2** (bin-0-first) — worth trying with adaptive toggle, measure carefully
4. **Idea 8** (batch Phase 6) — straightforward reuse of existing CUDA kernel
5. **Idea 4** (fused CUDA kernel) — only if Idea 5 fails to deliver
6. **Idea 6** (eliminate CPU grid) — modest win, watch for sync hazard
7. **Idea 9** (pre-allocate grid_states) — negligible due to CUDA caching allocator, skip unless touching that code anyway
8. **Idea 7** (batch vacancy) — math doesn't strongly support it, skip unless Phase 3 share grows
9. ~~**Idea 1**~~ (fp16 FFT) — already tested and rejected, skip

**Note:** Idea 1 (fp16) was previously benchmarked and shown to give 0% improvement due to power-of-2 padding requirements in cuFFT. Do not re-implement without first verifying that the cuFFT restriction has been lifted in a newer PyTorch/CUDA version.

After each change, run correctness checks as well as `profile_phases.py` on the remote GPU to measure impact:
```bash
python remote.py sync . BRKGA_FFT2 --ext .py
python remote.py run "python profile_phases.py 50 2 0 torch_gpu 5" --cwd /notebooks/BRKGA_FFT2 --timeout 300
```