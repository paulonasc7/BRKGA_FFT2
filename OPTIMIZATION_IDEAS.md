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

## Idea 1: Half-precision FFT (fp16)

**Target:** Phase 4 (66.1%)
**Expected savings:** 0.3-0.5s/gen
**Effort:** Medium
**Priority:** HIGH

**What:** The grids are binary (0/1) and the IFFT result is rounded to check for exact zeros. float16 can represent integers exactly up to 2048, which is far more than the max overlap value in a 300x250 grid with small parts. Switching from float32 to float16 halves GPU memory bandwidth — the main bottleneck for FFT on modern GPUs.

**Where to change:**
- `wave_batch_evaluator.py` — `_batch_fft_all_tests()`:
  - Line ~430: `torch.stack(test_part_ffts[...])` — ensure part FFTs are stored as complex32 (half-precision complex = `torch.complex32`)
  - Line ~433: `torch.fft.irfft2(batch_grid_ffts * batch_part_ffts, s=(H, W))` — this is the hot call
  - Line ~253: `torch.fft.rfft2(batch_grids)` in Phase 2 should also output fp16
- `collision_backend.py` — `prepare_part_fft()` (line ~75-83): Store part FFTs as complex32 instead of complex64
- `wave_batch_evaluator.py` — `_process_machine_batch()`:
  - Line ~130: `grid_ffts` tensor allocation should use `torch.complex32` instead of `torch.complex64`
  - Line ~129: `grid_states` can stay float32 (used by the CUDA kernel) OR also go float16

**Validation:** After implementing, compare placement results on 1-2 generations against the float32 version. The results should be bitwise identical since the rounding to 0/nonzero is the only check that matters.

**Caveat:** `torch.complex32` support in cuFFT may require PyTorch >= 2.0. Check with a quick test: `torch.fft.rfft2(torch.zeros(4,4, dtype=torch.float16, device='cuda'))`. If unsupported, an alternative is to keep FFT in float32 but use fp16 for the post-IFFT masking/scoring operations (smaller win but still helps).

---

## Idea 2: Speculative bin-0-first early exit

**Target:** Phase 4 (66.1%)
**Expected savings:** 0.3-0.8s/gen (instance-dependent)
**Effort:** Medium
**Priority:** HIGH

**What:** Currently all (context x bin x rotation) tests are batched into one IFFT call. But the algorithm is first-fit: if bin 0 has a valid placement, bins 1+ are irrelevant for that context. By processing bin 0 first, we can skip IFFT work for contexts that succeed.

**How:** Split `_process_wave_true_batch` Phase 3+4 into two passes:

1. **Pass 1 — bin 0 only:** Collect tests only for `bin_idx == 0`. Run `_batch_fft_all_tests` on those. Identify which contexts got a valid placement.
2. **Pass 2 — remaining bins:** For contexts that failed bin 0, collect tests for bins 1+. Run `_batch_fft_all_tests` again.

Phase 5 then merges results from both passes.

**Where to change:**
- `wave_batch_evaluator.py` — `_process_wave_true_batch()`, Phase 3 (lines ~258-300) and Phase 4 (lines ~303-309). The Phase 3 collection loop already iterates `for bin_idx, bin_state in enumerate(ctx.open_bins)` — split this into two collection passes.

**Key detail:** The scoring already encodes `bin_idx` as the most significant component (`-bin_idx * 1e9`), so bin 0 always wins over bin 1+ when valid. This means pass 1 results are final for any context that got a hit — no need to re-evaluate.

**When this helps most:** When bins are relatively empty (early waves) and most parts fit in bin 0. As bins fill up, more contexts need bin 1+, and the savings decrease. The first ~50% of waves typically have high bin-0 success rates.

**When this hurts:** If almost all contexts need bin 1+ anyway (heavily packed instances), you pay the overhead of two `_batch_fft_all_tests` calls instead of one. Consider making this adaptive: if the previous wave had >80% bin-0 success, use two-pass; otherwise use single-pass.

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

---

## Implementation Order

1. **Idea 3** (pre-allocate chunk tensors) — easiest, safest, do first
2. **Idea 5** (`torch.compile`) — low effort, test if it works
3. **Idea 1** (fp16 FFT) — biggest potential win, validate carefully
4. **Idea 2** (bin-0-first) — significant win, moderate refactor
5. **Idea 9** (pre-allocate grid_states) — trivial
6. **Idea 8** (batch Phase 6) — straightforward reuse of existing kernel
7. **Idea 6** (eliminate CPU grid) — medium refactor, modest win
8. **Idea 7** (batch vacancy) — only if Phase 3 becomes a bottleneck after other optimizations
9. **Idea 4** (fused CUDA kernel) — only if `torch.compile` doesn't deliver

After each change, run correctness checks as well as `profile_phases.py` on the remote GPU to measure impact:
```bash
python remote.py sync . BRKGA_FFT2 --ext .py
python remote.py run "python profile_phases.py 50 2 0 torch_gpu 5" --cwd /notebooks/BRKGA_FFT2 --timeout 300
```