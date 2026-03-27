# Optimization Ideas for Generation Speed

**Current baseline:** ~2.29s/gen (P50M2-0, 500 individuals, wave_batch, torch_gpu, RTX A4000)
**Previous baseline (pre-Idea 2):** ~3.32s/gen
**Original baseline (pre-all-optimizations):** ~3.80s/gen

**Phase breakdown (% of total time) — updated after Idea 2 (5 gens, 357 waves):**
- Phase 4a (P1 IFFT): **43.9%** (15.44ms/wave)
- Phase 5 (Best placements + grid updates): 17.9% (6.29ms/wave)
- Phase 6 (Open new bins): 16.9% (5.94ms/wave)
- Phase 3a (P1 collect, first valid bin): 8.5% (2.98ms/wave)
- Phase 2 (Batch grid FFTs): 7.3% (2.57ms/wave)
- Phase 3b (P2 collect, remaining bins): 4.8% (1.70ms/wave)
- Phase 4b (P2 IFFT): **~0%** (0.01ms/wave — nearly empty in practice)
- Phase 1 (Gather context): 0.6%

---

## ~~Idea 1: Half-precision FFT (fp16)~~ — PREVIOUSLY TESTED, REJECTED

**Status:** Already tested and rejected. See `PROJECT_DEEP_ANALYSIS.md` line 478 and `FFT_OPTIMIZATION_OPTIONS.md` line 142.

**Why it doesn't work:** cuFFT only supports fp16 for power-of-2 dimensions. Our grids are 300x250, which would need padding to 512x256. The increased FFT size cancels out the bandwidth savings from halved precision. Benchmarked result: 0% improvement.

**Possible revisit:** Only if a future PyTorch version adds non-power-of-2 fp16 cuFFT support, or if we find a way to make power-of-2 dimensions work naturally (e.g., if the problem instances change to have power-of-2 bin dimensions). Otherwise, skip.

---

## ~~Idea 2: First-valid-bin early exit~~ — IMPLEMENTED, HUGE WIN

**Target:** Phase 4 (was 58.9%)
**Expected savings:** 0.3-0.8s/gen
**Actual result:** 3.32s → 2.29s/gen (**−31%**) — COMMITTED (5233b1c)
**Effort:** Medium

**What:** Split Phase 3+4 into two passes. Pass 1 tests only the first vacancy-passing bin per context. Pass 2 (only for Pass-1 misses) tests remaining bins. Since `bin_idx` dominates the score (×1e9), any Pass-1 hit is globally optimal — no need to check later bins.

**Result in practice:** Phase 4b IFFT was **0.01ms/wave** (essentially zero). >99% of contexts found a valid geometric placement in their first valid bin, so Pass 2 was nearly always empty. This cut total IFFT time by ~43% (22ms → 15.44ms/wave for Phase 4a; 4b ≈ 0).

**Why "first valid bin", not "bin 0":** Different contexts may have a different first valid bin. Pass 1 uses the first bin that actually produces at least one vacancy-passing rotation, ensuring no wasted IFFT work.

---

## ~~Idea 3: Pre-allocate chunk tensors~~ — TESTED, NO BENEFIT

**Target:** Phase 4 (66.1%)
**Expected savings:** 0.1-0.15s/gen (predicted)
**Actual result:** ~0s improvement — REVERTED
**Effort:** Low

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

**Why it didn't work:** PyTorch's CUDA caching allocator reuses large memory blocks at near-zero cost (pointer lookup). The allocations we eliminated (~675 MB/chunk in theory) were already essentially free in practice. Implemented and benchmarked: gen time went from 3.80s → ~3.95s (within noise, no improvement). Reverted.

---

## Idea 4: Fused post-IFFT CUDA kernel

**Target:** Phase 4 (now 43.9% after Idea 2)
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

## Idea 5: `torch.compile` on Phase 4 inner loop — IMPLEMENTED ✓

**Target:** Phase 4 (66.1% → 58.1%)
**Expected savings:** 0.1-0.3s/gen
**Actual result:** **3.80s → 3.54s (−0.26s, −7%); Phase 4: 34.4 → 28.0 ms/wave (−19%)**
**Effort:** Low
**Status:** Committed (49b902a)

**What was done:** Extracted the post-IFFT block (round → mask → score → argmax, ~10 kernel launches) into a module-level `_post_ifft_score` function and compiled it with `torch.compile(dynamic=True, fullgraph=True)`. Triton fuses the elementwise ops into ~2 kernels, cutting VRAM bandwidth by ~5x for that block.

**Key implementation details:**
- `dynamic=True` instead of `mode="reduce-overhead"`: handles varying chunk sizes (last chunk is smaller) without padding. One compiled graph covers all chunk sizes and both machines.
- W derived from `col_idx.shape[-1]` (tensor shape) rather than passed as a Python int — avoids per-machine specialization.
- Guarded with `if hasattr(torch, 'compile')` for backwards compatibility.
- First run incurs a one-time Triton compilation spike (~3s extra in gen 2). Subsequent runs use the disk-cached kernel.
- PyTorch version on Paperspace was already new enough (no upgrade needed).

---

## ~~Idea 6: Eliminate CPU grid mirror~~ — TESTED, SEVERE REGRESSION, REVERTED

**Target:** Phase 5 (16.7%)
**Expected savings:** 0.05-0.1s/gen (predicted)
**Actual result:** Phase 5 exploded from 7.6→17.9 ms/wave (+137%), gen time 3.32→4.2s. **REVERTED.**
**Effort:** Medium

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

**Why it failed:** Each `.cpu()` call per placement forces a GPU stream flush, completely eliminating the CPU-GPU overlap that currently exists in Phase 5. The per-placement DtoH transfer (~30KB × 500 placements = 15MB/wave) added on top. The `bin_state.grid` CPU mirror is actually doing valuable work — it lets the CPU update the vacancy vector concurrently with the async GPU kernel. Do not attempt again.

---

## Idea 7: Batch Phase 3 vacancy checks

**Target:** Phase 3a + 3b (8.5% + 4.8% = 13.3% combined)
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

**Math check:** After Idea 2, Phase 3a+3b total is ~1.67s over 5 gens (357 waves). Phase 3 is now 13.3% of total — a meaningful share. With ~1038 tests/wave and vacancy checks being ~1µs/call, the Numba compute itself (scanning a 300-element array) dominates; Python call overhead is a small fraction. `prange` thread-pool overhead for individually-fast calls may not help. Worth revisiting now that Phase 3 share has grown relative to Phase 4.

---

## ~~Idea 8: Batch Phase 6 new-bin creation~~ — IMPLEMENTED ✓

**Target:** Phase 6 (8.9% → 8.7%)
**Expected savings:** 0.05-0.1s/gen
**Actual result:** **3.54s → 3.32s (−0.22s, −6%); Phase 6: 4.30→3.93 ms/wave (−8%)**
**Effort:** Low
**Status:** Committed (c4a6292)

**What was done:** Replaced serial `_start_new_bin()` loop with a batched approach:
1. Create all `BinState` objects (CPU) in one pass
2. Zero all new GPU grids with a single `grid_states.index_fill_(0, idx_tensor, 0.0)`
3. Place all first parts with one `_cuda_batch_update` kernel call
4. Do all CPU updates (numpy grid + vacancy vector) concurrently

First part always goes at bottom-left with `best_rotation` — no FFT needed, so the CUDA kernel handles the write directly.

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

1. ~~**Idea 3**~~ (pre-allocate chunk tensors) — **tested, no benefit, reverted**
2. ~~**Idea 5**~~ (`torch.compile`) — **implemented, −0.26s/gen (−7%), committed**
3. ~~**Idea 8**~~ (batch Phase 6) — **implemented, −0.22s/gen (−6%), committed**
4. ~~**Idea 6**~~ (eliminate CPU grid) — **tested, severe regression (+137% Phase 5), reverted**
5. ~~**Idea 2**~~ (first-valid-bin early exit) — **implemented, −1.03s/gen (−31%), committed**
6. ~~**Idea 4**~~ (fused CUDA kernel) — superseded by Idea 5
7. **Idea 9** (pre-allocate grid_states) — negligible due to CUDA caching allocator, skip unless touching that code anyway
8. **Idea 7** (batch vacancy) — Phase 3 now 13.3% of total after Idea 2, worth revisiting
9. ~~**Idea 1**~~ (fp16 FFT) — already tested and rejected, skip

**Note:** Idea 1 (fp16) was previously benchmarked and shown to give 0% improvement due to power-of-2 padding requirements in cuFFT. Do not re-implement without first verifying that the cuFFT restriction has been lifted in a newer PyTorch/CUDA version.

After each change, run correctness checks as well as `profile_phases.py` on the remote GPU to measure impact:
```bash
python remote.py sync . BRKGA_FFT2 --ext .py
python remote.py run "python profile_phases.py 50 2 0 torch_gpu 5" --cwd /notebooks/BRKGA_FFT2 --timeout 300
```