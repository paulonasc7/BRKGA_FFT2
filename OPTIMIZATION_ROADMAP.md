# Optimization Roadmap — BRKGA_FFT2

**Updated:** 2026-04-06
**Hardware:** RTX A4000 (16 GB VRAM), Paperspace Gradient

---

## Context

`full_native_decoder.py` moved the entire decoder — Phases 1–6 — into a single compiled C++/CUDA extension. BRKGA evolution stays in Python; a single `evaluate_batch(chromosomes)` call handles everything else.

**Performance baselines (full_native_decoder, after all completed optimizations):**

| Instance | Before (wave_batch) | After (native, current) | Speedup vs wave_batch |
|----------|--------------------|--------------------------|-----------------------|
| P50M2-0 (500 ind.) | 2.606s | **1.907s** (std 0.014s) | 1.37× |
| P75M2-0 (750 ind.) | 4.890s | **3.776s** (std 0.026s) | 1.29× |

---

## Completed optimizations

### A. CUDA selector kernel — DONE
**Result: −1.4% P50M2-0, −0.5% P75M2-0**

The selector path was doing 3 blocking CPU transfers per IFFT chunk (`.to(kCPU)` = `cudaStreamSynchronize`). The existing `_native_select_best_positions_kernel` computes best (row, col) entirely on GPU. Enabled permanently by changing the default in `_pack_problem_data` from `"0"` to `"1"`. Correctness verified via dual-check mode.

---

### B. Lazy valid_zeros — DONE
**Result: −17.4% P50M2-0 (2.292s→1.893s), −15.7% P75M2-0 (4.465s→3.763s)**

Six element-wise GPU ops (`round`, `eq`, `ge×2`, `logical_and×2`) ran unconditionally after each `irfft2` to produce `valid_zeros`. This mask was already handled internally by the CUDA selector kernel and was never consumed in the default path. Moving the six ops into a lazy `compute_valid_zeros` lambda (called only in fallback branches) eliminated ~10–13% of CUDA time per chunk at zero correctness risk.

---

### C. Pinned memory for CPU→GPU transfers — DONE (no measurable impact)
**Result: P50M2-0 1.907s vs 1.908s baseline — within noise**

Added 11 pinned CPU tensors (one per GPU workspace). Rewrote `load_workspace_*` to `memcpy` src → pinned → `out.copy_(pinned, non_blocking=true)`, replacing `torch::from_blob` on pageable heap memory. Also eliminated the intermediate `scratch_rot_i64_`, `scratch_h_i64_`, `scratch_w_i64_` vectors. No speedup because transfers are too small (~10 KB each) for staging overhead to matter on this hardware. The `non_blocking=true` and per-workspace pinned buffers lay groundwork for CPU/GPU overlap under parallel machines (§1).

---

### D. Selector output pre-allocation — DONE (baseline confirmed)
**Result: P50M2-0 1.908s, P75M2-0 3.798s**

1857 `torch::zeros({chunk_n})` calls per evaluate_batch (3 per IFFT chunk for `out_has_t`, `out_row_t`, `out_col_t`) replaced with `ensure_workspace_i32` calls into pre-allocated class-level scratch tensors (`ws_sel_has_i32_`, `ws_sel_row_i32_`, `ws_sel_col_i32_`). Safe because the CUDA selector kernel writes all positions unconditionally. Negligible timing impact but eliminates 1857 GPU allocations per call.

---

### §3a. CHUNK_SIZE = 1500 — DONE (reverted)
**Result: P50 within noise; P75 OOM**

Increasing CHUNK_SIZE from 750 to 1500 doubles `overlap_batch` (`chunk_n × H × W × 4 bytes`). P75M2-0 OOM'd (16 GB exceeded). Reverted to 750. Not worth pursuing without sparse grid allocation (§2) first.

---

## Remaining opportunities

---

### 1. Parallel machine processing — **HIGHEST PRIORITY**
**Expected: 1.5–2.5× on multi-machine instances**
**Effort: Medium**
**Prerequisite: §1a**

Machines are processed sequentially inside `evaluate_batch`. For a 4-machine instance, that's 4 × ~3s = 12s of serial work on fully independent placements. Two threads with two CUDA streams would let Machine 1's CPU work overlap with Machine 0's GPU work:

```
Time →
Machine 0:  [Ph3-CPU][Ph4-GPU·stream0][Ph5-CPU+GPU] ...
Machine 1:           [Ph3-CPU]        [Ph4-GPU·stream1][Ph5] ...
```

**Bottom line:** immediately viable for P50M2 and P75M2 (VRAM fits). Requires sparse allocation (§2) for P100M4+.

#### 1a. Prerequisite: MachineWorkspace refactor

All 30+ `scratch_*` vectors are class-level members shared across machine calls. Running two machines concurrently causes data races. Fix: move them into a `MachineWorkspace` struct, allocate one per thread:

```cpp
struct MachineWorkspace {
    std::vector<int> active, ctx_global, part_idx_local;
    std::vector<GpuPlacement> gpu_updates, phase6_gpu_updates;
    // ... all current scratch_* members
};
std::array<MachineWorkspace, 2> workspaces;
```

This is a mechanical refactor — logic doesn't change. The pinned CPU tensors added in §C also need to move into `MachineWorkspace` to be thread-safe.

---

### 2. Sparse grid allocation — HIGH PRIORITY (required for large instances)
**Expected: enables P100M4+; prerequisite for parallel machines on large grids**
**Effort: High**
**Risk: Medium**

`grid_states` is pre-allocated for `num_solutions × max_bins_per_sol` bins upfront. For P100M4 Machine 3 (400×400): `1000 × 33 × 400 × 400 × 4 bytes = 21 GB` — OOM. In practice, solutions open 6–12 bins on average. A pool allocator that hands out slots on demand reduces peak VRAM 3–5×.

#### 2a. ~~VRAM cap~~ — DONE

`_pack_problem_data` now passes `vram_total_bytes` (from `torch.cuda.get_device_properties`) into the C++ constructor. `process_machine_batch` caps `max_bins_per_sol` so that `grid_states + grid_ffts` stay within 50% of total VRAM (`budget / (num_solutions × bytes_per_bin)`). No impact on small instances (cap doesn't trigger); prevents OOM on large ones.

---

### 3. Phase 4 IFFT reduction — HIGH PRIORITY
**Expected: up to 25% of total time**
**Effort: Medium**
**Risk: Low**

Phase 4 (irfft2) is 52–56% of total time and is the hard GPU bottleneck.

#### 3b. Adaptive FFT size based on grid occupancy

Every IFFT is `irfft2(H × W)` regardless of how full the grid is. Early in a solution's life, only the bottom rows are occupied. We only need `irfft2((max_occupied_row + part_height) × W)` — potentially 2× smaller per call, reducing Phase 4 by ~25% of total time. Challenge: all tests in one batched `irfft2` call must share the same shape, so tests need to be grouped by fill level.

#### 3c. Pre-filter trivially-failing tests

After Phase 3 vacancy filtering, some tests still produce no valid placement (`has_result=0` after IFFT). If a fraction of these can be identified without running the IFFT (e.g. by a simple area or vacancy check), those IFFT calls can be skipped. Even 10–15% filtering reduces Phase 4 proportionally.

---

## What has been dropped / superseded

| Idea | Reason |
|------|--------|
| #3 Vectorize `_decode_sequences` | Done in C++ (`std::sort` + filter) |
| #5 Composite float64 score in Phase 5 | Done in C++ (`lex_better` struct comparison) |
| #7 Phase 5 Python overhead | Done — C++ linear scan |
| #8 Phase 6 numpy allocation | Done — C++ `std::vector::assign` |
| #9 C++/pybind11 for CPU phases | Done — this is `full_native_decoder.py` |
| #10 CUDA streams within a machine | Previously showed 23–29% regression; abandoned |
| #4 Early stopping across machines | Algorithmic change — removed from scope |

---

## Recommended order

**Target: P200M4 and above.** At P200M4, `grid_states` alone requires 16–42 GB depending on grid size — OOM before anything runs. §2 (sparse grid allocation) is therefore a hard prerequisite for the target instances, and must come before §1 (parallel machines).

| # | Action | Impact | Effort | Status |
|---|--------|--------|--------|--------|
| 1 | CUDA selector kernel (§A) | −1.4% P50 / −0.5% P75 | Very Low | **DONE** |
| 2 | Lazy valid_zeros (§B) | −17.4% P50 / −15.7% P75 | Low | **DONE** |
| 3 | Selector output pre-alloc (§D) | Negligible | Very Low | **DONE** |
| 4 | Pinned memory (§C) | Within noise | Low | **DONE** |
| 5 | CHUNK_SIZE = 1500 (§3a) | Within noise / OOM | Very Low | **DONE** (reverted) |
| 6 | VRAM cap in C++ evaluator (§2a) | Safety | Very Low | **DONE** — no impact on small instances, prevents OOM on large |
| 7 | Sparse grid allocation (§2) | **Required for P200M4+** | High | **TODO** — prerequisite for §1 on target instances |
| 8 | MachineWorkspace refactor (§1a) | ~0 alone | Medium | **TODO** — prerequisite for §1 |
| 9 | Parallel machine processing (§1) | **1.5–2.5× multi-machine** | Medium | **TODO** — requires §2 + §1a |
| 10 | Adaptive FFT size (§3b) | Up to 25% Phase 4 | Medium | **TODO** |
| 11 | Pre-filter failing tests (§3c) | 10–15% Phase 4 | Medium | **TODO** |
