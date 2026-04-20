# Parallel Machines Implementation Plan

**Created:** 2026-04-20
**Goal:** Enable multi-machine parallelism (§3.1 in OPTIMIZATION_CATALOG.md) to achieve 1.5–2.5× speedup on multi-machine instances, and unblock large instances (P100M4+, P200M4+) that currently OOM.

**Target sequencing:** Stage 1 (MachineWorkspace refactor) → Stage 2 (sparse grid allocation) → Stage 3 (parallel machines).

---

## Context

The current `full_native_decoder.py` processes machines sequentially inside `evaluate_batch`:
```cpp
for (int m = 0; m < nb_machines_; ++m) {
    auto makespan = process_machine_batch(chrom_ptr, num_solutions, m);
    ...
}
```

Machines are fully independent (each has its own assigned parts, own bin geometry, own grids) but they share two things:

1. **Scratch state on the evaluator class** (~60 `scratch_*` + `ws_*` members) — repurposed per machine, so they cannot be reused concurrently without data races.
2. **GPU VRAM** — `grid_states` for one machine alone can exceed 16 GB on P100M4+ due to the upfront `(num_solutions × max_bins_per_sol, H, W)` allocation.

Removing these two shared-state constraints is a prerequisite for running multiple machines concurrently.

---

## Stage 1 — `MachineWorkspace` struct (plumbing, no behavior change)

### Why this is stage 1

All scratch vectors and GPU workspace tensors are currently class-level members of the evaluator. Any attempt to run two machines concurrently would race on the same buffers. Refactoring these into a per-machine (later: per-thread) struct is a pure plumbing change that must land before any real optimization.

### What changes

1. Create `struct MachineWorkspace` in [full_native_decoder.py](full_native_decoder.py), containing every member currently in lines 579–689:

   **CPU scratch vectors (collections, pair data, FFT batch results):**
   - `CollectedTests scratch_p1_, scratch_p2_`
   - `FFTBatchResult scratch_fft_p1_, scratch_fft_p2_`
   - `std::vector<int> scratch_active_, scratch_ctx_global_, scratch_part_idx_local_`
   - `std::vector<int64_t> scratch_invalid_grid_indices_`
   - `std::vector<BinStateNative*> scratch_invalid_bins_`
   - `std::vector<int32_t> scratch_ctx_first_valid_bin_`
   - `std::vector<uint8_t> scratch_ctx_p1_hit_`
   - `std::vector<int32_t> scratch_test_ctx_local_, scratch_test_bin_local_, scratch_test_rot_global_`
   - `std::vector<uint8_t> scratch_placement_has_`
   - `std::vector<int32_t> scratch_placement_cols_, scratch_placement_rows_`
   - `std::vector<double> scratch_sc_bin_indices_, scratch_sc_densities_, scratch_sc_rows_, scratch_sc_cols_`
   - `std::vector<uint8_t> scratch_sc_valid_`
   - `std::vector<int32_t> scratch_best_ti_`
   - `std::vector<uint8_t> scratch_has_best_`
   - `std::vector<double> scratch_best_neg_bin_, scratch_best_density_, scratch_best_row_, scratch_best_neg_col_`
   - `std::vector<int32_t> scratch_place_ctx_local_, scratch_place_ti_, scratch_place_rows_, scratch_place_cols_`
   - `std::vector<int32_t> scratch_newbin_ctx_local_`
   - `std::vector<GpuPlacement> scratch_gpu_updates_, scratch_phase6_gpu_updates_`
   - `std::vector<int64_t> scratch_phase6_grid_indices_`
   - `std::vector<int32_t> scratch_cell_offsets_, scratch_grid_idxs_, scratch_y_starts_, scratch_x_starts_, scratch_part_widths_, scratch_part_offsets_`
   - `std::vector<int32_t> scratch_pair_vac_row_, scratch_pair_den_off_, scratch_pair_den_len_`
   - `std::vector<int8_t> scratch_pass_`

   **GPU workspace tensors (persistent, pre-allocated for reuse):**
   - `ws_grid_idx_long_, ws_rot_idx_long_, ws_h_long_, ws_w_long_, ws_wave_idx_long_`
   - `ws_cell_offsets_i32_, ws_grid_idxs_i32_, ws_y_starts_i32_, ws_x_starts_i32_, ws_part_widths_i32_, ws_part_offsets_i32_`
   - `ws_gpu_update_packed_, ws_fft_idx_packed_`
   - `ws_vacancy_gpu_`
   - `ws_vac_pairs_gpu_, ws_vac_out_pass_`
   - `ws_gpu_vacancy_recompute_`
   - `ws_fused_product_, ws_gather_out_float_`
   - `ws_sel_has_i32_, ws_sel_row_i32_, ws_sel_col_i32_`

   **Pinned CPU backing tensors:**
   - `ws_cpu_gpu_update_packed_, ws_cpu_fft_idx_packed_`
   - `ws_cpu_pair_buf_, ws_cpu_pass_buf_`
   - `ws_cpu_vacancy_recompute_`
   - `ws_cpu_grid_idx_, ws_cpu_rot_idx_, ws_cpu_h_, ws_cpu_w_, ws_cpu_wave_idx_`
   - `ws_cpu_cell_offsets_, ws_cpu_grid_idxs_, ws_cpu_y_starts_, ws_cpu_x_starts_, ws_cpu_part_widths_, ws_cpu_part_offsets_`

2. Add `std::vector<MachineWorkspace> workspaces_;` to the evaluator class, sized to `nb_machines_` at construction time.

3. Plumb `MachineWorkspace& ws` as a parameter through every method that touches scratch/ws state:
   - `process_machine_batch` (top of call tree — picks `workspaces_[machine_idx]`)
   - `process_wave`
   - `collect_tests_p1`, `collect_tests_p2`
   - `batch_fft_all_tests`
   - `apply_gpu_updates`
   - `recompute_vacancy_gpu`
   - `run_gpu_vacancy_check`
   - `load_workspace_*` helpers (`load_workspace_long_from_i64`, etc.)
   - `ensure_workspace_*` helpers (`ensure_workspace_i32`, `ensure_workspace_cfloat`, `ensure_workspace_float`)
   - `ensure_cpu_pinned_i32`

4. Replace every reference to `this->scratch_X_` / `this->ws_X_` with `ws.scratch_X_` / `ws.ws_X_`.

### What does NOT change

- Read-only problem data (machine geometry, part tables, rotation metadata, density tables, `flat_parts_gpu_`, `density_flat_gpu_`, `machine_ffts_dense_`) stays on the evaluator class — shared across machines, never mutated during a batch.
- Algorithmic flow is identical. One worker, sequential machine loop.
- No threading, no locks, no CUDA streams added yet.

### Milestone

- Fingerprints match baseline exactly on P50M2-0 and P75M2-0 (the two golden instances).
- Wall clock within noise of pre-refactor baseline.
- Instrument a quick sanity check: log `&workspaces_[m]` per machine to confirm per-machine isolation.

### Risks

- **Easy to miss a reference.** A stale `this->scratch_X_` left behind would compile fine and run correctly under Stage 1 (still single-threaded) but cause silent races at Stage 3. Mitigation: after the refactor, grep the entire file for `scratch_` and `ws_` and confirm every hit is either a member of `MachineWorkspace` or an access through `ws.` / `other_ws.`.
- **Constructor cost.** Each `MachineWorkspace` holds ~20 `torch::Tensor` members. These are default-constructed (empty tensors) so the cost is negligible, but the struct is large — allocate via `std::vector::resize(nb_machines_)` at construction, not per-batch.
- **Signature churn.** Many methods gain a `MachineWorkspace& ws` parameter. Risk of accidentally passing the wrong workspace. Mitigation: always pass `ws` as the last parameter, name it consistently, avoid ambiguous overloads.

### Definition of done

- [ ] `MachineWorkspace` struct defined, containing all ~60 members.
- [ ] `workspaces_` initialized in evaluator constructor (sized to `nb_machines_`).
- [ ] All methods that touch scratch/ws state accept `MachineWorkspace&` explicitly.
- [ ] Zero `this->scratch_` or `this->ws_` references remain.
- [ ] Fingerprints P50: `281426.499026`, P75: all `10000000000000000` match baseline.
- [ ] Wall clock P50 ≤ 0.90s/gen, P75 ≤ 1.70s/gen (within noise).

---

## Stage 2 — Sparse (bump) grid allocation

### Why this is stage 2

Stage 1 gives us per-machine isolation of scratch state, but the GPU tensors `grid_states` and `grid_ffts` are still sized upfront at `(num_solutions × max_bins_per_sol, H, W)` per machine. For P100M4 Machine 3 this is 21 GB — OOMs immediately. Sparse allocation reduces the upfront footprint 3-5× and is a prerequisite for fitting two machines' tensors in VRAM simultaneously (Stage 3).

### Observation that simplifies the design

Bins are **never closed** during a `process_machine_batch` call — `ctx.next_grid_idx` only increments, never returns slots to a pool. So the "free-list" described in the original roadmap is overkill. This is really just a **global bump counter** plus a growth fallback.

### What changes

1. **Slot assignment:** in `process_machine_batch`, replace
   ```cpp
   ctx.next_grid_idx = s * max_bins_per_sol;
   ```
   with per-machine-workspace counter:
   ```cpp
   ctx.next_grid_idx = -1;  // lazy — allocated at Phase 6 open-bin time
   ```
   and in Phase 6 (new bin creation, around line 1823):
   ```cpp
   const int grid_idx = ws.global_next_slot_++;
   ```

2. **Upfront sizing:** replace
   ```cpp
   const int max_total_bins = num_solutions * max_bins_per_sol;
   ```
   with a more conservative initial allocation:
   ```cpp
   const int expected_avg_bins = 12;  // tuned from instrumentation
   const int max_total_bins = num_solutions * expected_avg_bins;
   ```
   The VRAM cap logic is removed (or loosened) since we're no longer pre-allocating the worst case.

3. **Growth fallback:** if `ws.global_next_slot_` reaches the current capacity mid-batch, reallocate larger tensors and copy existing slots:
   ```cpp
   void grow_grid_storage(torch::Tensor& grid_states, torch::Tensor& grid_ffts,
                          torch::Tensor& ws_vacancy_gpu, int new_capacity) {
       auto new_states = torch::empty({new_capacity, H, W}, ...);
       new_states.narrow(0, 0, old_capacity).copy_(grid_states);
       grid_states = new_states;
       // same for grid_ffts and ws_vacancy_gpu
   }
   ```
   Trigger: detected in Phase 6 before slot allocation when `ws.global_next_slot_ + num_new_bins > capacity`. Grow by 2× (amortized O(1) per insert).

4. **Instrumentation (temporary, for tuning):** log `ws.global_next_slot_` high-water mark per machine per `process_machine_batch` call. Used to pick `expected_avg_bins` and validate that growth events are rare.

### What does NOT change

- Indexing semantics. `grid_idx` is still a global slot index into `grid_states`; all `index_select` / `index_copy_` / CUDA kernel calls work unchanged.
- Per-bin geometry (H, W), vacancy buffer shape per bin, FFT shape per bin.
- BinState layout. `grid_state_idx` field unchanged.
- Algorithmic behavior.

### Milestone

- Fingerprints match baseline exactly on P50M2-0, P75M2-0.
- P100M4-0 fits in VRAM and runs to completion (new capability).
- VRAM high-water mark reduced 2-4× on multi-machine instances.
- Wall clock P50/P75 flat (no regression — allocation isn't on critical path, but growth events could hurt if mis-tuned).
- Log confirms < 1% of batches trigger a growth event.

### Risks

- **Growth event cost.** A mid-batch reallocation copies `capacity × H × W × 4` bytes on GPU (~100 MB for mid-sized instance). If growth fires often, wall clock regresses. Mitigation: pick `expected_avg_bins` on the generous side (e.g., 1.5× measured peak), 2× growth factor so growth fires at most once per batch.
- **Reduced VRAM cap protection.** Removing the upfront VRAM cap means we could OOM during growth. Need a hard ceiling (e.g., 70% of total VRAM) and graceful failure with a clear error. Keep `vram_total_bytes_` check as a safety on growth calls.
- **Parallel-machine interaction (preview for Stage 3).** When two machines run concurrently, their allocations compete for the same VRAM pool. Stage 2's sizing must leave headroom for a second concurrent machine. Tune conservatively — `expected_avg_bins × 2 machines` must fit.
- **Slot ordering.** In the old scheme, `ctx.next_grid_idx` was deterministic per solution. In the bump scheme, slot assignment depends on Phase 6 iteration order. Should still be deterministic (Phase 6 iterates `newbin_ctx_local_` in order) but needs verification via fingerprints.

### Definition of done

- [ ] `global_next_slot_` counter added to `MachineWorkspace`, reset at start of `process_machine_batch`.
- [ ] Phase 6 acquires slots via `ws.global_next_slot_++`.
- [ ] Upfront `grid_states`/`grid_ffts`/`ws_vacancy_gpu_` sized to `num_solutions × expected_avg_bins`.
- [ ] Growth fallback implemented and tested (forcibly trigger via small initial capacity).
- [ ] P50/P75 fingerprints + wall clock match.
- [ ] P100M4-0 runs without OOM (was OOM'ing before).
- [ ] Growth event rate logged and < 1% of batches on tuned instances.

---

## Stage 3 — Parallel machine processing

### Why this is stage 3

With per-machine workspaces (Stage 1) and VRAM-efficient allocation (Stage 2), we can now run machines concurrently on separate CUDA streams and CPU threads without races or OOMs.

### What changes

1. **Thread pool:** replace the sequential loop in `evaluate_batch`:
   ```cpp
   for (int m = 0; m < nb_machines_; ++m) {
       machine_makespans[m] = process_machine_batch(chrom_ptr, num_solutions, m);
   }
   ```
   with a `std::thread` / `std::async` dispatch across ≤ `num_workers` threads (2 on A4000, tunable).

2. **CUDA streams:** each worker thread owns a `torch::cuda::CUDAStream` stored as `MachineWorkspace::stream_`. All GPU ops in `process_machine_batch` for that machine submit to its stream via `torch::cuda::CUDAStreamGuard`.

3. **Machine-to-worker assignment:** balance by estimated machine cost (larger grid = more work). Simple heuristic: sort machines by `H × W` descending, round-robin assign to workers. For 4 machines / 2 workers: worker 0 gets M0+M3, worker 1 gets M1+M2 (longest-first across workers).

4. **Synchronization:** `evaluate_batch` waits for all workers to finish before collecting makespans. Use `std::future::get()` or `std::thread::join()`. Each worker's final ops on its stream must be synced via `stream.synchronize()` before returning.

5. **GIL release.** The entire `evaluate_batch` call must release the GIL (PyBind: `py::gil_scoped_release`) so threads run truly concurrently. No Python callbacks during `process_machine_batch`.

6. **Deterministic result ordering.** `machine_makespans` must be indexed by machine_idx regardless of which thread processed which machine. Each worker writes to `machine_makespans[m]` directly; no reordering needed.

### What does NOT change

- Per-machine algorithmic flow. Each machine still runs its own wave loop, Phase 1–6, placements, etc.
- Chromosome data (read-only, shared safely across threads).
- Part/machine metadata (read-only).
- `flat_parts_gpu_`, `density_flat_gpu_`, `machine_ffts_dense_` (read-only on GPU; PyTorch tensor reads are thread-safe with proper stream ordering).

### Milestone

- Fingerprints match baseline exactly (results are order-independent across machines).
- P50M2-0 / P75M2-0 wall clock within noise (no regression on 2-machine instances with only 2 workers — negligible dispatch overhead).
- P100M4-0 gen time drops to ≈ max(single-machine times) + overhead, roughly 1.5-2× vs sequential.
- `nvidia-smi` shows increased GPU utilization during overlapping phases.

### Risks

- **Torch CUDA caching allocator cross-thread weirdness.** The allocator is stream-aware but has surfaced odd interactions in past experiments (see §2.5 "Multi-stream Phase 2 rfft2" — CUDA device-side assert). Must carefully use `CUDAStreamGuard` and never access a tensor from a stream that didn't produce it without explicit `recordStream()`.
- **GIL / Python callbacks.** If `process_machine_batch` or any helper invokes Python (via pybind11 casting of non-trivial objects), threads will serialize on the GIL and eat the parallelism win. Audit every call path for implicit Python calls.
- **cuFFT plan cache contention.** The plan cache is global per-device. Concurrent `rfft2`/`irfft2` from different streams may serialize at the cache lock. If observed, consider per-stream private caches (currently 4096 entries suffice for single-stream).
- **Thread-safe logging.** If any `std::cout` / `printf` appears in hot paths, it'll serialize. Should be gated behind profiling flags only.
- **Uneven machine loads.** For `nb_machines_ == 2` (P50M2, P75M2), with 2 workers, each worker does 1 machine — simplest case. For `nb_machines_ == 4` (P100M4), 2 workers each do 2 machines. For odd counts, one worker is idle for part of the time.
- **Pinned memory per workspace.** Each `MachineWorkspace` has ~15 pinned CPU tensors. With 2 workspaces processed concurrently, 2× the pinned buffers in flight. Size is small (~a few MB total) but verify no host-memory blowup.

### Definition of done

- [ ] `std::thread` or `std::async` thread pool dispatches `process_machine_batch` per machine.
- [ ] Each worker uses a dedicated `CUDAStream` bound via `CUDAStreamGuard`.
- [ ] `evaluate_batch` releases the GIL for the duration.
- [ ] Fingerprints match on P50/P75.
- [ ] P100M4-0 runs with ≥ 1.5× speedup vs Stage 2 baseline.
- [ ] No CUDA asserts, allocator warnings, or hangs across a 10-generation run.

---

## Testing strategy

Each stage validates on three axes before moving on:

1. **Correctness:** fingerprint match on P50M2-0 (`281426.499026`) and P75M2-0 (all `10000000000000000`). Zero tolerance for drift.
2. **Performance (no-regression on small instances):** wall clock P50 ≤ 0.90s/gen, P75 ≤ 1.70s/gen (current baseline is 0.87/1.63 per recent measurements).
3. **Scalability (new capability):** applicable at Stages 2 & 3:
   - Stage 2: P100M4-0 must run (currently OOMs).
   - Stage 3: P100M4-0 must show ≥ 1.5× speedup vs Stage 2 baseline.

Run procedure per stage:
```bash
# Sync
python remote.py sync . BRKGA_FFT2 --ext .py

# Correctness + wall clock on primary benchmarks
python remote.py run "ABRKGA_FULL_NATIVE_DECODER=1 python BRKGA_alg3.py 50 2 0 torch_gpu native_full 1 1 5" \
    --cwd /notebooks/BRKGA_FFT2 --timeout 900
python remote.py run "ABRKGA_FULL_NATIVE_DECODER=1 python BRKGA_alg3.py 75 2 0 torch_gpu native_full 1 1 5" \
    --cwd /notebooks/BRKGA_FFT2 --timeout 900

# Stages 2+: scalability
python remote.py run "ABRKGA_FULL_NATIVE_DECODER=1 python BRKGA_alg3.py 100 4 0 torch_gpu native_full 1 1 3" \
    --cwd /notebooks/BRKGA_FFT2 --timeout 1800
```

Each stage commits only after all three gates pass.

---

## Rollback strategy

Each stage is a separate commit (or coherent commit series). If a stage fails review or introduces issues not caught in testing, `git revert` drops it cleanly without affecting earlier stages.

**First-run compilation:** any changes to C++/CUDA source in `full_native_decoder.py` invalidate the `torch_extensions` cache. After each stage's first remote run, factor in 2-3 min compile time. Clear cache only if structural errors: `rm -rf /root/.cache/torch_extensions/py*/`.

---

## Out of scope

Following items are discussed elsewhere and not part of this plan:

- Adaptive FFT size based on occupancy (§3.3 in catalog) — independent optimization.
- Pre-filter trivially-failing IFFT tests (§3.4) — independent.
- BRKGA outer loop profiling (§3.13) — independent; user declined algorithmic changes.
- Early pruning across machines (§3.5) — algorithmic change; user declined outer-loop changes.
- `CHUNK_SIZE` retuning (§3.11) — worth retesting after Stage 2 frees VRAM, but separate work.

---

## Progress tracker

| Stage | Status | Commit | Notes |
|-------|--------|--------|-------|
| 1 — MachineWorkspace refactor | Not started | — | Plumbing only, no behavior change |
| 2 — Sparse grid allocation | Not started | — | Depends on Stage 1 |
| 3 — Parallel machines | Not started | — | Depends on Stages 1 & 2 |

Update this table as each stage lands.
