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
| 1 — MachineWorkspace refactor | **Done** | `9a4441f` | Plumbing only, no behavior change |
| 2 — Sparse grid allocation | **Split into 2a–2e (2026-04-21)** | — | See "Stage 2 — Revised plan" below |
| 2a — Rebase golden + high-water | **Done 2026-04-21** | pending commit | Baselines P50=0.890s, P75=1.658s, peak=6 bins/sol |
| 2b — observer counter | **Done 2026-04-21** | pending commit | Fingerprint exact match, P50=0.877s, P75=1.638s |
| 2c — bump-counter slots | **Done 2026-04-21** | pending commit | Fingerprint exact match, P50=0.881s, P75=1.639s — slot layout confirmed opaque |
| 2d — shrink + growth | Not started | — | |
| 2e — P100M4-0 scale test | Not started | — | |
| 3 — Parallel machines | Not started | — | Depends on Stages 1 & 2 |

---

## Stage 2 attempt — findings (2026-04-20)

Implemented full sparse allocation (bump counter + 2× growth fallback +
smaller initial `EXPECTED_AVG_BINS=16` cap).  Validated via
`profile_cpu_hotspots.py 50 2 0 torch_gpu 5` (seed 123, 5 reps — fingerprint
is first 5 makespans of the last rep).

### What went wrong

1. **Fingerprint divergence even for behavior-equivalent variants.**
   Preserving the exact old slot formula (`ctx.next_grid_idx = s *
   max_bins_per_sol`) while additionally tracking `ws.global_next_slot_` as a
   high-water observer also diverged.  This suggests the slot *layout* in
   `grid_states`/`grid_ffts` is not as opaque as expected — something
   downstream (Phase 3/4/5) produces different fitness values when
   grid-state indices differ, even though every direct use appears to be
   `index_select`/`index_copy_`.  Need to instrument to find what.

2. **The "golden" fingerprint in the plan (`281426.499026`) is stale.** Both
   current Stage-1 baseline (commit `9a4441f`) and the pre-Stage-1 commit
   (`066c3ca`) produce `['1e16', '315736.048454' | '312669.888854', '1e16',
   '1e16', '1e16']` — the 2nd makespan differs *between* those two commits
   as well.  This means Stage 1 already changed the fingerprint (despite
   BRKGA-loop validation passing) and the catalog reference is from an
   older optimization era.

   **Action before Stage 2 resumes:** confirm which baseline the user
   considers golden.  Options:
   - Accept `9a4441f` Stage-1 fingerprint as the new golden; validate any
     Stage 2 candidate against it.
   - Investigate the Stage 1 drift (see §Stage 1 remaining risk below).

3. **Growth/VRAM interactions on P50.** With 50% VRAM cap, P50 on A4000
   yields `max_bins_per_sol ≈ 13`; an over-eager growth trigger (fire on
   `global_next_slot_ + new_count > capacity`) caused doubling to ~13000
   slots mid-batch → 4.86 GiB allocation → OOM.  Growth logic must track
   "next *new* slot to be assigned" separately from "max slot index ever
   touched" to avoid this.

4. **Per-solution cap semantics are asymmetric in old code.** OLD:
   `ctx.next_grid_idx = s * max_bins_per_sol`; check `grid_idx >=
   max_total_bins`. So solution 0 may open up to `max_total_bins` bins
   (colliding into other sols' ranges), while solution `n-1` is capped at
   `max_bins_per_sol`.  In practice no solution exceeds ~12 bins, so the
   cap rarely fires.  Stage 2 must decide: (a) preserve this asymmetry
   (doesn't save VRAM), or (b) replace with a uniform per-sol cap +
   accept the fingerprint change.

### What to try next session

1. **Instrument grid_state_idx usage to find the non-opaque consumer.**
   Add a runtime assertion that writes and reads to each slot agree with
   the stored `grid_state_idx`.  Grep `grid_state_idx` and audit every
   use; check whether any downstream code assumes slot order matches
   solution order (e.g., batched gather indices sorted assumption, or a
   kernel that uses slot index for a tiebreaker).

2. **Rebase the "golden" reference.** Run
   `profile_cpu_hotspots.py 50 2 0 torch_gpu 5` and
   `profile_cpu_hotspots.py 75 2 0 torch_gpu 5` on `9a4441f` and commit
   those numbers as the new baseline fingerprint.  Re-verify Stage 1
   drift is acceptable (or debug it first if not).

3. **Minimal-risk Stage 2 alternative.** If slot layout truly matters,
   keep the OLD formula (no compaction) but lower the initial capacity
   to observed peak + 20% headroom with growth-on-overflow.  This saves
   VRAM only when VRAM cap was previously forcing `max_bins_per_sol` way
   below `needed_bins` — i.e., large instances.  Confirm via
   instrumentation that growth rarely fires on P50/P75.

4. **Split the growth trigger.** Use two separate bookkeeping values:
   `ws.peak_bin_index_` (max slot index ever *written* — used for growth
   check) vs `ws.new_bins_this_wave_` (count of bins about to be
   allocated this wave).  Fire growth only when
   `peak_bin_index_ + 1 + new_bins_this_wave_ > capacity`, not on an
   observer counter that already trails near capacity.

### Stage 1 remaining risk

The `profile_cpu_hotspots.py` fingerprint for `9a4441f` differs from
`066c3ca`:
- `066c3ca` 2nd makespan: `312669.888854`
- `9a4441f` 2nd makespan: `315736.048454`

Stage 1 validation was via `BRKGA_alg3.py` Gen 0–4 best-fitness match,
which did match.  But the `profile_cpu_hotspots.py` single-seed
`evaluate_batch` result diverges — meaning some chromosome's fitness
changed in Stage 1 even though the best fitness didn't.  This is either
a pre-existing non-determinism surfaced by the refactor (pinned-memory
data races, uninitialized workspace read, etc.) or a subtle Stage 1 bug.

Need to bisect/inspect before trusting any Stage 2 fingerprint.

**Update 2026-04-21:** the Stage 1 drift was almost certainly the CUDA
async race on reused pinned buffers (see [CROSS_ROW_LEAKAGE_FINDINGS.md](CROSS_ROW_LEAKAGE_FINDINGS.md)),
which was latent in both `066c3ca` and `9a4441f` and surfaced differently
under the refactored call pattern.  With the fix landed, the native
decoder is now validated element-wise vs pp for the full seed-42
population of P50M2-0 (500/500 chroms match,
`verify_full_population_vs_pp.py` 2026-04-21).  The correctness baseline
is therefore trustworthy going forward and we can rebase the Stage 2
golden against the post-fix `main`.

---

## Stage 2 — Revised plan (2026-04-21)

The prior single-landing attempt failed on three independent axes
(fingerprint divergence, VRAM cap interaction, ambiguous golden).  To
isolate failure modes, Stage 2 is broken into four sub-stages.  Each
sub-stage gates on `verify_full_population_vs_pp.py` passing (500/500
element-wise match on P50M2-0 seed 42) before proceeding to the next.

### Sub-stage 2a — Rebase the golden reference

**Code change:** minimal — `get_bin_stats()` / `reset_bin_stats()`
pybind methods + a `peak_bins_per_sol_` counter on `MachineWorkspace`,
updated at the end of every `process_machine_batch`.  Purely observer,
does not touch decoder logic.

Run on current post-fix `main`:
- `verify_full_population_vs_pp.py` → 500/500 match ✓
- `profile_cpu_hotspots.py 50 2 0 torch_gpu 5`
- `profile_cpu_hotspots.py 75 2 0 torch_gpu 5`

#### Baseline captured 2026-04-21 (post-fix `main`)

Fingerprint policy: "first 5 **feasible** makespans" (i.e., `< 1e15`).
Infeasible `1e16` entries are filtered out — they carry no
discriminative signal and would mask real drift if they dominate.

**P50M2-0** (seed 123, 5 reps, pop=500):
- Wall clock: **0.890s/gen** mean, std 0.012s
- Feasible: 241 / 500
- Fingerprint (last rep, first 5 feasible):
  `['368832.376454', '373724.029669', '344097.246008', '288449.665904', '315764.072077']`
- Bins-per-solution — M0: peak=6 avg=2.87; M1: peak=5 avg=3.02

**P75M2-0** (seed 123, 5 reps, pop=750):
- Wall clock: **1.662s/gen** mean, std 0.015s
- Feasible: 38 / 750
- Fingerprint (last rep, first 5 feasible):
  `['369976.676340', '321364.609932', '327184.488143', '377398.063238', '375592.968978']`
- Bins-per-solution — M0: peak=6 avg=2.04; M1: peak=6 avg=3.52

**Full-population element-wise vs pp (P50M2-0 seed 42, 500 chroms):**
500/500 match (max |diff| = 0.0).

#### Implications for sub-stage 2d sizing

Measured peak across P50/P75 is 6 bins/sol.  Current upfront
allocation uses `needed_bins = max(10, nb_parts / 3)` — for P50 that's
16, for P75 that's 25.  A conservative `expected_avg_bins` that still
gives headroom is peak × 1.5 ≈ 10, which is **1.6× smaller for P50
and 2.5× smaller for P75**.  Those are the VRAM savings 2d unlocks.

**Gate:** ✓ Passed.  Baseline fingerprints recorded above; high-water
data captured.  Instrumentation overhead is within noise (P50: 0.886 →
0.890s/gen, P75: 1.644 → 1.658s/gen, both <1% slower and within std).

### Sub-stage 2b — Add `global_next_slot_` counter with OLD sizing

**Code change:** plumbing only.  Add `int global_next_slot_` to
`MachineWorkspace`, reset at the start of `process_machine_batch`,
incremented in Phase 6 when a bin is opened.  **Preserve** the existing
slot formula (`ctx.next_grid_idx = s * max_bins_per_sol`) and the
existing upfront `max_total_bins = num_solutions * max_bins_per_sol`
allocation.  The counter is an observer, not the slot source.

**Rationale:** if fingerprints drift here, the regression is purely in
the refactor plumbing — NOT in slot layout, sizing, or growth logic.
Prior attempt showed the slot layout is not opaque to something
downstream, so this is the cleanest way to confirm the counter itself
is inert.

**Gate:** ✓ Passed 2026-04-21.
- `verify_full_population_vs_pp.py`: 500/500 match
- P50M2-0 fingerprint: exact match with 2a baseline; wall clock
  **0.877s/gen** (vs 0.890s baseline — within noise)
- P75M2-0 fingerprint: exact match with 2a baseline; wall clock
  **1.638s/gen** (vs 1.662s baseline — within noise)
- Peak `global_next_slot_` observed per batch: P50 M0=1446, M1=1517;
  P75 M0=1554, M1=2657.  These match `total_bins ÷ num_batches` exactly
  (e.g., P50 M0: 7172/5 = 1434.4 avg, 1446 peak — consistent with
  counter correctness).

### Sub-stage 2c — Switch Phase 6 to bump-counter slot assignment

**Code change:** Phase 6 opens new bins via `ws.global_next_slot_++`
instead of `ctx.next_grid_idx++`.  Upfront allocation stays at
`num_solutions × max_bins_per_sol`, so no growth logic yet.

This was the sub-stage most feared to surface the "non-opaque slot
consumer" bug seen in the prior single-landing attempt.

**Gate:** ✓ Passed 2026-04-21 on the first try.
- `verify_full_population_vs_pp.py`: 500/500 match
- P50M2-0: fingerprint exact match, wall clock **0.881s/gen**
- P75M2-0: fingerprint exact match, wall clock **1.639s/gen**

**Conclusion:** the slot layout **is** opaque.  Every
`grid_state_idx` consumer in the codebase (gather kernels,
`index_select`, vacancy buffer, FFT scatter) treats the index as an
opaque handle.  The prior Stage 2 attempt's divergence was the CUDA
async race on reused pinned buffers (fixed
2026-04-21; see [CROSS_ROW_LEAKAGE_FINDINGS.md](CROSS_ROW_LEAKAGE_FINDINGS.md)),
not a slot-layout dependency.  2d is now unblocked to reduce the
upfront allocation.

### Sub-stage 2d — Shrink initial capacity + growth fallback

**Code change:** replace `max_bins_per_sol` upfront sizing with
`expected_avg_bins` (tuned from 2a high-water data, e.g., 1.5× measured
peak).  Implement 2×-growth reallocation of `grid_states`, `grid_ffts`,
`ws_vacancy_gpu_` when `global_next_slot_ + new_bins_this_wave_ >
capacity`.  Keep a hard VRAM ceiling (~70% total) as a safety.

Key fix from prior attempt: separate `peak_bin_index_` (for growth
check) from `new_bins_this_wave_` (for headroom).  Fire growth only
when `peak_bin_index_ + 1 + new_bins_this_wave_ > capacity`, not on an
observer counter that already trails capacity.

**Gate:** `verify_full_population_vs_pp.py` passes 500/500.  Log shows
< 1% of batches trigger growth.  Wall clock flat on P50/P75.

**Status: LANDED 2026-04-21.** Initial capacity shrunk from `num_solutions × needed_bins`
(P50: 8000, P75: 18750) to `num_solutions × min(needed_bins, EXPECTED_AVG_BINS=10)`
(P50: 5000, P75: 7500). Added `grow_grid_storage()` helper that
`narrow+copy`-reallocates `grid_states`, `grid_ffts`, `ws_vacancy_gpu_`
with doubling and a 70%-VRAM hard ceiling. Growth check fires before
Phase 6 when `ws.global_next_slot_ + new_count > max_total_bins`.

Gate results:
- pp element-wise: **500/500** match (|diff|<1.0), max |diff|=0.0 ✓
- P50M2-0 fingerprint: exact match, **0 growths**, peak_global 1517 (<< 5000 capacity),
  wall 0.883s (vs 0.881s 2c baseline — flat) ✓
- P75M2-0 fingerprint: exact match, **0 growths**, peak_global 2657 (<< 7500 capacity),
  wall 1.640s (vs 1.639s 2c baseline — flat) ✓

Zero growth events on both instances at steady-state population confirms
`EXPECTED_AVG_BINS=10` is well-sized for the seed-123 profile. Growth
path exists only as a safety net for larger instances / outlier batches.

### Sub-stage 2e — Scale-test P100M4-0

**Code change:** loosen/remove the upfront VRAM cap (now unnecessary
since allocation tracks actual usage).  Confirm P100M4-0 fits and runs
to completion — the new capability that justifies Stage 2 in the first
place.

**Gate:** P100M4-0 runs without OOM.  VRAM high-water 2-4× lower than
pre-Stage 2.

### Ordering rationale

- 2a alone gives us trust in the gate.
- 2b isolates plumbing drift.
- 2c isolates slot-layout drift.
- 2d isolates growth/sizing drift.
- 2e proves the new capability.

If any gate fails, the regression is contained to one sub-stage's
change set and easy to `git revert`.
