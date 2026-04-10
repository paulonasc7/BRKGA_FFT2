# CPU-Side Optimization Candidates — full_native_decoder

**Created:** 2026-04-09
**Goal:** identify and rank CPU-bound optimization opportunities in `full_native_decoder.py` after ideas #1, #2, #5 from `NATIVE_OPTIMIZATION_IDEAS.md` are done and idea #6 was rejected.
**Context:** Profiling data in `NATIVE_OPTIMIZATION_IDEAS.md` shows **~34.7% of wall time (~675ms/seed) is CPU-only** (invisible to the GPU profiler). This document drills into that 34.7% to find concrete, measurable targets.

**Current baseline (P50M2-0, 500 ind.):** 1.103s/gen (after SIMD UVR + template grid write + bin pool). Previous: 1.487s/gen (−26%).
**Current baseline (P75M2-0, 750 ind.):** 2.173s/gen. Previous: 3.227s/gen (−33%). vs wave_batch: **2.15×** (up from 1.49× at session start).

---

## OPTIMIZATION HISTORY

### #1 SIMD `update_vacancy_rows_cpp` — DONE 2026-04-09
Replaced scalar row scan with runtime-dispatched AVX2 implementation using `_mm256_cmpeq_epi8` + `_mm256_movemask_epi8` + `__builtin_ctz` for branchless run-length-max over uint8 bytes. Kill-switch: `ABRKGA_NATIVE_UVR_SIMD=0`.

| Metric | Scalar | AVX2 | Speedup |
|---|---|---|---|
| `update_vacancy_rows` ms/gen | 430.9 | 89.1 | **4.84×** |
| per-row cost (ns) | 263 | 54 | 4.87× |
| wall clock ms/gen | 1499 | 1212 | 1.24× (−287 ms) |

Correctness verified: makespan fingerprints byte-identical between scalar and SIMD paths on 5-rep P50M2-0 run. The SIMD version shifts `update_vacancy_rows` out of the top slot.

### #4 Phase 6 bin pool — DONE 2026-04-10
Added `bin_pool_` (up to 4096 `BinStateNative` entries). After each `process_machine_batch` call, bins are moved into the pool instead of being destroyed. Phase 6 new-bin creation pops from the pool (if H/W match) and calls `std::memset` + `std::fill` on warm memory instead of `grid.assign` (which triggers OS page faults on first-touch).

Partial-zero variant (zero only dirty rows using `min_occupied_row`) was tested and was slightly slower — the conditional + potential cache miss on `min_occupied_row` outweighed the reduced memset volume. Kept full memset.

| Metric | Before pool | After pool | Delta |
|---|---|---|---|
| BinState ctor ms/gen | 31.5 | 22.2 | −9.3 ms |
| Phase 6 loop ms/gen | 46.7 | 37.9 | −8.8 ms |
| wall clock ms/gen (P50) | 1119 | 1103 | −16 ms (−1.4%) |
| wall clock ms/gen (P75) | 2.162s | 2.173s | within noise |

The P50 saving is real but modest because page faults were not the only cost — warm memset of 221MB/gen still takes ~5-6ms at DRAM bandwidth. P75 sees no meaningful change. Pool is still worth keeping (eliminates allocator churn and smooths variance).

### #3 Template + memcpy/AVX2-add grid write in `add_part_to_bin_cpu` — DONE 2026-04-10
Replaced the `if (overwrite)` branch inside the inner `(rr, cc)` loop with a `template <bool Overwrite>` instantiation. For `Overwrite=true`, inner loop becomes `std::memcpy(dst, src, pw)` — glibc uses NT stores and prefetch. For `Overwrite=false`, a runtime-dispatched `grid_add_row_avx2` uses `_mm256_add_epi8` over 32 bytes/iteration with scalar tail.

| Metric | Before | After | Delta |
|---|---|---|---|
| `add_part_to_bin_cpu` ms/gen | 200 | 142 | −58 ms |
| non-UVR grid write ms/gen | 111 | 53 | −58 ms (−52%) |
| wall clock ms/gen | 1212 | 1119 | **−93 ms (−7.7%)** |

Correctness verified: fingerprint unchanged on 5-rep P50M2-0 run.

### #2 Vacancy upload sync — INVESTIGATED 2026-04-10, SKIPPED

Measured: 97 ms/gen (8% of 1212ms wall clock). The sync at line 1534 of `run_gpu_vacancy_check` (`gpu_pairs.copy_(cpu_pairs, non_blocking=false)`) blocks CPU until the GPU drains. Per-call stats: 136.2 calls/gen, 0.71 ms per call.

**Why there's nothing cheap to gain:**

The 0.71ms per-call wait is mostly real GPU execution time, not preventable stall. The wave sequence is:
1. Phase 2: rfft2 fires async on GPU (~0.7 ms GPU time per wave for that batch)
2. Pass A: CPU loop builds pair list (~0.076 ms of CPU work)
3. Sync: upload pairs + run vacancy kernel. CPU waits the remaining GPU time.

When `batch_fft_all_tests` at the end of each wave reads results back (`.to(torch::kCPU)`), it fully drains the stream. So the stream is clean entering Phase 2 of the next wave. Pass A (0.076ms CPU) is too short to hide rfft2 (0.7ms GPU) — hence ~0.63ms of wait.

Making the upload `non_blocking=True` + CUDA event deferred sync would shift the wait from upload to readback, but the total wait doesn't change — we'd still block for ~0.71ms per call at readback. Net saving: ~0 ms.

**Reducing calls per wave** (merge p1+p2 into one check) is the only structural fix, but it's a significant algorithm restructure with ~48 ms/gen upside — bounded by the fact that p2 pairs are a subset of p1-miss contexts only.

**Decision: skip.** Cost-to-benefit ratio is unfavorable. Moving on to `add_part_to_bin_cpu` non-UVR work (~111 ms/gen).

---

## MEASURED RESULTS (2026-04-09) — ground truth from instrumented run

Ran `profile_cpu_hotspots.py` on P50M2-0 with 5 reps, seed 123. Instrumentation is a local uncommitted change using `std::chrono::steady_clock` around the hot spots; overhead is <5 ms/gen (well under noise floor).

| Hot spot | Per-gen | % of wall | My pre-profile guess |
|---|---|---|---|
| `update_vacancy_rows_cpp` | **383.4 ms** | **25.8%** | 100–200 ms (**underestimated**) |
| `add_part_to_bin_cpu` (total, incl. uvr) | 502.5 ms | 33.8% | — |
| `add_part_to_bin_cpu` (excl. uvr) | ~119 ms | 8.0% | — |
| Phase 6 new-bin loop | 106.3 ms | 7.1% | 100–300 ms (roughly right) |
| └ BinState ctor (grid.assign + vacancy.assign) | 33.2 ms | 2.2% | 220 ms (**7× overestimate**) |
| Vacancy upload sync (`non_blocking=false`) | 99.0 ms | 6.7% | — (this specific copy wasn't isolated) |
| Phase 3 Pass B total wait | 104.6 ms | 7.0% | 50–120 ms (right ballpark) |
| Vacancy readback sync | 6.9 ms | 0.5% | — |
| Phase 3 Pass A | 4.8 ms | 0.3% | 50–100 ms (**dead target**) |
| Phase 3 Pass C | 2.9 ms | 0.2% | 50–100 ms (**dead target**) |

**Event counts (per gen):**
- 2,944 new `BinStateNative` allocations (~10/wave), not 100/wave as estimated
- 21,614 `add_part_to_bin_cpu` calls
- 1.64M vacancy rows scanned at 234 ns/row
- 136 vacancy-check invocations per gen (2 per wave × ~68 waves)

### Key revisions to the initial plan

1. **`update_vacancy_rows_cpp` is the #1 target by a wide margin.** Not Phase 6 bin allocation. Single function eats 25.8% of wall time.
2. **Phase 3 Pass A/C merge is dead.** 8 ms/gen combined, not 100–200 ms. Don't touch.
3. **Phase 6 bin pool is demoted to #3.** BinState ctor is 33 ms, not 220 ms. New-bin rate is 10/wave not 100/wave.
4. **Vacancy upload sync (99 ms/gen) is the new #2.** The upload is the expensive sync, not the readback. Upload is `non_blocking=false` + the kernel is waiting on it, so CPU blocks until GPU drains.
5. **Vacancy readback sync is 6.9 ms/gen.** Skip — near noise.

---

## 1. Phase 6 bin allocation (pool / lazy grid) — TOP CANDIDATE

**Expected saving:** 100–300ms/seed
**Effort:** Medium
**Risk:** Medium
**Confidence:** High (simple arithmetic, well-localized)

### Where it lives

- `BinStateNative` struct definition: `full_native_decoder.py` around line 38
- New-bin creation in `process_wave`: Phase 6 section around lines 1861–1922
- `add_part_to_bin_cpu(..., overwrite=true)` is called on each new bin: line 1826

### What happens today

Each wave creates new `BinStateNative` objects for contexts that failed to find a fit in any existing open bin. For each new bin:

1. `BinStateNative` default-constructs `grid` as a `std::vector<uint8_t>` sized H×W = **75,000 bytes** of zero-initialized memory.
2. `vacancy` is sized to H = 300 × `int32_t` = 1,200 bytes, also zero-initialized.
3. The first part is written via `add_part_to_bin_cpu(..., overwrite=true)` into the fresh grid.

**Per-seed cost estimate:**
- ~100 new bins/wave × 300 waves = **~30,000 new `BinStateNative` allocations/seed**
- 30,000 × 75 KB memset = **~2.2 GB of CPU memset/seed**
- At ~10 GB/s sustained memset bandwidth → **~220 ms/seed** spent just zero-initializing grids that get immediately overwritten

This is plausibly the single biggest untouched CPU cost. None of the prior ideas (#1–#5) touched it.

### Why it's on the critical path

New bins must exist before Phase 6 can write parts into them. Phase 6 runs on the CPU thread between GPU Phase 5 kernels and the next wave's Phase 2 rfft2 launches — there's no GPU work to hide it behind at that exact point (GPU is idle waiting for Phase 2 inputs).

### Optimization candidates

**A. Object pool / arena allocation:**
- Pre-allocate a pool of `BinStateNative` objects with pre-sized `grid` and `vacancy` buffers. Recycle them across waves (when a bin is closed, return it to the pool).
- Still need `memset` on reuse, but the allocator churn goes away. Saves ~20–40ms.

**B. Lazy grid zeroing:**
- Don't zero `grid` on allocation. Track a "clean" flag that gets set to `true` after the next full wipe.
- On first placement, just write the part values (they're uint8 so you know what to write).
- The grid is read by `update_vacancy_rows_cpp` which scans the *entire* width — so you can't be lazy about the rows that overlap the part. But rows the part doesn't touch don't need to be written at all until a second part lands there.

**C. Eliminate grid zeroing by using a row-touched bitmap:**
- Replace `bin.grid` with a sparse representation for new bins: only materialize rows that have been touched.
- First part: write only `ph` rows, compute vacancy for only those rows, leave the rest as "all zeros" (vacancy = W).
- This is a bigger refactor but could eliminate the 75 KB zero-init entirely for fresh bins.

**D. Pool + lazy combination:**
- Pool the `BinStateNative` objects (A), and track a `grid_dirty_rows` bitmap so reused bins only memset the dirty rows from the previous user.

### Verification plan

Before implementing: add `std::chrono` instrumentation around the Phase 6 new-bin creation loop and around each `BinStateNative` constructor in a debug build. Run one seed and report:
- Total time in Phase 6 new-bin creation
- Total time in `BinStateNative` constructors
- Number of new bins created per seed
- Per-bin construction time

If the measured cost is <50ms/seed, deprioritize. If it's >150ms/seed, proceed with option A+B.

---

## 2. SIMD `update_vacancy_rows_cpp`

**Expected saving:** 100–200ms/seed
**Effort:** Low
**Risk:** Low
**Confidence:** High — the codebase already has AVX512/AVX2 infrastructure

### Where it lives

- `update_vacancy_rows_cpp`: `full_native_decoder.py` lines 838–862
- Called from `add_part_to_bin_cpu`: line 1143
- `add_part_to_bin_cpu` called ~500×/wave × 300 waves = **~150K calls/seed**

### What happens today

For each of the `ph` rows modified by a placement, a pure scalar loop scans all `W` = 250 cells counting max consecutive zeros (run-length max). No SIMD:

```cpp
for (int i = 0; i < num_rows; ++i) {
    int max_zeros = 0;
    int current_zeros = 0;
    const int row_idx = y_start + i;
    const int row_off = row_idx * width;
    for (int j = 0; j < width; ++j) {
        if (grid[row_off + j] == 0) {
            current_zeros += 1;
            if (current_zeros > max_zeros) max_zeros = current_zeros;
        } else {
            current_zeros = 0;
        }
    }
    vacancy_vector[row_idx] = max_zeros;
}
```

**Per-seed cost estimate:**
- 150K calls × ~20 rows avg × 250 cells = **~750M scalar ops/seed**
- At ~3 GHz scalar throughput → **~250 ms/seed**
- AVX512 byte-wise ops can do 64 cells/cycle → theoretical 16× speedup
- Realistic 3–5× on this function due to the run-length dependency

### Precedent in the codebase

`check_vacancy_fit_simple_cpp` already has AVX512 and AVX2 paths, runtime-dispatched via `__builtin_cpu_supports` (per your earlier note in idea #5's caveat). The `update_vacancy_rows_cpp` function should follow the same pattern.

### Optimization approach

**SIMD run-length max over uint8:**
- Load 32 or 64 bytes with `_mm256_loadu_si256` / `_mm512_loadu_si512`
- Compare to zero with `_mm256_cmpeq_epi8` → mask of zero positions
- Convert to bitmask with `_mm256_movemask_epi8` → 32-bit word where bit i = 1 iff byte i is zero
- Use `__builtin_clzll` / `__builtin_ctzll` to find run boundaries
- Maintain a running "current run" counter across chunk boundaries
- Track max across all chunks

**Alternative (simpler, possibly faster):**
- Since `W = 250`, it's 4 × 64-byte AVX512 iterations with a tail
- Or 8 × 32-byte AVX2 iterations
- A prefix-max-over-runlength via a single pass is tractable in ~30 lines of intrinsics

### Verification plan

Before implementing: instrument `update_vacancy_rows_cpp` with a per-call timer and a call counter. Run one seed:
- Total time in this function
- Number of calls
- Average time per call
- Average rows per call

If <80ms/seed, skip. If >150ms/seed, implement AVX2 version first (simpler), then AVX512 if still hot. Gate with `__builtin_cpu_supports` like the existing functions do.

### Risk notes

- Must match scalar output exactly (not within tolerance — vacancy vector is used for exact checks downstream).
- Edge case: rows shorter than one SIMD word (won't happen with W=250 and 32/64-byte words, but guard anyway).

---

## 3. Phase 3 Pass A / Pass C merge + SoA refactor

**Expected saving:** 50–100ms/seed
**Effort:** Low-Medium
**Risk:** Low
**Confidence:** Medium — wall-time gain depends on sync point position

### Where it lives

- `process_wave` Phase 3 p1: lines 1434–1521
- `process_wave` Phase 3 p2: lines 1538+
- `P1Pair` struct: local to `process_wave`, line 1440

### What happens today

**Pass A** (lines 1451–1478): triple-nested loop `(lc, bidx, rot)` — for every `context × open bin × rotation` triple, pushes into:
- Local `std::vector<P1Pair> p1_pairs` (NOT a scratch member — allocated per wave)
- `scratch_pair_vac_row_`, `scratch_pair_den_off_`, `scratch_pair_den_len_`

**Pass B** (line 1482): GPU vacancy check kernel + **synchronous readback** (`non_blocking=false` at line 1346)

**Pass C** (lines 1484–1521): walks the result mask. For each passing pair:
- Re-reads `contexts[ctx_global[lc]]`
- Re-reads `ctx.open_bins[bidx]`
- Pushes into **10 separate `scratch_p1_` vectors**

**Per-seed cost estimate:**
- ~1000 pairs/wave × 300 waves = 300K pairs/seed
- Pass C does 10 push_backs per passing pair + pointer chases into `open_bins` → ~10M push_backs + ~300K pointer chases
- Critical: **Pass C runs after a synchronous `cpu_pass.copy_(gpu_out)`** (line 1346), so it's on the critical path, not hidden behind GPU work.

### Hot spots

1. `p1_pairs` is a **local stack vector** — allocator churn every wave. Should be a scratch member.
2. **10 AoS push_backs per passing pair** in Pass C with no reserve ceiling tight enough.
3. **Pointer chases** into `ctx.open_bins[bidx]` in Pass C with poor cache locality (open_bins entries are ~1 KB each).
4. No SIMD/vectorization.

### Optimization candidates

**A. Make `p1_pairs` a scratch member:** trivial, saves allocator traffic. ~5ms/seed.

**B. Merge Pass A and Pass C:**
- In Pass A, record all metadata needed by Pass C (grid_state_idx, heights, widths, etc.) directly into the `scratch_p1_` vectors.
- In Pass C, use the mask to *compact in place* rather than push into new vectors.
- Eliminates the `ctx.open_bins[bidx]` reload in Pass C. ~30–60ms/seed.

**C. SoA refactor of BinStateNative fields accessed in Phase 3:**
- Extract `grid_state_idx`, `area`, `enclosure_box_length`, `bin_length` into parallel vectors indexed by `(ctx, bidx)`.
- Pass A becomes streaming reads over contiguous arrays instead of pointer-chasing into `std::vector<BinStateNative>`.
- ~20–40ms/seed.

**D. Skip Pass C entirely via in-Pass-A first-valid tracking:**
- Since Pass A iterates in `(lc, bidx, rot)` order, the first passing `(lc, bidx)` per `lc` is the first-valid bin.
- Track it in-line in Pass A and only build FFT test arrays for that bin after the vacancy check completes.
- More invasive but could eliminate Pass C's 10-push_back work entirely. ~50–100ms/seed.

### Verification plan

Instrument Pass A, Pass B wait, and Pass C separately with `std::chrono`. Run one seed:
- Time in Pass A
- Time blocked on Pass B readback
- Time in Pass C
- Passing pair count

If Pass C > 50ms/seed, implement B+C (merge + SoA). If Pass A > 50ms/seed, also do D.

### Interaction with #4

This optimization overlaps with #4 (vacancy check async). Fixing #4 reduces the sync at Pass B, which gives Pass C more time budget. If #4 is done first, #3's wall-time impact shrinks. Do #3 standalone only if #4 is too risky.

---

## 4. Vacancy check async + double-buffer

**Expected saving:** 50–120ms/seed
**Effort:** Medium
**Risk:** Medium
**Confidence:** Medium — depends on pipeline analysis

### Where it lives

- `run_gpu_vacancy_check`: lines 1318–1349
- Two **synchronous** copies: line 1332 (`non_blocking=false` for pair data upload) and line 1346 (`non_blocking=false` for result readback)
- Called twice per wave (p1 and p2) × 300 waves = **~600 sync points/seed**

### What happens today

```cpp
gpu_pairs.copy_(cpu_pairs, /*non_blocking=*/false);  // sync — kernel needs data immediately
// ... kernel launch ...
cpu_pass.copy_(gpu_out, /*non_blocking=*/false);     // sync — Pass C needs data immediately
```

Each sync point flushes GPU work in flight. If the GPU has Phase 2 rfft2 or Phase 4 IFFT work queued, CPU waits for it to drain. This is likely the main reason idea #5 showed only marginal wins: the vacancy kernel replaces hidden CPU work but introduces explicit sync points that weren't there before.

**Per-seed cost estimate:**
- 600 sync points × ~200 µs average GPU-drain wait = **~120 ms/seed**
- This is a rough upper bound — actual sync cost depends on what's in flight at each point

### Optimization candidates

**A. Double-buffer the vacancy check result:**
- Launch wave N's vacancy check asynchronously.
- Build wave N+1's Pass-A candidate list *on CPU* in parallel with wave N's GPU check.
- Read back wave N's result only when Pass C actually needs it.
- Hides the sync latency behind useful CPU work.

**B. CUDA event + deferred sync:**
- Replace `non_blocking=false` with `non_blocking=true` and a `cudaEvent_t`.
- In Pass C, first call `event.query()` — if ready, no sync. If not, sync.
- Often the result is ready because the GPU is fast → the sync becomes a no-op query.

**C. Merge p1 and p2 vacancy checks into one kernel launch:**
- Currently 2 sync points per wave. Could be 1 by interleaving p1 and p2 pair collection before the kernel launch.
- Requires careful restructuring because p2 is only collected for contexts that had no p1 hit — chicken-and-egg problem.
- Workaround: collect both p1 and p2 candidate pairs optimistically in the same pass, run one vacancy kernel, then apply first-valid-bin semantics post-hoc.

**D. Upload pair data via pinned memory + non-blocking copy:**
- The upload at line 1332 (`non_blocking=false`) could become `non_blocking=true` if the pair data is in pinned memory and the kernel is launched on the same stream. PyTorch auto-syncs stream dependencies. Simple change, maybe ~10–20ms.

### Verification plan

Add a CUDA event before/after each vacancy kernel launch. Measure:
- Total GPU time in vacancy kernels (should be small, ~5–10ms/seed)
- Total wall-clock time in `run_gpu_vacancy_check` (includes sync waits)
- The difference is the sync overhead we can reclaim

If sync overhead <30ms/seed, skip. If >80ms/seed, implement B (CUDA event) first, then A (double buffer) if still hot.

### Risk notes

- Double-buffering has subtle correctness hazards: Pass A for wave N+1 must not read state modified by wave N's Phase 5 updates. Since Phase 5 runs on GPU and wave N+1 Pass A runs on CPU, the `bin.grid_fft_valid` / `bin.vacancy_gpu_dirty` flags must be updated correctly.
- A regression test against single-buffered baseline is mandatory before merging.

---

## 5. Python ↔ C++ boundary — NOT A TARGET

**Status:** Already checked. Confirmed negligible.

### What happens

```python
def evaluate_batch(self, chromosomes: np.ndarray):
    chrom = np.ascontiguousarray(chromosomes, dtype=np.float32)
    out = self._decoder.evaluate_batch(chrom)
    return np.asarray(out, dtype=np.float64).tolist()
```

- `ascontiguousarray`: no-op if already contiguous float32 (usually is)
- `self._decoder.evaluate_batch(chrom)`: one pybind11 call, passes numpy buffer by pointer
- `np.asarray(out, dtype=np.float64).tolist()`: small result vector (num_solutions doubles, ~4 KB)

**Total cost:** ~microseconds per generation. Not a target.

### `decode_sequences_for_machine`

- Called once per machine × num_solutions at the start of `process_machine_batch` (not per wave)
- Per call: sort ~50 parts by random key (~280 comparisons)
- Per-seed: 500 × 2 = 1000 sorts = ~280K comparisons total
- **Estimated: <10 ms/seed**

Not worth optimizing in isolation. Could be batched across solutions via SoA but the gain is trivial compared to #1–#4.

---

## Summary ranking (REVISED 2026-04-10)

| # | Target | Measured cost | Status | Expected saving |
|---|--------|---------------|--------|----------------|
| ~~1~~ | ~~SIMD `update_vacancy_rows_cpp`~~ | 383 ms/gen (pre) → 89 ms/gen (post) | **DONE** −287 ms/gen | — |
| ~~2~~ | ~~Vacancy upload sync~~ | 97 ms/gen | **SKIPPED** — mostly real GPU time, no cheap fix | — |
| ~~3~~ | ~~Template + memcpy/AVX2-add grid write~~ | 111 ms/gen (pre) → 53 ms/gen (post) | **DONE** −58 ms/gen (−7.7%) | — |
| ~~4~~ | ~~Phase 6 bin pool~~ | 31.5 ms/gen ctor → 22.2 ms/gen | **DONE** −9.3 ms/gen ctor, −16 ms wall clock | — |
| — | ~~Phase 3 Pass A/C merge~~ | 8 ms/gen | Dead target | — |
| — | ~~Vacancy readback sync~~ | 7 ms/gen | Dead target | — |
| — | ~~Python-C++ boundary~~ | <1 ms/gen | Dead target | — |

**Current baseline after #1:** 1.212s/gen on P50M2-0.

---

## Implementation order (current)

1. ~~**Profile first**~~ — **DONE** (2026-04-09).
2. ~~**SIMD `update_vacancy_rows_cpp`**~~ — **DONE** (2026-04-09). −287 ms/gen.
3. ~~**Vacancy upload sync**~~ — **SKIPPED** (2026-04-10). See analysis above.
4. ~~**Template + memcpy/AVX2-add grid write**~~ — **DONE** (2026-04-10). −58 ms/gen.
5. **Phase 6 bin pool / lazy grid** — 106 ms/gen total (33 ms in ctor, 73 ms in add_part_to_bin_cpu for the first part). Evaluate whether BinStateNative ctor `grid.assign` is the bottleneck or if most time is in the first-part placement loop.
6. **Re-profile after any change.** Verify fingerprint.

All implementation work uses the local-only profiling instrumentation as the measurement harness. After #1+#2 are committed and validated, decide whether to commit or revert the instrumentation.

---

## Open questions / unknowns

- **What's the actual new-bin rate per wave?** Assumed ~100, but could be anywhere from 10 to 500 depending on instance geometry and BRKGA generation. Measure on P50 and P75.
- **Are `add_part_to_bin_cpu` calls hidden behind the Phase 5 GPU kernel?** If yes, SIMD-ing `update_vacancy_rows_cpp` only helps on larger instances where the CPU tail exceeds the GPU tail.
- **What's the actual GPU-drain wait at the vacancy sync points?** Could be 50µs (no-op) or 500µs (busy kernel). Determines #4's upside.
- **Does the `BinStateNative` allocator use tcmalloc/jemalloc?** If yes, allocation churn is cheaper than glibc malloc. Determines #1A's upside.

These should all be answered by a single instrumented run before any implementation work begins.
