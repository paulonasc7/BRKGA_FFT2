# Scalability Analysis — Arbitrary Parts & Machines

**Context:** `python BRKGA_alg3.py 300 10 0 torch_gpu wave_batch 1 1 3` is used as a representative large instance, but the issues described apply to any combination where `nbParts` and/or `nbMachines` is large enough to push GPU memory beyond capacity.

---

## Background: How `grid_states` is allocated

The wave-batch evaluator pre-allocates a single contiguous GPU tensor:

```
grid_states:  (num_solutions × max_bins_per_sol,  H, W)  — float32
grid_ffts:    (num_solutions × max_bins_per_sol,  H, W//2+1)  — complex64
```

Each solution `sol_idx` owns a fixed slice of rows `[sol_idx × max_bins_per_sol, (sol_idx+1) × max_bins_per_sol)`. When a solution opens its k-th bin, it writes into slot `sol_idx × max_bins_per_sol + k`.

**Bins are opened lazily** — a new bin only opens when a part cannot be placed in any currently open bin. The pre-allocation does not change the algorithm; it is purely a GPU memory reservation. The constraint is: if a solution ever needs more bins than `max_bins_per_sol`, the slot counter overflows the reserved region and causes a CUDA index-out-of-bounds error.

---

## Why it Breaks at Large Scale

### 1. Quadratic memory growth

Population size scales as `mult × nbParts` (currently `mult = 10`). The bin budget scales as `nbParts // 3`. So the required grid allocation scales as:

```
num_solutions × max_bins × bytes_per_bin
= (10 × nbParts) × (nbParts / 3) × bytes_per_bin
∝ nbParts²
```

For P50M2 (500 solutions, ~16 bins, H=300 W=250): **~2.3 GB** — fits comfortably.
For P300M10 (3000 solutions, ~100 bins, H=W=300): **~216 GB** — 13× beyond the GPU.

This is the fundamental ceiling. The current design cannot accommodate large instances on a 16 GB GPU without a structural change.

### 2. The `max_bins_per_sol` formula hits a dangerous floor

The current formula is:

```python
needed_bins  = max(10, self.nbParts // 3)
vram_cap     = max(5, int(total_vram * 0.50) // (bytes_per_bin * num_solutions))
max_bins_per_sol = min(needed_bins, vram_cap)
```

For P300M10 at H=W=300:
```
needed_bins = 100
vram_cap    = max(5, 8_000_000_000 // (722_400 × 3000)) = max(5, 3) = 5
max_bins_per_sol = min(100, 5) = 5
```

The `max(5, ...)` floor was added to protect small instances from getting too few bins. At large scale it backfires: it overrides the VRAM cap, causing the grid allocation to use **~10.8 GB** (67% of VRAM) instead of the budgeted 50%, while still providing only 5 bins for 300 parts. The algorithm then exhausts its bin slots after ~10–15 placements per solution and hits CUDA OOB errors.

### 3. No guard against bin slot overflow

There is no check in Phase 6 or `_start_new_bin` to detect when `ctx.next_grid_idx` is about to exceed its allocated region. When overflow happens, the code silently writes into an adjacent solution's grid slots or causes a CUDA assertion failure. Either way the results are wrong.

### 4. The 50% VRAM reserve is unnecessarily conservative

The formula reserves 50% of VRAM for IFFT working memory. The actual IFFT working memory is:

```
IFFT output tensor:  CHUNK_SIZE × H × W × 4 bytes
= 750 × 300 × 300 × 4 ≈ 270 MB
Other buffers (scores, indices, part FFTs): ~150–200 MB
Total: ~400–500 MB
```

Reserving 8 GB for ~500 MB of working memory wastes roughly 15× the actual need. This reservation was appropriate as a blunt safety margin for small instances but is wasteful when computing sub-batch sizes at large scale.

---

## Performance Impact of the Fix (Sub-batching)

The correct fix is **sub-batching**: split `num_solutions` into groups that fit in VRAM, run the full wave loop for each group sequentially, and accumulate makespans.

### Sub-batch size calculation (corrected)

```python
ifft_overhead  = CHUNK_SIZE × H × W × 4 + safety_margin  # ~500 MB–1 GB
available_vram = total_vram - ifft_overhead
sub_batch_size = available_vram // (needed_bins × bytes_per_bin)
```

For P300M10 at H=W=300:
```
available_vram ≈ 15 GB  (16 GB - 1 GB safety)
sub_batch_size = 15_000_000_000 // (100 × 722_400) ≈ 207 solutions
```

So ~207 solutions per sub-batch, ~15 sub-batches for 3000 solutions.

### Speed implications

| Aspect | Impact |
|--------|--------|
| IFFT batch size | Reduced by ~15× vs. processing all 3000 at once. However, 207 solutions × ~5 open bins × ~4 rotations × CHUNK_SIZE still produces thousands of IFFT operations per chunk — enough to saturate GPU memory bandwidth in most cases. |
| Sub-batch overhead | ~15 iterations of the Python wave loop, tensor allocation/deallocation, and vacancy check passes. At ~500 ms per machine per sub-batch this adds ~7.5 s/machine/generation. |
| Cross-solution parallelism | Fully preserved within each sub-batch; only reduced across sub-batches. |
| Phase 3 (vacancy checks) | With 207 contexts per sub-batch, Phase 3 scales proportionally. Already manageable. |

The throughput loss is real but acceptable — the alternative is a crash.

---

## Why the Dynamic Free-List is Not the Right Primary Fix

An alternative to sub-batching is a **dynamic free-list**: maintain a shared pool of GPU grid slots and allocate/free them as bins open and close, rather than pre-assigning a fixed slice per solution.

This could theoretically allow all 3000 solutions to run simultaneously by reclaiming slots from bins that are full (area-saturated and will never receive another part). However:

1. **Peak memory is not significantly reduced.** In the wave-based design, all solutions advance roughly in lockstep (one part per wave). At late waves, most solutions have many bins open simultaneously. The peak slot count is still close to `num_solutions × max_bins`. For P300M10 this is still ~216 GB.

2. **It doesn't eliminate the need for sub-batching** at this scale — you would need both the free-list complexity and sub-batching, worse on both dimensions.

3. **It complicates Phase 2/4 index lookups**, which today rely on a simple `grid_states[list_of_indices]` into one contiguous tensor.

The free-list is worth revisiting only as a secondary optimization once sub-batching is in place — specifically to reclaim slots from geometrically-full bins mid-wave, which could allow slightly larger sub-batches.

---

## Required Changes

### Design principle: VRAM-cap first, sub-batch as fallback

The right design is to keep the current approach as the primary path and only fall back to sub-batching when the VRAM cap is too restrictive:

1. **Primary path:** Try to fit all solutions at once with a VRAM-capped bin count (current behavior). This is what already works for P50M2, P75M2, and P100M4. For P100M4, the H=W=400 machine runs with only 6 bins per solution — enough because ~25 parts are assigned to that machine and 400×400 bins are large. No sub-batching overhead is incurred.

2. **Fallback:** If the VRAM-capped bin count falls below a minimum viable threshold (i.e., too few bins to realistically pack the assigned parts), switch to sub-batching with a larger bin budget per solution. This is what P300M10 would need — `vram_cap = 3` bins for 100 parts per machine is unworkable.

Combined with the overflow guard (Change 3), the primary path is safe even if the bin cap is occasionally too tight: solutions that overflow are marked infeasible (`makespan = 1e16`) rather than causing CUDA crashes. Sub-batching is only activated when the overflow rate would be unacceptably high.

The minimum viable threshold for switching to sub-batching is an open question — see "Estimating bin count upper bounds" below.

### Change 1: Sub-batch loop in `_process_machine_batch`

Split `chromosomes` into sub-batches of size `sub_batch_size`, run the existing wave loop for each, and concatenate makespans. Only activated when `vram_cap < needed_bins`.

```python
if vram_cap >= needed_bins:
    # Primary path: all solutions at once (current behavior, no overhead)
    max_bins_per_sol = needed_bins
    makespans = self._run_machine_wave_loop(chromosomes, ..., max_bins_per_sol)
else:
    # Fallback: sub-batching with full bin budget
    max_bins_per_sol = needed_bins
    sub_batch_size = max(1, available_vram // (needed_bins * bytes_per_bin))
    for start in range(0, num_solutions, sub_batch_size):
        end = min(start + sub_batch_size, num_solutions)
        sub_makespans = self._run_machine_wave_loop(chromosomes[start:end], ..., max_bins_per_sol)
        makespans[start:end] = sub_makespans
```

The inner `_run_machine_wave_loop` is the existing wave loop with no changes.

### Change 2: Remove the `max(5, ...)` floor from `vram_cap`

```python
# Before
vram_cap = max(5, int(total_vram * 0.50) // (bytes_per_bin * num_solutions))

# After
vram_cap = available_vram // (bytes_per_bin * num_solutions)
```

With the overflow guard in place and sub-batching as a fallback, the defensive floor is no longer needed. If `vram_cap` is small but sufficient in practice (like 6 bins for P100M4's H=W=400 machine), the primary path handles it. If it's too small, sub-batching activates.

### Change 3: Add a bin overflow guard

The guard must be applied in two places: the Phase 6 batch loop (`wave_batch_evaluator.py`, line ~532) and `_start_new_bin` (line ~698) — both locations where `next_grid_idx` is incremented.

```python
# In both Phase 6 and _start_new_bin, before incrementing next_grid_idx:
if ctx.next_grid_idx >= (ctx.solution_idx + 1) * max_bins_per_sol:
    ctx.is_feasible = False
    continue
grid_idx = ctx.next_grid_idx
ctx.next_grid_idx += 1
```

To make `max_bins_per_sol` visible inside `_process_wave_true_batch` (where Phase 6 lives), it either needs to be passed as a parameter or pre-computed as a per-context limit field (e.g., `ctx.max_grid_idx = (sol_idx + 1) * max_bins_per_sol`) stored at initialization in `_init_batch_contexts`.

**Runtime overhead:** Negligible. This is a single Python integer comparison that runs once per new bin opened. Phase 6 fires at most a few hundred times per generation total across all contexts — completely undetectable in profiling.

**When it triggers:** The context is marked `is_feasible = False`, causing it to return `makespan = 1e16`. The BRKGA treats that chromosome as very bad, it stays out of the elite set, and is eventually replaced. This is correct behavior — the alternative (without the guard) is silently corrupting the GPU grid of the neighboring solution in the batch, producing wrong makespans for two solutions with no error signal.

**Diagnostic use:** Adding a generation-level counter alongside the guard (increment each time it fires) is cheap and reveals the actual overflow rate in practice. For P100M4 on the H=W=400 machine with `max_bins_per_sol = 6`, this counter would immediately show how many chromosomes genuinely need more than 6 bins — informing whether the current cap is adequate or whether sub-batching is needed.

**Impact by instance:**
- **P50M2:** Never triggers. `max_bins_per_sol = 16` for ~25 parts/machine. Zero performance impact.
- **P100M4 (H=W=400 machine):** May trigger for chromosomes that assign an unusually large number of parts to that machine. Frequency unknown without the diagnostic counter.
- **P300M10:** Would trigger constantly with `vram_cap = 3–6` bins for 100 parts — this is the scenario where sub-batching must activate instead.

### Change 4: Replace the 50% VRAM heuristic with an explicit overhead calculation

```python
ifft_working_mb = (CHUNK_SIZE * H * W * 4) / 1e6  # IFFT output tensor
safety_mb       = 1024                              # 1 GB for PyTorch runtime + fragmentation
available_vram  = total_vram - int((ifft_working_mb + safety_mb) * 1e6)
```

This gives a tighter budget and larger sub-batches than the blanket 50% reserve.

### Estimating bin count upper bounds

The `needed_bins` formula (`max(10, nbParts // 3)`) is a rough heuristic that assumes ~3 parts per bin on average. This drives two critical decisions: (1) whether sub-batching is needed (`vram_cap < needed_bins?`), and (2) how large each sub-batch can be (`available_vram // (needed_bins × bytes_per_bin)`). An overestimate wastes VRAM and forces unnecessary sub-batching. An underestimate causes bin overflow (caught by the overflow guard, but those solutions return infeasible).

Getting this number right matters — it is the difference between P100M4 running in 1 batch (current, 6 bins, works fine) vs. 3 sub-batches (if `needed_bins = 33` is enforced). Better heuristics or adaptive approaches should be investigated:

**Data-driven heuristics from the problem instance:**
- **Area-based bound:** `ceil(sum_of_part_areas / bin_area)` gives a theoretical minimum number of bins assuming perfect packing (no wasted space). Multiply by a utilization factor (e.g., 1.5–2.0 for typical 2D irregular packing densities of 50–70%) to get a realistic upper bound. This uses actual part and bin dimensions rather than a fixed ratio.
- **Per-machine bound:** Parts are not evenly split across machines — the chromosome decides the assignment. The worst case is all parts assigned to one machine, but in practice the BRKGA's machine assignment keys distribute parts across machines. A per-machine bound could use `ceil(nbParts / nbMachines)` as the part count and compute the area-based bound from there.
- **Largest-part bound:** If the largest part occupies a significant fraction of the bin, the bin can hold fewer parts. `ceil(nbParts_on_machine / floor(bin_area / largest_part_area))` gives a rough bound.

**Adaptive approaches (runtime):**
- **Track actual max bins used** across generations and machines. Start with a conservative estimate, then tighten it based on observed behavior. If the maximum bins ever observed is 8, allocating 33 is wasteful.
- **Grow on demand:** Start with a small `max_bins_per_sol`, and if the overflow guard triggers for too many solutions in a generation, increase it for the next generation (reallocating the grid tensors). This avoids the need for an accurate upfront estimate at the cost of occasional wasted generations.
- **Profile the initial population:** Generation 0 evaluates all 500+ solutions. The maximum bins used in gen 0 is a strong signal for subsequent generations (the population doesn't change structure dramatically between generations). Use `max_bins_gen0 + margin` as the cap for gen 1+.

The choice of heuristic directly affects whether sub-batching activates and how many sub-batches are needed. This is worth investigating as part of the implementation, especially for instances where the current `nbParts // 3` is far from reality.

---

## Summary

| Issue | Root cause | Fix |
|-------|-----------|-----|
| CUDA OOB on large instances | Pre-allocated grid tensor is too small; no overflow guard | Sub-batching + overflow guard |
| `max(5, ...)` floor overrides VRAM budget | Defensive floor was sized for small instances | Remove floor; compute sub-batch size instead |
| 50% VRAM reserve wastes ~15× actual need | Blunt heuristic | Replace with explicit IFFT overhead calculation |
| Quadratic memory growth | `num_solutions × max_bins ∝ nbParts²` | Sub-batching caps peak allocation per machine call |

After these changes, the code will handle arbitrary `nbParts` and `nbMachines` — larger instances just run in more sub-batches with proportionally more time, but without crashes or incorrect results.

---

## Additional Considerations

The sections above cover the GPU VRAM problem and the sub-batching fix. The analysis below identifies further constraints and alternative approaches that should be evaluated before or alongside implementation.

### 5. CPU memory is also a binding constraint

Each `BinState` allocates a CPU grid and vacancy vector (`wave_batch_evaluator.py`, lines 537–538):

```python
grid=np.zeros((ctx.bin_length, ctx.bin_width), dtype=np.uint8)
vacancy_vector=np.zeros(ctx.bin_length, dtype=np.int32) + ctx.bin_width
```

For P300M10 (3000 solutions × 100 bins, H=300, W=300):
- CPU grids: `3000 × 100 × 300 × 300 × 1 byte = 27 GB`
- Vacancy vectors: `3000 × 100 × 300 × 4 bytes = 360 MB`
- Total CPU: **~27 GB**

Sub-batching solves this too (207 solutions × 100 bins × ~76 KB ≈ 1.6 GB per sub-batch), but on a system with limited RAM, CPU memory could be the binding constraint before GPU memory even comes into play. The sub-batch size calculation should account for both GPU and CPU budgets.

### 6. Phase 3 time complexity may dominate at large scale

The document's speed implications table (section "Performance Impact") states Phase 3 is "already manageable" with 207 contexts per sub-batch. This underestimates the problem. For a single sub-batch of 207 solutions on one machine:

- Per wave: ~207 contexts × ~10 open bins × ~4 rotations = **~8,280 Numba calls**
- At ~1 µs/call: ~8.3 ms/wave for vacancy checks alone
- Waves per machine: up to ~300 (one per part in the worst case, assuming ~100 parts/machine for P300M10)
- Phase 3 per machine per sub-batch: **~2.5 seconds**
- With ~15 sub-batches × 10 machines: **~375 seconds just for vacancy checks**

This excludes the Python-level loop overhead around those Numba calls (building the `p1_*` / `p2_*` lists with `.append()` per test). At P300M10 scale, Phase 3 could overtake Phase 4 as the dominant bottleneck. Sub-batching makes the system *correct*, but without also addressing Phase 3 (see Idea 7 in `OPTIMIZATION_IDEAS.md` — batch vacancy checks), P300M10 may be impractically slow.

### 7. CHUNK_SIZE needs to be adaptive for large machine dimensions

The hardcoded `CHUNK_SIZE = 750` (`wave_batch_evaluator.py`, line 607) was tuned for H=300, W=250 on the RTX A4000. For larger machines (e.g., H=W=500, plausible at P300M10 scale), the per-chunk intermediates grow significantly:

```
overlap_batch:   750 × 500 × 500 × 4 =  750 MB
batch_grid_ffts: 750 × 500 × 251 × 8 =  753 MB
batch_part_ffts: 750 × 500 × 251 × 8 =  753 MB
Total per chunk:                        ~2.3 GB
```

If the sub-batch grid allocation uses ~14 GB of a 16 GB GPU, only ~2 GB remains — not enough for a single chunk at H=W=500. The IFFT overhead calculation in Change 4 should determine `CHUNK_SIZE` dynamically based on machine dimensions, or the sub-batch size formula should account for the actual chunk memory cost at the given H and W.

### 8. The `needed_bins` heuristic may be inaccurate

The formula `needed_bins = max(10, nbParts // 3)` assumes ~3 parts per bin on average. This is a rough heuristic that could be significantly off:

- **Large bins with small parts** → 10+ parts per bin → formula overestimates by 3×+, wasting VRAM and reducing sub-batch size unnecessarily.
- **Small bins with large parts** → 1–2 parts per bin → formula underestimates, causing bin overflow (caught by the overflow guard from Change 3, which marks the solution infeasible rather than crashing).

With the overflow guard in place, overestimation wastes resources but underestimation produces incorrect (infeasible) results. A more robust formula could use `bin_area / avg_part_area` from the problem data, or the system could track the actual maximum bins used across generations and adapt.

### 9. Population size is a user-tunable lever

For P300M10, `mult=10` gives 3000 solutions, which drives the entire scaling problem. But `mult=3` gives 900 solutions → ~5 sub-batches instead of ~15, with better GPU utilization per sub-batch. This is a parameter choice, not a code change. Whether a smaller population is acceptable depends on solution quality (a larger population explores more of the solution space per generation), which is an algorithmic tradeoff. The point is that for very large instances, reducing `mult` may be a more practical lever than engineering the code to handle arbitrarily large populations.

---

## Alternative Approaches to Reduce Sub-batch Count

Sub-batching with the current memory layout requires ~15 sub-batches for P300M10. The approaches below could significantly reduce that number by shrinking per-bin GPU memory, potentially changing the performance picture.

### A. Drop `grid_ffts` caching — recompute on demand

`grid_ffts` is a cache of `rfft2(grid_states)`. It exists so Phase 2 only recomputes FFTs for bins that changed since the last wave (`grid_fft_valid` flag). Eliminating it would:

- **Cut per-bin GPU memory by ~47%** (from ~722 KB to ~360 KB for H=W=300).
- **Roughly double sub-batch size** (from ~207 to ~414 solutions for P300M10).
- **Halve the number of sub-batches** (from ~15 to ~8).

The cost: Phase 2 recomputes `rfft2` for ALL open bins every wave, not just invalid ones. Currently Phase 2 is 7.3% of total time. Even if it doubled or tripled, it would remain smaller than Phase 4 (43.9%). For large instances where sub-batch count is the performance limiter, this tradeoff is likely worthwhile.

**Implementation:** Remove the `grid_ffts` tensor and the `grid_fft_valid` flag. In Phase 2, always compute `rfft2` on `grid_states` for all bins needed by Phase 3/4. The `rfft2` output becomes a temporary tensor per wave rather than a persistent cache.

### B. Store `grid_states` as uint8, convert to float32 on demand

The GPU grid stores small integer values (0, 1, 2, ...) accumulated via addition, but uses float32 because `rfft2` requires floating-point input. Storing as uint8 and converting to float32 only when needed (Phase 2 before `rfft2`, and in the CUDA batch update kernel) would cut `grid_states` memory by 4×.

- Per-bin GPU memory: drops from ~360 KB (float32 only, no `grid_ffts`) to **~90 KB** (uint8 only).
- Sub-batch size for P300M10: ~1,660 solutions → **2 sub-batches instead of 15**.

The cost: Phase 2 needs a `.float()` conversion before `rfft2`, and the CUDA batch update kernel needs to write uint8 instead of float32. Both are straightforward changes. The `.float()` conversion produces a temporary float32 tensor of size `(active_bins, H, W)` — but this is per-wave and much smaller than the persistent allocation.

**Combined impact of A + B:** Per-bin GPU memory drops from 722 KB to ~90 KB — an 8× reduction. This dramatically changes the sub-batching math and could make P300M10 nearly as efficient (in terms of sub-batch count) as P50M2 is today.

### C. Pipeline machine processing across sub-batches

Currently machines are processed sequentially: the full set of sub-batches runs for machine 0, then machine 1, etc. With 15 sub-batches × 10 machines = 150 sequential runs. Since machines are independent, sub-batches from different machines could be interleaved. For example, while sub-batch 1 of machine 0 is doing CPU-bound work (Phase 3 vacancy checks, Phase 5 CPU grid updates), GPU work for machine 1's sub-batch could overlap via CUDA streams. This is a form of pipeline parallelism that doesn't require additional VRAM (only one machine's grid tensors are on GPU at a time, since they're freed between machines).

This is an advanced optimization with meaningful implementation complexity, but it addresses the serial nature of the 150-iteration outer loop.

---

## Summary

| Issue | Root cause | Fix |
|-------|-----------|-----|
| CUDA OOB on large instances | Pre-allocated grid tensor is too small; no overflow guard | Sub-batching + overflow guard |
| `max(5, ...)` floor overrides VRAM budget | Defensive floor was sized for small instances | Remove floor; compute sub-batch size instead |
| 50% VRAM reserve wastes ~15× actual need | Blunt heuristic | Replace with explicit IFFT overhead calculation |
| Quadratic memory growth | `num_solutions × max_bins ∝ nbParts²` | Sub-batching caps peak allocation per machine call |
| CPU memory (~27 GB for P300M10) | Each BinState has a CPU grid (uint8) + vacancy vector | Sub-batching also bounds CPU memory per sub-batch |
| Phase 3 may dominate at large scale | O(contexts × bins × rotations) Numba calls per wave | Batch vacancy checks (Idea 7) needed alongside sub-batching |
| CHUNK_SIZE too large for big machines | Hardcoded 750, tuned for H=300 W=250 | Make CHUNK_SIZE adaptive based on machine dimensions |
| `needed_bins` heuristic may be off | Assumes ~3 parts/bin average | Use problem data (bin_area / avg_part_area) or adapt at runtime |
| Per-bin GPU memory is 8× larger than necessary | float32 grid + complex64 FFT cache both persistent | Drop FFT cache (approach A) + use uint8 grids (approach B) |

After the core changes (sub-batching, overflow guard, tighter VRAM budget), the code will handle arbitrary `nbParts` and `nbMachines` correctly. Approaches A and B can further reduce the sub-batch count by up to 8×, improving throughput for large instances. Phase 3 optimization (Idea 7) is needed to make large instances not just correct but tractably fast.

---

## Impact on Existing Performance (P50M2 Regression Analysis)

**Baseline:** P50M2 runs at ~2.6 s/gen (RTX A4000, 16 GB VRAM). The analysis below traces through each proposed change with the exact P50M2 parameters to determine whether that baseline would be affected.

### P50M2 parameters

```
nbParts       = 50
nbMachines    = 2
mult          = 10
num_individuals = 500
num_elites    = 50
num_mutants   = 75

Gen 0:  cal_fitness(500 chromosomes)  → num_solutions = 500
Gen 1+: cal_fitness(450 offspring)    → num_solutions = 450

H = 300, W = 250  (both machines — same dimensions)
bytes_per_bin = 300 × 250 × 4 + 300 × 126 × 8 = 602,400 bytes
needed_bins   = max(10, 50 // 3) = 16
```

### Change 1: Sub-batch loop — NO IMPACT

With the proposed tighter budget (Change 4):

```
available_vram = 16 GB - 1 GB safety = 15 GB
sub_batch_size = 15,000,000,000 // (16 × 602,400) = 1,556
```

Since `num_solutions` is at most 500 (gen 0) and 450 (gen 1+), both are well below the sub-batch size of 1,556. **The sub-batch loop executes exactly once** — identical to the current code path. The only added overhead is the sub-batch size computation itself (a few integer operations), which is negligible.

**To guarantee this**: the implementation should detect the single-sub-batch case and skip the loop wrapper entirely, calling the existing wave loop directly. This makes the zero-overhead property explicit rather than relying on the loop running once.

### Change 2: Remove `max(5, ...)` floor — NO IMPACT

The current formula for P50M2:

```
vram_cap = max(5, int(16e9 × 0.50) // (602,400 × 450))
         = max(5, 8,000,000,000 // 271,080,000)
         = max(5, 29)
         = 29

max_bins_per_sol = min(16, 29) = 16
```

The floor of 5 is never active — `vram_cap` is already 29. Removing it changes nothing for P50M2. The binding constraint is `needed_bins = 16`, not the VRAM cap.

With the tighter budget (Change 4), `vram_cap` would be even higher (since 15 GB > 8 GB budget), further confirming that `needed_bins` remains the binding constraint.

### Change 3: Overflow guard — NO IMPACT

This adds one integer comparison per new bin creation:

```python
if ctx.next_grid_idx >= max_grid_idx:
    ctx.is_feasible = False
    continue
```

For P50M2 with `max_bins_per_sol = 16`, the guard never triggers (solutions rarely need more than ~8 bins for 25 parts/machine). The cost is one `if` check per `_start_new_bin` call and per Phase 6 entry — a few hundred integer comparisons per generation total. This is completely undetectable in profiling.

### Change 4: Tighter VRAM budget — NO IMPACT

Current allocation for P50M2:

```
max_bins_per_sol = 16  (capped by needed_bins, not VRAM)
max_total_bins   = 450 × 16 = 7,200
grid_states:  7,200 × 300 × 250 × 4 = 2.16 GB
grid_ffts:    7,200 × 300 × 126 × 8 = 2.18 GB
Total:  ~4.34 GB  (~27% of 16 GB)
```

Whether the VRAM budget is 50% (8 GB) or explicit (15 GB), the result is the same: `needed_bins = 16` is the binding constraint, and the grid tensors use ~4.34 GB. The tighter formula produces a higher `vram_cap`, but `min(needed_bins, vram_cap) = 16` regardless.

### CHUNK_SIZE adaptive sizing — conditional

For P50M2 (H=300, W=250), the current hardcoded `CHUNK_SIZE = 750` produces per-chunk intermediates of:

```
overlap_batch:   750 × 300 × 250 × 4 = 225 MB
batch_grid_ffts: 750 × 300 × 126 × 8 = 227 MB
batch_part_ffts: 750 × 300 × 126 × 8 = 227 MB
Total:  ~679 MB
```

With ~4.34 GB used by grids and ~0.7 GB per chunk, there is ample room. An adaptive formula would compute a CHUNK_SIZE ≥ 750 for these dimensions, so it would either use 750 (if capped) or a larger value (which benchmarks showed gives no speedup — the 750–1500 range was within noise). Either way, no regression.

**To guarantee this**: cap the computed CHUNK_SIZE at 750 (or whatever the current empirical optimum is) so that the adaptive formula can only shrink chunks for large machines, never change them for machines where the current value works.

### Approach A (drop `grid_ffts`) — potential SMALL regression

This is the one change that would affect P50M2's hot path. Currently, Phase 2 only recomputes `rfft2` for bins whose grids changed in the previous wave. In a typical P50M2 wave:

- **Total open bins across all contexts:** ~2,000–3,500 (450 contexts × 4–8 bins each)
- **Invalid bins (changed last wave):** ~450 (one placement per context per wave)
- **Phase 2 recomputes:** ~450 FFTs (the invalid ones)

Without caching, Phase 2 would recompute `rfft2` on ALL ~2,000–3,500 open bins every wave — roughly **4–8× more FFTs** in Phase 2.

Phase 2 is currently 7.3% of total time (~0.17s over 5 gens, or ~0.034s/gen). A 4–8× increase would add **~0.10–0.24s/gen**, bringing P50M2 from ~2.6s to ~2.7–2.8s/gen. This is a measurable but small regression (~4–9%).

**Mitigation**: Approach A is not a required change — it's an optimization for large instances. It can be made conditional: use `grid_ffts` caching when the full allocation fits in VRAM (single sub-batch), drop it only when sub-batching is needed. This gives the best of both worlds:

```python
use_fft_cache = (sub_batch_size >= num_solutions)  # single sub-batch → cache fits
if use_fft_cache:
    grid_ffts = torch.zeros(...)  # current behavior
else:
    grid_ffts = None              # recompute per wave
```

With this conditional, P50M2 always takes the cached path (single sub-batch), so regression is zero.

### Approach B (uint8 grids) — potential SMALL regression

Storing `grid_states` as uint8 requires a `.float()` conversion before `rfft2` in Phase 2, and changes the CUDA batch update kernel to write uint8 instead of float32.

The `.float()` conversion creates a temporary `(active_bins, H, W)` float32 tensor each wave. For P50M2 with ~450 active bins: `450 × 300 × 250 × 4 = 135 MB`. This is an allocation + elementwise copy. On an A4000, a 135 MB copy takes ~0.05 ms — negligible.

The CUDA kernel change (writing uint8 instead of float32) is a code change with no measurable performance difference for the same data volume.

The `.float()` temporary allocation could cause PyTorch caching allocator churn if the tensor size varies between waves (different number of active bins). In practice, the caching allocator handles this well for tensors of similar size.

**Estimated regression:** < 0.02s/gen. Likely undetectable.

**Mitigation**: Same as Approach A — can be made conditional on whether sub-batching is needed.

### Summary: impact on P50M2

| Change | Impact on P50M2 | Reason |
|--------|-----------------|--------|
| Change 1 (sub-batch loop) | **Zero** | Sub-batch size (1,556) > num_solutions (450); loop runs once |
| Change 2 (remove floor) | **Zero** | Floor was never active (vram_cap = 29 > 5) |
| Change 3 (overflow guard) | **Zero** | One integer comparison per new bin; never triggers |
| Change 4 (tighter budget) | **Zero** | `needed_bins` is the binding constraint, not VRAM cap |
| CHUNK_SIZE adaptive | **Zero** | Adaptive value ≥ 750 for H=300 W=250; capped at current optimum |
| Approach A (drop FFT cache) | **~0.10–0.24s/gen** if applied unconditionally | 4–8× more Phase 2 FFTs per wave |
| Approach B (uint8 grids) | **< 0.02s/gen** | Small `.float()` conversion overhead per wave |

**The four required changes (1–4) have zero impact on P50M2.** The sub-batch loop runs once, the floor and budget changes don't affect the final `max_bins_per_sol`, and the overflow guard is a single integer comparison.

**Approaches A and B** are optional optimizations for large instances. If applied unconditionally, Approach A causes a small but measurable regression (~4–9%). Both can be made conditional — use the current (cached, float32) path when everything fits in one sub-batch, switch to the memory-optimized path only when sub-batching is needed. With this conditional logic, **all changes have zero impact on P50M2**.

### Generalizing beyond P50M2

The zero-impact guarantee holds for any instance where `num_solutions ≤ sub_batch_size`, i.e., where all solutions fit in a single sub-batch. The threshold depends on `nbParts`, `nbMachines`, machine dimensions, and GPU VRAM:

```
sub_batch_size = available_vram // (needed_bins × bytes_per_bin)
```

For a 16 GB GPU with H=300, W=250:
- P50M2  (500 solutions, 16 bins):  sub_batch_size ≈ 1,556 → **single sub-batch** → no impact
- P75M2  (750 solutions, 25 bins):  sub_batch_size ≈ 997  → **single sub-batch** → no impact
- P100M4 (1000 solutions, 33 bins): sub_batch_size ≈ 753  → **single sub-batch for most machines** → no impact on those machines; machines with H=W=400 may need 2 sub-batches (sub_batch_size ≈ 362, 1000 solutions → 3 sub-batches), with proportional slowdown for those machines only

The key property of the sub-batching design is that it **degrades gracefully**: instances that currently work keep working at full speed; larger instances automatically split as needed rather than crashing.
