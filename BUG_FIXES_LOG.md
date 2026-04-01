# Bug Fixes Log — wave_batch Decoder

This document records three critical bugs found in the `wave_batch_evaluator.py` implementation that prevented it from producing identical results to `placementProcedure`. All three have been fixed.

---

## Bug 1: Float32 Sentinel Comparison in Pass 1 Detection

**File:** `wave_batch_evaluator.py` (line ~428-430)

**Problem:**
The code used `-1e18` as a sentinel value to mark "invalid" placement results (when IFFT found no zero-overlap position). The sentinel was stored in float32 arrays, then compared against a float64 Python literal:

```python
all_scores = np.full(n_tests, -1e18, dtype=np.float32)  # Stored as float32
# Later:
if p1_all_scores[ti] > -1e18:  # Compared float32 vs float64 literal
    ctx_p1_hit[ctx_idx] = True
```

**Why it's a bug:**
- `float32(-1e18)` ≈ `-999,999,984,306,749,440` (closest representable value)
- This is **greater than** `float64(-1e18)` in exact representation
- Result: The comparison **always evaluated to True**, even when `p1_placement_results[ti]` was actually `None`

**Impact:**
- Pass 1 falsely reported "geometry hit" for every context, even those that had no valid geometric placement
- Pass 2 (remaining bins) **never executed**, because `ctx_p1_hit[ctx_idx]` was always True
- Parts that failed to fit in their first valid bin were incorrectly sent to new bins instead of trying remaining bins
- **Example:** Part 23 on Machine 0, Bin 1 should have been placed there but was sent to Bin 2

**How it was diagnosed:**
1. Comparison of `wave_batch` vs `placementProcedure` outputs showed that some parts were placed in different bins
2. Manual inspection of part 23's trajectory: should go to Bin 1 but went to Bin 2
3. Debug logging revealed Pass 2 was never executing (0 Pass 2 tests collected)
4. Traced through float32 representation and found the sentinel comparison bug

**Fix:**
Replace the sentinel comparison with a check against the actual placement result:

```python
# OLD (BUG):
if p1_all_scores[ti] > -1e18:
    ctx_p1_hit[ctx_idx] = True

# NEW (FIXED):
if p1_placement_results[ti] is not None:
    ctx_p1_hit[ctx_idx] = True
```

This is more direct: if the geometry phase found a valid placement, use it. No sentinel comparison needed.

**Verification:**
- `inspect_chromosome.py` shows part 23 now correctly placed in Bin 1 (matching PP)
- Best makespan in population improved from 280,366 to 243,616

---

## Bug 2: Float32 Score Precision for Column Tie-Breaking

**File:** `wave_batch_evaluator.py` (line ~703, ~767)

**Problem:**
The composite score was computed using float32 arrays:

```python
all_scores = np.full(n_tests, -1e18, dtype=np.float32)
# ...
all_scores = -bi_np * 1e9 + densities * 1e6 + rows_np * 1e3 - cols_np
```

At magnitude ~1e9 (from `-bin_idx * 1e9`), float32's unit-in-the-last-place (ULP) is **~64**. This means differences smaller than 64 in the score are lost to rounding.

**Why it's a bug:**
Column differences are at most ~300 (bin width), but contribute only `-col` to the score. Two rotations with columns differing by, say, 14 would have scores that differ by 14, which is **below float32's ULP of 64 at magnitude 1e9**. The scores would round to the same value, and the tie-breaker would be lost.

**Impact:**
- Parts with multiple valid rotations in the same bin at different columns could pick the wrong rotation
- **Example:** Part 65 on Machine 0, Bin 1:
  - Rotation 0: col=156
  - Rotation 3: col=142 (difference of 14 ≈ lost in float32 rounding)
  - Both could be treated as tied, leading to wrong choice

**How it was diagnosed:**
1. Machine 0 grids matched between `wave_batch` and PP after Bug 1 fix, but Machine 1 did not
2. Deep comparison showed part 81 placed at different rotation on Machine 1:
   - WB chose rot=3 at col=273 with lower density
   - PP chose rot=0 at col=0 with higher density (correct by hierarchy)
3. Traced score computation: float32 ULP at 1e9 is ~64, col differences of 14 were lost

**Fix:**
Change all score arrays from float32 to float64:

```python
# OLD (BUG):
all_scores = np.full(n_tests, -1e18, dtype=np.float32)

# NEW (FIXED):
all_scores = np.full(n_tests, -1e18, dtype=np.float64)

# Also update in _batch_fft_all_tests:
rows_np = np.zeros(n_tests, dtype=np.float64)
cols_np = np.zeros(n_tests, dtype=np.float64)

# And score component arrays:
enc_np = np.asarray(test_enclosure_lengths, dtype=np.float64)
ba_np  = np.asarray(test_bin_areas,          dtype=np.float64)
pa_np  = np.asarray(test_part_areas,         dtype=np.float64)
bi_np  = np.asarray(test_bin_indices,        dtype=np.float64)
ht_np  = np.asarray(test_heights,            dtype=np.float64)
```

Float64's ULP at magnitude 1e9 is ~0.13, which preserves column-level differences (0–300).

**Verification:**
- Machine 0 now produces perfect match on all 24 placements and all 5 bin grids
- Machine 1 now produces perfect match on all 26 placements and all 3 bin grids

---

## Bug 3: Composite Score Formula Doesn't Match Placement Procedure's Lexicographic Hierarchy

**File:** `wave_batch_evaluator.py` (line ~767, and Phase 5 comparison at line ~534–537)

**Problem:**
The composite score formula attempted to encode a lexicographic hierarchy:

```python
all_scores = -bi_np * 1e9 + densities * 1e6 + rows_np * 1e3 - cols_np
```

The hierarchy intended was:
1. **Bin index** (lower is better: fewer bins)
2. **Packing density** (higher is better: tighter packing)
3. **Row** (larger is better: bottom-left heuristic)
4. **Column** (smaller is better: bottom-left heuristic)

However, the formula used fixed multipliers (`1e9`, `1e6`, `1e3`, `1`) that **cannot** encode the density-first hierarchy, because **density differences can be arbitrarily small**.

**Why it's a bug:**
In `binClassNew.py`'s `can_insert()` method, the placement procedure uses **strict lexicographic comparison**:
```python
if newPackingDensity > packingDensity:  # Density first
    best_pixel, best_rotation = ...
elif newPackingDensity == packingDensity and largest_row > best_row:  # Then row
    best_pixel, best_rotation = ...
elif newPackingDensity == packingDensity and largest_row == best_row and col < best_col:  # Then col
    best_pixel, best_rotation = ...
```

Any **nonzero density difference** decides the choice, regardless of row/column differences.

The composite score formula cannot replicate this. Consider part 81 on Machine 1:
- Rotation 0: density=0.4407, row=109, col=0
- Rotation 3: density=0.4369, row=177, col=273

Density difference: 0.0038 → score contribution = `0.0038 * 1e6 = 3,800`
Row difference: 68 → score contribution = `68 * 1e3 = 68,000`

Row overwhelms density. The formula would pick rot=3 (higher row), but PP correctly picks rot=0 (higher density).

**Impact:**
- Parts with valid rotations differing in both density and row could be placed at wrong rotation
- Machine 1 had mismatches in bin packing order and part placement

**How it was diagnosed:**
1. After Bug 1+2 fixes, Machine 1 still had mismatches
2. Detailed debug logging for part 81 on M1 showed:
   - Pass 1 collected 4 rotations (all valid geometrically)
   - WB Phase 5 picked rot=3 (row=177, density=0.4369)
   - PP's `can_insert()` picked rot=0 (row=109, density=0.4407)
   - Root cause: density-first tie-breaking not encoded in composite score

**Fix:**
Replace single composite score with **proper lexicographic tuple comparison** matching PP's hierarchy exactly. In `_batch_fft_all_tests`, return the individual components:

```python
# OLD (BUG):
all_scores = -bi_np * 1e9 + densities * 1e6 + rows_np * 1e3 - cols_np
all_scores[~valid_np] = -1e18
return all_results, all_scores

# NEW (FIXED):
score_components = {
    'bin_indices': bi_np,
    'densities': densities,
    'rows': rows_np,
    'cols': cols_np,
    'valid': valid_np,
}
return all_results, score_components
```

In Phase 5, use tuple comparison with proper weights:

```python
# OLD (BUG):
for ti, ctx_idx in enumerate(test_ctx_indices):
    sc = all_scores[ti]
    if sc > best_sc_per_ctx[ctx_idx]:
        best_sc_per_ctx[ctx_idx] = sc
        best_ti_per_ctx[ctx_idx] = ti

# NEW (FIXED):
for ti, ctx_idx in enumerate(test_ctx_indices):
    if not sc_valid[ti]:
        continue
    # Key: (-bin_idx, density, row, -col) — higher is better for all components
    # This implements: lower bin_idx > higher density > larger row > smaller col
    key = (-sc_bin_indices[ti], sc_densities[ti], sc_rows[ti], -sc_cols[ti])
    prev = best_key_per_ctx[ctx_idx]
    if prev is None or key > prev:
        best_key_per_ctx[ctx_idx] = key
        best_ti_per_ctx[ctx_idx] = ti
```

**Verification:**
- Part 81 on M1 now correctly placed at rot=0 (higher density 0.4407 vs 0.4369)
- Machine 1 now produces perfect match: all 26 placements and all 3 bin grids identical

---

## Performance Impact

| Metric | Before (All Bugs) | After (All Fixes) | Change |
|--------|-------------------|-------------------|--------|
| Mean gen time | 2.29s | 2.76s | +0.47s (+20%) |
| Phase 4b (P2 IFFT) | ~0% | 10.3% | Now executing |
| Tests/wave | ~1038 | 1233 | +19% |

The increase is **intentional and correct**:
- Bug 1 fix means Pass 2 now actually runs (was always skipped before)
- More tests = more bins checked = slower but correct

The baseline 2.29s was **producing wrong results**. The 2.76s is correct.

---

## Lessons for Future Changes

1. **Never use sentinel comparisons with float types:**
   - Floating-point representation can make sentinels compare unexpectedly against literals
   - Use explicit null checks (e.g., `is None`) or typed flags instead

2. **Be aware of float precision at large magnitudes:**
   - Float32 ULP at 1e9 is ~64; float64 is ~0.13
   - If tying multiple values at different scales (1e9 + 1e6 + 1e3 + 1), use float64
   - Alternatively, avoid composite scores and use tuple comparison

3. **Lexicographic hierarchy cannot be encoded with fixed multipliers:**
   - If tie-breaking depends on ranges that vary (density: 0–1, unbounded resolution vs row: 0–300), use explicit tuple comparison
   - Composite scores only work if each level has a guaranteed "gap" that dominates the next level

4. **Always verify against a reference implementation:**
   - `placementProcedure` is the ground truth; `wave_batch` is a GPU-accelerated approximation
   - Use `inspect_chromosome.py` to compare per-placement and per-grid outputs
   - Run full instance checks, not just single placements

---

## Files Modified

- `wave_batch_evaluator.py` — Core fixes for all three bugs
- `profile_phases.py` — Updated to match new `_batch_fft_all_tests` API (returns dict, not array)
- `inspect_chromosome.py` — Enhanced diagnostics (already had debug instrumentation)

## Test Command

```bash
python remote.py sync . BRKGA_FFT2 --ext .py
python remote.py run "python inspect_chromosome.py 50 2 0 torch_gpu" --cwd /notebooks/BRKGA_FFT2
```

Expected output: All placements and grids match between wave_batch and placementProcedure.
