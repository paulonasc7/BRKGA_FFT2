# Phase 3/4/5 Native Fusion Plan (Behavior-Preserving)

## Goal

Reduce runtime by removing Python orchestration overhead in the hot path, while keeping algorithm behavior and decisions exactly the same.

This is a code-path rewrite, not an algorithm rewrite.

## What "fusion" means here

Target architecture per wave:

1. Python makes one call into a native module.
2. Native module runs Phase 3 -> Phase 4 -> Phase 5 as one pipeline.
3. Native module returns final placement decisions (or "open new bin" decisions).
4. Python applies updates and continues.

There is still one boundary at call entry and one at call exit, but no repeated Python <-> native bouncing inside phases 3/4/5.

## Scope (and non-scope)

In scope:

- Fuse Phase 3 candidate collection + Phase 5 selection/orchestration into native.
- Keep Phase 4 FFT execution in the same fused flow (using existing GPU FFT path, not reimplementing FFT math from scratch).
- Replace Python list/object-heavy wave orchestration with packed, contiguous buffers.

Not in scope:

- Any heuristic or objective changes.
- Any candidate pruning not already present.
- Any tie-break rule changes.
- Any approximation.

## Behavior contract (must remain identical)

The fused path must preserve:

- Same candidate generation eligibility checks.
- Same candidate ordering.
- Same score calculation and tie-break precedence.
- Same deterministic selection outcome for a fixed seed/environment.
- Same phase-level decisions (bin/rotation/row/col/new-bin) for golden checks.

## Why this can help without changing the algorithm

Current bottleneck pattern is not only math cost; it is also orchestration cost:

- Python loops over many tests/contexts.
- Python object creation and list appends.
- Multiple marshaling passes between representations.

Fusion reduces repeated materialization and interpreter overhead by keeping intermediate state in native arrays through the whole 3/4/5 pipeline.

## Proposed module boundary

Native module input (per wave):

- Context metadata arrays.
- Bin metadata arrays.
- Rotation/shape/density metadata arrays.
- Pointers/handles to required GPU-side tensors for FFT flow.
- Any fixed constants (H, W, limits, thresholds).

Native module output (per wave):

- Placement decisions:
  - context index
  - test index or explicit (bin, rotation, row, col)
- New-bin decisions (context index list).
- Optional compact diagnostics counters (for debugging/validation only).

## Data layout strategy

Use struct-of-arrays style contiguous buffers:

- `int32/float32` arrays for indices, dimensions, offsets, scores.
- Reusable per-wave scratch buffers with capacity growth (avoid per-wave realloc).
- Stable index maps so decision ordering is deterministic.

Avoid:

- Python tuples/lists in hot loops.
- Reconstructing equivalent arrays multiple times per wave.
- Frequent small allocations.

## Phase-by-phase fused execution sketch

### Phase 3 (native)

- Collect valid tests for pass-1 and pass-2 using same existing rules.
- Build compact test arrays directly in native memory.

### Phase 4 (GPU FFT path invoked inside fused flow)

- Run existing FFT/IFFT computations for collected tests.
- Write results into native score/result buffers.

### Phase 5 (native)

- Perform same lexicographic selection with unchanged tie-break order.
- Emit final decisions and contexts requiring new bins.

Python then only applies decided updates.

## Determinism and correctness guardrails

Determinism guardrails:

- Preserve iteration order exactly.
- Preserve score comparison order exactly.
- Avoid unstable reductions where order can drift.

Correctness checks after each step:

- Golden-reference comparison on fixed seeds.
- Decision-level diff checks:
  - selected bin
  - rotation
  - placement coordinates
  - new-bin triggers

No rollout step is accepted if behavior diverges unexpectedly.

## Incremental implementation plan

1. Freeze current behavior contract and test corpus (golden seeds/instances).
2. Introduce packed-buffer path in Python first (same outputs, lower object churn).
3. Move Phase 3 collect to native with identical ordering.
4. Move Phase 5 best-per-context selection to native with identical tie-break logic.
5. Stitch Phase 3/4/5 into one native call boundary.
6. Keep a feature flag for fallback during validation.
7. Promote fused path to production default after passing correctness and perf gates.

## Risk areas

- Ordering drift causing different placements even if objective stays close.
- Floating-point comparison subtleties when changing execution context.
- Hidden marshaling overhead if boundary design is not coarse enough.

Mitigations:

- Explicit deterministic ordering tests.
- Decision-level golden diffs.
- Coarse-grained API boundary (one wave call, not many subcalls).

## Expected performance impact

This path targets interpreter/orchestration costs and can improve end-to-end time without changing decisions.

Practical expectation:

- Meaningful but workload-dependent gain.
- Best results when Python-side object churn is a large share of wave time.
- Still bounded by FFT time and unavoidable CPU work outside fused path.

## Benchmark protocol for this work

When evaluating fused vs current-best:

- Same seed(s), same instance(s), same generation count.
- Same environment flags (except fused toggle).
- Compare:
  - mean generation time
  - total wave-function time
  - per-phase ms/wave
  - correctness/golden diffs

## Acceptance criteria

The fused path is acceptable only if:

1. Decision behavior matches golden references on required checks.
2. No correctness regressions.
3. End-to-end speedup is positive and stable across repeated runs.

## Extension: Full Native Decoder (Phase 1-6) with Python BRKGA

It is reasonable to extend the same approach from fused Phase 3/4/5 to the full decoder (Phase 1-6), while keeping BRKGA evolution logic in Python.

### High-level architecture

Python responsibilities (unchanged):

- Population lifecycle (init, elites, crossover, mutants).
- Generation loop and stopping criteria.
- Calling evaluator once per batch/population.

Native responsibilities (expanded):

- Full decoding and fitness evaluation pipeline (Phase 1-6).
- Wave processing, placement decisions, bin updates, and final makespan/fitness outputs.

Call boundary:

- One coarse call per evaluated batch, for example:
  - `fitness = native_evaluate_population(chromosomes, problem_context, runtime_flags)`

This does not mean zero boundaries; it means one entry and one return per evaluation batch, instead of repeated per-phase round-trips.

### One module vs multiple modules

Both are valid:

- One monolithic native module exposing one evaluator API.
- Multiple native internal components hidden behind one stable evaluator API.

Recommended practical shape:

- External API: one evaluator entry point.
- Internal implementation: split files/components by concern (phase orchestration, FFT interface, scoring, bin updates) for maintainability.

### Behavior-preserving constraints

To keep algorithm decisions exactly the same:

- Preserve all phase semantics and ordering rules.
- Preserve tie-break precedence and stable iteration order.
- Preserve deterministic mapping from chromosome to fitness/placements.
- Keep fallback path for differential validation against current evaluator.

### Data ownership and marshaling strategy

For full-decoder nativeization to pay off, data movement must stay coarse-grained:

- Pass chromosomes as contiguous arrays.
- Keep decoder scratch/state in native memory across waves.
- Return compact outputs (fitness and optional debug traces).
- Avoid converting intermediate wave data into Python objects.

### Expected benefits vs Phase 3/4/5-only fusion

Potential additional gains:

- Less Python overhead in Phase 1/2/6 orchestration.
- Fewer temporary Python objects and lower allocation/GC pressure.
- More consistent runtime due to reduced interpreter involvement in hot loops.

### Additional complexity and risk

This is a larger migration than Phase 3/4/5-only fusion:

- Larger validation matrix for correctness equivalence.
- Higher integration complexity around FFT/GPU interfaces.
- Harder debugging without robust trace hooks.

### Suggested rollout sequence

1. Land fused Phase 3/4/5 path with strong golden checks.
2. Expand native boundary to include Phase 6 updates.
3. Expand to Phase 1/2 orchestration.
4. Switch production evaluator to full native decoder behind one flag.
5. Keep legacy Python evaluator for regression bisects until fully retired.

### Acceptance criteria for full-decoder move

1. Fitness and placement decisions match golden references for agreed seeds/instances.
2. Determinism is preserved under repeated runs with fixed seeds.
3. End-to-end gain is materially better than Phase 3/4/5-only fusion.
4. Operational complexity (build, deploy, debug) remains manageable.
