# Implementation Phases for FFT Placement Optimization

This document organizes the 17 identified optimization opportunities into phases based on **complexity**, **risk**, and **dependencies**.

---

## Phase 1: Quick Wins
**Time Estimate**: 15-30 minutes total  
**Risk Level**: 🟢 Very Low  
**Description**: Simple changes with no architectural impact. Can be done immediately with minimal testing.

| Issue # | Title | File(s) | Change Type |
|---------|-------|---------|-------------|
| #16 | Remove unused `grid2` array | `binClassNew.py`, `binClassInitialSol.py` | Delete line |
| #17 | Use `uint8` dtype for grid | `binClassNew.py`, `binClassInitialSol.py` | Change dtype |
| #11 | Pre-compute `best_rotation` | `BRKGA_alg3.py`, `placement.py` | Add field + simplify lookup |
| #10 | Make arrays contiguous at load time | `BRKGA_alg3.py`, `collision_backend.py` | Move `ascontiguousarray` |

---

## Phase 2: Low-Hanging Fruit
**Time Estimate**: 1-2 hours total  
**Risk Level**: 🟡 Low  
**Description**: Straightforward optimizations that improve hot paths without changing interfaces.

| Issue # | Title | File(s) | Change Type |
|---------|-------|---------|-------------|
| #7 | Track enclosure box bounds incrementally | `binClassNew.py`, `binClassInitialSol.py` | Add tracking variables |
| #5 | Vectorize density calculation | `BRKGA_alg3.py` | Replace `itertools.groupby` with NumPy |
| #3 | Create ThreadPoolExecutor once | `BRKGA_alg3.py` | Move executor to `__init__` |

---

## Phase 3: Data Loading Refactor
**Time Estimate**: 2-3 hours  
**Risk Level**: 🟡 Low-Medium  
**Description**: Restructure the data loading to avoid redundant computation. Affects initialization only.

| Issue # | Title | File(s) | Change Type |
|---------|-------|---------|-------------|
| #2 | Separate part loading from machine-specific FFT | `BRKGA_alg3.py` | Restructure nested loops |
| #15 | Sort parts once and reuse order | `BRKGA_alg3.py` | Extract and store sorted order |

---

## Phase 4: Hot Path Optimization
**Time Estimate**: 2-4 hours  
**Risk Level**: 🟠 Medium  
**Description**: Optimize frequently-called functions in the placement procedure. Requires careful testing.

| Issue # | Title | File(s) | Change Type |
|---------|-------|---------|-------------|
| #6 | Pre-allocate buffers for vacancy vector | `binClassNew.py`, `binClassInitialSol.py` | Add buffer management |
| #9 | Optimize sliding window / dictionary access | `binClassNew.py`, `binClassInitialSol.py` | Reduce lookups in loop |
| #4 | Improve initial solution construction | `BRKGA_alg3.py` | Cache makespan, early exit |

---

## Phase 5: Structural Refactoring
**Time Estimate**: 4-8 hours  
**Risk Level**: 🔴 High  
**Description**: Major data structure changes. Affects multiple files and requires extensive testing.

| Issue # | Title | File(s) | Change Type |
|---------|-------|---------|-------------|
| #1 | Replace string key dict lookups | All files | Change data access pattern |
| #14 | Use dataclasses for part/machine data | All files | Define new classes, migrate |

**Note**: Issues #1 and #14 are closely related and should be implemented together.

---

## Phase 6: Parallelization & GPU Optimization
**Time Estimate**: 4-8 hours  
**Risk Level**: 🔴 High  
**Description**: Exploit parallelism at machine and GPU level. Complex interactions with threading/CUDA.

| Issue # | Title | File(s) | Change Type |
|---------|-------|---------|-------------|
| #12 | Machine-level parallelism | `placement.py` | Add parallel execution |
| #13 | GPU batch size optimization | `collision_backend.py` | Batch multiple parts |
| #8 | Cache grid FFT between calls | `collision_backend.py`, `binClassNew.py` | Add caching layer |

---

## Dependency Graph

```
Phase 1 (Quick Wins)
    │
    ▼
Phase 2 (Low-Hanging Fruit)
    │
    ├──────────────────┐
    ▼                  ▼
Phase 3            Phase 4
(Data Loading)     (Hot Path)
    │                  │
    └────────┬─────────┘
             ▼
        Phase 5
    (Structural Refactor)
             │
             ▼
        Phase 6
    (Parallelization)
```

---

## Recommended Approach

1. **Complete Phase 1** entirely before moving on
2. **Benchmark** after each phase to measure improvement
3. **Phases 3 and 4** can be done in parallel (independent concerns)
4. **Phase 5** is optional but enables maximum performance
5. **Phase 6** should only be attempted after stabilizing earlier phases

---

## Checklist

- [x] **Phase 1**: Quick Wins ✅ COMPLETED
  - [x] #16 - Remove `grid2`
  - [x] #17 - Use `uint8` dtype
  - [x] #11 - Pre-compute `best_rotation`
  - [x] #10 - Contiguous arrays at load
  
- [x] **Phase 2**: Low-Hanging Fruit ✅ COMPLETED
  - [x] #7 - Incremental enclosure box
  - [x] #5 - Vectorize density
  - [x] #3 - Reuse ThreadPoolExecutor
  
- [x] **Phase 3**: Data Loading Refactor ✅ COMPLETED
  - [x] #2 - Separate part/FFT loading
  - [x] #15 - Cache sorted parts
  
- [x] **Phase 4**: Hot Path Optimization ✅ COMPLETED
  - [x] #6 - Pre-allocate vacancy buffers
  - [x] #9 - Optimize sliding window
  - [x] #4 - Improve initial solution
  
- [x] **Phase 5**: Structural Refactoring ✅ COMPLETED
  - [x] #1 + #14 - Dataclasses + integer indexing
  - Created `data_structures.py` with `PartData`, `MachinePartData`, `MachineData`, `ProblemData`
  - Updated all files to use attribute access instead of string-keyed dict lookups
  
- [x] **Phase 6**: Parallelization ✅ COMPLETED
  - [x] #12 - Parallel machines: Implemented `ThreadPoolExecutor` in `placement.py` to process machines concurrently
  - [x] #13 - GPU batching: Evaluated - current rotation batching already efficient; multi-part batching incompatible with greedy algorithm
  - [x] #8 - FFT caching: Evaluated - limited benefit since grid FFT is computed once per `can_insert` call and changes after each insertion

---

## All Phases Complete!

All 17 optimization recommendations have been evaluated and implemented where applicable. Key improvements:

1. **Memory**: 8x reduction via `uint8` dtype, removed duplicate `grid2`
2. **Data Structures**: Dataclasses replace string-keyed dicts (faster attribute access)
3. **Algorithms**: Incremental tracking (O(1) vs O(n×m)), vectorized operations
4. **Parallelism**: Machines processed concurrently, reused ThreadPoolExecutor
5. **Loading**: Separated part geometry from machine-specific FFT computation
