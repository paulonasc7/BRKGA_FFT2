# BRKGA_FFT2 Deep Analysis

Complete technical reference for the BRKGA-based 2D nesting optimizer. Written to give a future reader (or AI assistant) full context without needing to re-analyze the codebase.

**Last updated:** 2026-03-26
**Benchmark command:** `python BRKGA_alg3.py 50 2 0 torch_gpu wave_batch 1 1 3`

---

## 1. Problem Domain

### What this project solves

A **2D Nesting + Scheduling** problem for additive manufacturing (3D printing). Given:

- **N parts** (binary pixel matrices representing 2D footprints of 3D objects)
- **M machines** (each with a rectangular building plate / bin of specific dimensions)

The goal is to **assign parts to machines**, **pack them into bins (batches) on each machine's plate**, and **minimize the worst-case makespan** (the maximum total time any single machine takes to finish all its batches).

### What is a "part"?

A binary matrix (`.npy` file) where 1 = material, 0 = empty. Example: a 47x32 matrix representing the footprint of a 3D-printed component. Each part can be rotated in 2 or 4 orientations (0/90/180/270 degrees). If a part is symmetric under 180-degree rotation, only 2 rotations are considered.

### What is a "bin" / "batch"?

A building plate (e.g., 300x250 pixels). Multiple parts are packed side-by-side into a single bin. The machine processes one bin at a time. Each bin has:
- **Setup time** (fixed cost per bin)
- **Processing time** (sum of all parts' volume-based processing times)
- **Height-based time** (driven by the tallest part in the bin)

### What is "makespan"?

`makespan(machine) = sum_over_bins(setup_time + proc_time + proc_time_height)`
`fitness = max(makespan(machine_0), makespan(machine_1), ..., makespan(machine_M-1))`

Lower = better. The BRKGA minimizes this.

---

## 2. Algorithm: BRKGA

### Biased Random-Key Genetic Algorithm

A metaheuristic that encodes solutions as floating-point chromosomes and uses a greedy decoder to map them to concrete placements.

### Chromosome structure

Each chromosome has `2N` genes (floats in [0, 1]):

```
[SV_0, SV_1, ..., SV_{N-1}, MV_0, MV_1, ..., MV_{N-1}]
 \___ Sequencing Values ___/ \__ Machine assignment Values __/
```

- **SV (first N genes):** Determine the order in which parts are placed. Parts are sorted by their SV value (ascending), so the part with the lowest SV gets placed first.
- **MV (last N genes):** Determine which machine a part is assigned to. With M machines and thresholds `[1/M, 2/M, ..., (M-1)/M]`:
  - `MV[i] <= threshold[0]` → machine 0
  - `threshold[0] < MV[i] <= threshold[1]` → machine 1
  - `MV[i] > threshold[M-2]` → machine M-1

### BRKGA generational loop

```
1. INITIALIZE: random population of num_individuals chromosomes (float32)
   - population[0] = initial heuristic solution
2. EVALUATE: decode all chromosomes → calculate fitness (makespan)
3. FOR EACH GENERATION:
   a. PARTITION: split into elites (top 10%) and non-elites
   b. MATE: create offspring via biased crossover (70% elite gene probability)
   c. MUTATE: add random chromosomes (15% of population)
   d. EVALUATE offspring only (elite fitness carried over from previous generation)
   e. MERGE: new population = elites + offspring + mutants
   f. TRACK: record best fitness, mean fitness, elapsed time
4. RETURN best solution found
```

### Key parameters (for P50M2 with mult=10)

| Parameter | Value | Formula |
|-----------|-------|---------|
| `num_individuals` | 500 | `mult * nbParts = 10 * 50` |
| `num_elites` | 50 | `ceil(500 * 0.10)` |
| `num_mutants` | 75 | `ceil(500 * 0.15)` |
| `num_offspring` | 375 | `500 - 50 - 75` |
| `num_gene` | 100 | `2 * 50` |
| `eliteCProb` | 0.70 | Probability of inheriting elite parent's gene |

### Elite reuse optimization

Elites are NOT re-evaluated. In `fit()`:
```python
elites, non_elites, elite_fitness_list = self.partition(population, fitness_list)
offspring = np.concatenate((mutants, offsprings), axis=0)
offspring_fitness_list = self.cal_fitness(offspring)  # only 450 solutions, not 500
fitness_list = list(elite_fitness_list) + offspring_fitness_list  # elite fitness reused
```

---

## 3. File-by-File Analysis

### Files used when running `python BRKGA_alg3.py 50 2 0 torch_gpu wave_batch 1 1 3`

#### `BRKGA_alg3.py` (main entry point, 610 lines)

**Two sections:** the `BRKGA` class (lines 22-260) and the `__main__` script (lines 262-609).

**`__main__` section (lines 262-609):**
1. **Parse CLI args** (lines 262-275): `nbParts=50, nbMachines=2, instNumber=0, backend=torch_gpu, eval_mode=wave_batch, workers=1, chunksize=1, generations=3`
2. **Load instance** (lines 277-283): Read `P50M2-0.txt` → 50 part IDs (e.g., `38 19 38 37 55 ...`)
3. **Load job specs** (lines 286-300): Excel data cached to pickle after first load. Contains processing times, areas, rotation counts.
4. **Build ProblemData** (lines 306-418): Two-phase loading:
   - Phase 1 (lines 310-365): Load each unique part's binary matrix, compute rotations (2 or 4), densities (max consecutive 1s per row), pre-compute GPU tensors and uint8 versions, prepare JIT data structures.
   - Phase 2 (lines 367-410): For each machine, compute per-part FFTs (padded, flipped rotation convolved with bin dimensions), processing times.
5. **Build initial solution** (lines 424-581): Greedy constructive heuristic — sort parts by decreasing height then area, place each greedily into the machine/bin that minimizes makespan.
6. **Run BRKGA** (lines 584-608): Create BRKGA model with `mult=10`, run `fit()`, export history to Excel.

**`BRKGA` class (lines 22-260):**
- `__init__`: Store problem data, configure eval mode, create WaveBatchEvaluator (for wave_batch mode) or ThreadPoolExecutor (for thread mode). Fitness cache only initialized for non-wave_batch modes.
- `evaluate_solution(solution)`: Calls `placementProcedure()` — used only by serial/thread/process modes.
- `cal_fitness(population)`: Dispatches to `wave_batch_evaluator.evaluate_batch()` for wave_batch mode, or iterates with cache for other modes.
- `partition(population, fitness_list)`: O(n) `argpartition` to split into elites/non-elites. Elites sorted; non-elites unsorted.
- `mating(elites, non_elites)`: Fully vectorized — generates all offspring in one NumPy operation using a boolean crossover mask.
- `mutants()`: Returns `num_mutants` random chromosomes (float32).
- `fit(verbose)`: The main generational loop (see Section 2).
- `shutdown()`: Cleans up executor, reports cache stats (for non-wave_batch only).

#### `wave_batch_evaluator.py` (~500 lines) — THE CRITICAL FILE for wave_batch mode

This is where ~99% of per-generation time is spent in wave_batch mode.

**Core idea:** Instead of evaluating each of 450 solutions sequentially (each doing its own FFT calls), batch ALL solutions' FFT operations together into large GPU tensor operations.

**Data structures:**
- `BinState` (dataclass): Tracks one bin's grid (numpy + GPU), vacancy vector, area, enclosure bounds, processing times.
- `BatchPlacementContext` (dataclass): Tracks one solution-machine combination's progress through the greedy placement algorithm.

**`__init__` pre-computation:**
- Builds `self.flat_parts_gpu`: All part rotation matrices concatenated into a single float32 CUDA tensor at init time. Used by the custom CUDA kernel for batched grid updates.
- Builds `self.part_update_meta`: Dict mapping `(part_id, rot) → (flat_offset, h, w)` for indexing into `flat_parts_gpu`.
- Falls back gracefully if CUDA kernel compilation fails (`flat_parts_gpu = None`).

**`evaluate_batch(chromosomes)` flow:**
1. For each machine (0, 1): call `_process_machine_batch()`
2. Final fitness = max makespan across machines

**`_process_machine_batch()` flow:**
1. **Decode sequences** (`_decode_sequences`): For each of 450 chromosomes, determine which parts go to this machine and in what order (based on SV/MV values).
2. **Initialize contexts** (`_init_batch_contexts`): Create 450 `BatchPlacementContext` objects, each with empty bin lists.
3. **Allocate GPU tensors**: Pre-allocate `grid_states` (max 10 bins per solution × 450 solutions = 4500 grids of 300×250) and `grid_ffts` (same shape, complex64). Also cache `row_idx`, `col_idx`, `neg_inf` tensors per machine.
4. **Process waves**: Iterate until all contexts are done. Each wave places one part per active context.

**`_process_wave_true_batch(contexts, ...)` — the hot inner loop:**

This is the function that runs ~25-50 times per machine (once per part placement "depth"). Each wave:

- **Phase 1: Gather part info** — For each active context, identify the next part to place, check feasibility (size fits bin dimensions).
- **Phase 2: Batch grid FFT update** — Collect all bins whose grid FFT is invalid, compute FFT in ONE batched `torch.fft.rfft2()` call across all of them.
- **Phase 3: Collect all tests** — For each (context × bin × rotation) triple, check vacancy (Numba JIT), and if feasible, record the test into parallel arrays (`test_grid_indices`, `test_part_ffts`, `test_heights`, `test_widths`, etc.). Also collects `test_ctx_indices`, `test_enclosure_lengths`, `test_bin_areas`, `test_part_areas` for GPU-side scoring.
- **Phase 4: Batch FFT collision check** — Call `_batch_fft_all_tests()` which performs ONE chunked `torch.fft.irfft2()` for ALL tests, finds valid positions via GPU masking and scoring, transfers results to CPU in ONE `.cpu()` call per chunk. Returns both `placement_results` and per-test `all_scores` (NumPy-computed composite scores incorporating density, row, and column).
- **Phase 5: Best placement selection + execution** — Two-pass approach:
  1. Find best test per context using pre-computed scores (simple NumPy argmax, no Python density loop).
  2. Execute all GPU grid updates in one batch via custom CUDA kernel (`cuda_batch_update.py`), then all CPU updates (NumPy grid + Numba vacancy) which overlap with async GPU.
- **Phase 6: New bins** — Any context that found no valid placement gets a new bin.

**`_batch_fft_all_tests()` — the GPU workhorse:**
- Chunks tests into groups of 750 (tuned for RTX A4000, 16GB VRAM)
- Each chunk: gather grid FFTs and part FFTs via tensor indexing → batched `irfft2` → round → mask valid positions (row ≥ h-1, col ≥ w-1) → score = row*(W+1) - col (maximize row, minimize col) → single `torch.max()` for both value and index → transfer to CPU
- Returns list of `(col, row)` or `None` per test, plus per-test composite `all_scores` (numpy array)

**`_place_part_in_bin()` — the placement executor:**
- Updates numpy grid (uint8) for vacancy tracking
- Updates GPU grid tensor (float32) for FFT
- Updates vacancy vector via Numba JIT
- Updates area, enclosure bounds, processing times
- Invalidates grid FFT cache flag
- Still used by `_start_new_bin()` (Phase 6); Phase 5 uses the batched CUDA kernel path instead

#### `cuda_batch_update.py` (194 lines) — custom CUDA kernel for Phase 5

Replaces ~450 sequential CUDA kernel launches per wave with a single kernel launch that processes all placements in parallel.

**Architecture:**
- Uses `torch.utils.cpp_extension.load_inline` to JIT-compile CUDA code at first use
- Compiled `.so` cached in `~/.cache/torch_extensions/` (first run takes 2-3 min)
- Uses `name='_cuda_batch_update_ext'` to avoid Python module namespace collision with the `.py` file
- `total_cells` passed as explicit int parameter — never dereferences GPU pointer from host code (avoids segfault on discrete GPUs)

**Kernel design:**
- 1D thread grid, one thread per cell across all placements
- Binary search over prefix-sum `cell_offsets` array maps each thread to its placement
- No atomics needed — each placement writes to a unique `(grid_idx, row, col)` region
- Pre-computed `flat_parts_gpu` tensor (all part rotations concatenated) indexed via `part_update_meta`

**Python wrapper:** `batch_grid_update(grid_states, flat_parts_gpu, placements, H, W)` — builds per-wave index tensors from a list of `(grid_state_idx, y_start, x_start, flat_offset, h, w)` tuples and dispatches the kernel.

**`_start_new_bin()` — creates a fresh bin:**
- Allocates numpy grid + zeros the GPU grid slot
- Places first part at bottom-left using `best_rotation` (minimum height)

#### `collision_backend.py` (287 lines)

**Base class** (`BaseCollisionBackend`): Defines interface — `prepare_part_fft()`, `find_bottom_left_zero()`, `find_bottom_left_zero_batch()`, `create_grid_state()`, `update_grid_region()`.

**`TorchCollisionBackend` (primary, line 66-212):**
- Uses PyTorch FFT on GPU (CUDA) or CPU
- `prepare_part_fft(part_matrix, bin_length, bin_width)`: Flip part 180°, pad to bin dimensions, compute FFT → returns a tensor stored in GPU VRAM
- `prepare_rotation_tensor(part_matrix)`: Pre-transfer rotation matrix to GPU as float32 tensor
- `find_bottom_left_zero_batch()`: Batch FFT convolution for multiple rotations. Uses the same scoring approach as wave_batch (row*(W+1) - col), single GPU→CPU transfer. For ≤2 rotations, falls back to sequential (less overhead).
- `compute_grid_fft()`: Compute and return grid FFT (used for caching in BuildingPlate)

**`NumpyCollisionBackend` (fallback, line 215-251):**
- Pure NumPy implementation. Same algorithm, no GPU.

**`create_collision_backend(name)` (line 253-286):**
- Factory function. Supported: `torch_gpu`, `torch_gpu_unbatched`, `torch_cpu`, `torch_cpu_unbatched`, `numpy_cpu`, `cupy_gpu`, `cupy_gpu_optimized`

**Global settings (lines 1-39):**
- TF32 disabled by default (cuFFT doesn't use it)
- cuFFT plan cache size = 32 (configurable via env var)
- `torch.set_num_threads(1)` and `torch.set_grad_enabled(False)` set in `BRKGA_alg3.py`

#### `placement.py` (166 lines) — used by serial/thread/process modes, NOT wave_batch

Decodes a single chromosome into a placement and returns the makespan.

**`placementProcedure()`:**
1. Split chromosome into SV and MV
2. For each machine: determine assigned parts, sort by SV, create task tuple
3. If ≥4 machines: process in parallel via ThreadPoolExecutor; otherwise sequential
4. Return worst makespan across machines

**`_process_single_machine()`:**
- For each part in sorted order: try existing bins (area check + FFT collision), create new bin if none work
- Uses `BuildingPlate.can_insert()` (from `binClassNew.py`)

#### `binClassNew.py` (172 lines) — BuildingPlate for BRKGA evaluation

The bin/building plate representation used during BRKGA generation evaluation.

**State:**
- `grid`: numpy uint8 array (H×W) — binary occupancy map
- `grid_state`: GPU tensor (float32) — mirror of grid for FFT
- `vacancy_vector`: 1D int array — max contiguous zeros per row (fast feasibility filter)
- `_grid_fft_cache`: Cached FFT of grid_state (invalidated after each insert)
- `area`, `enclosure_box_length`, `min/max_occupied_row`: O(1) incremental tracking
- `processingTime`, `processingTimeHeight`, `partsAssigned`: Scheduling state

**`can_insert(part, machPart)` — the critical per-part check:**
1. **Vacancy filter** (Numba JIT): For each rotation, check if the part's density profile fits anywhere in the vacancy vector. Uses batched `check_rotations_feasibility()` from numba_utils.
2. **FFT collision check**: For feasible rotations, batch-compute FFT convolution to find exact valid positions. Uses cached grid FFT.
3. **Best position selection**: Among all valid positions across rotations, pick the one that maximizes packing density (area / enclosure_box_length × width), with tie-breaking by bottom-most row then left-most column.
4. **Execute placement**: If a valid position found, insert the part.

**`insert(x, y, partMatrix, shapes, partArea, gpu_tensor)`:**
- Update numpy grid (uint8 += partMatrix)
- Update GPU grid state via collision_backend
- Invalidate FFT cache
- Update enclosure bounds (O(1))
- Update vacancy vector (Numba JIT single-pass)

#### `binClassInitialSol.py` (143 lines) — BuildingPlate for initial solution

Nearly identical to `binClassNew.py` but used during initial solution construction in `__main__`. The `can_insert()` returns `(result, best_pixel, best_rotation)` instead of just `result` — the initial solution builder needs the coordinates and rotation for its own placement logic.

Both versions now use the same Numba JIT vacancy update (`update_vacancy_vector_rows`).

#### `data_structures.py` (106 lines) — typed data containers

Dataclasses replacing the original nested dict-of-dicts-of-dicts:

- **`PartData`**: id, area, nrot, rotations (list of numpy arrays), shapes, densities, best_rotation, rotations_gpu (pre-computed GPU tensors), rotations_uint8 (pre-cast), JIT-prepared data (densities_flat, density_offsets, shapes_heights, shapes_widths)
- **`MachinePartData`**: ffts (pre-computed FFTs per rotation), proc_time, proc_time_height
- **`MachineData`**: bin_length, bin_width, bin_area, setup_time, parts (dict[int → MachinePartData])
- **`ProblemData`**: parts (dict[int → PartData]), machines (list[MachineData]), instance_parts, instance_parts_unique

#### `numba_utils.py` (189 lines) — JIT-compiled performance functions

All functions use `@jit(nopython=True, cache=True)` for ahead-of-time compilation.

- **`check_vacancy_fit_simple(vacancy, density)`**: Sliding window check — can a part with given density profile fit somewhere in the vacancy vector? O(n × h) where n = bin length, h = part height.
- **`check_vacancy_fit_single(vacancy, density, shape_height)`**: Same but takes explicit height parameter.
- **`check_rotations_feasibility(vacancy, densities_flat, density_offsets, shapes_heights, shapes_widths, bin_length, bin_width, nrot)`**: Batch check all rotations in single JIT call, using pre-prepared flat arrays.
- **`update_vacancy_vector_rows(vacancy_vector, grid_rows, y_start)`**: After inserting a part, recompute max consecutive zeros for the modified rows. Single-pass O(rows × width).

#### `collision_backend_cupy.py` (410 lines) — CuPy alternative backend

Two classes: `CuPyCollisionBackend` (basic) and `CuPyCollisionBackendOptimized` (custom CUDA kernel for position extraction). **Benchmarked as 4.7x slower than PyTorch** due to worse cuFFT plan caching. Not used.

#### `profile_phases.py` (303 lines) — per-phase wall-clock profiler

Standalone script that provides accurate per-phase timing of `_process_wave_true_batch`. Subclasses `WaveBatchEvaluator` as `TimedEvaluator`, inserting `torch.cuda.synchronize()` + `time.perf_counter()` around each phase boundary. No cProfile distortion.

Usage: `python profile_phases.py 50 2 0 torch_gpu 5` (nbParts, nbMachines, instNumber, backend, n_generations)

Produces a breakdown table showing time, percentage, and ms/wave for all 6 phases. This is the authoritative profiling tool for wave_batch performance.

#### `profile_quick.py` (276 lines) — cProfile-based profiling utility

Standalone script that sets up the problem and profiles BRKGA generations using cProfile. Has `setup_problem()` function (useful for benchmarking) and two modes:
- `profile`: Runs 3 generations under cProfile, prints top functions by cumulative and self time
- `eval`: Times single solution evaluations

---

## 4. The FFT Collision Detection Algorithm

### How parts are placed using FFT

This is the mathematical core. To check if a part can be placed at position (x, y) on a grid without overlapping existing parts:

1. **Grid**: Binary matrix G where G[r][c] = 1 if occupied, 0 if free
2. **Part**: Binary matrix P (the rotation being tested)
3. **Flip**: Reverse P in both axes → P_flipped (180° rotation)
4. **Pad**: Pad P_flipped to grid dimensions → P_padded
5. **FFT convolution**: `overlap = IFFT(FFT(G) * FFT(P_padded)).real`
6. **Round**: `overlap = round(overlap)` (remove floating-point noise)
7. **Interpretation**: `overlap[r][c]` gives the number of overlapping cells if the part's bottom-right corner is placed at position (r, c). A value of 0 means NO overlap → valid placement.
8. **Valid region**: Only positions where `r >= h-1` and `c >= w-1` are valid (part must fit within grid bounds).
9. **Bottom-left preference**: Among valid zeros, find the one with the highest row (bottom-most), then lowest column (left-most). Scoring: `score = row * (W+1) - col`.

### Why this is efficient

Instead of checking every possible position with O(h × w) overlap tests, the FFT approach checks ALL positions simultaneously in O(H × W × log(H × W)) time, where H, W are the grid dimensions. For grids of 300×250, this is much faster than brute force.

### Pre-computation

Part FFTs are computed ONCE at startup (in Phase 2 of data loading) and stored in `MachinePartData.ffts`. The grid FFT is computed once per `can_insert()` call and cached until the grid changes.

---

## 5. Execution Modes

### Collision backends

| Backend | Device | Batching | Status |
|---------|--------|----------|--------|
| `torch_gpu` | CUDA | Yes (≥3 rotations) | **Primary**, fastest |
| `torch_gpu_unbatched` | CUDA | No | Available |
| `torch_cpu` | CPU | Yes | Available |
| `torch_cpu_unbatched` | CPU | No | Available |
| `numpy_cpu` | CPU | Yes (numpy) | Fallback |
| `cupy_gpu` | CUDA | Yes | 4.7x slower than torch, not recommended |
| `cupy_gpu_optimized` | CUDA | Yes + custom kernel | Same, not recommended |

### Evaluation modes

| Mode | How fitness is computed | Best for |
|------|------------------------|----------|
| `serial` | One solution at a time, sequentially | GPU backends (GPU already parallel) |
| `thread` | ThreadPoolExecutor with N workers | CPU backends |
| `process` | ProcessPoolExecutor (not compatible with GPU) | CPU-only, multi-core |
| `wave_batch` | Batch ALL solutions' FFTs in one GPU operation | GPU, highest throughput |
| `auto` | Selects `serial` for GPU, `thread` for CPU | Default |

### wave_batch vs serial performance

| Mode | Time/gen (P50M2, 500 individuals) |
|------|-----------------------------------|
| serial | ~18.2s |
| wave_batch | ~5.9s |
| **Speedup** | **~3.1x** |

The speedup comes from batching thousands of FFT operations into fewer, larger GPU operations, reducing Python loop overhead and GPU kernel launch overhead.

---

## 6. Data Flow Diagram

```
Instance File (P50M2-0.txt)
    │ "38 19 38 37 55 97 65 85 ..."
    ▼
Parse → instanceParts[50]        instancePartsUnique[~40]
                                        │
                         ┌──────────────┤
                         ▼              ▼
              Part Matrices (.npy)   Job Specs (Excel/pickle)
                         │              │
                         ▼              ▼
              ┌──── ProblemData ────────────────────┐
              │  parts: {id: PartData}              │
              │    - rotations[2-4] (int32 numpy)   │
              │    - rotations_gpu[2-4] (GPU tensor)│
              │    - rotations_uint8[2-4]           │
              │    - shapes, densities, best_rot    │
              │    - JIT data (flat/offsets)         │
              │  machines: [MachineData]             │
              │    - bin_length, bin_width, setup_t  │
              │    - parts: {id: MachinePartData}   │
              │      - ffts[2-4] (GPU tensors)      │
              │      - proc_time, proc_time_height  │
              └────────────┬────────────────────────┘
                           │
              Initial Solution (greedy heuristic)
                           │
                           ▼
                    ┌──── BRKGA ────┐
                    │ Population    │
                    │ [500 × 100]   │    500 chromosomes, each 100 genes
                    │ (float32)     │
                    └──────┬────────┘
                           │
           ┌───────────────┤ cal_fitness()
           │               │
    [wave_batch]      [serial/thread/process]
           │               │
           ▼               ▼
  WaveBatchEvaluator   placementProcedure()
           │               │
           │               ▼
           │          BuildingPlate (binClassNew.py)
           │            can_insert() → FFT collision
           │               │
           ▼               ▼
     Batched FFT on    Sequential FFT on
     ALL solutions     one solution at a time
           │               │
           └───────┬───────┘
                   ▼
            fitness_list[450]
                   │
                   ▼
            partition → elites[50] + non_elites[450]
            mating → offspring[375] (vectorized crossover)
            mutants → mutants[75] (random)
                   │
                   ▼
            Next generation
```

---

## 7. Performance Profile and Bottlenecks

### Current performance (after all optimizations through CUDA batch kernel)

| Metric | Value |
|--------|-------|
| Time per generation (wave_batch, P50M2, 500 individuals) | **~3.80s** |
| Time per generation (serial, P50M2, 500 individuals) | ~18.2s |
| wave_batch speedup over serial | ~4.8x |
| Total optimization speedup (from original ~40s/gen) | **~10.5x** |

### Where time is spent in wave_batch mode (5 generations, 360 waves, profiled with `profile_phases.py`)

| Phase | Time(s) | % | ms/wave | Description |
|-------|---------|---|---------|-------------|
| Phase 1 | 0.074 | 0.4% | 0.20 | Gather context info |
| Phase 2 | 0.900 | 4.8% | 2.50 | Batch grid FFTs (`rfft2`) |
| Phase 3 | 1.418 | 7.6% | 3.94 | Vacancy check + collect tests |
| **Phase 4** | **12.398** | **66.1%** | **34.44** | **Batch IFFT (`irfft2`) — dominant** |
| Phase 5 | 2.629 | 14.0% | 7.30 | Best placement + grid updates |
| Phase 6 | 1.347 | 7.2% | 3.74 | Open new bins |
| **TOTAL** | **18.766** | 100% | | |

### The fundamental bottleneck

Phase 4 (batched IFFT) now dominates at 66% of wave time. The FFT math itself is fast, but the chunked approach (750 tests per chunk) still incurs GPU sync stalls between chunks. The remaining Phase 5 time (~7.3ms/wave) is dominated by CPU updates (NumPy grid + Numba vacancy), not GPU launches — the custom CUDA kernel eliminated launch overhead.

### Remaining optimization opportunities

| ID | Description | Status | Expected Impact |
|----|-------------|--------|-----------------|
| IMP-3 | Eliminate dual numpy/GPU grid | Pending | ~0.6s (high effort) |
| Phase 4 chunk size | Increase CHUNK_SIZE if VRAM allows | Not started | Fewer sync points |
| Cheap prefilters | Bounding box / row-range checks before FFT | Not started | Instance-dependent |
| Hybrid collision | Direct overlap for small parts instead of FFT | Not started | Moderate |

### Tested and rejected optimizations

| Option | Result | Why rejected |
|--------|--------|-------------|
| TF32 mode | 0% improvement | cuFFT doesn't use TF32 |
| CUDA Streams (2-8) | 23-29% slower | GPU already saturated |
| CuPy backend | 4.7x slower | Worse cuFFT plan caching |
| FP16 FFT | 0% improvement | Requires power-of-2 padding (cancels gain) |
| Dynamic FFT size | Deferred | Complex, interacts with batching |
| GPU-side indexed gather (Phase 4) | No improvement | Overhead of extra tensors negated benefit |
| GPU-side score computation (scatter_reduce) | No improvement | Too few tests/context (~4) for GPU to win |

---

## 8. Optimization History

### Phase 1-6 (structural optimizations)

All 17 original recommendations addressed:

| Phase | Changes | Impact |
|-------|---------|--------|
| 1 (Quick Wins) | Remove unused `grid2`, use uint8 dtype, pre-compute best_rotation, contiguous arrays | Memory 8x reduction |
| 2 (Low-Hanging Fruit) | Incremental enclosure box O(1), vectorized density, reusable ThreadPoolExecutor | Algorithm efficiency |
| 3 (Data Loading) | Separate part loading from FFT, cache sorted parts | Startup speed |
| 4 (Hot Path) | Pre-allocate vacancy buffers, optimize sliding window, improve initial solution | Evaluation speed |
| 5 (Structural Refactor) | Dataclasses instead of dicts, integer indexing | ~2x faster data access |
| 6 (Parallelization) | Parallel machine processing, FFT caching | Moderate speedup |

### IMP-1 through IMP-10 (wave_batch specific)

| IMP | Change | Actual Saving |
|-----|--------|---------------|
| IMP-1 | Pre-computed GPU tensors in `_place_part_in_bin` | **-1.0s (-13%)** |
| IMP-2 | Cache `torch.arange` / scalar tensors | ~0s (noise) |
| IMP-4 | Parallel arrays instead of list-of-dicts | -0.09s |
| IMP-5 | Remove redundant `.astype(np.int32)` | -0.05s |
| IMP-6 | Increase CHUNK_SIZE to 750 | -0.04s |
| IMP-7 | Single `torch.max` instead of `argmax`+`max` | **-0.17s** |
| IMP-8 | Pre-cast uint8 rotations | -0.05s |
| IMP-9 | Numba JIT in binClassInitialSol | N/A (init only) |
| IMP-10 | Remove fitness cache for wave_batch | ~0s (was already bypassed) |
| IMP-3 | Eliminate dual numpy/GPU grid | **Pending** |

### Post-IMP optimizations (2026-03-25/26)

| Change | Before → After | Saving | File(s) |
|--------|----------------|--------|---------|
| `rfft2`/`irfft2` instead of `fft2`/`ifft2` | 5.74s → 4.53s | **-1.21s (-21%)** | `wave_batch_evaluator.py`, `collision_backend.py` |
| NumPy-side composite scoring (density + position) | 4.53s → 4.23s | **-0.30s (-7%)** | `wave_batch_evaluator.py` |
| Custom CUDA kernel for batched grid updates | 4.23s → 3.80s | **-0.43s (-10%)** | `cuda_batch_update.py`, `wave_batch_evaluator.py` |

### Cumulative wave_batch performance

| State | Mean time (500-sol batch) |
|-------|--------------------------|
| Original baseline | 7.30s |
| After IMP-1 | 6.33s |
| After IMP-1+2 | 6.26s |
| After IMP-1+2+4 | 6.17s |
| After +IMP-5 | 6.12s |
| After +IMP-6 | 6.08s |
| After +IMP-7 | 5.91s |
| After +IMP-8 | 5.86s |
| After rfft2/irfft2 | 4.53s |
| After numpy scoring | 4.23s |
| After CUDA batch kernel | **3.80s** |

---

## 9. Instance Data Format

### Instance files (`data/Instances/P{N}M{M}-{i}.txt`)

Single line of N space-separated integers. Each integer is a part ID (0-indexed). Parts can repeat (same physical part used multiple times).

Available instances:
- P25M2: 5 variants (0-4)
- P50M2: 5 variants (0-4)
- P75M2: 5 variants (0-4)
- P100M4: 5 variants (0-4)
- P150M4: 5 variants (0-4)
- P200M4: 5 variants (0-4)

### Part matrices (`data/partsMatrices/matrix_{id}.npy`)

NumPy binary arrays (0/1). Irregular shapes (e.g., 47x32, 85x60). About 100 unique parts exist.

### Machine/part specs (`data/PartsMachines/`)

- `part-machine-information.xlsx`: Two sheets — "part" (volume, support volume, height per part) and "machine" (L, W, setup time, processing time rates)
- `polygon_areas.xlsx`: Area per part
- `parts_rotations.xlsx`: Rotation count per part (2 or 4)
- `cached_specs.pkl`: Pickle cache of above (created on first run)

---

## 10. CLI Arguments Reference

```
python BRKGA_alg3.py <nbParts> <nbMachines> <instNumber> [backend] [eval_mode] [workers] [chunksize] [generations]
```

| Arg | Position | Default | Values |
|-----|----------|---------|--------|
| nbParts | 1 (required) | — | 25, 50, 75, 100, 150, 200 |
| nbMachines | 2 (required) | — | 2, 4 |
| instNumber | 3 (required) | — | 0-4 |
| backend | 4 | `torch_gpu` | `torch_gpu`, `torch_cpu`, `numpy_cpu`, `cupy_gpu` |
| eval_mode | 5 | `auto` | `auto`, `serial`, `thread`, `process`, `wave_batch` |
| workers | 6 | 4 | Any positive integer |
| chunksize | 7 | 1 | Any positive integer |
| generations | 8 | 30 | Any positive integer |

Population size is fixed at `mult * nbParts` where `mult` is hardcoded as `prob = [10]` (line 585). So:
- P25: 250 individuals
- P50: 500 individuals
- P100: 1000 individuals
- P200: 2000 individuals

---

## 11. Key Invariants and Gotchas

1. **Grid coordinates**: Y-axis is inverted (row 0 = top, row H-1 = bottom). "Bottom-left" placement means high row index, low column index.

2. **Part placement is greedy and sequential**: Each part depends on the grid state after all previous placements. This cannot be parallelized across parts — only across solutions.

3. **FFT output coordinates**: `overlap[r][c]` = overlap count when part's bottom-right corner is at (r, c). Valid placement requires `r >= h-1` and `c >= w-1`. The "real" coordinates for the caller are `smallest_col = c - (w-1)` and `largest_row_real = r`.

4. **Two BuildingPlate classes**: `binClassNew.py` (used during BRKGA evaluation) returns `True/False` from `can_insert()` and self-inserts. `binClassInitialSol.py` (used during initial solution) returns `(result, best_pixel, best_rotation)` and the caller handles insertion.

5. **Dual grid maintenance in wave_batch**: `BinState` maintains both `grid` (numpy, for vacancy) and `grid_states[idx]` (GPU, for FFT). This is IMP-3 — the remaining optimization opportunity.

6. **Float32 precision**: All chromosomes and populations use float32. FFT uses float32 on GPU, complex64 for frequency domain.

7. **Vacancy vector semantics**: `vacancy_vector[row]` = maximum number of contiguous zero cells in that row of the grid. Used as a cheap filter before the expensive FFT check.

8. **CHUNK_SIZE = 750**: The number of (context × bin × rotation) tests batched into a single GPU IFFT call. Tuned for RTX A4000 (16GB). May need adjustment for different GPUs.

---

## 12. Output

The program produces:
1. **Console output**: Initial solution makespan, per-generation timing and fitness
2. **Excel file**: `OriginalInitialSol_P{N}M{M}-{i}_prob_{mult}.xlsx` with sheets containing generation history (min fitness, mean fitness, elapsed time)
