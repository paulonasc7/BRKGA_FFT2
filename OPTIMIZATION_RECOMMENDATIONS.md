# FFT Placement / BRKGA Algorithm - Computational Efficiency Recommendations

## System Overview

This codebase implements a **2D Bin Packing + Scheduling Optimization** system using a **Biased Random-Key Genetic Algorithm (BRKGA)**. It solves a variant of the **Nesting Problem** for additive manufacturing where:

1. **Parts** (represented as binary matrices) must be placed on **building plates/bins** across multiple machines
2. **FFT-based collision detection** determines valid placement positions using convolution
3. **Objective**: Minimize the worst makespan (maximum completion time) across all machines

### Key Files
- `BRKGA_alg3.py` - Main genetic algorithm implementation and data loading
- `placement.py` - Solution decoder (chromosome → placement → fitness)
- `binClassNew.py` - BuildingPlate class used during BRKGA evaluation
- `binClassInitialSol.py` - BuildingPlate class used for initial solution construction
- `collision_backend.py` - FFT-based collision detection (NumPy/PyTorch/CUDA)

---

## 🔴 HIGH-LEVEL / STRUCTURAL ISSUES

### Issue #1: Repeated Dictionary String Key Lookups

**Location**: All files, particularly `placement.py` and `BRKGA_alg3.py`

**Current Code**:
```python
# In placement.py
partsDict[f'part{partInd}']['shapes0']
partsDict[f'part{partInd}']['area']
partsDict[i][f'part{partInd}']['procTime']

# In BRKGA_alg3.py
data[f'part{part}'][f'rot{rot}']
data[f'part{part}'][f'dens{rot}']
data[m][f'part{part}'][f'fft{rot}']
```

**Problem**: 
- Every access requires string formatting (`f'part{partInd}'`)
- Dictionary key hashing is performed on every lookup
- String creation allocates memory and involves character copying
- This happens thousands of times per fitness evaluation

**Estimated Impact**: 5-15% of CPU time in hot loops

**Recommended Solution**:
```python
# Option A: Use integer-indexed lists/arrays
parts_data = [None] * max_part_id  # Pre-allocate list
parts_data[part_id] = {
    'rotations': [rot0, rot1, rot2, rot3],
    'shapes': [(h0, w0), (h1, w1), ...],
    'densities': [dens0, dens1, ...],
    'area': area_value
}
# Access: parts_data[part_id]['rotations'][rot_index]

# Option B: Use dataclasses for structured access
@dataclass
class PartData:
    rotations: List[np.ndarray]
    shapes: List[Tuple[int, int]]
    densities: List[np.ndarray]
    area: float
    best_rotation: int
    id: int

# Option C: Pre-build lookup dictionary once
part_keys = {part_id: f'part{part_id}' for part_id in instance_parts_unique}
# Then use: partsDict[part_keys[partInd]]
```

---

### Issue #2: Redundant Part Data Loading Per Machine

**Location**: `BRKGA_alg3.py`, lines 205-250

**Current Code**:
```python
for m in range(nbMachines):
    data[m] = {}
    binLength = machSpec['L(mm)'].iloc[m]
    binWidth = machSpec['W(mm)'].iloc[m]
    # ...
    for part in instancePartsUnique:
        matrix = np.load(f'data/partsMatrices/matrix_{part}.npy')
        matrix = matrix.astype(np.int32)
        matrix = np.ascontiguousarray(matrix)
        
        if np.array_equal(matrix, np.rot90(matrix, 2)):
            nrot = 2
        else:
            nrot = 4
            
        data[f'part{part}'] = {}
        data[m][f'part{part}'] = {}
        
        for rot in range(nrot):
            data[f'part{part}'][f'rot{rot}'] = np.rot90(matrix, rot)
            data[f'part{part}'][f'dens{rot}'] = np.array([...])  # density calculation
            data[f'part{part}'][f'shapes{rot}'] = [...]
            
            data[m][f'part{part}'][f'fft{rot}'] = collision_backend.prepare_part_fft(
                data[f'part{part}'][f'rot{rot}'],
                binLength,
                binWidth,
            )
```

**Problem**:
- The matrix is loaded from disk **nbMachines times** per part
- Rotations (`rot0` to `rot3`) are computed **nbMachines times** per part
- Density vectors are computed **nbMachines times** per part
- Only the FFT computation actually depends on machine dimensions
- With 4 machines and 200 parts, that's 800 redundant file loads + rotations

**Estimated Impact**: 10-30% of initialization time

**Recommended Solution**:
```python
# Step 1: Load and process each part ONCE
part_data = {}
for part in instancePartsUnique:
    matrix = np.load(f'data/partsMatrices/matrix_{part}.npy')
    matrix = matrix.astype(np.int32)
    matrix = np.ascontiguousarray(matrix)
    
    nrot = 2 if np.array_equal(matrix, np.rot90(matrix, 2)) else 4
    
    part_data[part] = {
        'nrot': nrot,
        'rotations': [np.rot90(matrix, r) for r in range(nrot)],
        'shapes': [],
        'densities': [],
        'area': area[part],
        'id': part
    }
    
    for rot in range(nrot):
        part_data[part]['shapes'].append(part_data[part]['rotations'][rot].shape)
        part_data[part]['densities'].append(compute_density(part_data[part]['rotations'][rot]))
    
    part_data[part]['best_rotation'] = np.argmin([s[0] for s in part_data[part]['shapes']])

# Step 2: Compute machine-specific FFTs separately
machine_ffts = {}
for m in range(nbMachines):
    machine_ffts[m] = {}
    binLength = machSpec['L(mm)'].iloc[m]
    binWidth = machSpec['W(mm)'].iloc[m]
    
    for part in instancePartsUnique:
        machine_ffts[m][part] = []
        for rot in range(part_data[part]['nrot']):
            fft = collision_backend.prepare_part_fft(
                part_data[part]['rotations'][rot],
                binLength,
                binWidth
            )
            machine_ffts[m][part].append(fft)
```

---

### Issue #3: ThreadPoolExecutor Recreated Each Generation

**Location**: `BRKGA_alg3.py`, lines 68-77

**Current Code**:
```python
def cal_fitness(self, population):
    if self.eval_mode == "serial" or self.eval_workers <= 1:
        return [self.evaluate_solution(sol) for sol in population]

    if self.eval_mode == "thread":
        max_workers = self.eval_workers if self.eval_workers > 0 else min(32, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.evaluate_solution, population))
    # ...
```

**Problem**:
- A new `ThreadPoolExecutor` is created every time `cal_fitness` is called
- This happens **twice per generation** (once for initial eval, once for offspring)
- Thread pool creation involves:
  - Spawning worker threads
  - Setting up task queues
  - OS-level thread scheduling overhead
- With 30 generations, that's 60+ thread pool creations

**Estimated Impact**: 2-5% of total runtime, more on Windows

**Recommended Solution**:
```python
class BRKGA:
    def __init__(self, ...):
        # ... existing initialization ...
        
        # Create executor once at initialization
        self._executor = None
        if self.eval_mode == "thread" and self.eval_workers > 1:
            max_workers = self.eval_workers if self.eval_workers > 0 else min(32, (os.cpu_count() or 4))
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def cal_fitness(self, population):
        if self.eval_mode == "serial" or self.eval_workers <= 1:
            return [self.evaluate_solution(sol) for sol in population]

        if self.eval_mode == "thread":
            return list(self._executor.map(self.evaluate_solution, population))
        # ...
    
    def __del__(self):
        """Clean up executor on object destruction."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
    
    # Or use context manager pattern:
    def fit(self, verbose=False):
        # ... at the end of fit() ...
        if self._executor is not None:
            self._executor.shutdown(wait=True)
```

---

### Issue #4: Inefficient Initial Solution Construction

**Location**: `BRKGA_alg3.py`, lines 262-330

**Current Code**:
```python
for part in part_sortedSequence:
    best_makespan = 1000000000000000000
    bestBatch = []
    
    for mach in range(nbMachines):
        # Check if part fits in machine at all
        if (part doesn't fit in machine dimensions):
            continue
        
        # Calculate makespan for new batch
        machineMakespan = machines_dict[f'machine_{mach}']['makespan'] + ...
        newMakespan = max(max([machines_dict[f'machine_{i}']['makespan'] for i in range(nbMachines)]), machineMakespan)
        
        if newMakespan <= best_makespan:
            best_makespan = newMakespan
            bestBatch = ['new', mach, ...]
        
        # Try to place in existing batches
        for x, batch in enumerate(machines_dict[f'machine_{mach}']['batches']):
            res = batch.can_insert(data[f'part{part}'], data[mach][f'part{part}'])
            if res[0]:
                # ... calculate makespan and update if better
```

**Problem**:
- For **each part**, all batches across **all machines** are checked
- Time complexity: O(parts × machines × batches_per_machine)
- The inner `max([machines_dict[f'machine_{i}']['makespan'] for i in range(nbMachines)])` is computed multiple times per part
- Dictionary lookups with string keys add overhead

**Estimated Impact**: 10-20% of initial solution time

**Recommended Solution**:
```python
# Pre-compute and cache the current worst makespan
current_worst_makespan = 0

for part in part_sortedSequence:
    best_makespan = float('inf')
    bestBatch = None
    
    # Cache the current worst makespan (avoid recomputing in inner loop)
    other_machines_max = current_worst_makespan
    
    for mach in range(nbMachines):
        # Early exit: if this machine's current makespan + minimum possible addition
        # already exceeds the best found, skip
        min_addition = data[mach][f'part{part}']['procTime']
        if machines_dict[f'machine_{mach}']['makespan'] + min_addition >= best_makespan:
            continue
        
        # ... rest of logic ...
    
    # Update current_worst_makespan after placing part
    if bestBatch is not None:
        current_worst_makespan = max(current_worst_makespan, bestBatch[4])
```

**Additional optimization**: Maintain a priority queue / sorted structure of batches by available area to skip batches that definitely can't fit the part.

---

## 🟡 CODE-LEVEL INEFFICIENCIES

### Issue #5: Slow Density Vector Calculation with itertools.groupby

**Location**: `BRKGA_alg3.py`, line 231

**Current Code**:
```python
data[f'part{part}'][f'dens{rot}'] = np.array([
    max(len(list(g)) for k, g in itertools.groupby(row) if k) 
    for row in data[f'part{part}'][f'rot{rot}']
])
```

**Problem**:
- `itertools.groupby` is a Python iterator - slow compared to NumPy operations
- `list(g)` creates a new list for each group
- List comprehension with `max()` is O(n) per row
- This runs for every row in the matrix, for every rotation, for every part

**What it computes**: For each row, find the maximum length of consecutive 1s (the "density" - max horizontal run of occupied cells).

**Estimated Impact**: 5-10% of data loading time

**Recommended Solution** (fully vectorized):
```python
def compute_max_consecutive_ones_per_row(matrix):
    """
    Vectorized computation of maximum consecutive 1s per row.
    """
    if matrix.size == 0:
        return np.zeros(matrix.shape[0], dtype=np.int32)
    
    # Pad with zeros on left and right
    padded = np.pad(matrix, ((0, 0), (1, 1)), constant_values=0)
    
    # Find transitions: 0→1 (start) and 1→0 (end)
    diff = np.diff(padded.astype(np.int8), axis=1)
    
    # For each row, find start and end positions
    max_runs = np.zeros(matrix.shape[0], dtype=np.int32)
    
    for i in range(matrix.shape[0]):
        starts = np.where(diff[i] == 1)[0]
        ends = np.where(diff[i] == -1)[0]
        if len(starts) > 0:
            max_runs[i] = np.max(ends - starts)
    
    return max_runs

# Even faster with numba (if available):
# @numba.jit(nopython=True)
# def compute_max_consecutive_ones_per_row_numba(matrix): ...
```

**Alternative (pure NumPy, no loop)**:
```python
def compute_max_consecutive_ones_vectorized(matrix):
    """Fully vectorized - no Python loops."""
    # This is the same logic as the vacancy vector update
    padded = np.pad(matrix, ((0, 0), (1, 1)), constant_values=0)
    diffs = np.diff(padded.astype(np.int8), axis=1)
    
    start_indices = np.where(diffs == 1)
    end_indices = np.where(diffs == -1)
    
    run_lengths = end_indices[1] - start_indices[1]
    
    max_runs = np.zeros(matrix.shape[0], dtype=np.int32)
    np.maximum.at(max_runs, start_indices[0], run_lengths)
    
    return max_runs
```

---

### Issue #6: Vacancy Vector Update Allocates on Every Insertion

**Location**: `binClassNew.py`, lines 101-125

**Current Code**:
```python
def insert(self, x, y, partMatrix, shapes, partArea):
    self.area += partArea
    self.grid[y - shapes[0] + 1:y + 1, x:x + shapes[1]] += partMatrix
    self.collision_backend.update_grid_region(self.grid_state, x, y, partMatrix, shapes)

    # Step 1: Pad the matrix with ones
    binaryGrid = self.grid[y - shapes[0] + 1:y + 1, :]
    padded_matrix = np.pad(binaryGrid, ((0, 0), (1, 1)), constant_values=1)

    # Step 2: Compute differences
    diffs = np.diff(padded_matrix, axis=1)

    # Step 3: Identify start and end of zero runs
    start_indices = np.where(diffs == -1)
    end_indices = np.where(diffs == 1)

    # Step 4: Compute lengths of zero runs
    run_lengths = end_indices[1] - start_indices[1]

    # Step 5: Create a result array initialized with zeros
    max_zeros = np.zeros(binaryGrid.shape[0], dtype=int)

    # Step 6: Use np.maximum.at to find the maximum run length for each row
    np.maximum.at(max_zeros, start_indices[0], run_lengths)

    # Step 7: Update the vacancy vector
    self.vacancy_vector[y - shapes[0] + 1:y + 1] = max_zeros
```

**Problem**:
- `np.pad()` allocates a new array every call
- `np.diff()` allocates a new array every call
- `np.where()` allocates tuple of arrays (×2)
- `np.zeros()` allocates a new array every call
- This happens for **every part insertion** (hundreds/thousands per fitness evaluation)

**Estimated Impact**: 3-8% of placement time

**Recommended Solution**:
```python
class BuildingPlate:
    def __init__(self, width, length, collision_backend=None):
        # ... existing code ...
        
        # Pre-allocate reusable buffers
        self._padded_buffer = np.ones((length, width + 2), dtype=np.int8)
        self._diff_buffer = np.zeros((length, width + 1), dtype=np.int8)
        self._max_zeros_buffer = np.zeros(length, dtype=np.int32)
    
    def _update_vacancy_vector_rows(self, y_start, y_end):
        """Update vacancy vector for rows [y_start, y_end)."""
        num_rows = y_end - y_start
        
        # Use pre-allocated buffer (only the needed rows)
        padded = self._padded_buffer[:num_rows, :]
        padded[:, 1:-1] = self.grid[y_start:y_end, :]
        padded[:, 0] = 1  # Left pad
        padded[:, -1] = 1  # Right pad
        
        # Compute diff in-place if possible, or use buffer
        diffs = np.diff(padded, axis=1)
        
        # Find zero runs
        start_indices = np.where(diffs == -1)
        end_indices = np.where(diffs == 1)
        run_lengths = end_indices[1] - start_indices[1]
        
        # Use buffer for max computation
        max_zeros = self._max_zeros_buffer[:num_rows]
        max_zeros.fill(0)
        np.maximum.at(max_zeros, start_indices[0], run_lengths)
        
        # Update vacancy vector
        self.vacancy_vector[y_start:y_end] = max_zeros
    
    def insert(self, x, y, partMatrix, shapes, partArea):
        self.area += partArea
        self.grid[y - shapes[0] + 1:y + 1, x:x + shapes[1]] += partMatrix
        self.collision_backend.update_grid_region(self.grid_state, x, y, partMatrix, shapes)
        
        # Update only affected rows
        self._update_vacancy_vector_rows(y - shapes[0] + 1, y + 1)
```

---

### Issue #7: Enclosure Box Length Scans Entire Grid

**Location**: `binClassNew.py`, lines 89-91; `binClassInitialSol.py`, lines 96-100

**Current Code** (binClassNew.py):
```python
def calculate_enclosure_box_length(self):
    # Find the first row index with at least one 1
    first_row_with_one = np.where(self.grid.any(axis=1))[0][0]
    self.enclosure_box_length = self.length - first_row_with_one
```

**Current Code** (binClassInitialSol.py):
```python
def calculate_enclosure_box_length(self):
    occupied_rows = np.where(self.grid != 0)[0]
    if len(occupied_rows) == 0:
        return 0
    self.enclosure_box_length = np.max(occupied_rows) - np.min(occupied_rows) + 1
```

**Problem**:
- `self.grid.any(axis=1)` scans the **entire grid** (length × width operations)
- `np.where(self.grid != 0)` creates a boolean array then finds indices
- This is called after **every single part insertion**
- The enclosure box only ever grows (parts are added, never removed)

**Estimated Impact**: 2-5% of placement time for large grids

**Recommended Solution**:
```python
class BuildingPlate:
    def __init__(self, width, length, collision_backend=None):
        # ... existing code ...
        self.min_occupied_row = length  # Track incrementally
        self.max_occupied_row = -1      # Track incrementally
    
    def calculate_enclosure_box_length(self):
        """O(1) lookup instead of O(length × width) scan."""
        if self.min_occupied_row > self.max_occupied_row:
            self.enclosure_box_length = 0
        else:
            self.enclosure_box_length = self.max_occupied_row - self.min_occupied_row + 1
    
    def insert(self, x, y, partMatrix, shapes, partArea):
        self.area += partArea
        
        y_start = y - shapes[0] + 1
        y_end = y + 1
        
        # Update grid
        self.grid[y_start:y_end, x:x + shapes[1]] += partMatrix
        
        # Update enclosure box bounds incrementally - O(1)
        self.min_occupied_row = min(self.min_occupied_row, y_start)
        self.max_occupied_row = max(self.max_occupied_row, y)
        
        # ... rest of vacancy vector update ...
```

---

### Issue #8: Grid FFT Recomputed for Every Batch Check

**Location**: `collision_backend.py`, lines 108-113

**Current Code**:
```python
def find_bottom_left_zero_batch(self, grid, part_ffts, part_shapes, grid_state=None):
    # ...
    grid_tensor = grid_state if grid_state is not None else torch.as_tensor(grid, ...)
    grid_fft = torch.fft.fft2(grid_tensor)  # <-- Computed every call
    stacked_part_ffts = torch.stack(part_ffts, dim=0)
    overlap_batch = torch.fft.ifft2(grid_fft.unsqueeze(0) * stacked_part_ffts).real
```

**Problem**:
- The grid's FFT is computed every time collision detection is called
- Within a single `can_insert()` call, this is fine (grid doesn't change)
- But across multiple `can_insert()` calls for different parts in the same bin, the grid FFT could be cached

**Complexity**: This is non-trivial to fix because:
1. The grid changes after each insertion
2. Incremental FFT updates are mathematically complex
3. The current `grid_state` tensor is already an optimization

**Estimated Impact**: 5-15% of FFT computation time

**Recommended Solution** (cache at bin level):
```python
class BuildingPlate:
    def __init__(self, ...):
        # ... existing code ...
        self._grid_fft_valid = False
        self._cached_grid_fft = None
    
    def invalidate_fft_cache(self):
        self._grid_fft_valid = False
    
    def get_grid_fft(self):
        if not self._grid_fft_valid:
            self._cached_grid_fft = torch.fft.fft2(self.grid_state)
            self._grid_fft_valid = True
        return self._cached_grid_fft
    
    def insert(self, ...):
        # ... insert logic ...
        self.invalidate_fft_cache()  # Grid changed, invalidate cache
```

Then modify `collision_backend.find_bottom_left_zero_batch()` to accept a pre-computed `grid_fft` parameter.

**Note**: This optimization only helps when multiple parts are tried against the same bin without insertions in between. Current code flow may not benefit much.

---

### Issue #9: Sliding Window View Created Per Rotation

**Location**: `binClassNew.py`, lines 47-48

**Current Code**:
```python
for currRot in range(part['nrot']):
    subarrays = np.lib.stride_tricks.sliding_window_view(
        self.vacancy_vector, 
        part[f'shapes{currRot}'][0]
    )
    binaryResult = np.any(np.all(subarrays >= part[f'dens{currRot}'], axis=1))
    
    if binaryResult:
        feasible_rotations.append(currRot)
        # ...
```

**Problem**:
- Creates a sliding window view for each rotation (2-4 times per `can_insert` call)
- The comparison `subarrays >= part[f'dens{currRot}']` broadcasts and compares
- String key lookups in inner loop

**Estimated Impact**: 1-3% of `can_insert` time

**Recommended Solution**:
```python
def can_insert(self, part, machPart, plott=False):
    # Pre-extract values to avoid repeated dict lookups
    nrot = part['nrot']
    vacancy = self.vacancy_vector
    
    feasible_rotations = []
    feasible_shapes = []
    feasible_ffts = []
    
    for currRot in range(nrot):
        shape_height = part['shapes'][currRot][0]  # Assuming restructured data
        density = part['densities'][currRot]
        
        # Early exit: if height > vacancy vector length, skip
        if shape_height > len(vacancy):
            continue
            
        subarrays = np.lib.stride_tricks.sliding_window_view(vacancy, shape_height)
        
        # Check if any position is feasible
        if np.any(np.all(subarrays >= density, axis=1)):
            feasible_rotations.append(currRot)
            feasible_shapes.append(part['shapes'][currRot])
            feasible_ffts.append(machPart['ffts'][currRot])
    
    # ... rest of function
```

---

### Issue #10: Non-Contiguous Arrays at Collision Check Time

**Location**: `collision_backend.py`, line 88

**Current Code**:
```python
def update_grid_region(self, grid_state, x, y, part_matrix, shapes):
    if grid_state is None:
        return
    part_contig = np.ascontiguousarray(part_matrix)  # <-- Copy if non-contiguous
    part_tensor = torch.as_tensor(part_contig, dtype=torch.float32, device=self.device)
    # ...
```

**Problem**:
- `np.rot90()` returns a **non-contiguous view**
- Every time `update_grid_region` is called, `np.ascontiguousarray()` creates a copy
- This happens for every part insertion

**Estimated Impact**: 1-2% of insertion time

**Recommended Solution** (make contiguous at load time):
```python
# In BRKGA_alg3.py, when loading parts:
for rot in range(nrot):
    rotated = np.rot90(matrix, rot)
    data[f'part{part}'][f'rot{rot}'] = np.ascontiguousarray(rotated)  # Make contiguous once
```

Then remove the redundant `np.ascontiguousarray()` call in `collision_backend.py`:
```python
def update_grid_region(self, grid_state, x, y, part_matrix, shapes):
    if grid_state is None:
        return
    # part_matrix is already contiguous from data loading
    part_tensor = torch.as_tensor(part_matrix, dtype=torch.float32, device=self.device)
```

---

### Issue #11: Double Scan for Best Rotation

**Location**: `placement.py`, line 66; `BRKGA_alg3.py`, lines 280, 322

**Current Code**:
```python
best_rotation = partsDict[f'part{partInd}']['lengths'].index(min(partsDict[f'part{partInd}']['lengths']))
```

**Problem**:
- `min()` scans the list once to find minimum value
- `.index()` scans the list again to find the index of that value
- This is computed every time a new bin is created

**Estimated Impact**: <1% but easy fix

**Recommended Solution** (pre-compute at data loading):
```python
# In BRKGA_alg3.py data loading:
data[f'part{part}']['lengths'] = [data[f'part{part}'][f'shapes{currRot}'][0] for currRot in range(nrot)]
data[f'part{part}']['best_rotation'] = int(np.argmin(data[f'part{part}']['lengths']))

# Then in placement.py:
best_rotation = partsDict[f'part{partInd}']['best_rotation']  # O(1) lookup
```

---

## 🟢 PARALLELIZATION OPPORTUNITIES

### Issue #12: Machine-Level Parallelism Not Exploited

**Location**: `placement.py`, lines 17-97

**Current Code**:
```python
def placementProcedure(partsDict, nbParts, nbMachines, thresholds, chromosome, matching, collision_backend, plot=False):
    # ...
    for i in range(nbMachines):  # Sequential loop over machines
        # ... process all parts for machine i ...
        for partInd in sorted_sequence:
            # ... place part in bins for this machine ...
```

**Problem**:
- Machines are **completely independent** - parts assigned to machine 1 don't affect machine 2
- Sequential processing leaves CPU/GPU resources underutilized
- The bottleneck might be FFT computation on GPU with small batch sizes

**Estimated Impact**: Up to 2-4x speedup with proper parallelization (depends on hardware)

**Recommended Solution** (parallel machine processing):
```python
from concurrent.futures import ThreadPoolExecutor

def process_single_machine(args):
    """Process all parts for a single machine."""
    machine_idx, parts_for_machine, partsDict, collision_backend = args
    
    openBins = []
    for partInd in parts_for_machine:
        # ... placement logic for this machine ...
    
    # Return makespan for this machine
    makespan = sum(batch.processingTime + batch.processingTimeHeight + setupTime 
                   for batch in openBins)
    return machine_idx, makespan, openBins

def placementProcedure(partsDict, nbParts, nbMachines, thresholds, chromosome, matching, collision_backend, plot=False):
    # ... compute parts_per_machine (which parts go to which machine) ...
    
    # Parallel processing
    with ThreadPoolExecutor(max_workers=nbMachines) as executor:
        args_list = [
            (i, parts_per_machine[i], partsDict, collision_backend)
            for i in range(nbMachines)
        ]
        results = list(executor.map(process_single_machine, args_list))
    
    worstMakespan = max(r[1] for r in results)
    return worstMakespan
```

**Caveat**: If using GPU collision backend, multiple threads accessing the same GPU may cause contention. Consider using separate CUDA streams or CPU backend for parallel evaluation.

---

### Issue #13: GPU Batch Size Underutilization

**Location**: `collision_backend.py`, lines 106-130

**Current Code**:
```python
def find_bottom_left_zero_batch(self, grid, part_ffts, part_shapes, grid_state=None):
    # ...
    stacked_part_ffts = torch.stack(part_ffts, dim=0)  # Stack 2-4 rotations
    overlap_batch = torch.fft.ifft2(grid_fft.unsqueeze(0) * stacked_part_ffts).real
```

**Problem**:
- Only 2-4 rotations (of one part) are batched together
- GPU kernel launch overhead is high relative to small batch computation
- GPU is most efficient with larger batch sizes (32, 64, 128+)

**Estimated Impact**: 10-30% of GPU time depending on problem size

**Recommended Solution** (batch multiple parts together):
```python
def check_multiple_parts_batch(self, grid, parts_data, grid_state=None):
    """
    Check placement feasibility for multiple parts at once.
    
    parts_data: List of (part_ffts, part_shapes) tuples for different parts
    Returns: List of results, one per part
    """
    all_ffts = []
    all_shapes = []
    part_boundaries = [0]
    
    for part_ffts, part_shapes in parts_data:
        all_ffts.extend(part_ffts)
        all_shapes.extend(part_shapes)
        part_boundaries.append(len(all_ffts))
    
    if not all_ffts:
        return [[] for _ in parts_data]
    
    # Single large batch FFT operation
    grid_tensor = grid_state if grid_state is not None else torch.as_tensor(grid, ...)
    grid_fft = torch.fft.fft2(grid_tensor)
    stacked_ffts = torch.stack(all_ffts, dim=0)
    overlap_batch = torch.fft.ifft2(grid_fft.unsqueeze(0) * stacked_ffts).real
    rounded_batch = torch.round(overlap_batch)
    
    # Split results back per part
    results_per_part = []
    for i in range(len(parts_data)):
        start, end = part_boundaries[i], part_boundaries[i+1]
        part_results = []
        for j in range(start, end):
            # ... process cropped[j] ...
            part_results.append(result)
        results_per_part.append(part_results)
    
    return results_per_part
```

---

## 🔵 DATA STRUCTURE IMPROVEMENTS

### Issue #14: Use Dataclasses Instead of Nested Dictionaries

**Location**: All files

**Current Code**:
```python
data[f'part{part}'][f'rot{rot}']
data[f'part{part}'][f'dens{rot}']
data[f'part{part}']['area']
data[m][f'part{part}']['procTime']
```

**Problem**:
- String key lookups are slow
- No type checking or IDE autocomplete
- Easy to make typos in key names
- Harder to reason about data structure

**Recommended Solution**:
```python
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

@dataclass
class PartRotationData:
    matrix: np.ndarray          # The rotated binary matrix
    shape: Tuple[int, int]      # (height, width)
    density: np.ndarray         # Max consecutive 1s per row
    
@dataclass  
class PartData:
    id: int
    area: float
    rotations: List[PartRotationData]
    best_rotation: int          # Index of rotation with minimum height
    
    @property
    def nrot(self) -> int:
        return len(self.rotations)

@dataclass
class MachinePartData:
    proc_time: float
    proc_time_height: float
    ffts: List[torch.Tensor]    # Pre-computed FFTs for each rotation

@dataclass
class MachineData:
    bin_length: int
    bin_width: int
    bin_area: int
    setup_time: float
    parts: Dict[int, MachinePartData]  # part_id -> machine-specific data

# Usage:
parts: Dict[int, PartData] = {}
machines: List[MachineData] = []

# Access is cleaner and faster:
part = parts[part_id]
rotation = part.rotations[rot_idx]
shape = rotation.shape
```

---

### Issue #15: Sort Parts Once and Reuse Order

**Location**: `BRKGA_alg3.py`, lines 257-262

**Current Code**:
```python
partsInfo = jobSpec.loc[instanceParts]["height(mm)"]
partsAR = pd.read_excel(...).loc[instanceParts]["Area"]
conc = pd.concat([partsInfo, partsAR], axis=1)
sorted_df = conc.sort_values(by=['height(mm)', 'Area'], ascending=[False, False])
part_sortedSequence = sorted_df.index.to_list()
```

**Problem**:
- Sorts parts for initial solution construction
- If this order is useful elsewhere, it should be computed once and stored

**Recommended Solution**:
```python
# Compute once during data loading
sorted_parts_by_height_area = sorted(
    instancePartsUnique,
    key=lambda p: (-jobSpec["height(mm)"].loc[p], -area[p])
)

# Store as part of problem data
problem_data['sorted_parts'] = sorted_parts_by_height_area
```

---

## 📊 MEMORY EFFICIENCY

### Issue #16: Unused grid2 Array

**Location**: `binClassNew.py`, line 15; `binClassInitialSol.py`, line 15

**Current Code**:
```python
class BuildingPlate:
    def __init__(self, width, length, collision_backend=None):
        # ...
        self.grid = np.zeros((length, width), dtype=int)
        self.grid2 = np.zeros((length, width), dtype=int)  # <-- Never used
```

**Problem**:
- `grid2` is allocated but never read or written
- Wastes `length × width × 8` bytes per BuildingPlate (int64)
- With many plates and large dimensions, this adds up

**Estimated Impact**: Wastes memory, no CPU impact

**Recommended Solution**:
```python
# Simply remove the line:
# self.grid2 = np.zeros((length, width), dtype=int)
```

---

### Issue #17: Integer Type Too Large for Binary Grid

**Location**: `binClassNew.py`, line 14; `binClassInitialSol.py`, line 14

**Current Code**:
```python
self.grid = np.zeros((length, width), dtype=int)  # Default int is int64 (8 bytes)
```

**Problem**:
- Grid values are 0 or small positive integers (number of overlapping parts)
- Using 64-bit integers wastes 7 bytes per cell
- A 500×500 grid uses 2MB instead of 250KB

**Estimated Impact**: 8x memory usage for grids; potential cache efficiency impact

**Recommended Solution**:
```python
# Use smallest sufficient type
self.grid = np.zeros((length, width), dtype=np.uint8)   # For values 0-255
# or
self.grid = np.zeros((length, width), dtype=np.int16)   # If values can exceed 255
```

**Note**: If multiple parts can overlap at the same position and the count matters, ensure uint8 is sufficient (max 255 overlaps).

---

## Implementation Priority

### Quick Wins (< 1 hour, low risk)
1. ✅ **Issue #16**: Remove unused `grid2`
2. ✅ **Issue #17**: Change `dtype=int` to `dtype=np.uint8`
3. ✅ **Issue #11**: Pre-compute `best_rotation`
4. ✅ **Issue #10**: Make arrays contiguous at load time

### Medium Effort (1-4 hours, medium risk)
5. 🔧 **Issue #2**: Separate part loading from machine-specific FFT computation
6. 🔧 **Issue #5**: Vectorize density calculation
7. 🔧 **Issue #7**: Track enclosure box bounds incrementally
8. 🔧 **Issue #3**: Create ThreadPoolExecutor once

### High Effort (4+ hours, higher risk)
9. 🔨 **Issue #1**: Restructure data to use integer indexing / dataclasses
10. 🔨 **Issue #6**: Pre-allocate buffers for vacancy vector updates
11. 🔨 **Issue #12**: Parallelize machine processing
12. 🔨 **Issue #13**: Batch multiple parts in GPU collision detection

---

## Benchmarking Recommendations

Before implementing optimizations, establish baseline measurements:

```python
import cProfile
import pstats

# Profile the main execution
with cProfile.Profile() as pr:
    model.fit(verbose=True)

stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats(30)  # Top 30 functions by cumulative time
```

Also consider using `line_profiler` for hot functions:
```python
# Add @profile decorator to functions of interest
# Run with: kernprof -l -v BRKGA_alg3.py
```

Key metrics to track:
- Total runtime per generation
- Time in `placementProcedure`
- Time in `can_insert`
- Time in FFT operations
- Memory usage (via `tracemalloc`)
