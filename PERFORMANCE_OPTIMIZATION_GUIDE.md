# Performance Optimization Guide for BRKGA_FFT2

## Executive Summary

**Current State**: ~25-40 seconds per generation  
**Target State**: <5 seconds per generation  
**Required Speedup**: 5-8x  

This document provides a comprehensive analysis of performance bottlenecks and detailed implementation guides for each optimization.

---

## Table of Contents

1. [Profiling Results](#profiling-results)
2. [Bottleneck Analysis](#bottleneck-analysis)
3. [Optimization Roadmap](#optimization-roadmap)
4. [High-Impact Optimizations](#high-impact-optimizations)
   - [O1: Reduce GPU-CPU Synchronization](#o1-reduce-gpu-cpu-synchronization)
   - [O2: CuPy Backend (Alternative to PyTorch)](#o2-cupy-backend-alternative-to-pytorch)
   - [O3: Numba JIT for Vacancy Checking](#o3-numba-jit-for-vacancy-checking)
   - [O4: Batch Solution Evaluation](#o4-batch-solution-evaluation)
5. [Medium-Impact Optimizations](#medium-impact-optimizations)
   - [O5: CUDA Streams for Parallel Solution Evaluation](#o5-cuda-streams-for-parallel-solution-evaluation)
   - [O6: Population Size Tuning](#o6-population-size-tuning)
   - [O7: Early Termination Heuristics](#o7-early-termination-heuristics)
6. [Low-Impact Optimizations](#low-impact-optimizations)
   - [O8: Memory Pre-allocation](#o8-memory-pre-allocation)
   - [O9: Grid State Optimization](#o9-grid-state-optimization)
7. [Implementation Priority](#implementation-priority)
8. [Benchmarking Protocol](#benchmarking-protocol)

---

## Profiling Results

### Test Configuration
- **Instance**: P50M2-0 (50 parts, 2 machines)
- **Population**: 500 individuals
- **Generations Profiled**: 3
- **Hardware**: NVIDIA RTX A4000

### Timing Breakdown (120.38s total for 3 generations)

| Function | Self Time | % Total | Calls | Time/Call |
|----------|-----------|---------|-------|-----------|
| `find_bottom_left_zero_batch` | 21.15s | 17.6% | 110,888 | 0.19ms |
| `can_insert` | 14.07s | 11.7% | 110,888 | 0.13ms |
| `torch.nonzero()` | 13.80s | 11.5% | 527,910 | 0.026ms |
| `insert` | 12.10s | 10.0% | 86,328 | 0.14ms |
| `torch.any()` | 5.70s | 4.7% | 683,026 | 0.008ms |
| `sliding_window_view` | 4.42s | 3.7% | 434,982 | 0.010ms |
| `torch.fft.fft2` | 4.33s | 3.6% | 79,394 | 0.055ms |
| `torch.item()` | 4.28s | 3.6% | 527,910 | 0.008ms |
| `torch.fft.ifft2` | 2.85s | 2.4% | 91,552 | 0.031ms |
| `update_grid_region` | 2.94s | 2.4% | 86,328 | 0.034ms |
| `torch.max()` | 2.44s | 2.0% | 263,955 | 0.009ms |
| `torch.min()` | 2.31s | 1.9% | 263,955 | 0.009ms |
| `torch.stack()` | 1.99s | 1.7% | 91,552 | 0.022ms |

### Key Statistics
- **Total evaluations**: 1,850 solutions
- **Can_insert calls per evaluation**: ~60
- **GPU sync operations**: ~1,500,000+ (nonzero, item, max, min)
- **Cache hit rate**: 0% (all unique solutions)

---

## Bottleneck Analysis

### 1. GPU-CPU Synchronization Overhead (≈27s, 22%)

**Problem**: PyTorch operations like `.nonzero()`, `.item()`, `.max()`, `.min()` force GPU synchronization, blocking the CPU until the GPU operation completes.

**Location**: `collision_backend.py:126-155` in `find_bottom_left_zero_batch()`

```python
# Current code - 4 sync points per rotation
largest_row_tensor = rows_with_zeros.nonzero().max()      # Sync 1 & 2
smallest_col_tensor = (cropped[largest_row_tensor] == 0).nonzero().min()  # Sync 3 & 4
largest_row = largest_row_tensor.item()                   # Sync 5
smallest_col = smallest_col_tensor.item()                 # Sync 6
```

**Why it's slow**: Each sync waits for all pending GPU operations to complete before returning a value to CPU. With 110K+ `can_insert` calls, each checking multiple rotations, this adds up to millions of sync points.

### 2. Sequential Part Placement (≈14s, 12%)

**Problem**: Parts must be placed sequentially because each placement affects the grid state for subsequent placements. This is an algorithmic constraint that limits parallelization.

**Location**: `placement.py:29-80` in `_process_single_machine()`

**Impact**: ~60 can_insert checks per solution × 1,850 solutions = 110,888 calls

### 3. Redundant FFT Computation (≈7s, 6%)

**Problem**: The grid FFT is recomputed after every part insertion, even though multiple rotations are checked against the same grid state.

**Location**: `binClassNew.py:82-87`

```python
# Currently: FFT computed once per can_insert, but invalidated on insert
if self._grid_fft_cache is None:
    self._grid_fft_cache = self.collision_backend.compute_grid_fft(self.grid_state)
```

### 4. Vacancy Vector Checking (≈4s, 4%)

**Problem**: Pure Python loops with numpy operations for checking if parts can fit based on vacancy vectors.

**Location**: `binClassNew.py:60-77`

```python
for currRot in range(nrot):
    subarrays = np.lib.stride_tricks.sliding_window_view(vacancy, shape[0])
    binaryResult = np.any(np.all(subarrays >= dens, axis=1))
```

---

## Optimization Roadmap

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION ROADMAP                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 1: Quick Wins (1-2 days)                                  │
│  ├── O1: Reduce GPU sync points                    [Est: 2-3x]   │
│  └── O3: Numba JIT for vacancy checking            [Est: 1.2x]   │
│                                                                   │
│  Phase 2: Backend Migration (3-5 days)                           │
│  └── O2: CuPy backend implementation               [Est: 2-4x]   │
│                                                                   │
│  Phase 3: Algorithmic Improvements (2-3 days)                    │
│  ├── O4: Batch solution evaluation                 [Est: 1.5x]   │
│  └── O5: CUDA streams parallelism                  [Est: 1.3x]   │
│                                                                   │
│  Phase 4: Fine Tuning (1 day)                                    │
│  ├── O6: Population size tuning                    [Est: 1.5x]   │
│  └── O7-O9: Memory and early termination           [Est: 1.2x]   │
│                                                                   │
│  COMBINED ESTIMATED SPEEDUP: 6-10x                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## High-Impact Optimizations

### O1: Reduce GPU-CPU Synchronization

**Estimated Speedup**: 2-3x  
**Complexity**: Medium  
**Files to Modify**: `collision_backend.py`

#### Current Problem

```python
# In find_bottom_left_zero_batch() - called 110K+ times
for i, part_shape in enumerate(part_shapes):
    cropped = rounded_batch[i, part_shape[0] - 1 : grid.shape[0], part_shape[1] - 1 : grid.shape[1]]
    rows_with_zeros = (cropped == 0).any(dim=1)
    
    # SYNC POINT 1: .any() result check
    if not rows_with_zeros.any():
        results.append((False, None, None))
        continue
    
    # SYNC POINT 2-3: nonzero() + max()
    largest_row_tensor = rows_with_zeros.nonzero().max()
    
    # SYNC POINT 4-5: nonzero() + min()
    smallest_col_tensor = (cropped[largest_row_tensor] == 0).nonzero().min()
    
    # SYNC POINT 6-7: .item() calls
    largest_row = largest_row_tensor.item()
    smallest_col = smallest_col_tensor.item()
```

#### Solution: Batch Position Extraction

```python
def find_bottom_left_zero_batch(self, grid, part_ffts, part_shapes, grid_state=None, grid_fft=None):
    """Optimized version with minimal GPU-CPU sync points."""
    if not part_ffts:
        return []
    
    with torch.inference_mode():
        # Compute all FFT convolutions at once
        if grid_fft is None:
            grid_tensor = grid_state if grid_state is not None else torch.as_tensor(
                grid, dtype=torch.float32, device=self.device
            )
            grid_fft = torch.fft.fft2(grid_tensor)
        
        stacked_part_ffts = torch.stack(part_ffts, dim=0)
        overlap_batch = torch.fft.ifft2(grid_fft.unsqueeze(0) * stacked_part_ffts).real
        rounded_batch = torch.round(overlap_batch)
        
        # Pre-allocate results tensor on GPU
        # Format: [is_feasible, smallest_col, largest_row] for each rotation
        num_rotations = len(part_shapes)
        results_tensor = torch.full((num_rotations, 3), -1, dtype=torch.int32, device=self.device)
        
        # Process all rotations on GPU without syncing
        for i, part_shape in enumerate(part_shapes):
            cropped = rounded_batch[i, part_shape[0] - 1 : grid.shape[0], part_shape[1] - 1 : grid.shape[1]]
            zero_mask = (cropped == 0)
            rows_with_zeros = zero_mask.any(dim=1)
            
            # Check feasibility without CPU sync
            if rows_with_zeros.any():
                # Find largest row with zeros (bottom-most)
                largest_row = rows_with_zeros.nonzero(as_tuple=True)[0].max()
                
                # Find smallest col in that row (left-most)
                row_zeros = zero_mask[largest_row]
                smallest_col = row_zeros.nonzero(as_tuple=True)[0].min()
                
                results_tensor[i, 0] = 1  # feasible
                results_tensor[i, 1] = smallest_col
                results_tensor[i, 2] = largest_row + part_shape[0] - 1
        
        # SINGLE SYNC POINT: Transfer all results to CPU at once
        results_cpu = results_tensor.cpu().numpy()
        
        # Convert to expected format
        results = []
        for i, part_shape in enumerate(part_shapes):
            if results_cpu[i, 0] == 1:
                results.append((True, int(results_cpu[i, 1]), int(results_cpu[i, 2])))
            else:
                results.append((False, None, None))
        
        return results
```

#### Alternative: Use `torch.compile()` (PyTorch 2.0+)

```python
@torch.compile(mode="reduce-overhead")
def _compute_positions_compiled(rounded_batch, part_shapes, grid_shape):
    """JIT-compiled position computation."""
    results = []
    for i, part_shape in enumerate(part_shapes):
        cropped = rounded_batch[i, part_shape[0] - 1 : grid_shape[0], part_shape[1] - 1 : grid_shape[1]]
        zero_mask = (cropped == 0)
        rows_with_zeros = zero_mask.any(dim=1)
        
        if rows_with_zeros.any():
            largest_row = rows_with_zeros.nonzero(as_tuple=True)[0].max()
            smallest_col = zero_mask[largest_row].nonzero(as_tuple=True)[0].min()
            results.append((True, smallest_col, largest_row + part_shape[0] - 1))
        else:
            results.append((False, None, None))
    return results
```

#### Testing the Change

```python
# Add to profile_quick.py for A/B testing
def test_sync_optimization():
    # Time original vs optimized version
    pass
```

---

### O2: CuPy Backend (Alternative to PyTorch)

**Estimated Speedup**: 2-4x  
**Complexity**: High  
**Files to Modify**: New `collision_backend_cupy.py`, minor changes to `collision_backend.py`

#### Why CuPy?

| Aspect | PyTorch | CuPy |
|--------|---------|------|
| **Purpose** | Deep Learning | NumPy on GPU |
| **Overhead** | Higher (autograd, graph building) | Lower (direct CUDA calls) |
| **FFT Backend** | cuFFT (same) | cuFFT (same) |
| **Sync Behavior** | Lazy evaluation, explicit sync | More eager, controllable |
| **NumPy Compatibility** | Partial | Near 100% |
| **Installation** | `pip install torch` | `pip install cupy-cuda12x` |
| **Memory Management** | Managed | More control |

#### Key Advantages for This Codebase

1. **Lower Overhead**: CuPy is designed for numerical computing, not deep learning. No gradient tracking, no computation graph.

2. **Better NumPy Integration**: The codebase already uses NumPy extensively. CuPy is a drop-in replacement:
   ```python
   import cupy as cp
   # Instead of:
   # np.fft.fft2(grid)
   # Use:
   # cp.fft.fft2(cp.asarray(grid))
   ```

3. **Explicit Memory Control**: Can use memory pools to avoid allocation overhead:
   ```python
   mempool = cp.get_default_memory_pool()
   mempool.set_limit(size=4 * 1024**3)  # 4GB limit
   ```

4. **Fused Operations**: CuPy can fuse multiple operations into single CUDA kernels:
   ```python
   @cp.fuse()
   def fused_overlap_check(grid_fft, part_fft):
       return cp.round(cp.fft.ifft2(grid_fft * part_fft).real)
   ```

5. **Raw CUDA Kernel Access**: For critical paths, can write custom CUDA:
   ```python
   find_bottom_left_kernel = cp.RawKernel(r'''
   extern "C" __global__
   void find_bottom_left(const float* cropped, int rows, int cols, int* result) {
       // Custom CUDA kernel for finding bottom-left zero
   }
   ''', 'find_bottom_left')
   ```

#### Implementation Plan

##### Step 1: Create CuPy Backend

```python
# collision_backend_cupy.py
import numpy as np
import cupy as cp
from cupy import fft as cp_fft

class CuPyCollisionBackend:
    def __init__(self, device_id=0):
        self.name = "cupy_gpu"
        self.device = cp.cuda.Device(device_id)
        self.device.use()
        
        # Pre-allocate memory pool for consistent performance
        self.mempool = cp.get_default_memory_pool()
        self.pinned_mempool = cp.get_default_pinned_memory_pool()
        
        # Cache for FFT plans (cuFFT plan reuse)
        self._fft_plans = {}
    
    def prepare_part_fft(self, part_matrix, bin_length, bin_width):
        """Pre-compute FFT of flipped, padded part matrix."""
        # Transfer to GPU
        part_gpu = cp.asarray(part_matrix, dtype=cp.float32)
        
        # Flip and pad
        part_flipped = cp.flip(cp.flip(part_gpu, axis=0), axis=1)
        padded = cp.pad(
            part_flipped,
            ((0, bin_length - part_matrix.shape[0]), (0, bin_width - part_matrix.shape[1])),
            mode='constant',
            constant_values=0
        )
        
        return cp_fft.fft2(padded)
    
    def prepare_rotation_tensor(self, part_matrix):
        """Pre-transfer rotation matrix to GPU."""
        return cp.asarray(part_matrix, dtype=cp.float32)
    
    def create_grid_state(self, length, width):
        """Create GPU grid state."""
        return cp.zeros((length, width), dtype=cp.float32)
    
    def update_grid_region(self, grid_state, x, y, part_matrix, shapes, part_tensor=None):
        """Update grid with placed part."""
        if grid_state is None:
            return
        y0 = y - shapes[0] + 1
        y1 = y + 1
        x0 = x
        x1 = x + shapes[1]
        
        if part_tensor is None:
            part_tensor = cp.asarray(part_matrix, dtype=cp.float32)
        
        grid_state[y0:y1, x0:x1] += part_tensor
    
    def compute_grid_fft(self, grid_state):
        """Compute and cache grid FFT."""
        return cp_fft.fft2(grid_state)
    
    def find_bottom_left_zero_batch(self, grid, part_ffts, part_shapes, grid_state=None, grid_fft=None):
        """Find bottom-left zero position for multiple rotations."""
        if not part_ffts:
            return []
        
        # Compute grid FFT if not cached
        if grid_fft is None:
            grid_gpu = grid_state if grid_state is not None else cp.asarray(grid, dtype=cp.float32)
            grid_fft = cp_fft.fft2(grid_gpu)
        
        # Stack all part FFTs
        stacked = cp.stack(part_ffts, axis=0)
        
        # Batch FFT convolution
        overlap_batch = cp_fft.ifft2(grid_fft[cp.newaxis, ...] * stacked, axes=(-2, -1)).real
        rounded_batch = cp.round(overlap_batch)
        
        # Process results - minimize GPU->CPU transfers
        results = []
        
        for i, part_shape in enumerate(part_shapes):
            cropped = rounded_batch[i, part_shape[0] - 1 : grid.shape[0], part_shape[1] - 1 : grid.shape[1]]
            zero_mask = (cropped == 0)
            rows_with_zeros = cp.any(zero_mask, axis=1)
            
            if not cp.any(rows_with_zeros):
                results.append((False, None, None))
                continue
            
            # Find positions on GPU
            largest_row = int(cp.flatnonzero(rows_with_zeros).max())
            smallest_col = int(cp.flatnonzero(zero_mask[largest_row]).min())
            
            results.append((True, smallest_col, largest_row + part_shape[0] - 1))
        
        return results
    
    def find_bottom_left_zero(self, grid, part_fft, part_shape, grid_state=None, grid_fft=None):
        """Single rotation version."""
        results = self.find_bottom_left_zero_batch(grid, [part_fft], [part_shape], grid_state, grid_fft)
        return results[0]


# Alternative: Fused kernel for critical path
@cp.fuse()
def fused_overlap_round(grid_fft, part_fft):
    """Fused FFT convolution and rounding."""
    return cp.round(cp.fft.ifft2(grid_fft * part_fft).real)
```

##### Step 2: Advanced CuPy Optimizations

```python
# Custom CUDA kernel for finding bottom-left zero (maximum performance)
_find_bl_kernel = cp.RawKernel(r'''
extern "C" __global__
void find_bottom_left_zero(
    const float* __restrict__ cropped,
    const int rows,
    const int cols,
    int* __restrict__ largest_row,
    int* __restrict__ smallest_col,
    bool* __restrict__ found
) {
    // Shared memory for row results
    __shared__ bool row_has_zero[1024];
    __shared__ int col_positions[1024];
    
    int row = blockIdx.x;
    if (row >= rows) return;
    
    // Check if this row has any zeros
    bool has_zero = false;
    int first_zero_col = cols;
    
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        if (cropped[row * cols + col] == 0.0f) {
            has_zero = true;
            if (col < first_zero_col) {
                first_zero_col = col;
            }
        }
    }
    
    // Reduce within warp
    __shared__ bool any_zero;
    __shared__ int min_col;
    
    if (threadIdx.x == 0) {
        any_zero = false;
        min_col = cols;
    }
    __syncthreads();
    
    if (has_zero) {
        atomicOr((int*)&any_zero, 1);
        atomicMin(&min_col, first_zero_col);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        row_has_zero[row] = any_zero;
        col_positions[row] = min_col;
    }
    __syncthreads();
    
    // Final reduction to find largest row with zeros
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *found = false;
        for (int r = rows - 1; r >= 0; r--) {
            if (row_has_zero[r]) {
                *largest_row = r;
                *smallest_col = col_positions[r];
                *found = true;
                break;
            }
        }
    }
}
''', 'find_bottom_left_zero')


class CuPyCollisionBackendOptimized(CuPyCollisionBackend):
    """CuPy backend with custom CUDA kernels for maximum performance."""
    
    def __init__(self, device_id=0):
        super().__init__(device_id)
        self.name = "cupy_gpu_optimized"
        
        # Pre-allocate output buffers
        self._largest_row = cp.zeros(1, dtype=cp.int32)
        self._smallest_col = cp.zeros(1, dtype=cp.int32)
        self._found = cp.zeros(1, dtype=cp.bool_)
    
    def _find_bl_with_kernel(self, cropped):
        """Use custom CUDA kernel for position finding."""
        rows, cols = cropped.shape
        
        # Reset outputs
        self._found[0] = False
        
        # Launch kernel
        threads_per_block = min(256, cols)
        blocks = rows
        
        _find_bl_kernel(
            (blocks,), (threads_per_block,),
            (cropped.ravel(), rows, cols, self._largest_row, self._smallest_col, self._found)
        )
        
        # Single sync point
        cp.cuda.Stream.null.synchronize()
        
        if self._found[0]:
            return True, int(self._smallest_col[0]), int(self._largest_row[0])
        return False, None, None
```

##### Step 3: Integration

```python
# Modify collision_backend.py to support CuPy
def create_collision_backend(name):
    if name.startswith("torch_"):
        configure_torch_runtime()
    
    if name == "cupy_gpu":
        try:
            import cupy as cp
            from collision_backend_cupy import CuPyCollisionBackend
            return CuPyCollisionBackend()
        except ImportError:
            raise RuntimeError("CuPy not installed. Install with: pip install cupy-cuda12x")
    
    if name == "cupy_gpu_optimized":
        try:
            from collision_backend_cupy import CuPyCollisionBackendOptimized
            return CuPyCollisionBackendOptimized()
        except ImportError:
            raise RuntimeError("CuPy not installed. Install with: pip install cupy-cuda12x")
    
    # ... existing torch backends ...
```

##### Step 4: Benchmark Comparison Script

```python
# benchmark_backends.py
import time
import numpy as np
from collision_backend import create_collision_backend

def benchmark_backend(backend_name, num_iterations=100):
    backend = create_collision_backend(backend_name)
    
    # Create test data
    grid = np.random.randint(0, 2, (500, 300)).astype(np.float32)
    part = np.random.randint(0, 2, (50, 30)).astype(np.float32)
    
    # Prepare FFT
    part_fft = backend.prepare_part_fft(part, 500, 300)
    grid_state = backend.create_grid_state(500, 300)
    
    # Warmup
    for _ in range(10):
        backend.find_bottom_left_zero_batch(grid, [part_fft], [(50, 30)], grid_state)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        backend.find_bottom_left_zero_batch(grid, [part_fft] * 4, [(50, 30)] * 4, grid_state)
        times.append(time.perf_counter() - start)
    
    print(f"{backend_name}: {np.mean(times)*1000:.3f}ms ± {np.std(times)*1000:.3f}ms")

if __name__ == "__main__":
    benchmark_backend("torch_gpu")
    benchmark_backend("cupy_gpu")
    benchmark_backend("cupy_gpu_optimized")
```

#### CuPy Installation

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x

# Verify installation
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

---

### O3: Numba JIT for Vacancy Checking

**Estimated Speedup**: 1.2-1.5x on this section (overall ~1.1x)  
**Complexity**: Low  
**Files to Modify**: `binClassNew.py`, `binClassInitialSol.py`

#### Current Code

```python
# In can_insert() - called 110K+ times
for currRot in range(nrot):
    shape = part_shapes[currRot]
    
    if shape[0] > self.length or shape[1] > self.width:
        continue
    
    dens = part_densities[currRot]
    
    # This section is slow
    subarrays = np.lib.stride_tricks.sliding_window_view(vacancy, shape[0])
    binaryResult = np.any(np.all(subarrays >= dens, axis=1))
```

#### Optimized with Numba

```python
from numba import jit, prange
import numpy as np

@jit(nopython=True, cache=True, fastmath=True)
def check_vacancy_fit(vacancy, density, shape_height):
    """
    Check if a part with given density profile can fit in the vacancy vector.
    
    Args:
        vacancy: 1D array of max contiguous zeros per row
        density: 1D array of max consecutive 1s per row for the part
        shape_height: Height of the part
    
    Returns:
        True if part can potentially fit, False otherwise
    """
    n = len(vacancy)
    window_size = shape_height
    
    if window_size > n:
        return False
    
    # Slide window over vacancy vector
    for start in range(n - window_size + 1):
        fits = True
        for i in range(window_size):
            if vacancy[start + i] < density[i]:
                fits = False
                break
        if fits:
            return True
    
    return False


@jit(nopython=True, cache=True, parallel=True)
def check_all_rotations_parallel(vacancy, densities, shapes, length, width):
    """
    Check all rotations in parallel.
    
    Args:
        vacancy: 1D vacancy vector
        densities: List of density arrays for each rotation
        shapes: List of (height, width) tuples
        length: Bin length
        width: Bin width
    
    Returns:
        Array of booleans indicating feasibility of each rotation
    """
    n_rotations = len(shapes)
    results = np.zeros(n_rotations, dtype=np.bool_)
    
    for rot in prange(n_rotations):
        h, w = shapes[rot]
        if h <= length and w <= width:
            results[rot] = check_vacancy_fit(vacancy, densities[rot], h)
    
    return results
```

#### Integration in `binClassNew.py`

```python
# At top of file
from numba import jit
import numpy as np

# Add the JIT functions (see above)

# Modify can_insert():
def can_insert(self, part, machPart, plott=False):
    # ... existing setup code ...
    
    # Replace the for loop with:
    feasibility = check_all_rotations_parallel(
        self.vacancy_vector,
        part.densities,  # Need to ensure this is a list of numpy arrays
        part.shapes,     # List of tuples
        self.length,
        self.width
    )
    
    # Collect feasible rotations
    feasible_rotations = []
    feasible_shapes = []
    feasible_ffts = []
    
    for rot in range(part.nrot):
        if feasibility[rot]:
            feasible_rotations.append(rot)
            feasible_shapes.append(part.shapes[rot])
            feasible_ffts.append(machPart.ffts[rot])
    
    # ... rest of function ...
```

---

### O4: Batch Solution Evaluation

**Estimated Speedup**: 1.3-1.5x  
**Complexity**: High  
**Files to Modify**: `placement.py`, `BRKGA_alg3.py`

#### Concept

Instead of evaluating solutions one at a time, group solutions that assign similar parts to the same machines and share intermediate computations.

#### Implementation Sketch

```python
def evaluate_solutions_batched(problem_data, nbParts, nbMachines, thresholds, chromosomes, matching, collision_backend):
    """
    Evaluate multiple solutions at once, sharing computation where possible.
    
    Key insight: Solutions that assign the same parts to the same machines
    can share FFT computations for the first few insertions.
    """
    n_solutions = len(chromosomes)
    
    # Group solutions by machine assignments
    # MV (second half of chromosome) determines machine assignment
    machine_assignments = []
    for chrom in chromosomes:
        MV = chrom[nbParts:]
        assignment = tuple(
            np.searchsorted(thresholds, MV[i], side='right')
            for i in range(nbParts)
        )
        machine_assignments.append(assignment)
    
    # Group similar assignments together
    from collections import defaultdict
    groups = defaultdict(list)
    for i, assignment in enumerate(machine_assignments):
        # Use first 5 parts' assignments as group key
        key = assignment[:5]
        groups[key].append(i)
    
    # Evaluate each group, sharing early computations
    results = [None] * n_solutions
    
    for group_key, indices in groups.items():
        if len(indices) == 1:
            # Single solution, evaluate normally
            results[indices[0]] = placementProcedure(
                problem_data, nbParts, nbMachines, thresholds,
                chromosomes[indices[0]], matching, collision_backend
            )
        else:
            # Multiple solutions with similar assignments
            # Could share initial bin states
            for idx in indices:
                results[idx] = placementProcedure(
                    problem_data, nbParts, nbMachines, thresholds,
                    chromosomes[idx], matching, collision_backend
                )
    
    return results
```

---

## Medium-Impact Optimizations

### O5: CUDA Streams for Parallel Solution Evaluation

**Estimated Speedup**: 1.2-1.4x  
**Complexity**: Medium  
**Files to Modify**: `collision_backend.py`, `BRKGA_alg3.py`

#### Concept

Use multiple CUDA streams to evaluate 2-4 solutions concurrently on the GPU.

```python
import torch.cuda

class TorchCollisionBackendMultiStream(TorchCollisionBackend):
    def __init__(self, device, num_streams=4):
        super().__init__(device)
        self.name = f"torch_{device}_multistream"
        self.streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
        self.stream_idx = 0
    
    def get_next_stream(self):
        stream = self.streams[self.stream_idx]
        self.stream_idx = (self.stream_idx + 1) % len(self.streams)
        return stream
    
    def find_bottom_left_zero_batch_async(self, grid, part_ffts, part_shapes, grid_state=None, grid_fft=None):
        """Async version that uses dedicated stream."""
        stream = self.get_next_stream()
        with torch.cuda.stream(stream):
            return super().find_bottom_left_zero_batch(grid, part_ffts, part_shapes, grid_state, grid_fft)
```

---

### O6: Population Size Tuning

**Estimated Speedup**: 1.5-2.5x  
**Complexity**: Low  
**Files to Modify**: `BRKGA_alg3.py`

#### Analysis

Current: `num_individuals = mult * nbParts = 10 * 50 = 500`

Each generation evaluates `num_individuals - num_elites = 500 - 50 = 450` solutions.

#### Recommendations

| Population | Evaluations/Gen | Est. Time/Gen | Quality Impact |
|------------|-----------------|---------------|----------------|
| 500 | 450 | 25-40s | Baseline |
| 300 | 270 | 15-24s | Slight decrease |
| 200 | 180 | 10-16s | Moderate decrease |
| 100 | 90 | 5-8s | Noticeable decrease |

#### Adaptive Population Strategy

```python
class BRKGA:
    def __init__(self, ..., adaptive_population=True):
        self.adaptive_population = adaptive_population
        self.stagnation_counter = 0
        
    def fit(self, verbose=False):
        for g in range(self.num_generations):
            # ... existing code ...
            
            if self.adaptive_population:
                # Reduce population if stagnating
                if g > 10 and (best_fitness == prev_best):
                    self.stagnation_counter += 1
                    if self.stagnation_counter > 5:
                        # Reduce population by 20%
                        self.num_individuals = max(50, int(self.num_individuals * 0.8))
                        self.num_elites = max(5, int(self.num_individuals * 0.1))
                        self.num_mutants = max(10, int(self.num_individuals * 0.15))
                else:
                    self.stagnation_counter = 0
```

---

### O7: Early Termination Heuristics

**Estimated Speedup**: Variable (can skip many can_insert calls)  
**Complexity**: Low  
**Files to Modify**: `binClassNew.py`

```python
def can_insert(self, part, machPart, plott=False):
    # Early termination: if part is larger than remaining area
    remaining_area = (self.length * self.width) - self.area
    if part.area > remaining_area:
        return False
    
    # Early termination: if part height > remaining enclosure height
    remaining_height = self.length - self.enclosure_box_length
    min_part_height = min(s[0] for s in part.shapes)
    if min_part_height > remaining_height:
        return False
    
    # ... proceed with FFT check only if early tests pass ...
```

---

## Low-Impact Optimizations

### O8: Memory Pre-allocation

**Estimated Speedup**: 1.05-1.1x  
**Complexity**: Low

```python
class BuildingPlate:
    # Class-level pre-allocated buffers (shared across instances)
    _PREALLOCATED_GRID = None
    _PREALLOCATED_VACANCY = None
    
    @classmethod
    def preallocate_buffers(cls, max_length=600, max_width=400):
        """Call once at startup."""
        cls._PREALLOCATED_GRID = np.zeros((max_length, max_width), dtype=np.uint8)
        cls._PREALLOCATED_VACANCY = np.zeros(max_length, dtype=np.int32)
```

---

### O9: Grid State Optimization

**Estimated Speedup**: 1.05x  
**Complexity**: Low

```python
# Use float16 instead of float32 for grid state (half memory bandwidth)
def create_grid_state(self, length, width):
    return torch.zeros((length, width), dtype=torch.float16, device=self.device)
```

---

## Implementation Priority

### Phase 1: Quick Wins (Expected: 3-4x speedup)
1. **O1**: Reduce GPU sync points - 2 days
2. **O3**: Numba JIT - 0.5 days
3. **O7**: Early termination - 0.5 days

### Phase 2: CuPy Migration (Expected: additional 1.5-2x)
4. **O2**: CuPy backend - 3-5 days

### Phase 3: Fine Tuning (Expected: additional 1.2x)
5. **O6**: Population tuning - 0.5 days
6. **O5**: CUDA streams - 1 day

---

## Benchmarking Protocol

### Test Script

```python
# benchmark_optimizations.py
import time
import sys
import numpy as np

def run_benchmark(backend_name, num_generations=5, num_individuals=500):
    """Run standardized benchmark."""
    from profile_quick import setup_problem
    from BRKGA_alg3 import BRKGA
    
    problem_data, nbParts, nbMachines, thresholds, instanceParts, initial_sol, collision_backend = \
        setup_problem(backend_name=backend_name)
    
    model = BRKGA(
        problem_data, nbParts, nbMachines, thresholds, instanceParts, initial_sol,
        collision_backend=collision_backend,
        num_generations=num_generations,
        num_individuals=num_individuals,
        num_elites=int(num_individuals * 0.1),
        num_mutants=int(num_individuals * 0.15)
    )
    
    start = time.perf_counter()
    model.fit(verbose=True)
    total_time = time.perf_counter() - start
    
    print(f"\n{'='*60}")
    print(f"Backend: {backend_name}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Time per generation: {total_time/num_generations:.2f}s")
    print(f"Final fitness: {model.best_fitness:.2f}")
    print(f"{'='*60}")
    
    return total_time, model.best_fitness

if __name__ == "__main__":
    backends = ["torch_gpu", "cupy_gpu"]  # Add more as implemented
    
    for backend in backends:
        try:
            run_benchmark(backend)
        except Exception as e:
            print(f"Backend {backend} failed: {e}")
```

### Metrics to Track

1. **Time per generation** (primary metric)
2. **Final fitness** (quality check - should not degrade significantly)
3. **GPU memory usage** (`nvidia-smi`)
4. **Cache hit rate** (for solution caching)

---

## Summary

| Optimization | Difficulty | Time | Speedup | Cumulative |
|--------------|------------|------|---------|------------|
| O1: GPU sync reduction | Medium | 2 days | 2-3x | 2-3x |
| O3: Numba JIT | Low | 0.5 days | 1.1x | 2.2-3.3x |
| O7: Early termination | Low | 0.5 days | 1.1x | 2.4-3.6x |
| O2: CuPy backend | High | 4 days | 1.5-2x | 3.6-7.2x |
| O6: Population tuning | Low | 0.5 days | 1.3x | 4.7-9.4x |
| O5: CUDA streams | Medium | 1 day | 1.2x | 5.6-11x |

**Estimated final speedup: 5-10x** (from 25-40s to 3-5s per generation)
