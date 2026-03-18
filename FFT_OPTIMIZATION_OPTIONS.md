# FFT Optimization Options

## Current Status (March 2026)
- **Baseline**: ~40s/gen
- **Current**: ~13s/gen  
- **Target**: <5s/gen
- **Main bottleneck**: FFT computation (~10s, 77% of total time)

## Bottleneck Breakdown

| Component | Time/gen | % of 13s |
|-----------|----------|----------|
| FFT computation | ~10s | 77% |
| Position extraction | ~2s | 15% |
| Vacancy checks | ~0.5s | 4% |
| Other (loops, inserts) | ~0.5s | 4% |

---

## Option 1: Enable TF32 (Tensor Float 32)
**Status**: [ ] Not tested

**Description**: TF32 is a math mode on Ampere+ GPUs (RTX 30xx, A100, A4000, etc.) that uses 19-bit precision instead of 32-bit for faster computation. Currently disabled in `collision_backend.py`.

**Pros**:
- Quick to test (one-line change)
- Can provide 2-3x speedup for some operations
- No code changes needed

**Cons**:
- Minor precision loss (unlikely to affect integer collision detection)
- Only works on Ampere+ GPUs

**Implementation**: Set `DEFAULT_TORCH_TF32 = True` in `collision_backend.py`

---

## Option 2: Reduce FFT Size Dynamically
**Status**: [ ] Not tested

**Description**: Instead of FFT on full 300×250 grid every time, use only the occupied region + part size. Early in packing (empty grid), FFT could be much smaller.

**Pros**:
- Smaller FFT = faster (O(n log n))
- Adaptive - biggest gains early in packing

**Cons**:
- Requires tracking occupied bounds
- Need to handle coordinate transformations
- Part FFTs would need to be recomputed for different sizes

**Implementation**: Track max occupied row, compute FFT on grid[0:max_row+part_height, :]

---

## Option 3: Batch Grid FFTs Across Multiple Parts
**Status**: [ ] Not tested

**Description**: Pre-compute what the grid state would look like after placing each rotation, then batch FFT for the NEXT part placement decision.

**Pros**:
- Reduces sequential dependency
- Better GPU utilization

**Cons**:
- Complex implementation
- Memory overhead (multiple grid states)
- May not help if parts are placed one-at-a-time

**Implementation**: Speculative execution with rollback

---

## Option 4: CUDA Streams for Solution Parallelism
**Status**: [x] Tested - NOT viable

**Description**: Process 2-4 solutions simultaneously on different CUDA streams. Each stream handles a different chromosome evaluation.

**Pros**:
- Better GPU utilization (in theory)
- Hides latency between operations

**Cons**:
- Memory: Need separate grid states per stream
- Complexity: Managing multiple streams
- **GPU already saturated** - streams add overhead without benefit

**Test Results (March 2026)**:
- 2 streams: 0.77x (23% slower)
- 4 streams: 0.72x (28% slower)  
- 8 streams: 0.71x (29% slower)
- GPU: NVIDIA RTX A4000, 16.9GB
- Conclusion: FFT operations already saturate the GPU. **Skip this option.**

**Implementation**: N/A - not recommended

---

## Option 5: Early Termination (Algorithm Change)
**Status**: [ ] Not tested

**Description**: Stop checking rotations once a "good enough" placement is found (e.g., packing density > 0.8).

**Pros**:
- Reduces FFT calls significantly
- Simple to implement

**Cons**:
- **Changes algorithm semantics** - may find different solutions
- Could affect solution quality
- Need to tune threshold

**Implementation**: Add threshold check in rotation loop

---

## Option 6: CuPy Backend
**Status**: [x] Tested - NOT viable

**Description**: CuPy has lower Python overhead than PyTorch for raw FFT operations. We have `collision_backend_cupy.py` but haven't benchmarked recently.

**Pros**:
- Lower function call overhead (in theory)
- Potentially faster for pure FFT workloads

**Cons**:
- Separate dependency (CuPy)
- May not have all PyTorch optimizations
- Needs memory management

**Test Results (March 2026)**:
- CuPy 12.2.0: 1006ms for 10K FFTs
- PyTorch 2.1.1: 213ms for 10K FFTs
- **CuPy is 4.7x SLOWER** than PyTorch!
- Reason: PyTorch has better cuFFT plan caching and optimizations
- Conclusion: **Skip this option.** PyTorch is already optimal for FFT.

**Implementation**: N/A - not recommended

---

## Option 7: Half Precision (FP16) FFT
**Status**: [x] Tested - NOT viable

**Description**: Use FP16 (half precision) instead of FP32 to potentially double FFT throughput.

**Pros**:
- Half the memory bandwidth
- Could be 2x faster on supported sizes

**Cons**:
- Loss of precision (may affect collision detection accuracy)
- cuFFT **only supports FP16 for power-of-2 dimensions**

**Test Results (March 2026)**:
- Grid 300×250 → must pad to 512×256 (1.75x more elements)
- FP32 at 300×250: 199.6ms (10K calls)
- FP32 at 512×256: 262.3ms (10K calls)
- FP16 at 512×256: 200.8ms (10K calls)
- **Net speedup: 0.99x (no improvement)**
- Larger padded size cancels out FP16 benefit
- Conclusion: **Skip this option.** Padding overhead eliminates any gain.

**Implementation**: N/A - not recommended

---

## Testing Log

### Option 1: TF32
- Date: March 6, 2026
- Result: **No improvement** (0% difference, actually -1% in isolated FFT test)
- Notes: cuFFT does not use TF32. TF32 only affects matmul/convolution operations, not FFT. This option can be skipped.

### Option 2: Dynamic FFT Size
- Date: 
- Result: 
- Notes: Trade-off with batching makes this complex. Deferred.

### Option 4: CUDA Streams
- Date: March 6, 2026
- Result: **25-30% slower**
- Notes: GPU already saturated by FFT operations. Streams add overhead.

### Option 6: CuPy
- Date: March 6, 2026
- Result: **4.7x slower** than PyTorch
- Notes: PyTorch has better cuFFT plan caching.

### Option 7: FP16
- Date: March 6, 2026
- Result: **No improvement** (0.99x)
- Notes: Requires power-of-2 padding which cancels out FP16 benefits.

(etc.)
