import numpy as np
import torch

DEFAULT_TORCH_TF32 = False
DEFAULT_CUFFT_PLAN_CACHE = 32


def _parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_or_default(name, default):
    import os

    value = os.getenv(name)
    if value is None:
        return default
    return value


def configure_torch_runtime(tf32=None, cufft_plan_cache=None):
    if tf32 is None:
        tf32 = _parse_bool(_env_or_default("ABRKGA_TORCH_TF32", DEFAULT_TORCH_TF32))
    else:
        tf32 = _parse_bool(tf32)

    if cufft_plan_cache is None:
        cufft_plan_cache = int(_env_or_default("ABRKGA_CUFFT_PLAN_CACHE", DEFAULT_CUFFT_PLAN_CACHE))
    else:
        cufft_plan_cache = int(cufft_plan_cache)

    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    try:
        torch.backends.cuda.cufft_plan_cache.max_size = cufft_plan_cache
    except Exception:
        pass

    return tf32, cufft_plan_cache


class BaseCollisionBackend:
    def __init__(self, name):
        self.name = name

    def prepare_part_fft(self, part_matrix, bin_length, bin_width):
        raise NotImplementedError

    def prepare_rotation_tensor(self, part_matrix):
        """Pre-compute device tensor for a rotation matrix. Returns None for CPU backends."""
        return None

    def find_bottom_left_zero(self, grid, part_fft, part_shape):
        raise NotImplementedError

    def find_bottom_left_zero_batch(self, grid, part_ffts, part_shapes):
        raise NotImplementedError

    def create_grid_state(self, length, width):
        return None

    def update_grid_region(self, grid_state, x, y, part_matrix, shapes, part_tensor=None):
        return None


class TorchCollisionBackend(BaseCollisionBackend):
    def __init__(self, device, use_batch=True):
        suffix = "batched" if use_batch else "unbatched"
        super().__init__(f"torch_{device}_{suffix}")
        self.device = torch.device(device)
        self.use_batch = use_batch
        # Pre-allocate tensor cache for common operations
        self._tensor_cache = {}

    def prepare_part_fft(self, part_matrix, bin_length, bin_width):
        # Use non-blocking transfer for GPU
        part_tensor = torch.tensor(part_matrix.copy(), dtype=torch.float32, device=self.device)
        part_tensor_flipped = torch.flip(part_tensor, dims=[0, 1])
        padded = torch.nn.functional.pad(
            part_tensor_flipped,
            (0, bin_width - part_matrix.shape[1], 0, bin_length - part_matrix.shape[0]),
        )
        return torch.fft.fft2(padded)

    def prepare_rotation_tensor(self, part_matrix):
        """Pre-compute GPU tensor for a rotation matrix (avoids CPU->GPU transfer per insert)."""
        return torch.tensor(part_matrix, dtype=torch.float32, device=self.device)

    def create_grid_state(self, length, width):
        return torch.zeros((length, width), dtype=torch.float32, device=self.device)

    def update_grid_region(self, grid_state, x, y, part_matrix, shapes, part_tensor=None):
        if grid_state is None:
            return
        y0 = y - shapes[0] + 1
        y1 = y + 1
        x0 = x
        x1 = x + shapes[1]
        # Use pre-computed tensor if available, else create one
        if part_tensor is None:
            part_tensor = torch.tensor(part_matrix, dtype=torch.float32, device=self.device)
        grid_state[y0:y1, x0:x1] += part_tensor

    def compute_grid_fft(self, grid_state):
        """Compute and return the grid FFT for caching."""
        with torch.inference_mode():
            return torch.fft.fft2(grid_state)

    def find_bottom_left_zero(self, grid, part_fft, part_shape, grid_state=None, grid_fft=None):
        with torch.inference_mode():
            if grid_fft is None:
                grid_tensor = grid_state if grid_state is not None else torch.as_tensor(grid, dtype=torch.float32, device=self.device)
                grid_fft = torch.fft.fft2(grid_tensor)
            overlap = torch.fft.ifft2(grid_fft * part_fft).real
            cropped = torch.round(overlap[part_shape[0] - 1 : grid.shape[0], part_shape[1] - 1 : grid.shape[1]])

            zero_mask = (cropped == 0)
            rows_with_zeros = zero_mask.any(dim=1)
            if not rows_with_zeros.any():
                return False, None, None

            # Use argmax on reversed tensor to find bottom-most row
            num_rows = rows_with_zeros.shape[0]
            largest_row = num_rows - 1 - rows_with_zeros.flip(0).int().argmax().item()
            smallest_col = zero_mask[largest_row].int().argmax().item()
            largest_row_real = largest_row + part_shape[0] - 1
            return True, smallest_col, largest_row_real

    def find_bottom_left_zero_batch(self, grid, part_ffts, part_shapes, grid_state=None, grid_fft=None):
        if not part_ffts:
            return []
        # Use traditional approach for ≤2 rotations (faster due to lower overhead)
        # Batched approach wins at 3+ rotations
        if not self.use_batch or len(part_ffts) <= 2:
            return [
                self.find_bottom_left_zero(grid, part_ffts[i], part_shapes[i], grid_state=grid_state, grid_fft=grid_fft)
                for i in range(len(part_ffts))
            ]

        num_rot = len(part_ffts)
        H, W = grid.shape
        
        with torch.inference_mode():
            # Use cached grid FFT if provided, otherwise compute
            if grid_fft is None:
                grid_tensor = grid_state if grid_state is not None else torch.as_tensor(grid, dtype=torch.float32, device=self.device)
                grid_fft = torch.fft.fft2(grid_tensor)
            stacked_part_ffts = torch.stack(part_ffts, dim=0)
            overlap_batch = torch.fft.ifft2(grid_fft.unsqueeze(0) * stacked_part_ffts).real
            rounded_batch = torch.round(overlap_batch)
            
            # Zero mask for all rotations at once
            zero_mask = (rounded_batch == 0)  # (num_rot, H, W)
            
            # Part shape constraints - valid region where part fits
            part_heights = torch.tensor([s[0] for s in part_shapes], device=self.device)
            part_widths = torch.tensor([s[1] for s in part_shapes], device=self.device)
            
            # Create validity masks using broadcasting
            row_idx = torch.arange(H, device=self.device).view(1, H, 1)  # (1, H, 1)
            col_idx = torch.arange(W, device=self.device).view(1, 1, W)  # (1, 1, W)
            
            # Valid positions: row >= h-1, col >= w-1 for each rotation
            valid_row = row_idx >= (part_heights - 1).view(-1, 1, 1)  # (num_rot, H, 1)
            valid_col = col_idx >= (part_widths - 1).view(-1, 1, 1)   # (num_rot, 1, W)
            valid_mask = valid_row & valid_col  # (num_rot, H, W)
            
            # Combine: valid zeros only
            valid_zeros = zero_mask & valid_mask  # (num_rot, H, W)
            
            # Score: row * (W+1) - col (maximize row, minimize col as tiebreaker)
            score = torch.where(
                valid_zeros,
                row_idx.float() * (W + 1) - col_idx.float(),
                torch.tensor(-1e9, device=self.device)
            )
            
            # Find best position for each rotation (single operation)
            flat_scores = score.view(num_rot, -1)  # (num_rot, H*W)
            best_flat_idx = flat_scores.argmax(dim=1)  # (num_rot,)
            max_scores = flat_scores.max(dim=1).values  # (num_rot,)
            
            # Convert flat index to row, col
            best_row_full = best_flat_idx // W
            best_col_full = best_flat_idx % W
            
            # Has valid zero if max score is not -1e9
            has_valid = max_scores > -1e8
            
            # Convert to output coordinates:
            # smallest_col = col_full - (w - 1)  [cropped coordinate]
            # largest_row_real = row_full        [full coordinate, as expected by caller]
            smallest_col = best_col_full - (part_widths - 1)
            largest_row_real = best_row_full
            
            # SINGLE SYNC POINT: transfer all results to CPU at once
            results_tensor = torch.stack([
                has_valid.int(),
                smallest_col,
                largest_row_real
            ], dim=1)
            results_cpu = results_tensor.cpu().numpy()
        
        # Convert to expected format
        results = []
        for i in range(num_rot):
            if results_cpu[i, 0] == 1:
                results.append((True, int(results_cpu[i, 1]), int(results_cpu[i, 2])))
            else:
                results.append((False, None, None))
        
        return results


class NumpyCollisionBackend(BaseCollisionBackend):
    def __init__(self):
        super().__init__("numpy_cpu")

    def prepare_part_fft(self, part_matrix, bin_length, bin_width):
        flipped = np.flip(part_matrix, axis=(0, 1))
        padded = np.pad(flipped, ((0, bin_length - part_matrix.shape[0]), (0, bin_width - part_matrix.shape[1])))
        return np.fft.fft2(padded)

    def find_bottom_left_zero(self, grid, part_fft, part_shape, grid_state=None):
        results = self.find_bottom_left_zero_batch(grid, [part_fft], [part_shape], grid_state=grid_state)
        return results[0]

    def find_bottom_left_zero_batch(self, grid, part_ffts, part_shapes, grid_state=None):
        if not part_ffts:
            return []

        grid_fft = np.fft.fft2(grid)
        stacked_part_ffts = np.stack(part_ffts, axis=0)
        overlap_batch = np.fft.ifft2(grid_fft[np.newaxis, ...] * stacked_part_ffts, axes=(-2, -1)).real
        rounded_batch = np.round(overlap_batch)

        results = []
        for i, part_shape in enumerate(part_shapes):
            cropped = rounded_batch[i, part_shape[0] - 1 : grid.shape[0], part_shape[1] - 1 : grid.shape[1]]
            rows_with_zeros = np.any(cropped == 0, axis=1)
            if not rows_with_zeros.any():
                results.append((False, None, None))
                continue

            largest_row = np.flatnonzero(rows_with_zeros).max().item()
            smallest_col = np.flatnonzero(cropped[largest_row] == 0).min().item()
            largest_row_real = largest_row + part_shape[0] - 1
            results.append((True, smallest_col, largest_row_real))

        return results


def create_collision_backend(name):
    if name.startswith("torch_"):
        configure_torch_runtime()

    if name == "torch_gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("torch_gpu backend selected but CUDA is unavailable.")
        return TorchCollisionBackend("cuda", use_batch=True)
    if name == "torch_gpu_unbatched":
        if not torch.cuda.is_available():
            raise RuntimeError("torch_gpu_unbatched backend selected but CUDA is unavailable.")
        return TorchCollisionBackend("cuda", use_batch=False)
    if name == "torch_cpu":
        return TorchCollisionBackend("cpu", use_batch=True)
    if name == "torch_cpu_unbatched":
        return TorchCollisionBackend("cpu", use_batch=False)
    if name == "numpy_cpu":
        return NumpyCollisionBackend()
    
    # CuPy backends (lower overhead than PyTorch)
    if name == "cupy_gpu":
        try:
            from collision_backend_cupy import CuPyCollisionBackend
            return CuPyCollisionBackend()
        except ImportError as e:
            raise RuntimeError(f"CuPy backend unavailable: {e}. Install with: pip install cupy-cuda12x")
    if name == "cupy_gpu_optimized":
        try:
            from collision_backend_cupy import CuPyCollisionBackendOptimized
            return CuPyCollisionBackendOptimized()
        except ImportError as e:
            raise RuntimeError(f"CuPy backend unavailable: {e}. Install with: pip install cupy-cuda12x")
    
    raise ValueError(f"Unsupported collision backend: {name}")
