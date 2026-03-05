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

    def find_bottom_left_zero(self, grid, part_fft, part_shape):
        raise NotImplementedError

    def find_bottom_left_zero_batch(self, grid, part_ffts, part_shapes):
        raise NotImplementedError

    def create_grid_state(self, length, width):
        return None

    def update_grid_region(self, grid_state, x, y, part_matrix, shapes):
        return None


class TorchCollisionBackend(BaseCollisionBackend):
    def __init__(self, device, use_batch=True):
        suffix = "batched" if use_batch else "unbatched"
        super().__init__(f"torch_{device}_{suffix}")
        self.device = torch.device(device)
        self.use_batch = use_batch

    def prepare_part_fft(self, part_matrix, bin_length, bin_width):
        part_tensor = torch.tensor(part_matrix.copy(), dtype=torch.float32, device=self.device)
        part_tensor_flipped = torch.flip(part_tensor, dims=[0, 1])
        padded = torch.nn.functional.pad(
            part_tensor_flipped,
            (0, bin_width - part_matrix.shape[1], 0, bin_length - part_matrix.shape[0]),
        )
        return torch.fft.fft2(padded)

    def create_grid_state(self, length, width):
        return torch.zeros((length, width), dtype=torch.float32, device=self.device)

    def update_grid_region(self, grid_state, x, y, part_matrix, shapes):
        if grid_state is None:
            return
        # part_matrix is already contiguous from data loading
        part_tensor = torch.as_tensor(part_matrix, dtype=torch.float32, device=self.device)
        y0 = y - shapes[0] + 1
        y1 = y + 1
        x0 = x
        x1 = x + shapes[1]
        grid_state[y0:y1, x0:x1] += part_tensor

    def find_bottom_left_zero(self, grid, part_fft, part_shape, grid_state=None):
        grid_tensor = grid_state if grid_state is not None else torch.as_tensor(grid, dtype=torch.float32, device=self.device)
        overlap = torch.fft.ifft2(torch.fft.fft2(grid_tensor) * part_fft).real
        cropped = torch.round(overlap[part_shape[0] - 1 : grid.shape[0], part_shape[1] - 1 : grid.shape[1]])

        rows_with_zeros = (cropped == 0).any(dim=1)
        if not rows_with_zeros.any():
            return False, None, None

        largest_row = rows_with_zeros.nonzero().max().item()
        smallest_col = (cropped[largest_row] == 0).nonzero().min().item()
        largest_row_real = largest_row + part_shape[0] - 1
        return True, smallest_col, largest_row_real

    def find_bottom_left_zero_batch(self, grid, part_ffts, part_shapes, grid_state=None):
        if not part_ffts:
            return []
        if not self.use_batch:
            return [
                self.find_bottom_left_zero(grid, part_ffts[i], part_shapes[i], grid_state=grid_state)
                for i in range(len(part_ffts))
            ]

        grid_tensor = grid_state if grid_state is not None else torch.as_tensor(grid, dtype=torch.float32, device=self.device)
        grid_fft = torch.fft.fft2(grid_tensor)
        stacked_part_ffts = torch.stack(part_ffts, dim=0)
        overlap_batch = torch.fft.ifft2(grid_fft.unsqueeze(0) * stacked_part_ffts).real
        rounded_batch = torch.round(overlap_batch)

        results = []
        for i, part_shape in enumerate(part_shapes):
            cropped = rounded_batch[i, part_shape[0] - 1 : grid.shape[0], part_shape[1] - 1 : grid.shape[1]]
            rows_with_zeros = (cropped == 0).any(dim=1)
            if not rows_with_zeros.any():
                results.append((False, None, None))
                continue

            largest_row = rows_with_zeros.nonzero().max().item()
            smallest_col = (cropped[largest_row] == 0).nonzero().min().item()
            largest_row_real = largest_row + part_shape[0] - 1
            results.append((True, smallest_col, largest_row_real))

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
    raise ValueError(f"Unsupported collision backend: {name}")
