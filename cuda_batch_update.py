"""
Custom CUDA kernel for batched grid updates in Phase 5.

Replaces ~450 sequential CUDA kernel launches (one per placement) with a
single launch that processes all placements in parallel, allowing the GPU
command queue to be fully loaded before the CPU starts its updates.

The kernel uses a 1-D thread grid where each thread is responsible for
one cell (pixel) across the union of all part matrices being placed.
A binary search maps the global thread ID to the correct placement and
local (row, col) within that part.
"""
import torch

# ── C++ host wrapper (pybind entry point) ─────────────────────────────────────
_CPP_SRC = r"""
#include <torch/extension.h>

// total_cells is passed explicitly so the host code never dereferences
// a device pointer (which would segfault on discrete GPUs).
void batch_grid_update_cuda(
    torch::Tensor grid_flat,
    torch::Tensor parts_flat,
    torch::Tensor cell_offsets,
    torch::Tensor grid_idxs,
    torch::Tensor y_starts,
    torch::Tensor x_starts,
    torch::Tensor part_widths,
    torch::Tensor part_offsets,
    int n_placements, int total_cells, int H, int W
);

void batch_phase6_fused_cuda(
    torch::Tensor grid_flat,
    torch::Tensor parts_flat,
    torch::Tensor cell_offsets,
    torch::Tensor row_offsets,
    torch::Tensor grid_idxs,
    torch::Tensor y_starts,
    torch::Tensor part_heights,
    torch::Tensor part_widths,
    torch::Tensor part_offsets,
    torch::Tensor row_vacancy_out,
    int n_placements, int total_cells, int total_rows, int H, int W
);

void batch_grid_update(
    torch::Tensor grid_flat,
    torch::Tensor parts_flat,
    torch::Tensor cell_offsets,
    torch::Tensor grid_idxs,
    torch::Tensor y_starts,
    torch::Tensor x_starts,
    torch::Tensor part_widths,
    torch::Tensor part_offsets,
    int n_placements, int total_cells, int H, int W
) {
    batch_grid_update_cuda(grid_flat, parts_flat, cell_offsets,
                           grid_idxs, y_starts, x_starts,
                           part_widths, part_offsets,
                           n_placements, total_cells, H, W);
}

void batch_phase6_fused(
    torch::Tensor grid_flat,
    torch::Tensor parts_flat,
    torch::Tensor cell_offsets,
    torch::Tensor row_offsets,
    torch::Tensor grid_idxs,
    torch::Tensor y_starts,
    torch::Tensor part_heights,
    torch::Tensor part_widths,
    torch::Tensor part_offsets,
    torch::Tensor row_vacancy_out,
    int n_placements, int total_cells, int total_rows, int H, int W
) {
    batch_phase6_fused_cuda(
        grid_flat, parts_flat, cell_offsets, row_offsets,
        grid_idxs, y_starts, part_heights, part_widths, part_offsets,
        row_vacancy_out, n_placements, total_cells, total_rows, H, W
    );
}
"""

# ── CUDA kernel ────────────────────────────────────────────────────────────────
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void _batch_grid_update_kernel(
    float* __restrict__ grid_flat,
    const float* __restrict__ parts_flat,
    const int* __restrict__ cell_offsets,   // prefix sum length n_placements+1
    const int* __restrict__ grid_idxs,
    const int* __restrict__ y_starts,
    const int* __restrict__ x_starts,
    const int* __restrict__ part_widths,
    const int* __restrict__ part_offsets,   // offset into parts_flat per placement
    int n_placements, int H, int W
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total_cells = cell_offsets[n_placements];
    if (tid >= total_cells) return;

    // Binary search: find which placement owns this thread
    int lo = 0, hi = n_placements - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cell_offsets[mid + 1] <= tid) lo = mid + 1;
        else hi = mid;
    }
    int p = lo;

    int local_idx = tid - cell_offsets[p];
    int pw        = part_widths[p];
    int row       = local_idx / pw;
    int col       = local_idx % pw;

    int global_row = y_starts[p] + row;
    int x0 = (x_starts != nullptr) ? x_starts[p] : 0;
    int global_col = x0 + col;

    // No atomics needed: each placement maps to a unique (grid_idx, row, col) region
    grid_flat[(long)grid_idxs[p] * H * W + (long)global_row * W + global_col]
        += parts_flat[part_offsets[p] + local_idx];
}

void batch_grid_update_cuda(
    torch::Tensor grid_flat,
    torch::Tensor parts_flat,
    torch::Tensor cell_offsets,
    torch::Tensor grid_idxs,
    torch::Tensor y_starts,
    torch::Tensor x_starts,
    torch::Tensor part_widths,
    torch::Tensor part_offsets,
    int n_placements, int total_cells, int H, int W
) {
    if (total_cells == 0) return;

    const int threads = 256;
    const int blocks  = (total_cells + threads - 1) / threads;

    _batch_grid_update_kernel<<<blocks, threads>>>(
        grid_flat.data_ptr<float>(),
        parts_flat.data_ptr<float>(),
        cell_offsets.data_ptr<int>(),
        grid_idxs.data_ptr<int>(),
        y_starts.data_ptr<int>(),
        x_starts.data_ptr<int>(),
        part_widths.data_ptr<int>(),
        part_offsets.data_ptr<int>(),
        n_placements, H, W
    );
}

__global__ void _phase6_row_vacancy_kernel(
    const float* __restrict__ parts_flat,
    const int* __restrict__ row_offsets,    // prefix sum length n_placements+1
    const int* __restrict__ part_widths,
    const int* __restrict__ part_offsets,
    int* __restrict__ row_vacancy_out,
    int n_placements, int W
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total_rows = row_offsets[n_placements];
    if (tid >= total_rows) return;

    // Binary search: find which placement owns this row
    int lo = 0, hi = n_placements - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (row_offsets[mid + 1] <= tid) lo = mid + 1;
        else hi = mid;
    }
    int p = lo;

    int local_row = tid - row_offsets[p];
    int pw = part_widths[p];
    int row_base = part_offsets[p] + local_row * pw;

    int max_zeros = 0;
    int current_zeros = 0;
    for (int c = 0; c < W; ++c) {
        bool occupied = (c < pw) && (parts_flat[row_base + c] != 0.0f);
        if (!occupied) {
            current_zeros += 1;
            if (current_zeros > max_zeros) max_zeros = current_zeros;
        } else {
            current_zeros = 0;
        }
    }
    row_vacancy_out[tid] = max_zeros;
}

void batch_phase6_fused_cuda(
    torch::Tensor grid_flat,
    torch::Tensor parts_flat,
    torch::Tensor cell_offsets,
    torch::Tensor row_offsets,
    torch::Tensor grid_idxs,
    torch::Tensor y_starts,
    torch::Tensor part_heights,
    torch::Tensor part_widths,
    torch::Tensor part_offsets,
    torch::Tensor row_vacancy_out,
    int n_placements, int total_cells, int total_rows, int H, int W
) {
    if (n_placements == 0) return;

    if (total_cells > 0) {
        const int threads_cells = 256;
        const int blocks_cells  = (total_cells + threads_cells - 1) / threads_cells;
        _batch_grid_update_kernel<<<blocks_cells, threads_cells>>>(
            grid_flat.data_ptr<float>(),
            parts_flat.data_ptr<float>(),
            cell_offsets.data_ptr<int>(),
            grid_idxs.data_ptr<int>(),
            y_starts.data_ptr<int>(),
            nullptr,  // x_starts is always zero in Phase 6 fused path
            part_widths.data_ptr<int>(),
            part_offsets.data_ptr<int>(),
            n_placements, H, W
        );
    }

    if (total_rows > 0) {
        const int threads_rows = 256;
        const int blocks_rows = (total_rows + threads_rows - 1) / threads_rows;
        _phase6_row_vacancy_kernel<<<blocks_rows, threads_rows>>>(
            parts_flat.data_ptr<float>(),
            row_offsets.data_ptr<int>(),
            part_widths.data_ptr<int>(),
            part_offsets.data_ptr<int>(),
            row_vacancy_out.data_ptr<int>(),
            n_placements, W
        );
    }
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        from torch.utils.cpp_extension import load_inline
        # Use a different internal name so sys.modules['cuda_batch_update']
        # keeps pointing at this .py file, not the compiled C++ extension.
        _module = load_inline(
            name='_cuda_batch_update_ext_v2',
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=['batch_grid_update', 'batch_phase6_fused'],
            verbose=False,
        )
    return _module


def batch_grid_update(grid_states, flat_parts_gpu, placements, H, W):
    """
    Apply all placements to grid_states in a single CUDA kernel launch.

    Args:
        grid_states    : (max_total_bins, H, W) float32 CUDA tensor, modified in-place
        flat_parts_gpu : (total_cells,) float32 CUDA tensor — all part rotation matrices
                         concatenated in the order built by WaveBatchEvaluator.__init__
        placements     : list of (grid_state_idx, y_start, x_start, part_flat_offset,
                                  part_h, part_w)
        H, W           : grid dimensions
    """
    if not placements:
        return

    n = len(placements)
    grid_idxs_list    = []
    y_starts_list     = []
    x_starts_list     = []
    part_offsets_list = []
    part_widths_list  = []
    cell_offsets_list = [0]

    for grid_idx, y_start, x_start, part_offset, part_h, part_w in placements:
        grid_idxs_list.append(grid_idx)
        y_starts_list.append(y_start)
        x_starts_list.append(x_start)
        part_offsets_list.append(part_offset)
        part_widths_list.append(part_w)
        cell_offsets_list.append(cell_offsets_list[-1] + part_h * part_w)

    device = grid_states.device

    grid_idxs_t    = torch.tensor(grid_idxs_list,    dtype=torch.int32, device=device)
    y_starts_t     = torch.tensor(y_starts_list,     dtype=torch.int32, device=device)
    x_starts_t     = torch.tensor(x_starts_list,     dtype=torch.int32, device=device)
    part_offsets_t = torch.tensor(part_offsets_list, dtype=torch.int32, device=device)
    part_widths_t  = torch.tensor(part_widths_list,  dtype=torch.int32, device=device)
    cell_offsets_t = torch.tensor(cell_offsets_list, dtype=torch.int32, device=device)

    # grid_states is (max_total_bins, H, W) C-contiguous — flat view is safe
    grid_flat = grid_states.view(-1)

    total_cells = cell_offsets_list[-1]  # computed on CPU, never dereferences device ptr

    mod = _get_module()
    mod.batch_grid_update(
        grid_flat, flat_parts_gpu,
        cell_offsets_t, grid_idxs_t, y_starts_t, x_starts_t,
        part_widths_t, part_offsets_t,
        n, total_cells, H, W
    )


def batch_phase6_fused(grid_states, flat_parts_gpu, placements, H, W):
    """
    Fused Phase 6 helper:
      1) Applies all new-bin first placements to GPU grids.
      2) Computes row-wise vacancy values for each placed row on GPU.

    Args:
        placements: list of (grid_state_idx, y_start, part_flat_offset, part_h, part_w)

    Returns:
        (row_offsets_list, row_vacancy_cpu)
        - row_offsets_list: prefix sum offsets length n_placements+1
        - row_vacancy_cpu : 1D np.int32 with length row_offsets_list[-1]
    """
    if not placements:
        return [0], torch.empty(0, dtype=torch.int32).cpu().numpy()

    n = len(placements)
    grid_idxs_list = []
    y_starts_list = []
    part_offsets_list = []
    part_heights_list = []
    part_widths_list = []
    cell_offsets_list = [0]
    row_offsets_list = [0]

    for grid_idx, y_start, part_offset, part_h, part_w in placements:
        grid_idxs_list.append(grid_idx)
        y_starts_list.append(y_start)
        part_offsets_list.append(part_offset)
        part_heights_list.append(part_h)
        part_widths_list.append(part_w)
        cell_offsets_list.append(cell_offsets_list[-1] + part_h * part_w)
        row_offsets_list.append(row_offsets_list[-1] + part_h)

    device = grid_states.device

    grid_idxs_t = torch.tensor(grid_idxs_list, dtype=torch.int32, device=device)
    y_starts_t = torch.tensor(y_starts_list, dtype=torch.int32, device=device)
    part_offsets_t = torch.tensor(part_offsets_list, dtype=torch.int32, device=device)
    part_heights_t = torch.tensor(part_heights_list, dtype=torch.int32, device=device)
    part_widths_t = torch.tensor(part_widths_list, dtype=torch.int32, device=device)
    cell_offsets_t = torch.tensor(cell_offsets_list, dtype=torch.int32, device=device)
    row_offsets_t = torch.tensor(row_offsets_list, dtype=torch.int32, device=device)
    row_vacancy_t = torch.empty(row_offsets_list[-1], dtype=torch.int32, device=device)

    # Phase 6 always places at x=0, so x_starts is implicit in the fused kernel.
    grid_flat = grid_states.view(-1)
    total_cells = cell_offsets_list[-1]
    total_rows = row_offsets_list[-1]

    mod = _get_module()
    mod.batch_phase6_fused(
        grid_flat, flat_parts_gpu,
        cell_offsets_t, row_offsets_t,
        grid_idxs_t, y_starts_t,
        part_heights_t, part_widths_t, part_offsets_t,
        row_vacancy_t,
        n, total_cells, total_rows, H, W
    )
    return row_offsets_list, row_vacancy_t.cpu().numpy()
