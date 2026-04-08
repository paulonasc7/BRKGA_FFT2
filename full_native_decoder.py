"""
Full native decoder (Phase 1-6) for BRKGA wave-batch fitness evaluation.

This module exposes a C++/pybind evaluator that keeps BRKGA evolution in Python
while moving decoder orchestration into a single native call boundary.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch

_CPP_SRC = r"""
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace py = pybind11;
using torch::indexing::Slice;

struct BinStateNative {
    int bin_idx = 0;
    std::vector<uint8_t> grid;
    std::vector<int32_t> vacancy;
    int grid_state_idx = 0;
    double area = 0.0;
    int enclosure_box_length = 0;
    int min_occupied_row = 0;
    int max_occupied_row = -1;
    double proc_time = 0.0;
    double proc_time_height = 0.0;
    bool grid_fft_valid = false;
    int bin_length = 0;
    int bin_width = 0;
};

struct ContextNative {
    int solution_idx = 0;
    int machine_idx = 0;
    std::vector<int> parts_sequence;
    int current_part_idx = 0;
    int bin_length = 0;
    int bin_width = 0;
    double bin_area = 0.0;
    std::vector<BinStateNative> open_bins;
    int next_grid_idx = 0;
    bool is_done = false;
    bool is_feasible = true;
};

struct CollectedTests {
    std::vector<int64_t> grid_indices;
    std::vector<int32_t> heights;
    std::vector<int32_t> widths;
    std::vector<double> bin_indices;
    std::vector<int32_t> bin_local;
    std::vector<int32_t> rot_global;
    std::vector<int32_t> ctx_local;
    std::vector<double> enclosure_lengths;
    std::vector<double> bin_areas;
    std::vector<double> part_areas;
};

struct FFTBatchResult {
    std::vector<uint8_t> has_result;
    std::vector<int32_t> cols;
    std::vector<int32_t> rows;
    std::vector<double> densities;
    std::vector<double> sc_rows;
    std::vector<double> sc_cols;
    std::vector<uint8_t> sc_valid;
};

template <typename T>
static std::vector<T> vec_from_1d(const py::array_t<T, py::array::c_style | py::array::forcecast>& arr) {
    if (arr.ndim() != 1) {
        throw std::runtime_error("expected 1D array");
    }
    const auto n = static_cast<size_t>(arr.shape(0));
    std::vector<T> out(n);
    const T* ptr = arr.data();
    std::copy(ptr, ptr + n, out.begin());
    return out;
}

template <typename T>
static std::vector<T> vec_from_2d_rowmajor(const py::array_t<T, py::array::c_style | py::array::forcecast>& arr) {
    if (arr.ndim() != 2) {
        throw std::runtime_error("expected 2D array");
    }
    const auto n = static_cast<size_t>(arr.shape(0) * arr.shape(1));
    std::vector<T> out(n);
    const T* ptr = arr.data();
    std::copy(ptr, ptr + n, out.begin());
    return out;
}

static inline bool lex_better(
    double neg_bin, double density, double row, double neg_col,
    double best_neg_bin, double best_density, double best_row, double best_neg_col
) {
    if (neg_bin != best_neg_bin) return neg_bin > best_neg_bin;
    if (density != best_density) return density > best_density;
    if (row != best_row) return row > best_row;
    return neg_col > best_neg_col;
}

void native_batch_grid_update_cuda(
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

void native_select_best_positions_cuda(
    torch::Tensor overlap_batch,
    torch::Tensor part_h,
    torch::Tensor part_w,
    int H,
    int W,
    torch::Tensor out_has,
    torch::Tensor out_row,
    torch::Tensor out_col_start
);

inline void native_batch_grid_update(
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
    native_batch_grid_update_cuda(
        grid_flat, parts_flat, cell_offsets, grid_idxs, y_starts, x_starts,
        part_widths, part_offsets, n_placements, total_cells, H, W
    );
}

inline void native_select_best_positions(
    torch::Tensor overlap_batch,
    torch::Tensor part_h,
    torch::Tensor part_w,
    int H,
    int W,
    torch::Tensor out_has,
    torch::Tensor out_row,
    torch::Tensor out_col_start
) {
    native_select_best_positions_cuda(
        overlap_batch, part_h, part_w, H, W, out_has, out_row, out_col_start
    );
}

class NativeFullDecoder {
public:
    NativeFullDecoder(py::dict packed, int nb_parts, int nb_machines, const std::string& device)
        : nb_parts_(nb_parts),
          nb_machines_(nb_machines),
          device_(torch::Device(device)) {
        if (packed.contains("use_fast_selector")) {
            use_fast_selector_ = packed["use_fast_selector"].cast<bool>();
        }
        if (packed.contains("use_cuda_selector_kernel")) {
            use_cuda_selector_kernel_ = packed["use_cuda_selector_kernel"].cast<bool>();
        }
        if (packed.contains("selector_dual_check")) {
            selector_dual_check_ = packed["selector_dual_check"].cast<bool>();
        }
        if (packed.contains("vram_total_bytes")) {
            vram_total_bytes_ = packed["vram_total_bytes"].cast<int64_t>();
        }
        thresholds_ = vec_from_1d<double>(
            packed["thresholds"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>()
        );
        instance_parts_idx_ = vec_from_1d<int32_t>(
            packed["instance_parts_idx"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>()
        );
        if (static_cast<int>(instance_parts_idx_.size()) != nb_parts_) {
            throw std::runtime_error("instance_parts_idx length must match nb_parts");
        }

        machine_bin_length_ = vec_from_1d<int32_t>(
            packed["machine_bin_length"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>()
        );
        machine_bin_width_ = vec_from_1d<int32_t>(
            packed["machine_bin_width"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>()
        );
        machine_bin_area_ = vec_from_1d<double>(
            packed["machine_bin_area"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>()
        );
        machine_setup_time_ = vec_from_1d<double>(
            packed["machine_setup_time"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>()
        );
        if (static_cast<int>(machine_bin_length_.size()) != nb_machines_ ||
            static_cast<int>(machine_bin_width_.size()) != nb_machines_ ||
            static_cast<int>(machine_bin_area_.size()) != nb_machines_ ||
            static_cast<int>(machine_setup_time_.size()) != nb_machines_) {
            throw std::runtime_error("machine metadata size mismatch");
        }

        part_area_ = vec_from_1d<double>(
            packed["part_area"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>()
        );
        part_nrot_ = vec_from_1d<int32_t>(
            packed["part_nrot"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>()
        );
        part_best_rot_ = vec_from_1d<int32_t>(
            packed["part_best_rot"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>()
        );
        part_rot_offsets_ = vec_from_1d<int32_t>(
            packed["part_rot_offsets"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>()
        );

        n_part_types_ = static_cast<int>(part_area_.size());
        if (static_cast<int>(part_nrot_.size()) != n_part_types_ ||
            static_cast<int>(part_best_rot_.size()) != n_part_types_ ||
            static_cast<int>(part_rot_offsets_.size()) != n_part_types_ + 1) {
            throw std::runtime_error("part metadata size mismatch");
        }

        rot_h_ = vec_from_1d<int32_t>(
            packed["rot_h"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>()
        );
        rot_w_ = vec_from_1d<int32_t>(
            packed["rot_w"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>()
        );
        rot_density_offsets_ = vec_from_1d<int32_t>(
            packed["rot_density_offsets"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>()
        );
        density_flat_ = vec_from_1d<int32_t>(
            packed["density_flat"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>()
        );
        rot_matrix_offsets_ = vec_from_1d<int32_t>(
            packed["rot_matrix_offsets"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>()
        );
        rot_matrix_flat_u8_ = vec_from_1d<uint8_t>(
            packed["rot_matrix_flat_u8"].cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>()
        );

        n_rot_total_ = static_cast<int>(rot_h_.size());
        if (static_cast<int>(rot_w_.size()) != n_rot_total_ ||
            static_cast<int>(rot_density_offsets_.size()) != n_rot_total_ + 1 ||
            static_cast<int>(rot_matrix_offsets_.size()) != n_rot_total_ + 1) {
            throw std::runtime_error("rotation metadata size mismatch");
        }

        machine_proc_time_ = vec_from_2d_rowmajor<double>(
            packed["machine_proc_time"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>()
        );
        machine_proc_time_height_ = vec_from_2d_rowmajor<double>(
            packed["machine_proc_time_height"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>()
        );
        if (static_cast<int>(machine_proc_time_.size()) != nb_machines_ * n_part_types_ ||
            static_cast<int>(machine_proc_time_height_.size()) != nb_machines_ * n_part_types_) {
            throw std::runtime_error("machine proc-time matrix size mismatch");
        }

        rot_flat_offsets_ = vec_from_1d<int32_t>(
            packed["rot_flat_offsets"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>()
        );
        if (static_cast<int>(rot_flat_offsets_.size()) != n_rot_total_ + 1) {
            throw std::runtime_error("rot_flat_offsets size mismatch");
        }
        flat_parts_gpu_ = packed["flat_parts_gpu"].cast<torch::Tensor>();
        if (!flat_parts_gpu_.defined() || !flat_parts_gpu_.is_cuda() || flat_parts_gpu_.numel() == 0) {
            throw std::runtime_error("flat_parts_gpu must be a non-empty CUDA tensor");
        }

        machine_ffts_dense_.assign(nb_machines_, {});
        py::list machine_fft_dense = packed["machine_fft_dense"].cast<py::list>();
        if (static_cast<int>(py::len(machine_fft_dense)) != nb_machines_) {
            throw std::runtime_error("machine_fft_dense machine count mismatch");
        }
        for (int m = 0; m < nb_machines_; ++m) {
            auto dense = machine_fft_dense[m].cast<torch::Tensor>();
            if (dense.dim() != 3 || dense.size(0) != n_rot_total_) {
                throw std::runtime_error("machine_fft_dense tensor shape mismatch");
            }
            machine_ffts_dense_[m] = dense.contiguous();
        }
    }

    py::array_t<double> evaluate_batch(
        py::array_t<float, py::array::c_style | py::array::forcecast> chromosomes
    ) {
        c10::InferenceMode guard(true);
        if (chromosomes.ndim() != 2) {
            throw std::runtime_error("chromosomes must be a 2D float32 array");
        }
        const int num_solutions = static_cast<int>(chromosomes.shape(0));
        if (chromosomes.shape(1) != static_cast<ssize_t>(2 * nb_parts_)) {
            throw std::runtime_error("chromosomes second dimension must be 2 * nb_parts");
        }

        const float* chrom_ptr = chromosomes.data();
        std::vector<double> final_makespans(static_cast<size_t>(num_solutions), 0.0);

        for (int m = 0; m < nb_machines_; ++m) {
            std::vector<double> machine_makespan = process_machine_batch(chrom_ptr, num_solutions, m);
            for (int s = 0; s < num_solutions; ++s) {
                if (m == 0) {
                    final_makespans[s] = machine_makespan[s];
                } else {
                    final_makespans[s] = std::max(final_makespans[s], machine_makespan[s]);
                }
            }
        }

        py::array_t<double> out(num_solutions);
        auto out_u = out.mutable_unchecked<1>();
        for (int i = 0; i < num_solutions; ++i) {
            out_u(i) = final_makespans[i];
        }
        return out;
    }

private:
    int nb_parts_ = 0;
    int nb_machines_ = 0;
    int n_part_types_ = 0;
    int n_rot_total_ = 0;
    torch::Device device_;
    bool use_fast_selector_ = true;
    bool use_cuda_selector_kernel_ = false;
    bool selector_dual_check_ = false;
    int64_t vram_total_bytes_ = 0;  // total VRAM; used to cap max_bins_per_sol

    std::vector<double> thresholds_;
    std::vector<int32_t> instance_parts_idx_;

    std::vector<int32_t> machine_bin_length_;
    std::vector<int32_t> machine_bin_width_;
    std::vector<double> machine_bin_area_;
    std::vector<double> machine_setup_time_;

    std::vector<double> part_area_;
    std::vector<int32_t> part_nrot_;
    std::vector<int32_t> part_best_rot_;
    std::vector<int32_t> part_rot_offsets_;

    std::vector<int32_t> rot_h_;
    std::vector<int32_t> rot_w_;
    std::vector<int32_t> rot_density_offsets_;
    std::vector<int32_t> density_flat_;
    std::vector<int32_t> rot_matrix_offsets_;
    std::vector<uint8_t> rot_matrix_flat_u8_;

    std::vector<double> machine_proc_time_;
    std::vector<double> machine_proc_time_height_;
    std::vector<torch::Tensor> machine_ffts_dense_;
    std::vector<int32_t> rot_flat_offsets_;
    torch::Tensor flat_parts_gpu_;

    struct GpuPlacement {
        int grid_idx;
        int y_start;
        int x_start;
        int flat_offset;
        int ph;
        int pw;
    };

    CollectedTests scratch_p1_;
    CollectedTests scratch_p2_;
    FFTBatchResult scratch_fft_p1_;
    FFTBatchResult scratch_fft_p2_;

    std::vector<int> scratch_active_;
    std::vector<int> scratch_ctx_global_;
    std::vector<int> scratch_part_idx_local_;
    std::vector<int64_t> scratch_invalid_grid_indices_;
    std::vector<BinStateNative*> scratch_invalid_bins_;
    std::vector<int32_t> scratch_ctx_first_valid_bin_;
    std::vector<uint8_t> scratch_ctx_p1_hit_;

    std::vector<int32_t> scratch_test_ctx_local_;
    std::vector<int32_t> scratch_test_bin_local_;
    std::vector<int32_t> scratch_test_rot_global_;
    std::vector<uint8_t> scratch_placement_has_;
    std::vector<int32_t> scratch_placement_cols_;
    std::vector<int32_t> scratch_placement_rows_;
    std::vector<double> scratch_sc_bin_indices_;
    std::vector<double> scratch_sc_densities_;
    std::vector<double> scratch_sc_rows_;
    std::vector<double> scratch_sc_cols_;
    std::vector<uint8_t> scratch_sc_valid_;

    std::vector<int32_t> scratch_best_ti_;
    std::vector<uint8_t> scratch_has_best_;
    std::vector<double> scratch_best_neg_bin_;
    std::vector<double> scratch_best_density_;
    std::vector<double> scratch_best_row_;
    std::vector<double> scratch_best_neg_col_;

    std::vector<int32_t> scratch_place_ctx_local_;
    std::vector<int32_t> scratch_place_ti_;
    std::vector<int32_t> scratch_place_rows_;
    std::vector<int32_t> scratch_place_cols_;
    std::vector<int32_t> scratch_newbin_ctx_local_;

    std::vector<GpuPlacement> scratch_gpu_updates_;
    std::vector<GpuPlacement> scratch_phase6_gpu_updates_;
    std::vector<int64_t> scratch_phase6_grid_indices_;

    std::vector<int32_t> scratch_cell_offsets_;
    std::vector<int32_t> scratch_grid_idxs_;
    std::vector<int32_t> scratch_y_starts_;
    std::vector<int32_t> scratch_x_starts_;
    std::vector<int32_t> scratch_part_widths_;
    std::vector<int32_t> scratch_part_offsets_;
    torch::Tensor ws_grid_idx_long_;
    torch::Tensor ws_rot_idx_long_;
    torch::Tensor ws_h_long_;
    torch::Tensor ws_w_long_;
    torch::Tensor ws_wave_idx_long_;
    torch::Tensor ws_cell_offsets_i32_;
    torch::Tensor ws_grid_idxs_i32_;
    torch::Tensor ws_y_starts_i32_;
    torch::Tensor ws_x_starts_i32_;
    torch::Tensor ws_part_widths_i32_;
    torch::Tensor ws_part_offsets_i32_;
    // Output buffers for the CUDA selector kernel — pre-allocated once and
    // reused every chunk so we avoid repeated GPU alloc + fill_(0) per chunk.
    torch::Tensor ws_sel_has_i32_;
    torch::Tensor ws_sel_row_i32_;
    torch::Tensor ws_sel_col_i32_;
    // CPU-side pinned backing tensors for CPU→GPU transfers.
    // Pinned (page-locked) memory allows direct DMA without a staging copy.
    // One per GPU workspace so buffers are never overwritten before consumed.
    torch::Tensor ws_cpu_grid_idx_;
    torch::Tensor ws_cpu_rot_idx_;
    torch::Tensor ws_cpu_h_;
    torch::Tensor ws_cpu_w_;
    torch::Tensor ws_cpu_wave_idx_;
    torch::Tensor ws_cpu_cell_offsets_;
    torch::Tensor ws_cpu_grid_idxs_;
    torch::Tensor ws_cpu_y_starts_;
    torch::Tensor ws_cpu_x_starts_;
    torch::Tensor ws_cpu_part_widths_;
    torch::Tensor ws_cpu_part_offsets_;

    inline int rot_global(int part_idx, int rot) const {
        return part_rot_offsets_[static_cast<size_t>(part_idx)] + rot;
    }

    inline double proc_time(int machine_idx, int part_idx) const {
        return machine_proc_time_[static_cast<size_t>(machine_idx) * static_cast<size_t>(n_part_types_) +
                                  static_cast<size_t>(part_idx)];
    }

    inline double proc_time_height(int machine_idx, int part_idx) const {
        return machine_proc_time_height_[static_cast<size_t>(machine_idx) * static_cast<size_t>(n_part_types_) +
                                         static_cast<size_t>(part_idx)];
    }

    torch::Tensor ensure_workspace_long(torch::Tensor& ws, int64_t n) {
        auto opts = torch::TensorOptions().dtype(torch::kLong).device(device_);
        if (!ws.defined() || ws.scalar_type() != torch::kLong || ws.device() != device_ || ws.numel() < n) {
            ws = torch::empty({std::max<int64_t>(n, 1)}, opts);
        }
        return ws.narrow(0, 0, n);
    }

    torch::Tensor ensure_workspace_i32(torch::Tensor& ws, int64_t n) {
        auto opts = torch::TensorOptions().dtype(torch::kInt32).device(device_);
        if (!ws.defined() || ws.scalar_type() != torch::kInt32 || ws.device() != device_ || ws.numel() < n) {
            ws = torch::empty({std::max<int64_t>(n, 1)}, opts);
        }
        return ws.narrow(0, 0, n);
    }

    torch::Tensor ensure_cpu_pinned_long(torch::Tensor& ws, int64_t n) {
        auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU).pinned_memory(true);
        if (!ws.defined() || ws.numel() < n) {
            ws = torch::empty({std::max<int64_t>(n, 64)}, opts);
        }
        return ws.narrow(0, 0, n);
    }

    torch::Tensor ensure_cpu_pinned_i32(torch::Tensor& ws, int64_t n) {
        auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
        if (!ws.defined() || ws.numel() < n) {
            ws = torch::empty({std::max<int64_t>(n, 64)}, opts);
        }
        return ws.narrow(0, 0, n);
    }

    torch::Tensor load_workspace_long_from_i64(
        torch::Tensor& ws,
        torch::Tensor& cpu_pinned_ws,
        const std::vector<int64_t>& src,
        int64_t n
    ) {
        auto out = ensure_workspace_long(ws, n);
        if (n > 0) {
            auto pinned = ensure_cpu_pinned_long(cpu_pinned_ws, n);
            std::memcpy(pinned.data_ptr<int64_t>(), src.data(), static_cast<size_t>(n) * sizeof(int64_t));
            out.copy_(pinned, /*non_blocking=*/true);
        }
        return out;
    }

    torch::Tensor load_workspace_long_from_i32(
        torch::Tensor& ws,
        torch::Tensor& cpu_pinned_ws,
        const std::vector<int32_t>& src,
        int64_t n
    ) {
        auto out = ensure_workspace_long(ws, n);
        if (n > 0) {
            auto pinned = ensure_cpu_pinned_long(cpu_pinned_ws, n);
            int64_t* dst = pinned.data_ptr<int64_t>();
            for (int64_t i = 0; i < n; ++i) {
                dst[i] = static_cast<int64_t>(src[static_cast<size_t>(i)]);
            }
            out.copy_(pinned, /*non_blocking=*/true);
        }
        return out;
    }

    torch::Tensor load_workspace_i32_from_i32(
        torch::Tensor& ws,
        torch::Tensor& cpu_pinned_ws,
        const std::vector<int32_t>& src,
        int64_t n
    ) {
        auto out = ensure_workspace_i32(ws, n);
        if (n > 0) {
            auto pinned = ensure_cpu_pinned_i32(cpu_pinned_ws, n);
            std::memcpy(pinned.data_ptr<int32_t>(), src.data(), static_cast<size_t>(n) * sizeof(int32_t));
            out.copy_(pinned, /*non_blocking=*/true);
        }
        return out;
    }

    inline bool belongs_to_machine(float mv, int machine_idx) const {
        if (nb_machines_ <= 1) {
            return true;
        }
        if (machine_idx == 0) {
            return mv <= thresholds_[0];
        }
        if (machine_idx == nb_machines_ - 1) {
            return mv > thresholds_[static_cast<size_t>(machine_idx - 1)];
        }
        return (mv > thresholds_[static_cast<size_t>(machine_idx - 1)]) &&
               (mv <= thresholds_[static_cast<size_t>(machine_idx)]);
    }

    static bool check_vacancy_fit_simple_cpp(
        const std::vector<int32_t>& vacancy,
        const int32_t* density,
        int density_len
    ) {
        const int n = static_cast<int>(vacancy.size());
        if (density_len > n) {
            return false;
        }

        // Optional kill-switch for A/B benchmarks and debugging.
        const bool simd_enabled = []() {
            const char* env = std::getenv("ABRKGA_NATIVE_VACANCY_SIMD");
            if (env == nullptr) {
                return true;
            }
            const std::string s(env);
            return !(s == "0" || s == "false" || s == "False");
        }();

        if (!simd_enabled) {
            return check_vacancy_fit_simple_scalar_cpp(vacancy, density, density_len);
        }

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
        static int simd_level = -1;  // 0=scalar, 1=avx2, 2=avx512
        if (simd_level < 0) {
            simd_level = 0;
            if (__builtin_cpu_supports("avx512f")) {
                simd_level = 2;
            } else if (__builtin_cpu_supports("avx2")) {
                simd_level = 1;
            }
        }

        if (simd_level == 2 && density_len >= 16) {
            return check_vacancy_fit_simple_avx512_cpp(vacancy, density, density_len);
        }
        if (simd_level >= 1 && density_len >= 8) {
            return check_vacancy_fit_simple_avx2_cpp(vacancy, density, density_len);
        }
#endif

        return check_vacancy_fit_simple_scalar_cpp(vacancy, density, density_len);
    }

    static bool check_vacancy_fit_simple_scalar_cpp(
        const std::vector<int32_t>& vacancy,
        const int32_t* density,
        int density_len
    ) {
        const int n = static_cast<int>(vacancy.size());
        if (density_len > n) {
            return false;
        }
        const int max_start = n - density_len;
        for (int start = 0; start <= max_start; ++start) {
            bool fits = true;
            for (int i = 0; i < density_len; ++i) {
                if (vacancy[static_cast<size_t>(start + i)] < density[i]) {
                    fits = false;
                    break;
                }
            }
            if (fits) {
                return true;
            }
        }
        return false;
    }

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    __attribute__((target("avx2")))
    static inline bool window_fits_avx2_cpp(
        const int32_t* vacancy_ptr,
        const int32_t* density_ptr,
        int density_len
    ) {
        int i = 0;
        for (; i + 8 <= density_len; i += 8) {
            const auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vacancy_ptr + i));
            const auto d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(density_ptr + i));
            const auto bad = _mm256_cmpgt_epi32(d, v);  // bad when density > vacancy
            if (_mm256_movemask_ps(_mm256_castsi256_ps(bad)) != 0) {
                return false;
            }
        }
        for (; i < density_len; ++i) {
            if (vacancy_ptr[i] < density_ptr[i]) {
                return false;
            }
        }
        return true;
    }

    __attribute__((target("avx2")))
    static bool check_vacancy_fit_simple_avx2_cpp(
        const std::vector<int32_t>& vacancy,
        const int32_t* density,
        int density_len
    ) {
        const int n = static_cast<int>(vacancy.size());
        const int max_start = n - density_len;
        const int32_t* vacancy_ptr = vacancy.data();
        for (int start = 0; start <= max_start; ++start) {
            if (window_fits_avx2_cpp(vacancy_ptr + start, density, density_len)) {
                return true;
            }
        }
        return false;
    }

    __attribute__((target("avx512f")))
    static inline bool window_fits_avx512_cpp(
        const int32_t* vacancy_ptr,
        const int32_t* density_ptr,
        int density_len
    ) {
        int i = 0;
        for (; i + 16 <= density_len; i += 16) {
            const auto v = _mm512_loadu_si512(reinterpret_cast<const void*>(vacancy_ptr + i));
            const auto d = _mm512_loadu_si512(reinterpret_cast<const void*>(density_ptr + i));
            const auto bad = _mm512_cmpgt_epi32_mask(d, v);  // bad when density > vacancy
            if (bad != 0) {
                return false;
            }
        }
        for (; i < density_len; ++i) {
            if (vacancy_ptr[i] < density_ptr[i]) {
                return false;
            }
        }
        return true;
    }

    __attribute__((target("avx512f")))
    static bool check_vacancy_fit_simple_avx512_cpp(
        const std::vector<int32_t>& vacancy,
        const int32_t* density,
        int density_len
    ) {
        const int n = static_cast<int>(vacancy.size());
        const int max_start = n - density_len;
        const int32_t* vacancy_ptr = vacancy.data();
        for (int start = 0; start <= max_start; ++start) {
            if (window_fits_avx512_cpp(vacancy_ptr + start, density, density_len)) {
                return true;
            }
        }
        return false;
    }
#endif

    static void update_vacancy_rows_cpp(
        std::vector<int32_t>& vacancy_vector,
        const std::vector<uint8_t>& grid,
        int y_start,
        int num_rows,
        int width
    ) {
        for (int i = 0; i < num_rows; ++i) {
            int max_zeros = 0;
            int current_zeros = 0;
            const int row_idx = y_start + i;
            const int row_off = row_idx * width;
            for (int j = 0; j < width; ++j) {
                if (grid[static_cast<size_t>(row_off + j)] == 0) {
                    current_zeros += 1;
                    if (current_zeros > max_zeros) {
                        max_zeros = current_zeros;
                    }
                } else {
                    current_zeros = 0;
                }
            }
            vacancy_vector[static_cast<size_t>(row_idx)] = max_zeros;
        }
    }

    void decode_sequences_for_machine(
        const float* chrom_row,
        int machine_idx,
        std::vector<int>& out_seq
    ) const {
        std::vector<std::pair<float, int>> pairs;
        pairs.reserve(static_cast<size_t>(nb_parts_));
        for (int j = 0; j < nb_parts_; ++j) {
            const float mv = chrom_row[nb_parts_ + j];
            if (!belongs_to_machine(mv, machine_idx)) {
                continue;
            }
            const float sv = chrom_row[j];
            const int part_idx = instance_parts_idx_[static_cast<size_t>(j)];
            pairs.emplace_back(sv, part_idx);
        }

        std::sort(pairs.begin(), pairs.end(),
                  [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                      return a.first < b.first;
                  });

        out_seq.clear();
        out_seq.reserve(pairs.size());
        for (const auto& p : pairs) {
            out_seq.push_back(p.second);
        }
    }

    void batch_fft_all_tests(
        int n_tests,
        const std::vector<int64_t>& test_grid_indices,
        const std::vector<int32_t>& test_rot_global,
        const std::vector<int32_t>& test_heights,
        const std::vector<int32_t>& test_widths,
        const std::vector<double>& test_bin_indices,
        const std::vector<double>& test_enclosure_lengths,
        const std::vector<double>& test_bin_areas,
        const std::vector<double>& test_part_areas,
        int machine_idx,
        torch::Tensor& grid_ffts,
        int H,
        int W,
        const torch::Tensor& row_idx,
        const torch::Tensor& col_idx,
        const torch::Tensor& row_idx_f,
        const torch::Tensor& col_idx_f,
        const torch::Tensor& neg_inf,
        FFTBatchResult& out
    ) {
        out.has_result.assign(static_cast<size_t>(n_tests), 0);
        out.cols.assign(static_cast<size_t>(n_tests), 0);
        out.rows.assign(static_cast<size_t>(n_tests), 0);
        out.densities.assign(static_cast<size_t>(n_tests), 0.0);
        out.sc_rows.assign(static_cast<size_t>(n_tests), 0.0);
        out.sc_cols.assign(static_cast<size_t>(n_tests), 0.0);
        out.sc_valid.assign(static_cast<size_t>(n_tests), 0);

        if (n_tests <= 0) {
            return;
        }

        auto all_grid_idx_t = load_workspace_long_from_i64(
            ws_grid_idx_long_, ws_cpu_grid_idx_, test_grid_indices, static_cast<int64_t>(n_tests)
        );
        auto all_rot_idx_t = load_workspace_long_from_i32(
            ws_rot_idx_long_, ws_cpu_rot_idx_, test_rot_global, static_cast<int64_t>(n_tests)
        );
        auto all_h_t = load_workspace_long_from_i32(
            ws_h_long_, ws_cpu_h_, test_heights, static_cast<int64_t>(n_tests)
        );
        auto all_w_t = load_workspace_long_from_i32(
            ws_w_long_, ws_cpu_w_, test_widths, static_cast<int64_t>(n_tests)
        );

        constexpr int CHUNK_SIZE = 750;
        for (int chunk_start = 0; chunk_start < n_tests; chunk_start += CHUNK_SIZE) {
            const int chunk_end = std::min(chunk_start + CHUNK_SIZE, n_tests);
            const int chunk_n = chunk_end - chunk_start;

            auto grid_idx_t = all_grid_idx_t.narrow(0, chunk_start, chunk_n);
            auto rot_idx_t = all_rot_idx_t.narrow(0, chunk_start, chunk_n);
            auto part_h_t = all_h_t.narrow(0, chunk_start, chunk_n);
            auto part_w_t = all_w_t.narrow(0, chunk_start, chunk_n);

            auto batch_grid_ffts = grid_ffts.index_select(0, grid_idx_t);
            auto batch_part_ffts = machine_ffts_dense_[static_cast<size_t>(machine_idx)].index_select(0, rot_idx_t);
            auto overlap_batch = torch::fft::irfft2(batch_grid_ffts * batch_part_ffts, {H, W});

            // Compute the valid-placement boolean mask lazily.  When the CUDA
            // selector kernel is active it performs round/eq/ge masking
            // internally, so computing valid_zeros here would be dead work.
            // Calling compute_valid_zeros() only in the branches that need it
            // eliminates 6 element-wise GPU kernels per chunk in the default path.
            auto compute_valid_zeros = [&]() -> torch::Tensor {
                auto zero_mask  = overlap_batch.round().eq(0);
                auto valid_row_ = row_idx.ge((part_h_t - 1).view({-1, 1, 1}));
                auto valid_col_ = col_idx.ge((part_w_t - 1).view({-1, 1, 1}));
                return zero_mask.logical_and(valid_row_).logical_and(valid_col_);
            };

            if (use_fast_selector_) {
                auto compute_fast_selector_cpu_i32 = [&]() {
                    auto valid_zeros   = compute_valid_zeros();
                    auto valid_any_row = valid_zeros.any(2);
                    auto has_valid = valid_any_row.any(1);
                    auto rev_first = valid_any_row.flip(1).to(torch::kInt64).argmax(1);
                    auto best_row = (H - 1) - rev_first;
                    auto selected_row = valid_zeros.gather(
                        1,
                        best_row.view({-1, 1, 1}).to(torch::kLong).expand({-1, 1, W})
                    ).squeeze(1);
                    auto best_col = selected_row.to(torch::kInt64).argmax(1);
                    auto best_col_start = best_col - (part_w_t - 1).to(torch::kInt64);
                    return std::make_tuple(
                        has_valid.to(torch::kInt32).to(torch::kCPU),
                        best_row.to(torch::kInt32).to(torch::kCPU),
                        best_col_start.to(torch::kInt32).to(torch::kCPU)
                    );
                };

                torch::Tensor has_cpu_i32;
                torch::Tensor row_cpu_i32;
                torch::Tensor col_cpu_i32;

                if (use_cuda_selector_kernel_ && overlap_batch.is_cuda()) {
                    auto out_has_t = ensure_workspace_i32(ws_sel_has_i32_, chunk_n);
                    auto out_row_t = ensure_workspace_i32(ws_sel_row_i32_, chunk_n);
                    auto out_col_t = ensure_workspace_i32(ws_sel_col_i32_, chunk_n);
                    auto overlap_contig = overlap_batch.contiguous();
                    auto part_h_contig = part_h_t.contiguous();
                    auto part_w_contig = part_w_t.contiguous();
                    native_select_best_positions(
                        overlap_contig, part_h_contig, part_w_contig, H, W,
                        out_has_t, out_row_t, out_col_t
                    );
                    has_cpu_i32 = out_has_t.to(torch::kCPU);
                    row_cpu_i32 = out_row_t.to(torch::kCPU);
                    col_cpu_i32 = out_col_t.to(torch::kCPU);

                    if (selector_dual_check_) {
                        auto ref = compute_fast_selector_cpu_i32();
                        auto ref_has = std::get<0>(ref);
                        auto ref_row = std::get<1>(ref);
                        auto ref_col = std::get<2>(ref);
                        auto got_has = has_cpu_i32.accessor<int32_t, 1>();
                        auto got_row = row_cpu_i32.accessor<int32_t, 1>();
                        auto got_col = col_cpu_i32.accessor<int32_t, 1>();
                        auto chk_has = ref_has.accessor<int32_t, 1>();
                        auto chk_row = ref_row.accessor<int32_t, 1>();
                        auto chk_col = ref_col.accessor<int32_t, 1>();
                        for (int i = 0; i < chunk_n; ++i) {
                            if (got_has[i] != chk_has[i]) {
                                throw std::runtime_error("selector dual-check mismatch");
                            }
                            if (got_has[i] != 0 &&
                                (got_row[i] != chk_row[i] || got_col[i] != chk_col[i])) {
                                throw std::runtime_error("selector dual-check mismatch");
                            }
                        }
                    }
                } else {
                    auto ref = compute_fast_selector_cpu_i32();
                    has_cpu_i32 = std::get<0>(ref);
                    row_cpu_i32 = std::get<1>(ref);
                    col_cpu_i32 = std::get<2>(ref);
                }

                auto has_acc = has_cpu_i32.accessor<int32_t, 1>();
                auto row_acc = row_cpu_i32.accessor<int32_t, 1>();
                auto col_acc = col_cpu_i32.accessor<int32_t, 1>();
                for (int i = 0; i < chunk_n; ++i) {
                    const int gi = chunk_start + i;
                    if (has_acc[i] != 0) {
                        out.has_result[static_cast<size_t>(gi)] = 1;
                        out.cols[static_cast<size_t>(gi)] = static_cast<int32_t>(col_acc[i]);
                        out.rows[static_cast<size_t>(gi)] = static_cast<int32_t>(row_acc[i]);
                        out.sc_cols[static_cast<size_t>(gi)] = static_cast<double>(col_acc[i]);
                        out.sc_rows[static_cast<size_t>(gi)] = static_cast<double>(row_acc[i]);
                        out.sc_valid[static_cast<size_t>(gi)] = 1;
                    }
                }
            } else {
                auto valid_zeros = compute_valid_zeros();
                auto score = torch::where(
                    valid_zeros,
                    row_idx_f * static_cast<float>(W + 1) - col_idx_f,
                    neg_inf
                );
                auto score2 = score.view({chunk_n, -1});
                auto max_pair = score2.max(1);
                auto max_scores = std::get<0>(max_pair);
                auto best_flat_idx = std::get<1>(max_pair);
                auto best_row = best_flat_idx / W;
                auto best_col = best_flat_idx % W;
                auto has_valid = max_scores.gt(-1e8);
                auto res = torch::stack(
                    {
                        has_valid.to(torch::kInt64),
                        (best_col - (part_w_t - 1)).to(torch::kInt64),
                        best_row.to(torch::kInt64)
                    },
                    1
                ).to(torch::kCPU);

                auto acc = res.accessor<int64_t, 2>();
                for (int i = 0; i < chunk_n; ++i) {
                    const int gi = chunk_start + i;
                    if (acc[i][0] == 1) {
                        out.has_result[static_cast<size_t>(gi)] = 1;
                        out.cols[static_cast<size_t>(gi)] = static_cast<int32_t>(acc[i][1]);
                        out.rows[static_cast<size_t>(gi)] = static_cast<int32_t>(acc[i][2]);
                        out.sc_cols[static_cast<size_t>(gi)] = static_cast<double>(acc[i][1]);
                        out.sc_rows[static_cast<size_t>(gi)] = static_cast<double>(acc[i][2]);
                        out.sc_valid[static_cast<size_t>(gi)] = 1;
                    }
                }
            }
        }

        for (int i = 0; i < n_tests; ++i) {
            if (!out.sc_valid[static_cast<size_t>(i)]) {
                continue;
            }
            const double row = out.sc_rows[static_cast<size_t>(i)];
            const double h = static_cast<double>(test_heights[static_cast<size_t>(i)]);
            const double y_start = row - h + 1.0;
            const double new_len = std::max(
                test_enclosure_lengths[static_cast<size_t>(i)],
                static_cast<double>(H) - y_start
            );
            out.densities[static_cast<size_t>(i)] =
                (test_bin_areas[static_cast<size_t>(i)] + test_part_areas[static_cast<size_t>(i)]) /
                (new_len * static_cast<double>(W));
        }
    }

    void add_part_to_bin_cpu(
        BinStateNative& bin,
        int x,
        int y,
        int rotg,
        int part_idx,
        int machine_idx,
        int machine_width,
        bool overwrite
    ) {
        const int ph = rot_h_[static_cast<size_t>(rotg)];
        const int pw = rot_w_[static_cast<size_t>(rotg)];
        const int y_start = y - ph + 1;
        const int m_off = rot_matrix_offsets_[static_cast<size_t>(rotg)];
        const uint8_t* mat_ptr = rot_matrix_flat_u8_.data() + m_off;

        for (int rr = 0; rr < ph; ++rr) {
            const int row_off = (y_start + rr) * machine_width;
            const int mat_off = rr * pw;
            for (int cc = 0; cc < pw; ++cc) {
                const int gi = row_off + x + cc;
                const uint8_t pv = mat_ptr[static_cast<size_t>(mat_off + cc)];
                if (overwrite) {
                    bin.grid[static_cast<size_t>(gi)] = pv;
                } else {
                    bin.grid[static_cast<size_t>(gi)] =
                        static_cast<uint8_t>(bin.grid[static_cast<size_t>(gi)] + pv);
                }
            }
        }

        update_vacancy_rows_cpp(bin.vacancy, bin.grid, y_start, ph, machine_width);
        bin.area += part_area_[static_cast<size_t>(part_idx)];
        bin.min_occupied_row = std::min(bin.min_occupied_row, y_start);
        bin.max_occupied_row = std::max(bin.max_occupied_row, y);
        if (overwrite) {
            bin.enclosure_box_length = ph;
            bin.proc_time += proc_time(machine_idx, part_idx);
            bin.proc_time_height = proc_time_height(machine_idx, part_idx);
        } else {
            bin.enclosure_box_length = bin.bin_length - bin.min_occupied_row;
            bin.proc_time += proc_time(machine_idx, part_idx);
            bin.proc_time_height = std::max(bin.proc_time_height, proc_time_height(machine_idx, part_idx));
        }
        bin.grid_fft_valid = false;
    }

    std::vector<double> process_machine_batch(
        const float* chrom_ptr,
        int num_solutions,
        int machine_idx
    ) {
        const int H = machine_bin_length_[static_cast<size_t>(machine_idx)];
        const int W = machine_bin_width_[static_cast<size_t>(machine_idx)];
        const double bin_area = machine_bin_area_[static_cast<size_t>(machine_idx)];
        const int needed_bins = std::max(10, nb_parts_ / 3);
        int max_bins_per_sol = needed_bins;
        // Cap max_bins_per_sol so grid_states + grid_ffts stay within 50% of total VRAM.
        if (vram_total_bytes_ > 0 && num_solutions > 0) {
            const int64_t bytes_per_bin =
                static_cast<int64_t>(H) * W * 4 +           // grid_states: float32
                static_cast<int64_t>(H) * (W / 2 + 1) * 8; // grid_ffts: complex float (2×float32)
            const int64_t budget = vram_total_bytes_ / 2;
            const int64_t denom  = static_cast<int64_t>(num_solutions) * bytes_per_bin;
            if (denom > 0) {
                const int cap = static_cast<int>(budget / denom);
                max_bins_per_sol = std::min(max_bins_per_sol, std::max(1, cap));
            }
        }
        const int max_total_bins = std::max(1, num_solutions * max_bins_per_sol);

        std::vector<ContextNative> contexts(static_cast<size_t>(num_solutions));
        int max_seq_len = 0;
        for (int s = 0; s < num_solutions; ++s) {
            ContextNative ctx;
            ctx.solution_idx = s;
            ctx.machine_idx = machine_idx;
            decode_sequences_for_machine(chrom_ptr + static_cast<size_t>(s) * static_cast<size_t>(2 * nb_parts_),
                                         machine_idx, ctx.parts_sequence);
            ctx.current_part_idx = 0;
            ctx.bin_length = H;
            ctx.bin_width = W;
            ctx.bin_area = bin_area;
            ctx.next_grid_idx = s * max_bins_per_sol;
            ctx.is_done = ctx.parts_sequence.empty();
            ctx.is_feasible = true;
            max_seq_len = std::max(max_seq_len, static_cast<int>(ctx.parts_sequence.size()));
            contexts[static_cast<size_t>(s)] = std::move(ctx);
        }

        auto grid_states = torch::zeros(
            {max_total_bins, H, W},
            torch::TensorOptions().dtype(torch::kFloat32).device(device_)
        );
        auto grid_ffts = torch::zeros(
            {max_total_bins, H, (W / 2) + 1},
            torch::TensorOptions().dtype(torch::kComplexFloat).device(device_)
        );
        auto row_idx = torch::arange(H, torch::TensorOptions().dtype(torch::kLong).device(device_)).view({1, H, 1});
        auto col_idx = torch::arange(W, torch::TensorOptions().dtype(torch::kLong).device(device_)).view({1, 1, W});
        auto row_idx_f = row_idx.to(torch::kFloat);
        auto col_idx_f = col_idx.to(torch::kFloat);
        auto neg_inf = torch::tensor(-1e9f, torch::TensorOptions().dtype(torch::kFloat32).device(device_));

        const int max_waves = max_seq_len * 3;
        for (int wave = 0; wave < max_waves; ++wave) {
            scratch_active_.clear();
            scratch_active_.reserve(static_cast<size_t>(num_solutions));
            for (int i = 0; i < num_solutions; ++i) {
                auto& ctx = contexts[static_cast<size_t>(i)];
                if (!ctx.is_done && ctx.is_feasible) {
                    scratch_active_.push_back(i);
                }
            }
            if (scratch_active_.empty()) {
                break;
            }
            process_wave(scratch_active_, contexts, machine_idx, max_total_bins, H, W,
                         grid_states, grid_ffts, row_idx, col_idx, row_idx_f, col_idx_f, neg_inf);
        }

        std::vector<double> makespans(static_cast<size_t>(num_solutions), 0.0);
        for (int i = 0; i < num_solutions; ++i) {
            auto& ctx = contexts[static_cast<size_t>(i)];
            if (!ctx.is_feasible) {
                makespans[static_cast<size_t>(i)] = 1e16;
                continue;
            }
            double total = 0.0;
            for (const auto& b : ctx.open_bins) {
                if (b.area > 0.0) {
                    total += b.proc_time + b.proc_time_height + machine_setup_time_[static_cast<size_t>(machine_idx)];
                }
            }
            makespans[static_cast<size_t>(i)] = total;
        }
        return makespans;
    }

    void process_wave(
        const std::vector<int>& active_indices,
        std::vector<ContextNative>& contexts,
        int machine_idx,
        int max_total_bins,
        int H,
        int W,
        torch::Tensor& grid_states,
        torch::Tensor& grid_ffts,
        const torch::Tensor& row_idx,
        const torch::Tensor& col_idx,
        const torch::Tensor& row_idx_f,
        const torch::Tensor& col_idx_f,
        const torch::Tensor& neg_inf
    ) {
        scratch_ctx_global_.clear();
        scratch_part_idx_local_.clear();
        scratch_ctx_global_.reserve(active_indices.size());
        scratch_part_idx_local_.reserve(active_indices.size());

        for (int gidx : active_indices) {
            auto& ctx = contexts[static_cast<size_t>(gidx)];
            if (ctx.current_part_idx >= static_cast<int>(ctx.parts_sequence.size())) {
                ctx.is_done = true;
                continue;
            }
            const int part_idx = ctx.parts_sequence[static_cast<size_t>(ctx.current_part_idx)];
            const int rg0 = rot_global(part_idx, 0);
            const int h0 = rot_h_[static_cast<size_t>(rg0)];
            const int w0 = rot_w_[static_cast<size_t>(rg0)];
            if (((h0 > H) || (w0 > W)) && ((w0 > H) || (h0 > W))) {
                ctx.is_feasible = false;
                continue;
            }
            scratch_ctx_global_.push_back(gidx);
            scratch_part_idx_local_.push_back(part_idx);
        }

        const int n_contexts = static_cast<int>(scratch_ctx_global_.size());
        if (n_contexts == 0) {
            return;
        }

        scratch_invalid_grid_indices_.clear();
        scratch_invalid_bins_.clear();
        for (int lc = 0; lc < n_contexts; ++lc) {
            auto& ctx = contexts[static_cast<size_t>(scratch_ctx_global_[static_cast<size_t>(lc)])];
            for (auto& b : ctx.open_bins) {
                if (!b.grid_fft_valid) {
                    scratch_invalid_grid_indices_.push_back(static_cast<int64_t>(b.grid_state_idx));
                    scratch_invalid_bins_.push_back(&b);
                }
            }
        }
        if (!scratch_invalid_grid_indices_.empty()) {
            auto idx_t = load_workspace_long_from_i64(
                ws_wave_idx_long_, ws_cpu_wave_idx_,
                scratch_invalid_grid_indices_,
                static_cast<int64_t>(scratch_invalid_grid_indices_.size())
            );
            auto batch_grids = grid_states.index_select(0, idx_t);
            auto batch_ffts = torch::fft::rfft2(batch_grids);
            grid_ffts.index_copy_(0, idx_t, batch_ffts);
            for (auto* b : scratch_invalid_bins_) {
                b->grid_fft_valid = true;
            }
        }

        scratch_ctx_first_valid_bin_.assign(static_cast<size_t>(n_contexts), -1);
        auto& p1 = scratch_p1_;
        p1.grid_indices.clear();
        p1.heights.clear();
        p1.widths.clear();
        p1.bin_indices.clear();
        p1.bin_local.clear();
        p1.rot_global.clear();
        p1.ctx_local.clear();
        p1.enclosure_lengths.clear();
        p1.bin_areas.clear();
        p1.part_areas.clear();
        for (int lc = 0; lc < n_contexts; ++lc) {
            auto& ctx = contexts[static_cast<size_t>(scratch_ctx_global_[static_cast<size_t>(lc)])];
            const int part_idx = scratch_part_idx_local_[static_cast<size_t>(lc)];
            const double p_area = part_area_[static_cast<size_t>(part_idx)];
            const int nrot = part_nrot_[static_cast<size_t>(part_idx)];
            const int r0 = part_rot_offsets_[static_cast<size_t>(part_idx)];

            for (int bidx = 0; bidx < static_cast<int>(ctx.open_bins.size()); ++bidx) {
                auto& b = ctx.open_bins[static_cast<size_t>(bidx)];
                if (b.area + p_area > ctx.bin_area) {
                    continue;
                }
                bool has_passing = false;
                for (int rot = 0; rot < nrot; ++rot) {
                    const int rg = r0 + rot;
                    const int ph = rot_h_[static_cast<size_t>(rg)];
                    const int pw = rot_w_[static_cast<size_t>(rg)];
                    if (ph > H || pw > W) {
                        continue;
                    }
                    const int d0 = rot_density_offsets_[static_cast<size_t>(rg)];
                    const int d1 = rot_density_offsets_[static_cast<size_t>(rg + 1)];
                    if (check_vacancy_fit_simple_cpp(
                            b.vacancy,
                            density_flat_.data() + d0,
                            d1 - d0)) {
                        has_passing = true;
                        p1.grid_indices.push_back(static_cast<int64_t>(b.grid_state_idx));
                        p1.heights.push_back(rot_h_[static_cast<size_t>(rg)]);
                        p1.widths.push_back(rot_w_[static_cast<size_t>(rg)]);
                        p1.bin_indices.push_back(static_cast<double>(bidx));
                        p1.bin_local.push_back(bidx);
                        p1.rot_global.push_back(rg);
                        p1.ctx_local.push_back(lc);
                        p1.enclosure_lengths.push_back(static_cast<double>(b.enclosure_box_length));
                        p1.bin_areas.push_back(b.area);
                        p1.part_areas.push_back(p_area);
                    }
                }
                if (has_passing) {
                    scratch_ctx_first_valid_bin_[static_cast<size_t>(lc)] = bidx;
                    break;
                }
            }
        }

        batch_fft_all_tests(
            static_cast<int>(p1.grid_indices.size()),
            p1.grid_indices, p1.rot_global, p1.heights, p1.widths,
            p1.bin_indices, p1.enclosure_lengths, p1.bin_areas, p1.part_areas,
            machine_idx, grid_ffts, H, W, row_idx, col_idx, row_idx_f, col_idx_f, neg_inf, scratch_fft_p1_
        );

        scratch_ctx_p1_hit_.assign(static_cast<size_t>(n_contexts), 0);
        for (size_t i = 0; i < p1.ctx_local.size(); ++i) {
            if (scratch_fft_p1_.has_result[i]) {
                const int lc = p1.ctx_local[i];
                scratch_ctx_p1_hit_[static_cast<size_t>(lc)] = 1;
            }
        }

        auto& p2 = scratch_p2_;
        p2.grid_indices.clear();
        p2.heights.clear();
        p2.widths.clear();
        p2.bin_indices.clear();
        p2.bin_local.clear();
        p2.rot_global.clear();
        p2.ctx_local.clear();
        p2.enclosure_lengths.clear();
        p2.bin_areas.clear();
        p2.part_areas.clear();
        for (int lc = 0; lc < n_contexts; ++lc) {
            if (scratch_ctx_p1_hit_[static_cast<size_t>(lc)]) {
                continue;
            }
            auto& ctx = contexts[static_cast<size_t>(scratch_ctx_global_[static_cast<size_t>(lc)])];
            const int part_idx = scratch_part_idx_local_[static_cast<size_t>(lc)];
            const double p_area = part_area_[static_cast<size_t>(part_idx)];
            const int nrot = part_nrot_[static_cast<size_t>(part_idx)];
            const int r0 = part_rot_offsets_[static_cast<size_t>(part_idx)];
            const int first_valid = scratch_ctx_first_valid_bin_[static_cast<size_t>(lc)];

            for (int bidx = 0; bidx < static_cast<int>(ctx.open_bins.size()); ++bidx) {
                if (bidx == first_valid) {
                    continue;
                }
                auto& b = ctx.open_bins[static_cast<size_t>(bidx)];
                if (b.area + p_area > ctx.bin_area) {
                    continue;
                }
                for (int rot = 0; rot < nrot; ++rot) {
                    const int rg = r0 + rot;
                    const int ph = rot_h_[static_cast<size_t>(rg)];
                    const int pw = rot_w_[static_cast<size_t>(rg)];
                    if (ph > H || pw > W) {
                        continue;
                    }
                    const int d0 = rot_density_offsets_[static_cast<size_t>(rg)];
                    const int d1 = rot_density_offsets_[static_cast<size_t>(rg + 1)];
                    if (!check_vacancy_fit_simple_cpp(
                            b.vacancy,
                            density_flat_.data() + d0,
                            d1 - d0)) {
                        continue;
                    }
                    p2.grid_indices.push_back(static_cast<int64_t>(b.grid_state_idx));
                    p2.heights.push_back(rot_h_[static_cast<size_t>(rg)]);
                    p2.widths.push_back(rot_w_[static_cast<size_t>(rg)]);
                    p2.bin_indices.push_back(static_cast<double>(bidx));
                    p2.bin_local.push_back(bidx);
                    p2.rot_global.push_back(rg);
                    p2.ctx_local.push_back(lc);
                    p2.enclosure_lengths.push_back(static_cast<double>(b.enclosure_box_length));
                    p2.bin_areas.push_back(b.area);
                    p2.part_areas.push_back(p_area);
                }
            }
        }

        batch_fft_all_tests(
            static_cast<int>(p2.grid_indices.size()),
            p2.grid_indices, p2.rot_global, p2.heights, p2.widths,
            p2.bin_indices, p2.enclosure_lengths, p2.bin_areas, p2.part_areas,
            machine_idx, grid_ffts, H, W, row_idx, col_idx, row_idx_f, col_idx_f, neg_inf, scratch_fft_p2_
        );

        auto& test_ctx_local = scratch_test_ctx_local_;
        auto& test_bin_local = scratch_test_bin_local_;
        auto& test_rot_global = scratch_test_rot_global_;
        auto& placement_has = scratch_placement_has_;
        auto& placement_cols = scratch_placement_cols_;
        auto& placement_rows = scratch_placement_rows_;
        auto& sc_bin_indices = scratch_sc_bin_indices_;
        auto& sc_densities = scratch_sc_densities_;
        auto& sc_rows = scratch_sc_rows_;
        auto& sc_cols = scratch_sc_cols_;
        auto& sc_valid = scratch_sc_valid_;

        test_ctx_local.clear();
        test_bin_local.clear();
        test_rot_global.clear();
        placement_has.clear();
        placement_cols.clear();
        placement_rows.clear();
        sc_bin_indices.clear();
        sc_densities.clear();
        sc_rows.clear();
        sc_cols.clear();
        sc_valid.clear();

        const size_t n_tests_total = p1.ctx_local.size() + p2.ctx_local.size();
        test_ctx_local.reserve(n_tests_total);
        test_bin_local.reserve(n_tests_total);
        test_rot_global.reserve(n_tests_total);
        placement_has.reserve(n_tests_total);
        placement_cols.reserve(n_tests_total);
        placement_rows.reserve(n_tests_total);
        sc_bin_indices.reserve(n_tests_total);
        sc_densities.reserve(n_tests_total);
        sc_rows.reserve(n_tests_total);
        sc_cols.reserve(n_tests_total);
        sc_valid.reserve(n_tests_total);

        auto append_tests = [&](
            const CollectedTests& c,
            const FFTBatchResult& f
        ) {
            for (size_t i = 0; i < c.ctx_local.size(); ++i) {
                test_ctx_local.push_back(c.ctx_local[i]);
                test_bin_local.push_back(c.bin_local[i]);
                test_rot_global.push_back(c.rot_global[i]);
                placement_has.push_back(f.has_result[i]);
                placement_cols.push_back(f.cols[i]);
                placement_rows.push_back(f.rows[i]);
                sc_bin_indices.push_back(c.bin_indices[i]);
                sc_densities.push_back(f.densities[i]);
                sc_rows.push_back(f.sc_rows[i]);
                sc_cols.push_back(f.sc_cols[i]);
                sc_valid.push_back(f.sc_valid[i]);
            }
        };
        append_tests(p1, scratch_fft_p1_);
        append_tests(p2, scratch_fft_p2_);

        auto& best_ti = scratch_best_ti_;
        auto& has_best = scratch_has_best_;
        auto& best_neg_bin = scratch_best_neg_bin_;
        auto& best_density = scratch_best_density_;
        auto& best_row = scratch_best_row_;
        auto& best_neg_col = scratch_best_neg_col_;
        best_ti.assign(static_cast<size_t>(n_contexts), -1);
        has_best.assign(static_cast<size_t>(n_contexts), 0);
        best_neg_bin.assign(static_cast<size_t>(n_contexts), 0.0);
        best_density.assign(static_cast<size_t>(n_contexts), 0.0);
        best_row.assign(static_cast<size_t>(n_contexts), 0.0);
        best_neg_col.assign(static_cast<size_t>(n_contexts), 0.0);

        for (int ti = 0; ti < static_cast<int>(test_ctx_local.size()); ++ti) {
            if (!sc_valid[static_cast<size_t>(ti)]) {
                continue;
            }
            const int lc = test_ctx_local[static_cast<size_t>(ti)];
            const double neg_bin = -sc_bin_indices[static_cast<size_t>(ti)];
            const double den = sc_densities[static_cast<size_t>(ti)];
            const double row = sc_rows[static_cast<size_t>(ti)];
            const double neg_col = -sc_cols[static_cast<size_t>(ti)];
            const size_t c = static_cast<size_t>(lc);
            if (!has_best[c] || lex_better(
                    neg_bin, den, row, neg_col,
                    best_neg_bin[c], best_density[c], best_row[c], best_neg_col[c])) {
                has_best[c] = 1;
                best_neg_bin[c] = neg_bin;
                best_density[c] = den;
                best_row[c] = row;
                best_neg_col[c] = neg_col;
                best_ti[c] = ti;
            }
        }

        auto& place_ctx_local = scratch_place_ctx_local_;
        auto& place_ti = scratch_place_ti_;
        auto& place_rows = scratch_place_rows_;
        auto& place_cols = scratch_place_cols_;
        auto& newbin_ctx_local = scratch_newbin_ctx_local_;
        place_ctx_local.clear();
        place_ti.clear();
        place_rows.clear();
        place_cols.clear();
        newbin_ctx_local.clear();
        place_ctx_local.reserve(static_cast<size_t>(n_contexts));
        place_ti.reserve(static_cast<size_t>(n_contexts));
        place_rows.reserve(static_cast<size_t>(n_contexts));
        place_cols.reserve(static_cast<size_t>(n_contexts));
        newbin_ctx_local.reserve(static_cast<size_t>(n_contexts));

        for (int lc = 0; lc < n_contexts; ++lc) {
            const int ti = best_ti[static_cast<size_t>(lc)];
            if (ti >= 0 && placement_has[static_cast<size_t>(ti)]) {
                place_ctx_local.push_back(lc);
                place_ti.push_back(ti);
                place_rows.push_back(placement_rows[static_cast<size_t>(ti)]);
                place_cols.push_back(placement_cols[static_cast<size_t>(ti)]);
            } else {
                newbin_ctx_local.push_back(lc);
            }
        }

        auto& gpu_updates = scratch_gpu_updates_;
        gpu_updates.clear();
        gpu_updates.reserve(place_ctx_local.size());

        for (size_t i = 0; i < place_ctx_local.size(); ++i) {
            const int lc = place_ctx_local[i];
            const int ctx_g = scratch_ctx_global_[static_cast<size_t>(lc)];
            const int part_idx = scratch_part_idx_local_[static_cast<size_t>(lc)];
            const int ti = place_ti[i];
            const int row = place_rows[i];
            const int col = place_cols[i];
            const int rotg = test_rot_global[static_cast<size_t>(ti)];
            const int bidx = test_bin_local[static_cast<size_t>(ti)];

            auto& ctx = contexts[static_cast<size_t>(ctx_g)];
            if (bidx < 0 || bidx >= static_cast<int>(ctx.open_bins.size())) {
                ctx.is_feasible = false;
                continue;
            }
            auto& bin = ctx.open_bins[static_cast<size_t>(bidx)];
            const int y_start = row - rot_h_[static_cast<size_t>(rotg)] + 1;
            bin.grid_fft_valid = false;
            add_part_to_bin_cpu(bin, col, row, rotg, part_idx, machine_idx, W, false);
            ctx.current_part_idx += 1;
            gpu_updates.push_back({
                bin.grid_state_idx,
                y_start,
                col,
                rot_flat_offsets_[static_cast<size_t>(rotg)],
                rot_h_[static_cast<size_t>(rotg)],
                rot_w_[static_cast<size_t>(rotg)]
            });
        }

        if (!gpu_updates.empty()) {
            apply_gpu_updates(grid_states, gpu_updates, H, W);
        }

        auto& phase6_gpu_updates = scratch_phase6_gpu_updates_;
        auto& phase6_grid_indices = scratch_phase6_grid_indices_;
        phase6_gpu_updates.clear();
        phase6_grid_indices.clear();
        phase6_gpu_updates.reserve(newbin_ctx_local.size());
        phase6_grid_indices.reserve(newbin_ctx_local.size());
        for (int lc : newbin_ctx_local) {
            const int ctx_g = scratch_ctx_global_[static_cast<size_t>(lc)];
            const int part_idx = scratch_part_idx_local_[static_cast<size_t>(lc)];
            auto& ctx = contexts[static_cast<size_t>(ctx_g)];

            const int grid_idx = ctx.next_grid_idx;
            ctx.next_grid_idx += 1;
            if (grid_idx < 0 || grid_idx >= max_total_bins) {
                ctx.is_feasible = false;
                continue;
            }

            BinStateNative new_bin;
            new_bin.bin_idx = static_cast<int>(ctx.open_bins.size());
            new_bin.grid.assign(static_cast<size_t>(H * W), 0u);
            new_bin.vacancy.assign(static_cast<size_t>(H), static_cast<int32_t>(W));
            new_bin.grid_state_idx = grid_idx;
            new_bin.area = 0.0;
            new_bin.enclosure_box_length = 0;
            new_bin.min_occupied_row = H;
            new_bin.max_occupied_row = -1;
            new_bin.proc_time = 0.0;
            new_bin.proc_time_height = 0.0;
            new_bin.grid_fft_valid = false;
            new_bin.bin_length = H;
            new_bin.bin_width = W;

            const int best_rot = part_best_rot_[static_cast<size_t>(part_idx)];
            const int rotg = rot_global(part_idx, best_rot);
            const int ph = rot_h_[static_cast<size_t>(rotg)];
            const int y = H - 1;
            const int y_start = H - ph;

            add_part_to_bin_cpu(new_bin, 0, y, rotg, part_idx, machine_idx, W, true);
            ctx.open_bins.push_back(std::move(new_bin));
            ctx.current_part_idx += 1;
            phase6_grid_indices.push_back(static_cast<int64_t>(grid_idx));
            phase6_gpu_updates.push_back({
                grid_idx,
                y_start,
                0,
                rot_flat_offsets_[static_cast<size_t>(rotg)],
                rot_h_[static_cast<size_t>(rotg)],
                rot_w_[static_cast<size_t>(rotg)]
            });
        }

        if (!phase6_gpu_updates.empty()) {
            auto idx_t = load_workspace_long_from_i64(
                ws_wave_idx_long_, ws_cpu_wave_idx_,
                phase6_grid_indices,
                static_cast<int64_t>(phase6_grid_indices.size())
            );
            grid_states.index_fill_(0, idx_t, 0.0);
            apply_gpu_updates(grid_states, phase6_gpu_updates, H, W);
        }
    }

    template <typename PlacementVec>
    void apply_gpu_updates(
        torch::Tensor& grid_states,
        const PlacementVec& updates,
        int H,
        int W
    ) {
        const int n = static_cast<int>(updates.size());
        if (n <= 0) {
            return;
        }
        if (!grid_states.is_cuda()) {
            for (const auto& u : updates) {
                const auto rot_len = static_cast<int64_t>(u.ph) * static_cast<int64_t>(u.pw);
                auto part_view = flat_parts_gpu_.index({
                    Slice(static_cast<int64_t>(u.flat_offset), static_cast<int64_t>(u.flat_offset) + rot_len)
                }).view({u.ph, u.pw});
                grid_states.index({u.grid_idx, Slice(u.y_start, u.y_start + u.ph), Slice(u.x_start, u.x_start + u.pw)})
                    .add_(part_view);
            }
            return;
        }

        auto& cell_offsets = scratch_cell_offsets_;
        auto& grid_idxs = scratch_grid_idxs_;
        auto& y_starts = scratch_y_starts_;
        auto& x_starts = scratch_x_starts_;
        auto& part_widths = scratch_part_widths_;
        auto& part_offsets = scratch_part_offsets_;
        cell_offsets.assign(static_cast<size_t>(n + 1), 0);
        grid_idxs.resize(static_cast<size_t>(n));
        y_starts.resize(static_cast<size_t>(n));
        x_starts.resize(static_cast<size_t>(n));
        part_widths.resize(static_cast<size_t>(n));
        part_offsets.resize(static_cast<size_t>(n));

        int total_cells = 0;
        for (int i = 0; i < n; ++i) {
            const auto& u = updates[static_cast<size_t>(i)];
            grid_idxs[static_cast<size_t>(i)] = static_cast<int32_t>(u.grid_idx);
            y_starts[static_cast<size_t>(i)] = static_cast<int32_t>(u.y_start);
            x_starts[static_cast<size_t>(i)] = static_cast<int32_t>(u.x_start);
            part_widths[static_cast<size_t>(i)] = static_cast<int32_t>(u.pw);
            part_offsets[static_cast<size_t>(i)] = static_cast<int32_t>(u.flat_offset);
            cell_offsets[static_cast<size_t>(i)] = total_cells;
            total_cells += static_cast<int>(u.ph) * static_cast<int>(u.pw);
        }
        cell_offsets[static_cast<size_t>(n)] = total_cells;
        if (total_cells <= 0) {
            return;
        }

        auto cell_offsets_t = load_workspace_i32_from_i32(
            ws_cell_offsets_i32_, ws_cpu_cell_offsets_, cell_offsets, static_cast<int64_t>(n + 1)
        );
        auto grid_idxs_t = load_workspace_i32_from_i32(
            ws_grid_idxs_i32_, ws_cpu_grid_idxs_, grid_idxs, static_cast<int64_t>(n)
        );
        auto y_starts_t = load_workspace_i32_from_i32(
            ws_y_starts_i32_, ws_cpu_y_starts_, y_starts, static_cast<int64_t>(n)
        );
        auto x_starts_t = load_workspace_i32_from_i32(
            ws_x_starts_i32_, ws_cpu_x_starts_, x_starts, static_cast<int64_t>(n)
        );
        auto part_widths_t = load_workspace_i32_from_i32(
            ws_part_widths_i32_, ws_cpu_part_widths_, part_widths, static_cast<int64_t>(n)
        );
        auto part_offsets_t = load_workspace_i32_from_i32(
            ws_part_offsets_i32_, ws_cpu_part_offsets_, part_offsets, static_cast<int64_t>(n)
        );

        native_batch_grid_update(
            grid_states,
            flat_parts_gpu_,
            cell_offsets_t,
            grid_idxs_t,
            y_starts_t,
            x_starts_t,
            part_widths_t,
            part_offsets_t,
            n,
            total_cells,
            H,
            W
        );
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<NativeFullDecoder>(m, "NativeFullDecoder")
        .def(py::init<py::dict, int, int, const std::string&>())
        .def("evaluate_batch", &NativeFullDecoder::evaluate_batch);
}
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>

__global__ void _native_batch_grid_update_kernel(
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
    int global_col = x_starts[p] + col;

    grid_flat[(long)grid_idxs[p] * H * W + (long)global_row * W + global_col]
        += parts_flat[part_offsets[p] + local_idx];
}

__global__ void _native_select_best_positions_kernel(
    const float* __restrict__ overlap,
    const int64_t* __restrict__ part_h,
    const int64_t* __restrict__ part_w,
    int n_tests, int H, int W,
    int* __restrict__ out_has,
    int* __restrict__ out_row,
    int* __restrict__ out_col_start
) {
    int test_idx = blockIdx.x;
    if (test_idx >= n_tests) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    int h = (int)part_h[test_idx];
    int w = (int)part_w[test_idx];
    int min_row = h - 1;
    int min_col = w - 1;
    int HW = H * W;

    int best_score = INT_MIN;
    const float* base = overlap + (long long)test_idx * (long long)HW;
    for (int idx = tid; idx < HW; idx += stride) {
        int row = idx / W;
        int col = idx - row * W;
        if (row < min_row || col < min_col) continue;
        float v = base[idx];
        if ((int)llroundf(v) == 0) {
            // Maximize row first, then minimize col.
            int score = row * (W + 1) + (W - col);
            if (score > best_score) best_score = score;
        }
    }

    extern __shared__ int s_best[];
    s_best[tid] = best_score;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (s_best[tid + offset] > s_best[tid]) {
                s_best[tid] = s_best[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int score = s_best[0];
        if (score == INT_MIN) {
            out_has[test_idx] = 0;
            out_row[test_idx] = 0;
            out_col_start[test_idx] = 0;
            return;
        }
        int row = score / (W + 1);
        int rem = score - row * (W + 1);
        int col = W - rem;
        out_has[test_idx] = 1;
        out_row[test_idx] = row;
        out_col_start[test_idx] = col - (w - 1);
    }
}

void native_batch_grid_update_cuda(
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
    _native_batch_grid_update_kernel<<<blocks, threads>>>(
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

void native_select_best_positions_cuda(
    torch::Tensor overlap_batch,
    torch::Tensor part_h,
    torch::Tensor part_w,
    int H,
    int W,
    torch::Tensor out_has,
    torch::Tensor out_row,
    torch::Tensor out_col_start
) {
    const int n_tests = (int)overlap_batch.size(0);
    if (n_tests <= 0) return;
    const int threads = 256;
    const int blocks = n_tests;
    const size_t shmem = (size_t)threads * sizeof(int);
    _native_select_best_positions_kernel<<<blocks, threads, shmem>>>(
        overlap_batch.data_ptr<float>(),
        part_h.data_ptr<int64_t>(),
        part_w.data_ptr<int64_t>(),
        n_tests, H, W,
        out_has.data_ptr<int>(),
        out_row.data_ptr<int>(),
        out_col_start.data_ptr<int>()
    );
}
"""

_module = None
_load_failed = False


def _get_module():
    global _module, _load_failed
    if _module is not None:
        return _module
    if _load_failed:
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        _module = load_inline(
            name="_full_native_decoder_ext",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=None,
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )
    except Exception:
        _module = None
        _load_failed = True
    return _module


def _pack_problem_data(problem_data, nb_machines, thresholds, instance_parts, device: str) -> Dict[str, object]:
    use_fast_selector = os.getenv("ABRKGA_NATIVE_FAST_SELECTOR", "1").strip() not in {"0", "false", "False"}
    use_cuda_selector_kernel = os.getenv("ABRKGA_NATIVE_SELECTOR_KERNEL", "1").strip() not in {"0", "false", "False"}
    selector_dual_check = os.getenv("ABRKGA_NATIVE_SELECTOR_DUAL_CHECK", "0").strip() not in {"0", "false", "False"}
    part_ids = [int(x) for x in np.asarray(problem_data.instance_parts_unique, dtype=np.int64)]
    part_id_to_idx = {pid: i for i, pid in enumerate(part_ids)}
    n_parts_unique = len(part_ids)

    part_area: List[float] = []
    part_nrot: List[int] = []
    part_best_rot: List[int] = []
    part_rot_offsets: List[int] = [0]

    rot_h: List[int] = []
    rot_w: List[int] = []
    rot_density_offsets: List[int] = [0]
    density_flat: List[np.ndarray] = []
    rot_matrix_offsets: List[int] = [0]
    rot_matrix_flat: List[np.ndarray] = []
    rot_gpu_tensors: List[torch.Tensor] = []
    rot_flat_offsets: List[int] = [0]

    machine_fft_dense: List[torch.Tensor] = []
    machine_proc_time = np.zeros((nb_machines, n_parts_unique), dtype=np.float64)
    machine_proc_time_height = np.zeros((nb_machines, n_parts_unique), dtype=np.float64)

    for pidx, pid in enumerate(part_ids):
        pd = problem_data.parts[pid]
        nrot = int(pd.nrot)
        part_area.append(float(pd.area))
        part_nrot.append(nrot)
        part_best_rot.append(int(pd.best_rotation))
        part_rot_offsets.append(part_rot_offsets[-1] + nrot)

        for rot in range(nrot):
            h, w = pd.shapes[rot]
            rot_h.append(int(h))
            rot_w.append(int(w))

            den = np.ascontiguousarray(np.asarray(pd.densities[rot], dtype=np.int32))
            density_flat.append(den)
            rot_density_offsets.append(rot_density_offsets[-1] + int(den.size))

            mat_u8 = np.ascontiguousarray(np.asarray(pd.rotations_uint8[rot], dtype=np.uint8))
            rot_matrix_flat.append(mat_u8.reshape(-1))
            rot_matrix_offsets.append(rot_matrix_offsets[-1] + int(mat_u8.size))

            if pd.rotations_gpu is not None and pd.rotations_gpu[rot] is not None:
                t = pd.rotations_gpu[rot]
            else:
                t = torch.as_tensor(pd.rotations[rot], dtype=torch.float32)
            if device.startswith("cuda") and (not t.is_cuda or str(t.device) != device):
                t = t.to(device)
            tc = t.contiguous()
            rot_gpu_tensors.append(tc)
            rot_flat_offsets.append(rot_flat_offsets[-1] + int(tc.numel()))

    for m in range(nb_machines):
        mach = problem_data.machines[m]
        machine_fft_rot: List[torch.Tensor] = []
        for pidx, pid in enumerate(part_ids):
            mpd = mach.parts[pid]
            machine_proc_time[m, pidx] = float(mpd.proc_time)
            machine_proc_time_height[m, pidx] = float(mpd.proc_time_height)
            nrot = int(problem_data.parts[pid].nrot)
            for rot in range(nrot):
                fft = mpd.ffts[rot]
                if device.startswith("cuda") and (not fft.is_cuda or str(fft.device) != device):
                    fft = fft.to(device)
                machine_fft_rot.append(fft.contiguous())
        machine_fft_dense.append(torch.stack(machine_fft_rot, dim=0).contiguous())

    if rot_gpu_tensors:
        flat_parts_gpu = torch.cat([t.reshape(-1) for t in rot_gpu_tensors], dim=0).contiguous()
    else:
        flat_parts_gpu = torch.empty((0,), dtype=torch.float32, device=device if device.startswith("cuda") else "cpu")

    packed = {
        "thresholds": np.ascontiguousarray(np.asarray(thresholds, dtype=np.float64)),
        "instance_parts_idx": np.ascontiguousarray(
            np.asarray([part_id_to_idx[int(pid)] for pid in np.asarray(instance_parts, dtype=np.int64)], dtype=np.int32)
        ),
        "machine_bin_length": np.ascontiguousarray(
            np.asarray([problem_data.machines[m].bin_length for m in range(nb_machines)], dtype=np.int32)
        ),
        "machine_bin_width": np.ascontiguousarray(
            np.asarray([problem_data.machines[m].bin_width for m in range(nb_machines)], dtype=np.int32)
        ),
        "machine_bin_area": np.ascontiguousarray(
            np.asarray([problem_data.machines[m].bin_area for m in range(nb_machines)], dtype=np.float64)
        ),
        "machine_setup_time": np.ascontiguousarray(
            np.asarray([problem_data.machines[m].setup_time for m in range(nb_machines)], dtype=np.float64)
        ),
        "part_area": np.ascontiguousarray(np.asarray(part_area, dtype=np.float64)),
        "part_nrot": np.ascontiguousarray(np.asarray(part_nrot, dtype=np.int32)),
        "part_best_rot": np.ascontiguousarray(np.asarray(part_best_rot, dtype=np.int32)),
        "part_rot_offsets": np.ascontiguousarray(np.asarray(part_rot_offsets, dtype=np.int32)),
        "rot_h": np.ascontiguousarray(np.asarray(rot_h, dtype=np.int32)),
        "rot_w": np.ascontiguousarray(np.asarray(rot_w, dtype=np.int32)),
        "rot_density_offsets": np.ascontiguousarray(np.asarray(rot_density_offsets, dtype=np.int32)),
        "density_flat": np.ascontiguousarray(
            np.concatenate(density_flat).astype(np.int32) if density_flat else np.zeros(0, dtype=np.int32)
        ),
        "rot_matrix_offsets": np.ascontiguousarray(np.asarray(rot_matrix_offsets, dtype=np.int32)),
        "rot_matrix_flat_u8": np.ascontiguousarray(
            np.concatenate(rot_matrix_flat).astype(np.uint8) if rot_matrix_flat else np.zeros(0, dtype=np.uint8)
        ),
        "machine_proc_time": np.ascontiguousarray(machine_proc_time),
        "machine_proc_time_height": np.ascontiguousarray(machine_proc_time_height),
        "rot_flat_offsets": np.ascontiguousarray(np.asarray(rot_flat_offsets, dtype=np.int32)),
        "flat_parts_gpu": flat_parts_gpu,
        "machine_fft_dense": machine_fft_dense,
        "use_fast_selector": bool(use_fast_selector),
        "use_cuda_selector_kernel": bool(use_cuda_selector_kernel),
        "selector_dual_check": bool(selector_dual_check),
        "vram_total_bytes": _get_vram_total_bytes(device),
    }
    return packed


def _get_vram_total_bytes(device: str) -> int:
    """Return total VRAM in bytes for the given device string, or 0 for CPU."""
    if not device.startswith("cuda"):
        return 0
    dev_idx = int(device.split(":")[-1]) if ":" in device else 0
    return int(torch.cuda.get_device_properties(dev_idx).total_memory)


class FullNativeDecoderEvaluator:
    """Python wrapper around the C++ NativeFullDecoder class."""

    def __init__(
        self,
        problem_data,
        nb_parts: int,
        nb_machines: int,
        thresholds,
        instance_parts,
        collision_backend,
        device: str = "cuda",
    ):
        del collision_backend  # kept for interface parity with WaveBatchEvaluator
        mod = _get_module()
        if mod is None:
            raise RuntimeError("Failed to build/load _full_native_decoder_ext")
        packed = _pack_problem_data(problem_data, int(nb_machines), thresholds, instance_parts, device)
        self._decoder = mod.NativeFullDecoder(packed, int(nb_parts), int(nb_machines), str(device))

    def evaluate_batch(self, chromosomes: np.ndarray):
        chrom = np.ascontiguousarray(chromosomes, dtype=np.float32)
        out = self._decoder.evaluate_batch(chrom)
        return np.asarray(out, dtype=np.float64).tolist()
