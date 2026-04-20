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
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
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

// ── Profiling instrumentation (LOCAL, UNCOMMITTED) ───────────────────────────
// Accumulators for CPU hot spot profiling.  See CPU_OPTIMIZATION_CANDIDATES.md.
// Turn off by commenting out PROFILE_CPU_HOTSPOTS.
#define PROFILE_CPU_HOTSPOTS 1

#if PROFILE_CPU_HOTSPOTS
struct WaveProfile {
    // Phase 6 new-bin creation
    int64_t phase6_loop_ns = 0;         // whole Phase 6 new-bin loop (per wave)
    int64_t binstate_ctor_ns = 0;       // sum of BinStateNative field init + grid.assign
    int64_t binstate_count = 0;

    // add_part_to_bin_cpu (whole function, includes uvr)
    int64_t aptb_ns = 0;
    int64_t aptb_count = 0;

    // Phase 3 p1
    int64_t p3_pass_a_ns = 0;           // pair collection loop
    int64_t p3_pass_b_wait_ns = 0;      // time blocked waiting for vacancy kernel result
    int64_t p3_pass_c_ns = 0;           // mask → test-list build

    // run_gpu_vacancy_check sync copy_ calls
    int64_t vac_upload_sync_ns = 0;     // non_blocking=false upload
    int64_t vac_readback_sync_ns = 0;   // non_blocking=false readback
    int64_t vac_call_count = 0;

    // Comprehensive phase breakdown
    int64_t total_wave_ns = 0;
    int64_t wave_count = 0;
    int64_t phase2_rfft2_ns = 0;         // rfft2 of invalid grids
    int64_t phase4_p1_fft_ns = 0;        // batch_fft_all_tests for p1
    int64_t phase4_p2_fft_ns = 0;        // batch_fft_all_tests for p2
    int64_t phase5_select_ns = 0;        // best selection + placement CPU loop
    int64_t phase5_gpu_update_ns = 0;    // apply_gpu_updates + recompute_vacancy
    int64_t phase3_p2_collect_ns = 0;    // p2 pair collection + vacancy check
    int64_t decode_seq_ns = 0;           // decode_sequences_for_machine total
    int64_t phase1_active_ns = 0;        // Phase 1: gather active contexts + invalid FFTs

    void reset() { *this = WaveProfile{}; }

    std::string summary() const {
        auto ms = [](int64_t ns) { return static_cast<double>(ns) / 1e6; };
        std::ostringstream os;
        os << "\n═══ CPU Hotspot Profile ═══\n";
        os << "  Total wave time      : " << ms(total_wave_ns) << " ms  (" << wave_count << " waves)\n";
        os << "  Phase 1 (active+FFT) : " << ms(phase1_active_ns) << " ms\n";
        os << "  Phase 2 (rfft2)      : " << ms(phase2_rfft2_ns) << " ms\n";
        os << "  Phase 3 Pass A       : " << ms(p3_pass_a_ns) << " ms\n";
        os << "  Phase 3 Pass B wait  : " << ms(p3_pass_b_wait_ns) << " ms\n";
        os << "  Phase 3 Pass C       : " << ms(p3_pass_c_ns) << " ms\n";
        os << "  Phase 3 p2 collect   : " << ms(phase3_p2_collect_ns) << " ms\n";
        os << "  Phase 4 p1 (irfft2)  : " << ms(phase4_p1_fft_ns) << " ms\n";
        os << "  Phase 4 p2 (irfft2)  : " << ms(phase4_p2_fft_ns) << " ms\n";
        os << "  Phase 5 (select+place): " << ms(phase5_select_ns) << " ms\n";
        os << "  Phase 5 GPU updates  : " << ms(phase5_gpu_update_ns) << " ms\n";
        os << "  Phase 6 new-bin loop : " << ms(phase6_loop_ns) << " ms  ("
           << binstate_count << " bins)\n";
        os << "    └ BinState ctor    : " << ms(binstate_ctor_ns) << " ms  ("
           << binstate_count << " calls, "
           << (binstate_count ? ms(binstate_ctor_ns) / static_cast<double>(binstate_count) * 1e3 : 0.0)
           << " µs/call)\n";
        os << "  add_part_to_bin_cpu  : " << ms(aptb_ns) << " ms  ("
           << aptb_count << " calls, "
           << (aptb_count ? ms(aptb_ns) / static_cast<double>(aptb_count) * 1e3 : 0.0)
           << " µs/call)\n";
        os << "  decode_sequences     : " << ms(decode_seq_ns) << " ms\n";
        os << "  Vacancy upload sync  : " << ms(vac_upload_sync_ns) << " ms\n";
        os << "  Vacancy readback sync: " << ms(vac_readback_sync_ns) << " ms\n";
        os << "  (" << vac_call_count << " vacancy-check invocations)\n";
        double accounted = ms(phase1_active_ns) + ms(phase2_rfft2_ns) + ms(p3_pass_a_ns)
            + ms(p3_pass_b_wait_ns) + ms(p3_pass_c_ns) + ms(phase3_p2_collect_ns)
            + ms(phase4_p1_fft_ns) + ms(phase4_p2_fft_ns) + ms(phase5_select_ns)
            + ms(phase5_gpu_update_ns) + ms(phase6_loop_ns);
        os << "  Accounted in waves   : " << accounted << " ms / " << ms(total_wave_ns) << " ms ("
           << (ms(total_wave_ns) > 0 ? 100.0 * accounted / ms(total_wave_ns) : 0.0) << "%)\n";
        return os.str();
    }
};
static WaveProfile g_profile;

#define PROF_NOW() std::chrono::steady_clock::now()
#define PROF_NS_SINCE(t0) \
    std::chrono::duration_cast<std::chrono::nanoseconds>(PROF_NOW() - (t0)).count()
#else
#define PROF_NOW() 0
#define PROF_NS_SINCE(t0) 0
#endif
// ─────────────────────────────────────────────────────────────────────────────

struct BinStateNative {
    int bin_idx = 0;
    // NOTE: No CPU grid mirror.  grid_states (GPU) is the authoritative grid.
    // Vacancy is computed on GPU via recompute_vacancy_gpu.
    // Grid FFTs are computed on GPU via rfft2(grid_states).
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

void native_gather_grids_cuda(
    torch::Tensor grid_states,   // (max_bins, H, W) float32
    torch::Tensor idx,           // (N,) int64
    torch::Tensor output,        // (N, H, W) float32, pre-allocated
    int N, int H, int W
);

void native_scatter_ffts_cuda(
    torch::Tensor input,         // (N, H, W2) complex64
    torch::Tensor idx,           // (N,) int64
    torch::Tensor grid_ffts,     // (max_bins, H, W2) complex64
    int N, int H, int W2
);

void native_fused_gather_multiply_cuda(
    torch::Tensor grid_ffts,
    torch::Tensor part_ffts,
    torch::Tensor grid_idx,
    torch::Tensor rot_idx,
    torch::Tensor out,
    int chunk_n,
    int fft_size
);

void native_batch_vacancy_check_cuda(
    torch::Tensor vacancy_flat,   // (max_bins, H) int32
    torch::Tensor density_flat,   // (total_density_len,) int32
    torch::Tensor pair_vac_row,   // (n_pairs,) int32
    torch::Tensor pair_den_off,   // (n_pairs,) int32
    torch::Tensor pair_den_len,   // (n_pairs,) int32
    torch::Tensor out_pass,       // (n_pairs,) int8
    int H
);

void native_compute_vacancy_rows_cuda(
    torch::Tensor grid_states,    // (max_bins, H, W) float32
    torch::Tensor vacancy_flat,   // (max_bins, H) int32
    torch::Tensor grid_idxs,      // (n_rows,) int32 — which bin
    torch::Tensor row_idxs,       // (n_rows,) int32 — which row
    int H, int W, int n_rows
);

inline void native_batch_vacancy_check(
    torch::Tensor vacancy_flat,
    torch::Tensor density_flat,
    torch::Tensor pair_vac_row,
    torch::Tensor pair_den_off,
    torch::Tensor pair_den_len,
    torch::Tensor out_pass,
    int H
) {
    native_batch_vacancy_check_cuda(
        vacancy_flat, density_flat, pair_vac_row, pair_den_off, pair_den_len, out_pass, H
    );
}

inline void native_compute_vacancy_rows(
    torch::Tensor grid_states,
    torch::Tensor vacancy_flat,
    torch::Tensor grid_idxs,
    torch::Tensor row_idxs,
    int H, int W, int n_rows
) {
    native_compute_vacancy_rows_cuda(
        grid_states, vacancy_flat, grid_idxs, row_idxs, H, W, n_rows
    );
}

inline void native_fused_gather_multiply(
    torch::Tensor grid_ffts,
    torch::Tensor part_ffts,
    torch::Tensor grid_idx,
    torch::Tensor rot_idx,
    torch::Tensor out,
    int chunk_n,
    int fft_size
) {
    native_fused_gather_multiply_cuda(
        grid_ffts, part_ffts, grid_idx, rot_idx, out, chunk_n, fft_size
    );
}

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

inline void native_gather_grids(
    torch::Tensor grid_states, torch::Tensor idx, torch::Tensor output,
    int N, int H, int W
) {
    native_gather_grids_cuda(grid_states, idx, output, N, H, W);
}

inline void native_scatter_ffts(
    torch::Tensor input, torch::Tensor idx, torch::Tensor grid_ffts,
    int N, int H, int W2
) {
    native_scatter_ffts_cuda(input, idx, grid_ffts, N, H, W2);
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
        n_rot_total_ = static_cast<int>(rot_h_.size());
        if (static_cast<int>(rot_w_.size()) != n_rot_total_ ||
            static_cast<int>(rot_density_offsets_.size()) != n_rot_total_ + 1) {
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

        // Upload density_flat to GPU once — rotation densities never change.
        if (device_.is_cuda() && !density_flat_.empty()) {
            auto cpu_t = torch::from_blob(
                const_cast<int32_t*>(density_flat_.data()),
                {static_cast<int64_t>(density_flat_.size())},
                torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
            );
            density_flat_gpu_ = cpu_t.to(device_);
        }

        // One MachineWorkspace per machine.  Each holds its own scratch vectors,
        // GPU/CPU workspace tensors, and pinned buffers so machines can be
        // processed concurrently without data races (Stage 3).
        workspaces_.resize(static_cast<size_t>(nb_machines_));
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

    std::vector<double> machine_proc_time_;
    std::vector<double> machine_proc_time_height_;
    std::vector<torch::Tensor> machine_ffts_dense_;
    std::vector<int32_t> rot_flat_offsets_;
    torch::Tensor flat_parts_gpu_;
    torch::Tensor density_flat_gpu_;   // GPU copy of density_flat_, uploaded once in constructor

    struct GpuPlacement {
        int grid_idx;
        int y_start;
        int x_start;
        int flat_offset;
        int ph;
        int pw;
    };

    // Per-machine workspace: all scratch + GPU/pinned-CPU buffers used during
    // a single process_machine_batch call.  One instance per machine (indexed
    // by machine_idx), isolating state across concurrent machine processing.
    struct MachineWorkspace {
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
        // Packed GPU+CPU buffers for apply_gpu_updates (single copy_ instead of 6).
        torch::Tensor ws_gpu_update_packed_;
        torch::Tensor ws_cpu_gpu_update_packed_;
        // Packed GPU+CPU buffers for batch_fft_all_tests index uploads (single copy_ instead of 4).
        torch::Tensor ws_fft_idx_packed_;
        torch::Tensor ws_cpu_fft_idx_packed_;
        // GPU vacancy buffer: (max_total_bins, H) int32.
        // Allocated once per process_machine_batch call; rows updated lazily via dirty flags.
        torch::Tensor ws_vacancy_gpu_;
        // GPU scratch for vacancy-check pair inputs/output (packed: 3*n_pairs int32).
        torch::Tensor ws_vac_pairs_gpu_;      // (3*n_pairs,) int32: vac_row | den_off | den_len
        torch::Tensor ws_vac_out_pass_;       // (n_pairs,) int8
        // Pinned CPU backing for vacancy upload and pair data.
        torch::Tensor ws_cpu_pair_buf_;       // pinned, int32 — packed [vac_row | den_off | den_len]
        torch::Tensor ws_cpu_pass_buf_;       // pinned, int8  — result readback
        // GPU vacancy recompute workspaces (packed grid_idxs + row_idxs).
        torch::Tensor ws_gpu_vacancy_recompute_;     // (2*n_rows,) int32 on GPU
        torch::Tensor ws_cpu_vacancy_recompute_;     // (2*n_rows,) int32 pinned CPU
        // CPU-side scratch for pair building.
        std::vector<int32_t> scratch_pair_vac_row_;
        std::vector<int32_t> scratch_pair_den_off_;
        std::vector<int32_t> scratch_pair_den_len_;
        std::vector<int8_t>  scratch_pass_;
        // Pre-allocated complex64 buffer for fused gather-multiply output.
        // Shape: (CHUNK_SIZE * fft_size,) viewed as (chunk_n, H, W/2+1) per chunk.
        torch::Tensor ws_fused_product_;
        // Pre-allocated float32 buffer for Phase 2 gather output (dirty grids).
        torch::Tensor ws_gather_out_float_;
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
    };

    // One workspace per machine.  Sized in the constructor once nb_machines_ is known.
    std::vector<MachineWorkspace> workspaces_;

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

    torch::Tensor ensure_workspace_cfloat(torch::Tensor& ws, int64_t n) {
        auto opts = torch::TensorOptions().dtype(torch::kComplexFloat).device(device_);
        if (!ws.defined() || ws.scalar_type() != torch::kComplexFloat || ws.device() != device_ || ws.numel() < n) {
            ws = torch::empty({std::max<int64_t>(n, 1)}, opts);
        }
        return ws.narrow(0, 0, n);
    }

    torch::Tensor ensure_workspace_float(torch::Tensor& ws, int64_t n) {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
        if (!ws.defined() || ws.scalar_type() != torch::kFloat32 || ws.device() != device_ || ws.numel() < n) {
            ws = torch::empty({std::max<int64_t>(n, 1)}, opts);
        }
        return ws.narrow(0, 0, n);
    }

    torch::Tensor ensure_workspace_i8(torch::Tensor& ws, int64_t n) {
        auto opts = torch::TensorOptions().dtype(torch::kInt8).device(device_);
        if (!ws.defined() || ws.scalar_type() != torch::kInt8 || ws.device() != device_ || ws.numel() < n) {
            ws = torch::empty({std::max<int64_t>(n, 1)}, opts);
        }
        return ws.narrow(0, 0, n);
    }

    torch::Tensor ensure_cpu_pinned_i8(torch::Tensor& ws, int64_t n) {
        auto opts = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU).pinned_memory(true);
        if (!ws.defined() || ws.numel() < n) {
            ws = torch::empty({std::max<int64_t>(n, 64)}, opts);
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
        FFTBatchResult& out,
        MachineWorkspace& ws
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

        // Pack all 4 index arrays into one pinned buffer and upload in a
        // single copy_ call.  Layout (int64): grid_idx[n] | rot_idx[n] | h[n] | w[n]
        const int64_t nt = static_cast<int64_t>(n_tests);
        auto gpu_fft_packed = ensure_workspace_long(ws.ws_fft_idx_packed_, 4 * nt);
        {
            auto cpu_packed = ensure_cpu_pinned_long(ws.ws_cpu_fft_idx_packed_, 4 * nt);
            int64_t* p = cpu_packed.data_ptr<int64_t>();
            // grid_idx is already int64
            std::memcpy(p, test_grid_indices.data(), static_cast<size_t>(nt) * sizeof(int64_t));
            // rot, h, w need widening from int32→int64
            int64_t* dst_rot = p + nt;
            int64_t* dst_h   = p + 2 * nt;
            int64_t* dst_w   = p + 3 * nt;
            for (int64_t i = 0; i < nt; ++i) {
                dst_rot[i] = static_cast<int64_t>(test_rot_global[static_cast<size_t>(i)]);
                dst_h[i]   = static_cast<int64_t>(test_heights[static_cast<size_t>(i)]);
                dst_w[i]   = static_cast<int64_t>(test_widths[static_cast<size_t>(i)]);
            }
            gpu_fft_packed.copy_(cpu_packed, /*non_blocking=*/true);
        }
        auto all_grid_idx_t = gpu_fft_packed.narrow(0, 0,      nt);
        auto all_rot_idx_t  = gpu_fft_packed.narrow(0, nt,     nt);
        auto all_h_t        = gpu_fft_packed.narrow(0, 2 * nt, nt);
        auto all_w_t        = gpu_fft_packed.narrow(0, 3 * nt, nt);

        constexpr int CHUNK_SIZE = 750;
        for (int chunk_start = 0; chunk_start < n_tests; chunk_start += CHUNK_SIZE) {
            const int chunk_end = std::min(chunk_start + CHUNK_SIZE, n_tests);
            const int chunk_n = chunk_end - chunk_start;

            auto grid_idx_t = all_grid_idx_t.narrow(0, chunk_start, chunk_n);
            auto rot_idx_t = all_rot_idx_t.narrow(0, chunk_start, chunk_n);
            auto part_h_t = all_h_t.narrow(0, chunk_start, chunk_n);
            auto part_w_t = all_w_t.narrow(0, chunk_start, chunk_n);

            const int fft_size = H * ((W / 2) + 1);
            auto product_flat = ensure_workspace_cfloat(ws.ws_fused_product_, static_cast<int64_t>(chunk_n) * fft_size);
            auto product = product_flat.view({chunk_n, H, (W / 2) + 1});
            if (grid_ffts.is_cuda()) {
                native_fused_gather_multiply(
                    grid_ffts, machine_ffts_dense_[static_cast<size_t>(machine_idx)],
                    grid_idx_t, rot_idx_t, product, chunk_n, fft_size
                );
            } else {
                auto batch_grid_ffts = grid_ffts.index_select(0, grid_idx_t);
                auto batch_part_ffts = machine_ffts_dense_[static_cast<size_t>(machine_idx)].index_select(0, rot_idx_t);
                product.copy_(batch_grid_ffts * batch_part_ffts);
            }
            auto overlap_batch = torch::fft::irfft2(product, {H, W}, {-2, -1}, "forward");

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
                    auto out_has_t = ensure_workspace_i32(ws.ws_sel_has_i32_, chunk_n);
                    auto out_row_t = ensure_workspace_i32(ws.ws_sel_row_i32_, chunk_n);
                    auto out_col_t = ensure_workspace_i32(ws.ws_sel_col_i32_, chunk_n);
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

    template <bool Overwrite>
    void add_part_to_bin_impl(
        BinStateNative& bin,
        int x,
        int y,
        int rotg,
        int part_idx,
        int machine_idx,
        int machine_width
    ) {
        const int ph = rot_h_[static_cast<size_t>(rotg)];
        const int y_start = y - ph + 1;
        // No CPU grid write — grid_states (GPU) is authoritative.
        // GPU is updated via apply_gpu_updates after all placements.

        bin.area += part_area_[static_cast<size_t>(part_idx)];
        bin.min_occupied_row = std::min(bin.min_occupied_row, y_start);
        bin.max_occupied_row = std::max(bin.max_occupied_row, y);
        if constexpr (Overwrite) {
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
#if PROFILE_CPU_HOTSPOTS
        auto _prof_t0 = PROF_NOW();
#endif
        if (overwrite) {
            add_part_to_bin_impl<true>(bin, x, y, rotg, part_idx, machine_idx, machine_width);
        } else {
            add_part_to_bin_impl<false>(bin, x, y, rotg, part_idx, machine_idx, machine_width);
        }
#if PROFILE_CPU_HOTSPOTS
        g_profile.aptb_ns += PROF_NS_SINCE(_prof_t0);
        g_profile.aptb_count += 1;
#endif
    }

    std::vector<double> process_machine_batch(
        const float* chrom_ptr,
        int num_solutions,
        int machine_idx
    ) {
        MachineWorkspace& ws = workspaces_[static_cast<size_t>(machine_idx)];
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

#if PROFILE_CPU_HOTSPOTS
        auto _prof_decode_t0 = PROF_NOW();
#endif
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
#if PROFILE_CPU_HOTSPOTS
        g_profile.decode_seq_ns += PROF_NS_SINCE(_prof_decode_t0);
#endif

        // torch::empty instead of torch::zeros — Phase 6 zeroes each bin slot
        // via index_fill_ before use, making the upfront bulk memset redundant.
        auto grid_states = torch::empty(
            {max_total_bins, H, W},
            torch::TensorOptions().dtype(torch::kFloat32).device(device_)
        );
        auto grid_ffts = torch::empty(
            {max_total_bins, H, (W / 2) + 1},
            torch::TensorOptions().dtype(torch::kComplexFloat).device(device_)
        );
        // Vacancy GPU buffer: one row per open bin slot, length H each.
        // Filled lazily via dirty flags before Phase 3 each wave.
        ws.ws_vacancy_gpu_ = torch::empty(
            {max_total_bins, H},
            torch::TensorOptions().dtype(torch::kInt32).device(device_)
        );
        auto row_idx = torch::arange(H, torch::TensorOptions().dtype(torch::kLong).device(device_)).view({1, H, 1});
        auto col_idx = torch::arange(W, torch::TensorOptions().dtype(torch::kLong).device(device_)).view({1, 1, W});
        auto row_idx_f = row_idx.to(torch::kFloat);
        auto col_idx_f = col_idx.to(torch::kFloat);
        auto neg_inf = torch::tensor(-1e9f, torch::TensorOptions().dtype(torch::kFloat32).device(device_));

        const int max_waves = max_seq_len * 3;
        for (int wave = 0; wave < max_waves; ++wave) {
            ws.scratch_active_.clear();
            ws.scratch_active_.reserve(static_cast<size_t>(num_solutions));
            for (int i = 0; i < num_solutions; ++i) {
                auto& ctx = contexts[static_cast<size_t>(i)];
                if (!ctx.is_done && ctx.is_feasible) {
                    ws.scratch_active_.push_back(i);
                }
            }
            if (ws.scratch_active_.empty()) {
                break;
            }
            process_wave(ws.scratch_active_, contexts, machine_idx, max_total_bins, H, W,
                         grid_states, grid_ffts, row_idx, col_idx, row_idx_f, col_idx_f, neg_inf, ws);
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

    // Evaluate vacancy checks for all (bin, rotation) pairs on GPU.
    // Returns a bool pass/fail for each pair via ws.scratch_pass_.
    void run_gpu_vacancy_check(int n_pairs, int H, MachineWorkspace& ws) {
        ws.scratch_pass_.assign(static_cast<size_t>(n_pairs), 0);
        if (n_pairs == 0) return;
        const int64_t np = static_cast<int64_t>(n_pairs);

        // Transfer pair arrays CPU→GPU in one packed buffer.
        // Layout: vac_row[n] | den_off[n] | den_len[n]
        auto cpu_pairs = ensure_cpu_pinned_i32(ws.ws_cpu_pair_buf_, 3 * np);
        int32_t* pp = cpu_pairs.data_ptr<int32_t>();
        std::memcpy(pp,           ws.scratch_pair_vac_row_.data(), static_cast<size_t>(np) * sizeof(int32_t));
        std::memcpy(pp + np,      ws.scratch_pair_den_off_.data(), static_cast<size_t>(np) * sizeof(int32_t));
        std::memcpy(pp + 2 * np,  ws.scratch_pair_den_len_.data(), static_cast<size_t>(np) * sizeof(int32_t));

        auto gpu_pairs = ensure_workspace_i32(ws.ws_vac_pairs_gpu_, 3 * np);
#if PROFILE_CPU_HOTSPOTS
        auto _prof_vup_t0 = PROF_NOW();
#endif
        gpu_pairs.copy_(cpu_pairs, /*non_blocking=*/true);  // pinned→GPU on default stream; ordered before kernel
#if PROFILE_CPU_HOTSPOTS
        g_profile.vac_upload_sync_ns += PROF_NS_SINCE(_prof_vup_t0);
#endif
        auto gpu_vac_row = gpu_pairs.narrow(0, 0,       np);
        auto gpu_den_off = gpu_pairs.narrow(0, np,      np);
        auto gpu_den_len = gpu_pairs.narrow(0, 2 * np,  np);

        auto gpu_out = ensure_workspace_i8(ws.ws_vac_out_pass_, np);
        native_batch_vacancy_check(
            ws.ws_vacancy_gpu_, density_flat_gpu_,
            gpu_vac_row, gpu_den_off, gpu_den_len,
            gpu_out, H
        );

        // Readback result (small: one int8 per pair).
        auto cpu_pass = ensure_cpu_pinned_i8(ws.ws_cpu_pass_buf_, np);
#if PROFILE_CPU_HOTSPOTS
        auto _prof_vrb_t0 = PROF_NOW();
#endif
        cpu_pass.copy_(gpu_out, /*non_blocking=*/false);  // sync
#if PROFILE_CPU_HOTSPOTS
        g_profile.vac_readback_sync_ns += PROF_NS_SINCE(_prof_vrb_t0);
        g_profile.vac_call_count += 1;
#endif
        const int8_t* pass_ptr = cpu_pass.data_ptr<int8_t>();
        ws.scratch_pass_.assign(pass_ptr, pass_ptr + n_pairs);
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
        const torch::Tensor& neg_inf,
        MachineWorkspace& ws
    ) {
#if PROFILE_CPU_HOTSPOTS
        auto _prof_wave_t0 = PROF_NOW();
        g_profile.wave_count += 1;
#endif
        ws.scratch_ctx_global_.clear();
        ws.scratch_part_idx_local_.clear();
        ws.scratch_ctx_global_.reserve(active_indices.size());
        ws.scratch_part_idx_local_.reserve(active_indices.size());

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
            ws.scratch_ctx_global_.push_back(gidx);
            ws.scratch_part_idx_local_.push_back(part_idx);
        }

        const int n_contexts = static_cast<int>(ws.scratch_ctx_global_.size());
        if (n_contexts == 0) {
            return;
        }

        ws.scratch_invalid_grid_indices_.clear();
        ws.scratch_invalid_bins_.clear();
        for (int lc = 0; lc < n_contexts; ++lc) {
            auto& ctx = contexts[static_cast<size_t>(ws.scratch_ctx_global_[static_cast<size_t>(lc)])];
            for (auto& b : ctx.open_bins) {
                if (!b.grid_fft_valid) {
                    ws.scratch_invalid_grid_indices_.push_back(static_cast<int64_t>(b.grid_state_idx));
                    ws.scratch_invalid_bins_.push_back(&b);
                }
            }
        }
#if PROFILE_CPU_HOTSPOTS
        g_profile.phase1_active_ns += PROF_NS_SINCE(_prof_wave_t0);
        auto _prof_rfft2_t0 = PROF_NOW();
#endif
        if (!ws.scratch_invalid_grid_indices_.empty()) {
            const int64_t N = static_cast<int64_t>(ws.scratch_invalid_grid_indices_.size());
            auto idx_t = load_workspace_long_from_i64(
                ws.ws_wave_idx_long_, ws.ws_cpu_wave_idx_,
                ws.scratch_invalid_grid_indices_, N
            );
            torch::Tensor batch_grids;
            if (grid_states.is_cuda()) {
                // Custom gather kernel → pre-allocated workspace (N, H, W) float.
                // Avoids the ~37ms/gen P50 spent in aten::index_select +
                // per-wave allocation churn.
                const int H_ = static_cast<int>(grid_states.size(1));
                const int W_ = static_cast<int>(grid_states.size(2));
                auto gather_flat = ensure_workspace_float(
                    ws.ws_gather_out_float_, N * H_ * W_
                );
                native_gather_grids(grid_states, idx_t, gather_flat,
                                    static_cast<int>(N), H_, W_);
                batch_grids = gather_flat.view({N, H_, W_});
            } else {
                batch_grids = grid_states.index_select(0, idx_t);
            }
            auto batch_ffts = torch::fft::rfft2(batch_grids);
            if (grid_ffts.is_cuda()) {
                const int H_ = static_cast<int>(grid_ffts.size(1));
                const int W2_ = static_cast<int>(grid_ffts.size(2));
                native_scatter_ffts(batch_ffts, idx_t, grid_ffts,
                                    static_cast<int>(N), H_, W2_);
            } else {
                grid_ffts.index_copy_(0, idx_t, batch_ffts);
            }
            for (auto* b : ws.scratch_invalid_bins_) {
                b->grid_fft_valid = true;
            }
        }
#if PROFILE_CPU_HOTSPOTS
        g_profile.phase2_rfft2_ns += PROF_NS_SINCE(_prof_rfft2_t0);
#endif

        // Vacancy is now maintained on GPU — no flush needed.

        ws.scratch_ctx_first_valid_bin_.assign(static_cast<size_t>(n_contexts), -1);
        auto& p1 = ws.scratch_p1_;
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
        // ── Phase 3 p1: GPU vacancy check ──────────────────────────────────────
        // Pass A: collect all (bin, rotation) pairs that pass area+dimension
        // filters, without vacancy check.  Record their bin grid_state_idx,
        // density offset+length, and structural metadata for later FFT collection.
        // Each pair is tagged with (lc, bidx, rg).

        struct P1Pair {
            int lc, bidx, rg;
            double p_area;
        };
        std::vector<P1Pair> p1_pairs;
        p1_pairs.reserve(static_cast<size_t>(n_contexts) * 4);

        ws.scratch_pair_vac_row_.clear();
        ws.scratch_pair_den_off_.clear();
        ws.scratch_pair_den_len_.clear();

#if PROFILE_CPU_HOTSPOTS
        auto _prof_passA_t0 = PROF_NOW();
#endif
        for (int lc = 0; lc < n_contexts; ++lc) {
            auto& ctx = contexts[static_cast<size_t>(ws.scratch_ctx_global_[static_cast<size_t>(lc)])];
            const int part_idx = ws.scratch_part_idx_local_[static_cast<size_t>(lc)];
            const double p_area = part_area_[static_cast<size_t>(part_idx)];
            const int nrot = part_nrot_[static_cast<size_t>(part_idx)];
            const int r0 = part_rot_offsets_[static_cast<size_t>(part_idx)];

            for (int bidx = 0; bidx < static_cast<int>(ctx.open_bins.size()); ++bidx) {
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
                    p1_pairs.push_back({lc, bidx, rg, p_area});
                    ws.scratch_pair_vac_row_.push_back(static_cast<int32_t>(b.grid_state_idx));
                    ws.scratch_pair_den_off_.push_back(static_cast<int32_t>(d0));
                    ws.scratch_pair_den_len_.push_back(static_cast<int32_t>(d1 - d0));
                }
            }
        }
#if PROFILE_CPU_HOTSPOTS
        g_profile.p3_pass_a_ns += PROF_NS_SINCE(_prof_passA_t0);
#endif

        // Pass B: GPU vacancy check for all pairs.
        const int n_p1_pairs = static_cast<int>(p1_pairs.size());
#if PROFILE_CPU_HOTSPOTS
        auto _prof_passB_t0 = PROF_NOW();
#endif
        run_gpu_vacancy_check(n_p1_pairs, H, ws);
#if PROFILE_CPU_HOTSPOTS
        g_profile.p3_pass_b_wait_ns += PROF_NS_SINCE(_prof_passB_t0);
#endif

        // Pass C: reconstruct p1 test list using mask, preserving first-valid-bin semantics.
#if PROFILE_CPU_HOTSPOTS
        auto _prof_passC_t0 = PROF_NOW();
#endif
        for (int lc = 0; lc < n_contexts; ++lc) {
            auto& ctx = contexts[static_cast<size_t>(ws.scratch_ctx_global_[static_cast<size_t>(lc)])];
            const int part_idx = ws.scratch_part_idx_local_[static_cast<size_t>(lc)];
            const double p_area = part_area_[static_cast<size_t>(part_idx)];
            ws.scratch_ctx_first_valid_bin_[static_cast<size_t>(lc)] = -1;
        }
        for (int pi = 0; pi < n_p1_pairs; ++pi) {
            if (!ws.scratch_pass_[static_cast<size_t>(pi)]) continue;
            const auto& pair = p1_pairs[static_cast<size_t>(pi)];
            const int lc = pair.lc;
            const int bidx = pair.bidx;
            const int rg = pair.rg;
            const double p_area = pair.p_area;
            // Enforce first-valid-bin: only add tests for bidx <= current first_valid
            const int first_valid = ws.scratch_ctx_first_valid_bin_[static_cast<size_t>(lc)];
            if (first_valid >= 0 && bidx > first_valid) {
                continue;  // already found a better bin
            }
            auto& ctx = contexts[static_cast<size_t>(ws.scratch_ctx_global_[static_cast<size_t>(lc)])];
            auto& b = ctx.open_bins[static_cast<size_t>(bidx)];
            if (first_valid < 0 || bidx < first_valid) {
                // New first-valid bin for this context: drop any p1 tests from later bins
                // that may have slipped in from previous pairs (shouldn't happen since we
                // iterate pairs in (lc, bidx, rot) order, but guard for safety).
                ws.scratch_ctx_first_valid_bin_[static_cast<size_t>(lc)] = bidx;
            }
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
#if PROFILE_CPU_HOTSPOTS
        g_profile.p3_pass_c_ns += PROF_NS_SINCE(_prof_passC_t0);
        auto _prof_p4p1_t0 = PROF_NOW();
#endif

        batch_fft_all_tests(
            static_cast<int>(p1.grid_indices.size()),
            p1.grid_indices, p1.rot_global, p1.heights, p1.widths,
            p1.bin_indices, p1.enclosure_lengths, p1.bin_areas, p1.part_areas,
            machine_idx, grid_ffts, H, W, row_idx, col_idx, row_idx_f, col_idx_f, neg_inf, ws.scratch_fft_p1_, ws
        );
#if PROFILE_CPU_HOTSPOTS
        g_profile.phase4_p1_fft_ns += PROF_NS_SINCE(_prof_p4p1_t0);
#endif

        ws.scratch_ctx_p1_hit_.assign(static_cast<size_t>(n_contexts), 0);
        for (size_t i = 0; i < p1.ctx_local.size(); ++i) {
            if (ws.scratch_fft_p1_.has_result[i]) {
                const int lc = p1.ctx_local[i];
                ws.scratch_ctx_p1_hit_[static_cast<size_t>(lc)] = 1;
            }
        }

#if PROFILE_CPU_HOTSPOTS
        auto _prof_p3p2_t0 = PROF_NOW();
#endif
        // ── Phase 3 p2: GPU vacancy check ──────────────────────────────────────
        auto& p2 = ws.scratch_p2_;
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

        struct P2Pair {
            int lc, bidx, rg;
            double p_area;
        };
        std::vector<P2Pair> p2_pairs;
        p2_pairs.reserve(32);

        ws.scratch_pair_vac_row_.clear();
        ws.scratch_pair_den_off_.clear();
        ws.scratch_pair_den_len_.clear();

        for (int lc = 0; lc < n_contexts; ++lc) {
            if (ws.scratch_ctx_p1_hit_[static_cast<size_t>(lc)]) {
                continue;
            }
            auto& ctx = contexts[static_cast<size_t>(ws.scratch_ctx_global_[static_cast<size_t>(lc)])];
            const int part_idx = ws.scratch_part_idx_local_[static_cast<size_t>(lc)];
            const double p_area = part_area_[static_cast<size_t>(part_idx)];
            const int nrot = part_nrot_[static_cast<size_t>(part_idx)];
            const int r0 = part_rot_offsets_[static_cast<size_t>(part_idx)];
            const int first_valid = ws.scratch_ctx_first_valid_bin_[static_cast<size_t>(lc)];

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
                    p2_pairs.push_back({lc, bidx, rg, p_area});
                    ws.scratch_pair_vac_row_.push_back(static_cast<int32_t>(b.grid_state_idx));
                    ws.scratch_pair_den_off_.push_back(static_cast<int32_t>(d0));
                    ws.scratch_pair_den_len_.push_back(static_cast<int32_t>(d1 - d0));
                }
            }
        }

        const int n_p2_pairs = static_cast<int>(p2_pairs.size());
        run_gpu_vacancy_check(n_p2_pairs, H, ws);

        for (int pi = 0; pi < n_p2_pairs; ++pi) {
            if (!ws.scratch_pass_[static_cast<size_t>(pi)]) continue;
            const auto& pair = p2_pairs[static_cast<size_t>(pi)];
            const int lc = pair.lc;
            const int bidx = pair.bidx;
            const int rg = pair.rg;
            const double p_area = pair.p_area;
            auto& ctx = contexts[static_cast<size_t>(ws.scratch_ctx_global_[static_cast<size_t>(lc)])];
            auto& b = ctx.open_bins[static_cast<size_t>(bidx)];
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

#if PROFILE_CPU_HOTSPOTS
        g_profile.phase3_p2_collect_ns += PROF_NS_SINCE(_prof_p3p2_t0);
        auto _prof_p4p2_t0 = PROF_NOW();
#endif
        batch_fft_all_tests(
            static_cast<int>(p2.grid_indices.size()),
            p2.grid_indices, p2.rot_global, p2.heights, p2.widths,
            p2.bin_indices, p2.enclosure_lengths, p2.bin_areas, p2.part_areas,
            machine_idx, grid_ffts, H, W, row_idx, col_idx, row_idx_f, col_idx_f, neg_inf, ws.scratch_fft_p2_, ws
        );
#if PROFILE_CPU_HOTSPOTS
        g_profile.phase4_p2_fft_ns += PROF_NS_SINCE(_prof_p4p2_t0);
        auto _prof_p5_t0 = PROF_NOW();
#endif

        auto& test_ctx_local = ws.scratch_test_ctx_local_;
        auto& test_bin_local = ws.scratch_test_bin_local_;
        auto& test_rot_global = ws.scratch_test_rot_global_;
        auto& placement_has = ws.scratch_placement_has_;
        auto& placement_cols = ws.scratch_placement_cols_;
        auto& placement_rows = ws.scratch_placement_rows_;
        auto& sc_bin_indices = ws.scratch_sc_bin_indices_;
        auto& sc_densities = ws.scratch_sc_densities_;
        auto& sc_rows = ws.scratch_sc_rows_;
        auto& sc_cols = ws.scratch_sc_cols_;
        auto& sc_valid = ws.scratch_sc_valid_;

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
        append_tests(p1, ws.scratch_fft_p1_);
        append_tests(p2, ws.scratch_fft_p2_);

        auto& best_ti = ws.scratch_best_ti_;
        auto& has_best = ws.scratch_has_best_;
        auto& best_neg_bin = ws.scratch_best_neg_bin_;
        auto& best_density = ws.scratch_best_density_;
        auto& best_row = ws.scratch_best_row_;
        auto& best_neg_col = ws.scratch_best_neg_col_;
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

        auto& place_ctx_local = ws.scratch_place_ctx_local_;
        auto& place_ti = ws.scratch_place_ti_;
        auto& place_rows = ws.scratch_place_rows_;
        auto& place_cols = ws.scratch_place_cols_;
        auto& newbin_ctx_local = ws.scratch_newbin_ctx_local_;
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

        auto& gpu_updates = ws.scratch_gpu_updates_;
        gpu_updates.clear();
        gpu_updates.reserve(place_ctx_local.size());

        for (size_t i = 0; i < place_ctx_local.size(); ++i) {
            const int lc = place_ctx_local[i];
            const int ctx_g = ws.scratch_ctx_global_[static_cast<size_t>(lc)];
            const int part_idx = ws.scratch_part_idx_local_[static_cast<size_t>(lc)];
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

#if PROFILE_CPU_HOTSPOTS
        g_profile.phase5_select_ns += PROF_NS_SINCE(_prof_p5_t0);
        auto _prof_gpu_upd_t0 = PROF_NOW();
#endif
        if (!gpu_updates.empty()) {
            apply_gpu_updates(grid_states, gpu_updates, H, W, ws);
            recompute_vacancy_gpu(grid_states, gpu_updates, H, W, ws);
        }
#if PROFILE_CPU_HOTSPOTS
        g_profile.phase5_gpu_update_ns += PROF_NS_SINCE(_prof_gpu_upd_t0);
#endif

        auto& phase6_gpu_updates = ws.scratch_phase6_gpu_updates_;
        auto& phase6_grid_indices = ws.scratch_phase6_grid_indices_;
        phase6_gpu_updates.clear();
        phase6_grid_indices.clear();
        phase6_gpu_updates.reserve(newbin_ctx_local.size());
        phase6_grid_indices.reserve(newbin_ctx_local.size());
#if PROFILE_CPU_HOTSPOTS
        auto _prof_phase6_t0 = PROF_NOW();
#endif
        for (int lc : newbin_ctx_local) {
            const int ctx_g = ws.scratch_ctx_global_[static_cast<size_t>(lc)];
            const int part_idx = ws.scratch_part_idx_local_[static_cast<size_t>(lc)];
            auto& ctx = contexts[static_cast<size_t>(ctx_g)];

            const int grid_idx = ctx.next_grid_idx;
            ctx.next_grid_idx += 1;
            if (grid_idx < 0 || grid_idx >= max_total_bins) {
                ctx.is_feasible = false;
                continue;
            }

#if PROFILE_CPU_HOTSPOTS
            auto _prof_ctor_t0 = PROF_NOW();
#endif
            BinStateNative new_bin;
            // No CPU grid allocation needed — GPU grid_states is authoritative.
            new_bin.bin_idx = static_cast<int>(ctx.open_bins.size());
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
#if PROFILE_CPU_HOTSPOTS
            g_profile.binstate_ctor_ns += PROF_NS_SINCE(_prof_ctor_t0);
            g_profile.binstate_count += 1;
#endif

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
#if PROFILE_CPU_HOTSPOTS
        g_profile.phase6_loop_ns += PROF_NS_SINCE(_prof_phase6_t0);
#endif

        if (!phase6_gpu_updates.empty()) {
            auto idx_t = load_workspace_long_from_i64(
                ws.ws_wave_idx_long_, ws.ws_cpu_wave_idx_,
                phase6_grid_indices,
                static_cast<int64_t>(phase6_grid_indices.size())
            );
            grid_states.index_fill_(0, idx_t, 0.0);
            // Initialize vacancy for new bins: all rows = W (empty grid).
            ws.ws_vacancy_gpu_.index_fill_(0, idx_t, static_cast<int32_t>(W));
            apply_gpu_updates(grid_states, phase6_gpu_updates, H, W, ws);
            // Recompute vacancy for the rows where parts were placed.
            recompute_vacancy_gpu(grid_states, phase6_gpu_updates, H, W, ws);
        }
#if PROFILE_CPU_HOTSPOTS
        g_profile.total_wave_ns += PROF_NS_SINCE(_prof_wave_t0);
#endif
    }

    // Recompute vacancy rows on GPU for affected (grid_idx, row) pairs.
    // Called after apply_gpu_updates to keep ws.ws_vacancy_gpu_ in sync.
    template <typename PlacementVec>
    void recompute_vacancy_gpu(
        torch::Tensor& grid_states,
        const PlacementVec& updates,
        int H, int W,
        MachineWorkspace& ws
    ) {
        int total_rows = 0;
        for (const auto& u : updates) {
            total_rows += u.ph;
        }
        if (total_rows <= 0) return;

        const int64_t n = static_cast<int64_t>(total_rows);
        auto cpu_packed = ensure_cpu_pinned_i32(ws.ws_cpu_vacancy_recompute_, 2 * n);
        int32_t* p = cpu_packed.data_ptr<int32_t>();
        int off = 0;
        for (const auto& u : updates) {
            for (int r = 0; r < u.ph; ++r) {
                p[off] = static_cast<int32_t>(u.grid_idx);
                p[n + off] = static_cast<int32_t>(u.y_start + r);
                ++off;
            }
        }

        auto gpu_packed = ensure_workspace_i32(ws.ws_gpu_vacancy_recompute_, 2 * n);
        gpu_packed.copy_(cpu_packed, /*non_blocking=*/true);
        auto grid_idxs_t = gpu_packed.narrow(0, 0, n);
        auto row_idxs_t  = gpu_packed.narrow(0, n, n);

        native_compute_vacancy_rows(
            grid_states, ws.ws_vacancy_gpu_, grid_idxs_t, row_idxs_t, H, W, total_rows
        );
    }

    template <typename PlacementVec>
    void apply_gpu_updates(
        torch::Tensor& grid_states,
        const PlacementVec& updates,
        int H,
        int W,
        MachineWorkspace& ws
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

        auto& cell_offsets = ws.scratch_cell_offsets_;
        auto& grid_idxs = ws.scratch_grid_idxs_;
        auto& y_starts = ws.scratch_y_starts_;
        auto& x_starts = ws.scratch_x_starts_;
        auto& part_widths = ws.scratch_part_widths_;
        auto& part_offsets = ws.scratch_part_offsets_;
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

        // Pack all 6 parameter arrays into one contiguous buffer and upload
        // in a single pinned→GPU copy_ call.  Layout (int32):
        //   cell_offsets[n+1] | grid_idxs[n] | y_starts[n] | x_starts[n] | part_widths[n] | part_offsets[n]
        const int64_t packed_len = static_cast<int64_t>(n + 1) + 5 * static_cast<int64_t>(n);
        auto gpu_packed = ensure_workspace_i32(ws.ws_gpu_update_packed_, packed_len);
        {
            auto cpu_packed = ensure_cpu_pinned_i32(ws.ws_cpu_gpu_update_packed_, packed_len);
            int32_t* p = cpu_packed.data_ptr<int32_t>();
            int64_t off = 0;
            std::memcpy(p + off, cell_offsets.data(), static_cast<size_t>(n + 1) * sizeof(int32_t)); off += n + 1;
            std::memcpy(p + off, grid_idxs.data(),    static_cast<size_t>(n)     * sizeof(int32_t)); off += n;
            std::memcpy(p + off, y_starts.data(),      static_cast<size_t>(n)     * sizeof(int32_t)); off += n;
            std::memcpy(p + off, x_starts.data(),      static_cast<size_t>(n)     * sizeof(int32_t)); off += n;
            std::memcpy(p + off, part_widths.data(),   static_cast<size_t>(n)     * sizeof(int32_t)); off += n;
            std::memcpy(p + off, part_offsets.data(),  static_cast<size_t>(n)     * sizeof(int32_t));
            gpu_packed.copy_(cpu_packed, /*non_blocking=*/true);
        }
        int64_t off = 0;
        auto cell_offsets_t = gpu_packed.narrow(0, off, n + 1); off += n + 1;
        auto grid_idxs_t    = gpu_packed.narrow(0, off, n);     off += n;
        auto y_starts_t     = gpu_packed.narrow(0, off, n);     off += n;
        auto x_starts_t     = gpu_packed.narrow(0, off, n);     off += n;
        auto part_widths_t  = gpu_packed.narrow(0, off, n);     off += n;
        auto part_offsets_t = gpu_packed.narrow(0, off, n);

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
#if PROFILE_CPU_HOTSPOTS
    m.def("get_profile_summary", []() { return g_profile.summary(); });
    m.def("reset_profile", []() { g_profile.reset(); });
#else
    m.def("get_profile_summary", []() { return std::string("(profiling disabled)"); });
    m.def("reset_profile", []() {});
#endif
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

// Gather grids by index into a compact (N, H, W) float buffer.
// Replaces grid_states.index_select(0, idx) which dispatches a general
// indexSelectLargeIndex kernel; this specialized version is pure coalesced
// reads/writes with no dtype dispatch or broadcasting logic.
__global__ void _native_gather_grids_kernel(
    const float* __restrict__ grid_states,
    const int64_t* __restrict__ idx,
    float* __restrict__ output,
    int N, int H, int W
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int HW = H * W;
    int total = N * HW;
    if (tid >= total) return;
    int i = tid / HW;
    int rem = tid - i * HW;
    int src_bin = (int)idx[i];
    output[tid] = grid_states[(long long)src_bin * HW + rem];
}

// Scatter complex FFT results into grid_ffts[idx[i]].
// Replaces grid_ffts.index_copy_(0, idx, batch_ffts).
__global__ void _native_scatter_ffts_kernel(
    const float2* __restrict__ input,
    const int64_t* __restrict__ idx,
    float2* __restrict__ grid_ffts,
    int N, int H, int W2
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int HW2 = H * W2;
    int total = N * HW2;
    if (tid >= total) return;
    int i = tid / HW2;
    int rem = tid - i * HW2;
    int dst_bin = (int)idx[i];
    grid_ffts[(long long)dst_bin * HW2 + rem] = input[tid];
}

void native_gather_grids_cuda(
    torch::Tensor grid_states,
    torch::Tensor idx,
    torch::Tensor output,
    int N, int H, int W
) {
    if (N <= 0) return;
    const int total = N * H * W;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    _native_gather_grids_kernel<<<blocks, threads>>>(
        grid_states.data_ptr<float>(),
        idx.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        N, H, W
    );
}

void native_scatter_ffts_cuda(
    torch::Tensor input,
    torch::Tensor idx,
    torch::Tensor grid_ffts,
    int N, int H, int W2
) {
    if (N <= 0) return;
    const int total = N * H * W2;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    _native_scatter_ffts_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float2*>(input.data_ptr<c10::complex<float>>()),
        idx.data_ptr<int64_t>(),
        reinterpret_cast<float2*>(grid_ffts.data_ptr<c10::complex<float>>()),
        N, H, W2
    );
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

// ── Fused gather-multiply kernel ─────────────────────────────────────────────
// Replaces: index_select(grid_ffts, grid_idx) * index_select(part_ffts, rot_idx)
// Reads directly from the source tensors, multiplies complex values in-register,
// writes the product once.  Eliminates 2 intermediate tensor allocations and
// reduces memory traffic from ~7 passes to 3.

__global__ void _fused_gather_multiply_kernel(
    const float2* __restrict__ grid_ffts,   // (N_grid, fft_size)
    const float2* __restrict__ part_ffts,   // (N_part, fft_size)
    const int64_t* __restrict__ grid_idx,   // (chunk_n,)
    const int64_t* __restrict__ rot_idx,    // (chunk_n,)
    float2* __restrict__ out,               // (chunk_n, fft_size)
    int fft_size
) {
    int test = blockIdx.y;
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= fft_size) return;

    long long g_off = (long long)grid_idx[test] * fft_size + elem;
    long long p_off = (long long)rot_idx[test]  * fft_size + elem;

    float2 g = __ldg(&grid_ffts[g_off]);
    float2 p = __ldg(&part_ffts[p_off]);

    // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    float2 r;
    r.x = g.x * p.x - g.y * p.y;
    r.y = g.x * p.y + g.y * p.x;

    out[(long long)test * fft_size + elem] = r;
}

void native_fused_gather_multiply_cuda(
    torch::Tensor grid_ffts,
    torch::Tensor part_ffts,
    torch::Tensor grid_idx,
    torch::Tensor rot_idx,
    torch::Tensor out,
    int chunk_n,
    int fft_size
) {
    if (chunk_n <= 0 || fft_size <= 0) return;
    const int threads = 256;
    dim3 blocks((fft_size + threads - 1) / threads, chunk_n);
    _fused_gather_multiply_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float2*>(grid_ffts.data_ptr()),
        reinterpret_cast<const float2*>(part_ffts.data_ptr()),
        grid_idx.data_ptr<int64_t>(),
        rot_idx.data_ptr<int64_t>(),
        reinterpret_cast<float2*>(out.data_ptr()),
        fft_size
    );
}

// ── Batch vacancy check kernel ────────────────────────────────────────────────
// One block per (bin, rotation) pair.
// Each thread checks one starting position of the density array against the
// vacancy vector.  Shared-memory OR-reduction: if any position passes, the
// pair passes.
//
// vacancy_flat: (max_bins, H) row-major int32
// density_flat: (total_density_len,) int32
// pair_vac_row: (n_pairs,) int32 — row index into vacancy_flat
// pair_den_off: (n_pairs,) int32 — offset into density_flat
// pair_den_len: (n_pairs,) int32 — density array length (= part height)
// out_pass:     (n_pairs,) int8  — 1 if any window fits, else 0

__global__ void _batch_vacancy_check_kernel(
    const int* __restrict__ vacancy_flat,
    const int* __restrict__ density_flat,
    const int* __restrict__ pair_vac_row,
    const int* __restrict__ pair_den_off,
    const int* __restrict__ pair_den_len,
    int8_t* __restrict__ out_pass,
    int H
) {
    const int pair_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const int vac_row = pair_vac_row[pair_idx];
    const int den_off = pair_den_off[pair_idx];
    const int den_len = pair_den_len[pair_idx];
    const int max_start = H - den_len;

    const int* vac = vacancy_flat + (long long)vac_row * H;
    const int* den = density_flat + den_off;

    // Each thread checks one or more starting positions.
    int found = 0;
    for (int start = tid; start <= max_start; start += stride) {
        int ok = 1;
        for (int i = 0; i < den_len; ++i) {
            if (vac[start + i] < den[i]) { ok = 0; break; }
        }
        if (ok) { found = 1; break; }
    }

    // Shared memory OR-reduction across threads.
    extern __shared__ int s_found[];
    s_found[tid] = found;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (s_found[tid + offset]) s_found[tid] = 1;
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_pass[pair_idx] = (int8_t)s_found[0];
    }
}

void native_batch_vacancy_check_cuda(
    torch::Tensor vacancy_flat,
    torch::Tensor density_flat,
    torch::Tensor pair_vac_row,
    torch::Tensor pair_den_off,
    torch::Tensor pair_den_len,
    torch::Tensor out_pass,
    int H
) {
    const int n_pairs = (int)pair_vac_row.size(0);
    if (n_pairs <= 0) return;
    // Use 64 threads per block — density arrays are short (≤ H = 300),
    // so max_start = H - den_len ≤ 298; 64 threads covers all windows
    // in at most ceil(299/64) = 5 iterations.
    const int threads = 64;
    const size_t shmem = (size_t)threads * sizeof(int);
    _batch_vacancy_check_kernel<<<n_pairs, threads, shmem>>>(
        vacancy_flat.data_ptr<int>(),
        density_flat.data_ptr<int>(),
        pair_vac_row.data_ptr<int>(),
        pair_den_off.data_ptr<int>(),
        pair_den_len.data_ptr<int>(),
        out_pass.data_ptr<int8_t>(),
        H
    );
}

// ── Vacancy row recompute kernel ─────────────────────────────────────────────
// For each (grid_idx, row_idx) pair, scan grid_states[grid_idx][row_idx][0..W-1]
// and compute max consecutive zeros.  One thread per row.
__global__ void _compute_vacancy_rows_kernel(
    const float* __restrict__ grid_flat,   // (max_bins * H * W)
    int* __restrict__ vacancy_flat,         // (max_bins * H)
    const int* __restrict__ grid_idxs,      // (n_rows,) which bin
    const int* __restrict__ row_idxs,       // (n_rows,) which row within the bin
    int H, int W, int n_rows
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n_rows) return;

    int grid_idx = grid_idxs[tid];
    int row_idx  = row_idxs[tid];

    const float* row = grid_flat + (long long)grid_idx * H * W + (long long)row_idx * W;

    int max_zeros = 0;
    int cur_zeros = 0;
    for (int j = 0; j < W; ++j) {
        if (row[j] == 0.0f) {
            cur_zeros++;
            if (cur_zeros > max_zeros) max_zeros = cur_zeros;
        } else {
            cur_zeros = 0;
        }
    }

    vacancy_flat[(long long)grid_idx * H + row_idx] = max_zeros;
}

void native_compute_vacancy_rows_cuda(
    torch::Tensor grid_states,
    torch::Tensor vacancy_flat,
    torch::Tensor grid_idxs,
    torch::Tensor row_idxs,
    int H, int W, int n_rows
) {
    if (n_rows <= 0) return;
    const int threads = 256;
    const int blocks = (n_rows + threads - 1) / threads;
    _compute_vacancy_rows_kernel<<<blocks, threads>>>(
        grid_states.data_ptr<float>(),
        vacancy_flat.data_ptr<int>(),
        grid_idxs.data_ptr<int>(),
        row_idxs.data_ptr<int>(),
        H, W, n_rows
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
        H_m, W_m = mach.bin_length, mach.bin_width
        for pidx, pid in enumerate(part_ids):
            mpd = mach.parts[pid]
            pd_part = problem_data.parts[pid]
            machine_proc_time[m, pidx] = float(mpd.proc_time)
            machine_proc_time_height[m, pidx] = float(mpd.proc_time_height)
            nrot = int(pd_part.nrot)
            for rot in range(nrot):
                fft = mpd.ffts[rot]
                if device.startswith("cuda") and (not fft.is_cuda or str(fft.device) != device):
                    fft = fft.to(device)
                machine_fft_rot.append(fft.contiguous())

        fft_dense = torch.stack(machine_fft_rot, dim=0).contiguous()
        fft_dense = fft_dense / float(H_m * W_m)
        machine_fft_dense.append(fft_dense)

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
        # Enlarge cuFFT plan cache so all unique (H, W, batch) plans stay cached
        # across waves.  Default is often small (e.g. 16–32), causing repeated
        # cuModuleLoad/Unload (~275ms/rep on P75M2-0 per nsys).
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.cufft_plan_cache[0].max_size = 4096
            except Exception:
                pass
        mod = _get_module()
        if mod is None:
            raise RuntimeError("Failed to build/load _full_native_decoder_ext")
        packed = _pack_problem_data(problem_data, int(nb_machines), thresholds, instance_parts, device)
        self._decoder = mod.NativeFullDecoder(packed, int(nb_parts), int(nb_machines), str(device))

    def evaluate_batch(self, chromosomes: np.ndarray):
        chrom = np.ascontiguousarray(chromosomes, dtype=np.float32)
        out = self._decoder.evaluate_batch(chrom)
        return np.asarray(out, dtype=np.float64).tolist()

    @staticmethod
    def get_profile_summary() -> str:
        mod = _get_module()
        if mod is None or not hasattr(mod, "get_profile_summary"):
            return "(profiling not available)"
        return mod.get_profile_summary()

    @staticmethod
    def reset_profile() -> None:
        mod = _get_module()
        if mod is not None and hasattr(mod, "reset_profile"):
            mod.reset_profile()
