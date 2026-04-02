"""
Native Phase 3 vacancy-test collector.

This extension accelerates Phase 3 by collecting vacancy-feasible tests for many
contexts in a single native call per pass.

Set `ABRKGA_PHASE3_CPP=1` to enable native mode.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

_CPP_SRC = r"""
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

static inline bool vacancy_fit(
    const int32_t* vacancy_row,
    int64_t vacancy_len,
    const int32_t* density,
    int32_t shape_height
) {
    if (shape_height <= 0 || shape_height > vacancy_len) {
        return false;
    }

    const int64_t max_start = vacancy_len - shape_height;
    for (int64_t start = 0; start <= max_start; ++start) {
        bool fits = true;
        for (int32_t i = 0; i < shape_height; ++i) {
            if (vacancy_row[start + i] < density[i]) {
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

py::tuple collect_phase3_tests_batch(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> vacancy_matrix,
    py::array_t<double, py::array::c_style | py::array::forcecast> row_bin_areas,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> row_bin_local_idx,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> ctx_bin_offsets,
    py::array_t<double, py::array::c_style | py::array::forcecast> ctx_part_areas,
    py::array_t<double, py::array::c_style | py::array::forcecast> ctx_bin_area_limits,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> ctx_skip_bins,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> ctx_rot_offsets,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> rot_heights,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> rot_widths,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> rot_density_offsets,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> densities_flat,
    int32_t H,
    int32_t W,
    int32_t mode
) {
    if (mode != 0 && mode != 1) {
        throw std::runtime_error("mode must be 0 (pass1) or 1 (pass2)");
    }
    if (vacancy_matrix.ndim() != 2) {
        throw std::runtime_error("vacancy_matrix must be 2D");
    }
    if (row_bin_areas.ndim() != 1 || row_bin_local_idx.ndim() != 1 ||
        ctx_bin_offsets.ndim() != 1 || ctx_part_areas.ndim() != 1 ||
        ctx_bin_area_limits.ndim() != 1 || ctx_skip_bins.ndim() != 1 ||
        ctx_rot_offsets.ndim() != 1 || rot_heights.ndim() != 1 ||
        rot_widths.ndim() != 1 || rot_density_offsets.ndim() != 1 ||
        densities_flat.ndim() != 1) {
        throw std::runtime_error("all non-vacancy inputs must be 1D");
    }

    const int64_t n_rows = vacancy_matrix.shape(0);
    const int64_t vac_len = vacancy_matrix.shape(1);
    const int64_t n_ctx = ctx_part_areas.shape(0);
    const int64_t n_rots = rot_heights.shape(0);

    if (row_bin_areas.shape(0) != n_rows || row_bin_local_idx.shape(0) != n_rows) {
        throw std::runtime_error("row arrays must match vacancy_matrix rows");
    }
    if (ctx_bin_offsets.shape(0) != n_ctx + 1 || ctx_rot_offsets.shape(0) != n_ctx + 1) {
        throw std::runtime_error("ctx offsets must have length n_ctx + 1");
    }
    if (ctx_bin_area_limits.shape(0) != n_ctx || ctx_skip_bins.shape(0) != n_ctx) {
        throw std::runtime_error("ctx arrays must have length n_ctx");
    }
    if (rot_widths.shape(0) != n_rots || rot_density_offsets.shape(0) != n_rots + 1) {
        throw std::runtime_error("rotation arrays length mismatch");
    }

    auto vac_v = vacancy_matrix.unchecked<2>();
    auto area_v = row_bin_areas.unchecked<1>();
    auto row_bin_v = row_bin_local_idx.unchecked<1>();
    auto ctx_bin_off_v = ctx_bin_offsets.unchecked<1>();
    auto ctx_part_area_v = ctx_part_areas.unchecked<1>();
    auto ctx_bin_limit_v = ctx_bin_area_limits.unchecked<1>();
    auto ctx_skip_v = ctx_skip_bins.unchecked<1>();
    auto ctx_rot_off_v = ctx_rot_offsets.unchecked<1>();
    auto rh_v = rot_heights.unchecked<1>();
    auto rw_v = rot_widths.unchecked<1>();
    auto rd_off_v = rot_density_offsets.unchecked<1>();
    auto den_v = densities_flat.unchecked<1>();

    std::vector<int32_t> first_valid_bins(static_cast<size_t>(n_ctx), -1);
    std::vector<int32_t> out_ctx_local;
    std::vector<int32_t> out_bin_local;
    std::vector<int32_t> out_rot_local;
    out_ctx_local.reserve(static_cast<size_t>(n_rows * 2));
    out_bin_local.reserve(static_cast<size_t>(n_rows * 2));
    out_rot_local.reserve(static_cast<size_t>(n_rows * 2));

    for (int32_t c = 0; c < n_ctx; ++c) {
        const int32_t row_start = ctx_bin_off_v(c);
        const int32_t row_end = ctx_bin_off_v(c + 1);
        const int32_t rot_start = ctx_rot_off_v(c);
        const int32_t rot_end = ctx_rot_off_v(c + 1);

        if (row_start < 0 || row_end < row_start || row_end > n_rows) {
            throw std::runtime_error("invalid ctx_bin_offsets range");
        }
        if (rot_start < 0 || rot_end < rot_start || rot_end > n_rots) {
            throw std::runtime_error("invalid ctx_rot_offsets range");
        }

        const double part_area = ctx_part_area_v(c);
        const double bin_area_limit = ctx_bin_limit_v(c);
        const int32_t skip_bin = (mode == 1) ? ctx_skip_v(c) : -1;

        for (int32_t rix = row_start; rix < row_end; ++rix) {
            const int32_t bin_local = row_bin_v(rix);
            if (bin_local == skip_bin) {
                continue;
            }
            if (area_v(rix) + part_area > bin_area_limit) {
                continue;
            }

            bool any_rot_pass = false;
            const int32_t* vacancy_row = &vac_v(rix, 0);

            for (int32_t gro = rot_start; gro < rot_end; ++gro) {
                const int32_t h = rh_v(gro);
                const int32_t w = rw_v(gro);
                if (h > H || w > W) {
                    continue;
                }

                const int32_t den_start = rd_off_v(gro);
                const int32_t den_end = rd_off_v(gro + 1);
                if (den_start < 0 || den_end < den_start || den_end > densities_flat.shape(0)) {
                    throw std::runtime_error("invalid rot_density_offsets range");
                }
                if ((den_end - den_start) != h) {
                    continue;
                }

                const int32_t* density_ptr = &den_v(den_start);
                if (vacancy_fit(vacancy_row, vac_len, density_ptr, h)) {
                    any_rot_pass = true;
                    out_ctx_local.push_back(c);
                    out_bin_local.push_back(bin_local);
                    out_rot_local.push_back(gro - rot_start);  // local rot index for this context
                }
            }

            if (mode == 0 && any_rot_pass) {
                first_valid_bins[static_cast<size_t>(c)] = bin_local;
                break;
            }
        }
    }

    py::array_t<int32_t> first_valid_arr(n_ctx);
    py::array_t<int32_t> out_ctx_arr(out_ctx_local.size());
    py::array_t<int32_t> out_bin_arr(out_bin_local.size());
    py::array_t<int32_t> out_rot_arr(out_rot_local.size());

    auto fv = first_valid_arr.mutable_unchecked<1>();
    for (int32_t i = 0; i < n_ctx; ++i) {
        fv(i) = first_valid_bins[static_cast<size_t>(i)];
    }

    auto oc = out_ctx_arr.mutable_unchecked<1>();
    auto ob = out_bin_arr.mutable_unchecked<1>();
    auto orr = out_rot_arr.mutable_unchecked<1>();
    for (size_t i = 0; i < out_ctx_local.size(); ++i) {
        oc(i) = out_ctx_local[i];
        ob(i) = out_bin_local[i];
        orr(i) = out_rot_local[i];
    }

    return py::make_tuple(first_valid_arr, out_ctx_arr, out_bin_arr, out_rot_arr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("collect_phase3_tests_batch", &collect_phase3_tests_batch,
          "Collect Phase 3 vacancy-feasible tests in one native batch");
}
"""

_module = None
_load_failed = False


def _enabled() -> bool:
    return os.getenv("ABRKGA_PHASE3_CPP", "0").strip() not in {"0", "false", "False"}


def _get_module():
    global _module, _load_failed
    if _module is not None:
        return _module
    if _load_failed or not _enabled():
        return None

    try:
        from torch.utils.cpp_extension import load_inline

        _module = load_inline(
            name="_phase3_collector_ext",
            cpp_sources=_CPP_SRC,
            functions=None,
            extra_cflags=["-O3"],
            with_cuda=False,
            verbose=False,
        )
    except Exception:
        _load_failed = True
        _module = None
    return _module


def collect_phase3_tests_batch(
    vacancy_matrix,
    row_bin_areas,
    row_bin_local_idx,
    ctx_bin_offsets,
    ctx_part_areas,
    ctx_bin_area_limits,
    ctx_skip_bins,
    ctx_rot_offsets,
    rot_heights,
    rot_widths,
    rot_density_offsets,
    densities_flat,
    H: int,
    W: int,
    mode: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns tuple:
      (first_valid_bins, out_ctx_local_idx, out_bin_local_idx, out_rot_local_idx)
    or None if native extension is unavailable.
    """
    mod = _get_module()
    if mod is None:
        return None

    vac = np.asarray(vacancy_matrix, dtype=np.int32)
    rba = np.asarray(row_bin_areas, dtype=np.float64)
    rbl = np.asarray(row_bin_local_idx, dtype=np.int32)
    cbo = np.asarray(ctx_bin_offsets, dtype=np.int32)
    cpa = np.asarray(ctx_part_areas, dtype=np.float64)
    cbl = np.asarray(ctx_bin_area_limits, dtype=np.float64)
    csb = np.asarray(ctx_skip_bins, dtype=np.int32)
    cro = np.asarray(ctx_rot_offsets, dtype=np.int32)
    rh = np.asarray(rot_heights, dtype=np.int32)
    rw = np.asarray(rot_widths, dtype=np.int32)
    rdo = np.asarray(rot_density_offsets, dtype=np.int32)
    den = np.asarray(densities_flat, dtype=np.int32)

    return mod.collect_phase3_tests_batch(
        vac, rba, rbl, cbo, cpa, cbl, csb, cro, rh, rw, rdo, den,
        int(H), int(W), int(mode)
    )
