"""
Native planner for Phase 5/6 context handling.

Helpers:
  - partition contexts by selected test index
  - build placement/new-bin plan using selected test indices + validity/coords

Set `ABRKGA_PHASE56_CPP=1` to enable native mode.
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

py::tuple partition_phase56(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> best_ti_per_ctx
) {
    if (best_ti_per_ctx.ndim() != 1) {
        throw std::runtime_error("best_ti_per_ctx must be 1D");
    }

    const int64_t n_ctx = best_ti_per_ctx.shape(0);
    auto best_v = best_ti_per_ctx.unchecked<1>();

    std::vector<int64_t> place_ctx;
    std::vector<int64_t> place_ti;
    std::vector<int64_t> newbin_ctx;
    place_ctx.reserve(static_cast<size_t>(n_ctx));
    place_ti.reserve(static_cast<size_t>(n_ctx));
    newbin_ctx.reserve(static_cast<size_t>(n_ctx));

    for (int64_t c = 0; c < n_ctx; ++c) {
        const int64_t ti = best_v(c);
        if (ti >= 0) {
            place_ctx.push_back(c);
            place_ti.push_back(ti);
        } else {
            newbin_ctx.push_back(c);
        }
    }

    py::array_t<int64_t> place_ctx_arr(place_ctx.size());
    py::array_t<int64_t> place_ti_arr(place_ti.size());
    py::array_t<int64_t> newbin_ctx_arr(newbin_ctx.size());
    auto pc = place_ctx_arr.mutable_unchecked<1>();
    auto pt = place_ti_arr.mutable_unchecked<1>();
    auto nc = newbin_ctx_arr.mutable_unchecked<1>();

    for (size_t i = 0; i < place_ctx.size(); ++i) {
        pc(i) = place_ctx[i];
        pt(i) = place_ti[i];
    }
    for (size_t i = 0; i < newbin_ctx.size(); ++i) {
        nc(i) = newbin_ctx[i];
    }

    return py::make_tuple(place_ctx_arr, place_ti_arr, newbin_ctx_arr);
}

py::tuple plan_phase56(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> best_ti_per_ctx,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> sc_valid,
    py::array_t<double, py::array::c_style | py::array::forcecast> sc_rows,
    py::array_t<double, py::array::c_style | py::array::forcecast> sc_cols
) {
    if (best_ti_per_ctx.ndim() != 1 || sc_valid.ndim() != 1 ||
        sc_rows.ndim() != 1 || sc_cols.ndim() != 1) {
        throw std::runtime_error("all inputs must be 1D arrays");
    }

    const int64_t n_ctx = best_ti_per_ctx.shape(0);
    const int64_t n_tests = sc_valid.shape(0);
    if (sc_rows.shape(0) != n_tests || sc_cols.shape(0) != n_tests) {
        throw std::runtime_error("score arrays must have the same length");
    }

    auto best_v = best_ti_per_ctx.unchecked<1>();
    auto val_v = sc_valid.unchecked<1>();
    auto row_v = sc_rows.unchecked<1>();
    auto col_v = sc_cols.unchecked<1>();

    std::vector<int64_t> place_ctx;
    std::vector<int64_t> place_ti;
    std::vector<int64_t> place_row;
    std::vector<int64_t> place_col;
    std::vector<int64_t> newbin_ctx;
    place_ctx.reserve(static_cast<size_t>(n_ctx));
    place_ti.reserve(static_cast<size_t>(n_ctx));
    place_row.reserve(static_cast<size_t>(n_ctx));
    place_col.reserve(static_cast<size_t>(n_ctx));
    newbin_ctx.reserve(static_cast<size_t>(n_ctx));

    for (int64_t c = 0; c < n_ctx; ++c) {
        const int64_t ti = best_v(c);
        if (ti < 0 || ti >= n_tests || val_v(ti) == 0) {
            newbin_ctx.push_back(c);
            continue;
        }
        place_ctx.push_back(c);
        place_ti.push_back(ti);
        place_row.push_back(static_cast<int64_t>(row_v(ti)));
        place_col.push_back(static_cast<int64_t>(col_v(ti)));
    }

    py::array_t<int64_t> place_ctx_arr(place_ctx.size());
    py::array_t<int64_t> place_ti_arr(place_ti.size());
    py::array_t<int64_t> place_row_arr(place_row.size());
    py::array_t<int64_t> place_col_arr(place_col.size());
    py::array_t<int64_t> newbin_ctx_arr(newbin_ctx.size());
    auto pc = place_ctx_arr.mutable_unchecked<1>();
    auto pt = place_ti_arr.mutable_unchecked<1>();
    auto pr = place_row_arr.mutable_unchecked<1>();
    auto pcol = place_col_arr.mutable_unchecked<1>();
    auto nc = newbin_ctx_arr.mutable_unchecked<1>();

    for (size_t i = 0; i < place_ctx.size(); ++i) {
        pc(i) = place_ctx[i];
        pt(i) = place_ti[i];
        pr(i) = place_row[i];
        pcol(i) = place_col[i];
    }
    for (size_t i = 0; i < newbin_ctx.size(); ++i) {
        nc(i) = newbin_ctx[i];
    }

    return py::make_tuple(place_ctx_arr, place_ti_arr, place_row_arr, place_col_arr, newbin_ctx_arr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("partition_phase56", &partition_phase56,
          "Partition contexts into placement/new-bin sets");
    m.def("plan_phase56", &plan_phase56,
          "Build placement/new-bin plan from selected tests and validity");
}
"""

_module = None
_load_failed = False


def _enabled() -> bool:
    return os.getenv("ABRKGA_PHASE56_CPP", "0").strip() not in {"0", "false", "False"}


def _get_module():
    global _module, _load_failed
    if _module is not None:
        return _module
    if _load_failed or not _enabled():
        return None

    try:
        from torch.utils.cpp_extension import load_inline

        _module = load_inline(
            name="_phase56_planner_ext",
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


def partition_phase56(best_ti_per_ctx) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns tuple of arrays:
      (placement_ctx_indices, placement_test_indices, newbin_ctx_indices)
    or None if native extension is unavailable.
    """
    mod = _get_module()
    if mod is None:
        return None

    best = np.asarray(best_ti_per_ctx, dtype=np.int64)
    return mod.partition_phase56(best)


def plan_phase56(
    best_ti_per_ctx,
    sc_valid,
    sc_rows,
    sc_cols,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns tuple of arrays:
      (placement_ctx_indices, placement_test_indices, placement_rows, placement_cols, newbin_ctx_indices)
    or None if native extension is unavailable.
    """
    mod = _get_module()
    if mod is None:
        return None

    best = np.asarray(best_ti_per_ctx, dtype=np.int64)
    valid = np.asarray(sc_valid, dtype=np.uint8)
    rows = np.asarray(sc_rows, dtype=np.float64)
    cols = np.asarray(sc_cols, dtype=np.float64)
    return mod.plan_phase56(best, valid, rows, cols)
