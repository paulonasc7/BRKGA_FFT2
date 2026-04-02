"""
Native selector for Phase 5 best-per-context placement choice.

This module provides a tiny C++/pybind11 extension (built with
torch.utils.cpp_extension.load_inline) that replaces the Python tuple-comparison
loop used to select the winning test index per context.

Set `ABRKGA_PHASE5_CPP=1` to enable native mode.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

_CPP_SRC = r"""
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

static inline bool better_key(
    double neg_bin, double density, double row, double neg_col,
    double best_neg_bin, double best_density, double best_row, double best_neg_col
) {
    if (neg_bin != best_neg_bin) {
        return neg_bin > best_neg_bin;
    }
    if (density != best_density) {
        return density > best_density;
    }
    if (row != best_row) {
        return row > best_row;
    }
    return neg_col > best_neg_col;
}

py::array_t<int64_t> select_best_per_context(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> ctx_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> bin_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> densities,
    py::array_t<double, py::array::c_style | py::array::forcecast> rows,
    py::array_t<double, py::array::c_style | py::array::forcecast> cols,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> valid,
    int64_t n_contexts
) {
    if (n_contexts < 0) {
        throw std::runtime_error("n_contexts must be non-negative");
    }

    if (ctx_indices.ndim() != 1 || bin_indices.ndim() != 1 || densities.ndim() != 1 ||
        rows.ndim() != 1 || cols.ndim() != 1 || valid.ndim() != 1) {
        throw std::runtime_error("all inputs must be 1D arrays");
    }

    const ssize_t n_tests = ctx_indices.shape(0);
    if (bin_indices.shape(0) != n_tests || densities.shape(0) != n_tests ||
        rows.shape(0) != n_tests || cols.shape(0) != n_tests || valid.shape(0) != n_tests) {
        throw std::runtime_error("input arrays must have the same length");
    }

    auto ctx_v = ctx_indices.unchecked<1>();
    auto bin_v = bin_indices.unchecked<1>();
    auto den_v = densities.unchecked<1>();
    auto row_v = rows.unchecked<1>();
    auto col_v = cols.unchecked<1>();
    auto val_v = valid.unchecked<1>();

    py::array_t<int64_t> best_idx(n_contexts);
    auto out = best_idx.mutable_unchecked<1>();

    std::vector<uint8_t> has_best(static_cast<size_t>(n_contexts), 0);
    std::vector<double> best_neg_bin(static_cast<size_t>(n_contexts), 0.0);
    std::vector<double> best_density(static_cast<size_t>(n_contexts), 0.0);
    std::vector<double> best_row(static_cast<size_t>(n_contexts), 0.0);
    std::vector<double> best_neg_col(static_cast<size_t>(n_contexts), 0.0);

    for (int64_t c = 0; c < n_contexts; ++c) {
        out(c) = -1;
    }

    for (ssize_t i = 0; i < n_tests; ++i) {
        if (val_v(i) == 0) {
            continue;
        }

        const int64_t ctx = ctx_v(i);
        if (ctx < 0 || ctx >= n_contexts) {
            continue;
        }

        const size_t c = static_cast<size_t>(ctx);
        const double neg_bin = -bin_v(i);
        const double density = den_v(i);
        const double row = row_v(i);
        const double neg_col = -col_v(i);

        if (!has_best[c] ||
            better_key(
                neg_bin, density, row, neg_col,
                best_neg_bin[c], best_density[c], best_row[c], best_neg_col[c]
            )) {
            has_best[c] = 1;
            best_neg_bin[c] = neg_bin;
            best_density[c] = density;
            best_row[c] = row;
            best_neg_col[c] = neg_col;
            out(ctx) = static_cast<int64_t>(i);
        }
    }

    return best_idx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("select_best_per_context", &select_best_per_context,
          "Select best test index per context using lexicographic key");
}
"""

_module = None
_load_failed = False


def _enabled() -> bool:
    # Opt-in by default to avoid unexpected toolchain/runtime failures.
    return os.getenv("ABRKGA_PHASE5_CPP", "0").strip() not in {"0", "false", "False"}


def _get_module():
    global _module, _load_failed
    if _module is not None:
        return _module
    if _load_failed or not _enabled():
        return None

    try:
        from torch.utils.cpp_extension import load_inline

        _module = load_inline(
            name="_phase5_selector_ext",
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


def select_best_per_context(
    test_ctx_indices,
    sc_bin_indices,
    sc_densities,
    sc_rows,
    sc_cols,
    sc_valid,
    n_contexts: int,
) -> Optional[np.ndarray]:
    """
    Return best test index per context as int64 array of shape (n_contexts,).
    Returns None if native extension is unavailable.
    """
    mod = _get_module()
    if mod is None:
        return None

    ctx = np.asarray(test_ctx_indices, dtype=np.int64)
    bi = np.asarray(sc_bin_indices, dtype=np.float64)
    de = np.asarray(sc_densities, dtype=np.float64)
    ro = np.asarray(sc_rows, dtype=np.float64)
    co = np.asarray(sc_cols, dtype=np.float64)
    va = np.asarray(sc_valid, dtype=np.uint8)

    return mod.select_best_per_context(ctx, bi, de, ro, co, va, int(n_contexts))
