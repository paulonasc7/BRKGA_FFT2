"""
Native planner for Phase 5/6 context partitioning.

This helper partitions contexts based on best test index:
  - contexts with placement (`best_ti >= 0`)
  - contexts needing new bin (`best_ti < 0`)

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("partition_phase56", &partition_phase56,
          "Partition contexts into placement/new-bin sets");
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
