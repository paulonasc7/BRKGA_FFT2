"""
Native batched CPU updater for Phase 6 new-bin initialization.

Set `ABRKGA_PHASE6_CPU_CPP=1` to enable.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

_CPP_SRC = r"""
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

void apply_phase6_cpu_update_batch(
    py::list grids,
    py::list vacancy_vectors,
    py::list part_mats,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> y_starts
) {
    const py::ssize_t n = py::len(grids);
    if (py::len(vacancy_vectors) != n || py::len(part_mats) != n) {
        throw std::runtime_error("grids/vacancy_vectors/part_mats must have same length");
    }
    if (y_starts.ndim() != 1 || y_starts.shape(0) != n) {
        throw std::runtime_error("y_starts must be 1D and match batch length");
    }
    auto ys = y_starts.unchecked<1>();

    for (py::ssize_t i = 0; i < n; ++i) {
        auto grid = py::cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(grids[i]);
        auto vac = py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(vacancy_vectors[i]);
        auto part = py::cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(part_mats[i]);

        if (grid.ndim() != 2 || vac.ndim() != 1 || part.ndim() != 2) {
            throw std::runtime_error("grid/part must be 2D and vacancy_vector must be 1D");
        }

        const int H = static_cast<int>(grid.shape(0));
        const int W = static_cast<int>(grid.shape(1));
        if (vac.shape(0) != H) {
            throw std::runtime_error("vacancy_vector length must equal grid height");
        }

        const int h = static_cast<int>(part.shape(0));
        const int w = static_cast<int>(part.shape(1));
        const int y0 = ys(i);
        if (y0 < 0 || y0 + h > H || w > W) {
            throw std::runtime_error("invalid placement bounds in Phase 6 batch");
        }

        auto g = grid.mutable_unchecked<2>();
        auto v = vac.mutable_unchecked<1>();
        auto p = part.unchecked<2>();

        // New bins are empty; Phase 6 always places at x=0.
        // Write grid rows and recompute vacancy for touched rows in one pass.
        for (int r = 0; r < h; ++r) {
            int max_zeros = 0;
            int current_zeros = 0;

            for (int c = 0; c < w; ++c) {
                const uint8_t cell = p(r, c);
                g(y0 + r, c) = cell;
                if (cell == 0) {
                    current_zeros += 1;
                    if (current_zeros > max_zeros) max_zeros = current_zeros;
                } else {
                    current_zeros = 0;
                }
            }

            // Remaining columns [w, W) are zeros in a fresh bin.
            const int tail = W - w;
            if (tail > 0) {
                current_zeros += tail;
                if (current_zeros > max_zeros) max_zeros = current_zeros;
            }
            v(y0 + r) = max_zeros;
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_phase6_cpu_update_batch", &apply_phase6_cpu_update_batch,
          "Batched CPU Phase 6 grid + vacancy update");
}
"""

_module = None
_load_failed = False


def _enabled() -> bool:
    return os.getenv("ABRKGA_PHASE6_CPU_CPP", "0").strip() not in {"0", "false", "False"}


def _get_module():
    global _module, _load_failed
    if _module is not None:
        return _module
    if _load_failed or not _enabled():
        return None

    try:
        from torch.utils.cpp_extension import load_inline

        _module = load_inline(
            name="_phase6_cpu_update_ext",
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


def apply_phase6_cpu_update_batch(
    grids: List[np.ndarray],
    vacancy_vectors: List[np.ndarray],
    part_mats: List[np.ndarray],
    y_starts: np.ndarray,
) -> bool:
    """
    Returns True if native path executed, else False (fallback should be used).
    """
    mod = _get_module()
    if mod is None:
        return False
    if not grids:
        return True

    ys = np.asarray(y_starts, dtype=np.int32)
    mod.apply_phase6_cpu_update_batch(grids, vacancy_vectors, part_mats, ys)
    return True
