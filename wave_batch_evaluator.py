"""
Wave-based batch evaluator for BRKGA - Version 3.

KEY INSIGHT: All part FFTs for a given machine have the SAME dimensions (H x W),
regardless of which part they are. This means we can batch FFT operations
across ALL chromosomes, even when they're placing DIFFERENT parts.

TRUE BATCHING: Instead of sequential per-context FFTs, we:
1. Collect ALL (context, bin, rotation) combinations to test
2. Do ONE batched IFFT for everything
3. Process results to determine best placement per context

This turns 500 contexts × 3 bins × 4 rotations = 6000 FFTs into ONE batched call!
"""

import os
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from numba_utils import check_vacancy_fit_simple, update_vacancy_vector_rows
from phase5_selector import select_best_per_context as _select_best_per_context_cpp
from phase3_collector import collect_phase3_tests_batch as _collect_phase3_tests_batch_cpp
from phase56_planner import plan_phase56 as _plan_phase56_cpp

# Lazy-load custom CUDA kernel for batched grid updates (compiles on first use)
try:
    from cuda_batch_update import batch_grid_update as _cuda_batch_update
    _HAS_CUDA_KERNEL = True
except Exception:
    _cuda_batch_update = None
    _HAS_CUDA_KERNEL = False


def _post_ifft_score(overlap_batch, part_heights, part_widths, row_idx, col_idx, neg_inf):
    """
    Fused post-IFFT scoring: round → mask → score → argmax.

    Decorated with torch.compile so Triton fuses the ~10 elementwise kernels into
    1-2 kernels, cutting VRAM bandwidth by ~5-6x for this block.

    W is derived from col_idx so dynamic=True handles all chunk sizes and both
    machines with a single compiled graph.
    """
    chunk_n = overlap_batch.shape[0]
    W = col_idx.shape[-1]
    zero_mask = (overlap_batch.round() == 0)
    valid_row = row_idx >= (part_heights - 1).view(-1, 1, 1)
    valid_col = col_idx >= (part_widths - 1).view(-1, 1, 1)
    valid_zeros = zero_mask & valid_row & valid_col
    score = torch.where(
        valid_zeros,
        row_idx.float() * (W + 1) - col_idx.float(),
        neg_inf
    )
    max_scores, best_flat_idx = score.view(chunk_n, -1).max(dim=1)
    best_row = best_flat_idx // W
    best_col = best_flat_idx % W
    has_valid = max_scores > -1e8
    return has_valid, best_col - (part_widths - 1), best_row


# Compile if torch.compile is available (PyTorch >= 2.0).
# dynamic=True: one compiled graph handles all chunk sizes and both machines.
# fullgraph=True: error rather than silently falling back to eager on a graph break.
# First call triggers Triton compilation (~30s); subsequent calls use the cached kernel.
if hasattr(torch, 'compile'):
    _post_ifft_score = torch.compile(_post_ifft_score, dynamic=True, fullgraph=True)
    _HAS_TORCH_COMPILE = True
else:
    _HAS_TORCH_COMPILE = False


@dataclass
class BinState:
    """State for a single bin."""
    bin_idx: int
    grid: np.ndarray
    vacancy_vector: np.ndarray
    grid_state_idx: int
    area: float
    enclosure_box_length: int
    min_occupied_row: int
    max_occupied_row: int
    proc_time: float
    proc_time_height: float
    grid_fft_valid: bool
    parts_assigned: List[int]
    bin_length: int
    bin_width: int


@dataclass
class BatchPlacementContext:
    """Tracks state for a single solution-machine combination."""
    solution_idx: int
    machine_idx: int
    parts_sequence: np.ndarray
    current_part_idx: int
    bin_length: int
    bin_width: int
    bin_area: float
    open_bins: List[BinState] = field(default_factory=list)
    next_grid_idx: int = 0
    is_done: bool = False
    is_feasible: bool = True


class WaveBatchEvaluator:
    """Evaluates many solutions in parallel using true cross-solution FFT batching."""
    
    def __init__(self, problem_data, nbParts, nbMachines, thresholds, 
                 instance_parts, collision_backend, device='cuda'):
        self.problem_data = problem_data
        self.nbParts = nbParts
        self.nbMachines = nbMachines
        self.thresholds = np.array(thresholds)
        self.instance_parts = instance_parts
        self.collision_backend = collision_backend
        self.device = device
        self.machines = problem_data.machines
        self.parts = problem_data.parts

        # Opt-in context collection for inspection (default off — zero overhead on normal runs).
        # Set _collect_contexts = True before evaluate_batch to populate _last_contexts.
        self._collect_contexts = False
        self._last_contexts = {}

        # Debug tracing: set _debug_part_ids to a set of part IDs to trace.
        # When the current wave processes any of these parts, detailed diagnostics
        # are printed (vacancy, IFFT results, scoring) for every bin tested.
        self._debug_part_ids = set()

        # Placement log: when _collect_contexts is True, records every placement
        # as (part_id, bin_idx, col, row, rot, shape) per machine.
        self._placement_log = {}  # machine_idx -> list of tuples

        # Structured Phase 6 CPU path (enabled by default, can disable with
        # ABRKGA_PHASE6_STRUCTURED=0 for A/B validation).
        self._phase6_structured = os.getenv("ABRKGA_PHASE6_STRUCTURED", "1").strip() not in {
            "0", "false", "False"
        }
        self._phase6_row_vacancy_cache = {}

        # Pre-build flat GPU tensor of all part rotation matrices for the batch kernel.
        # part_update_meta: (part_id, rot) -> (flat_offset, height, width)
        self.flat_parts_gpu = None
        self.part_update_meta = {}
        if _HAS_CUDA_KERNEL and torch.cuda.is_available():
            flat_list = []
            offset = 0
            ok = True
            for part_id in problem_data.instance_parts_unique:
                pd_ = problem_data.parts[part_id]
                if not pd_.rotations_gpu:
                    ok = False; break
                for rot in range(pd_.nrot):
                    t = pd_.rotations_gpu[rot]
                    if t is None or not t.is_cuda:
                        ok = False; break
                    h, w = pd_.shapes[rot]
                    self.part_update_meta[(part_id, rot)] = (offset, h, w)
                    flat_list.append(t.flatten())
                    offset += h * w
                if not ok:
                    break
            if ok and flat_list:
                self.flat_parts_gpu = torch.cat(flat_list)
            else:
                self.part_update_meta = {}

    def evaluate_batch(self, chromosomes: np.ndarray) -> List[float]:
        num_solutions = len(chromosomes)
        machine_makespans = []
        
        for machine_idx in range(self.nbMachines):
            makespans = self._process_machine_batch(chromosomes, machine_idx, num_solutions)
            machine_makespans.append(makespans)
        
        final_makespans = np.maximum.reduce(machine_makespans)
        return final_makespans.tolist()
    
    def _process_machine_batch(self, chromosomes, machine_idx, num_solutions):
        mach_data = self.machines[machine_idx]
        H, W = mach_data.bin_length, mach_data.bin_width

        # When machine dimensions change between calls the caching allocator cannot
        # reuse the previous machine's blocks (different shape), so they pile up.
        # Only empty the cache in that case; same-dimension machines reuse blocks
        # for free and don't need the OS round-trip cost of empty_cache().
        bytes_per_bin = H * W * 4 + H * (W // 2 + 1) * 8  # float32 + complex64
        if torch.cuda.is_available() and getattr(self, '_prev_machine_dims', None) != (H, W):
            torch.cuda.empty_cache()
        self._prev_machine_dims = (H, W)

        # max_bins_per_sol: start from the empirical lower bound (nbParts // 3 covers
        # ~3 parts/bin worst case), then cap so grid_states + grid_ffts together use
        # at most 50% of total GPU VRAM.  The remaining 50% covers IFFT intermediates
        # (~2 × CHUNK_SIZE × bytes_per_bin), part FFTs, and PyTorch runtime overhead.
        # Minimum of 5; if even that doesn't fit the instance is too large for this GPU.
        needed_bins = max(10, self.nbParts // 3)
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(self.device).total_memory
            vram_cap = max(5, int(total_vram * 0.50) // (bytes_per_bin * num_solutions))
        else:
            vram_cap = needed_bins
        max_bins_per_sol = min(needed_bins, vram_cap)

        sequences = self._decode_sequences(chromosomes, machine_idx)
        contexts = self._init_batch_contexts(sequences, machine_idx, num_solutions,
                                             mach_data, max_bins_per_sol)

        max_total_bins = num_solutions * max_bins_per_sol
        grid_states = torch.zeros((max_total_bins, H, W), dtype=torch.float32, device=self.device)
        grid_ffts = torch.zeros((max_total_bins, H, W // 2 + 1), dtype=torch.complex64, device=self.device)

        # Cache index tensors once per machine (reused across all waves and chunks)
        row_idx = torch.arange(H, device=self.device).view(1, H, 1)
        col_idx = torch.arange(W, device=self.device).view(1, 1, W)
        neg_inf = torch.tensor(-1e9, device=self.device)

        # Process waves
        max_waves = max(len(seq) for seq in sequences) * 3 if sequences else 0
        for wave in range(max_waves):
            active = [c for c in contexts if not c.is_done and c.is_feasible]
            if not active:
                break
            self._process_wave_true_batch(active, mach_data, grid_states, grid_ffts,
                                          row_idx, col_idx, neg_inf)
        
        # Collect makespans
        makespans = np.zeros(num_solutions)
        for ctx in contexts:
            if not ctx.is_feasible:
                makespans[ctx.solution_idx] = 1e16
            else:
                total = sum(b.proc_time + b.proc_time_height + mach_data.setup_time 
                           for b in ctx.open_bins if b.area > 0)
                makespans[ctx.solution_idx] = total
        
        if self._collect_contexts:
            self._last_contexts[machine_idx] = contexts

        return makespans

    def _decode_sequences(self, chromosomes, machine_idx):
        sequences = []
        for sol_idx in range(len(chromosomes)):
            chrom = chromosomes[sol_idx]
            SV, MV = chrom[:self.nbParts], chrom[self.nbParts:]
            
            if machine_idx == 0:
                mask = MV <= self.thresholds[0]
            elif machine_idx == self.nbMachines - 1:
                mask = MV > self.thresholds[-1]
            else:
                mask = (MV > self.thresholds[machine_idx-1]) & (MV <= self.thresholds[machine_idx])
            
            part_indices = np.where(mask)[0]
            values = SV[part_indices]
            actual_parts = self.instance_parts[part_indices]
            sorted_sequence = actual_parts[np.argsort(values)]
            sequences.append(sorted_sequence)
        
        return sequences
    
    def _init_batch_contexts(self, sequences, machine_idx, num_solutions, mach_data,
                             max_bins_per_sol):
        contexts = []
        
        for sol_idx in range(num_solutions):
            ctx = BatchPlacementContext(
                solution_idx=sol_idx,
                machine_idx=machine_idx,
                parts_sequence=sequences[sol_idx],
                current_part_idx=0,
                bin_length=mach_data.bin_length,
                bin_width=mach_data.bin_width,
                bin_area=mach_data.bin_area,
                open_bins=[],
                next_grid_idx=sol_idx * max_bins_per_sol,
                is_done=len(sequences[sol_idx]) == 0,
                is_feasible=True
            )
            contexts.append(ctx)
        
        return contexts

    def _select_best_tests_per_context(
        self,
        test_ctx_indices,
        sc_bin_indices,
        sc_densities,
        sc_rows,
        sc_cols,
        sc_valid,
        n_contexts,
    ):
        """
        Select winning test index per context with lexicographic ordering:
        lower bin_idx > higher density > larger row > smaller col.
        """
        native_best = _select_best_per_context_cpp(
            test_ctx_indices=test_ctx_indices,
            sc_bin_indices=sc_bin_indices,
            sc_densities=sc_densities,
            sc_rows=sc_rows,
            sc_cols=sc_cols,
            sc_valid=sc_valid,
            n_contexts=n_contexts,
        )
        if native_best is not None:
            return native_best.astype(np.int64, copy=False)

        # Python fallback (exactly the same lexicographic semantics).
        best_ti_per_ctx = np.full(n_contexts, -1, dtype=np.int64)
        best_key_per_ctx = [None] * n_contexts
        for ti, ctx_idx in enumerate(test_ctx_indices):
            if not sc_valid[ti]:
                continue
            key = (-sc_bin_indices[ti], sc_densities[ti], sc_rows[ti], -sc_cols[ti])
            prev = best_key_per_ctx[ctx_idx]
            if prev is None or key > prev:
                best_key_per_ctx[ctx_idx] = key
                best_ti_per_ctx[ctx_idx] = ti
        return best_ti_per_ctx

    def _collect_phase3_tests_native_batch(
        self,
        context_info,
        ctx_indices,
        H,
        W,
        mode,
        ctx_first_valid_bin,
    ):
        if not ctx_indices:
            return None

        vacancy_rows = []
        row_bin_areas = []
        row_bin_local_idx = []
        ctx_bin_offsets = [0]
        ctx_part_areas = []
        ctx_bin_area_limits = []
        ctx_skip_bins = []

        ctx_rot_offsets = [0]
        rot_heights = []
        rot_widths = []
        rot_density_offsets = [0]
        density_chunks = []

        for ctx_idx in ctx_indices:
            ctx, part_data, _ = context_info[ctx_idx]

            for b_local, bin_state in enumerate(ctx.open_bins):
                vacancy_rows.append(bin_state.vacancy_vector.astype(np.int32, copy=False))
                row_bin_areas.append(bin_state.area)
                row_bin_local_idx.append(b_local)
            ctx_bin_offsets.append(len(vacancy_rows))

            ctx_part_areas.append(part_data.area)
            ctx_bin_area_limits.append(ctx.bin_area)
            ctx_skip_bins.append(ctx_first_valid_bin[ctx_idx] if mode == 1 else -1)

            for rot in range(part_data.nrot):
                rot_heights.append(int(part_data.shapes_heights[rot]))
                rot_widths.append(int(part_data.shapes_widths[rot]))
                d0 = int(part_data.density_offsets[rot])
                d1 = int(part_data.density_offsets[rot + 1])
                d_chunk = part_data.densities_flat[d0:d1]
                density_chunks.append(d_chunk)
                rot_density_offsets.append(rot_density_offsets[-1] + len(d_chunk))
            ctx_rot_offsets.append(len(rot_heights))

        vacancy_matrix = (
            np.asarray(vacancy_rows, dtype=np.int32)
            if vacancy_rows
            else np.zeros((0, H), dtype=np.int32)
        )
        densities_flat = (
            np.concatenate(density_chunks).astype(np.int32, copy=False)
            if density_chunks
            else np.zeros(0, dtype=np.int32)
        )

        native = _collect_phase3_tests_batch_cpp(
            vacancy_matrix=vacancy_matrix,
            row_bin_areas=np.asarray(row_bin_areas, dtype=np.float64),
            row_bin_local_idx=np.asarray(row_bin_local_idx, dtype=np.int32),
            ctx_bin_offsets=np.asarray(ctx_bin_offsets, dtype=np.int32),
            ctx_part_areas=np.asarray(ctx_part_areas, dtype=np.float64),
            ctx_bin_area_limits=np.asarray(ctx_bin_area_limits, dtype=np.float64),
            ctx_skip_bins=np.asarray(ctx_skip_bins, dtype=np.int32),
            ctx_rot_offsets=np.asarray(ctx_rot_offsets, dtype=np.int32),
            rot_heights=np.asarray(rot_heights, dtype=np.int32),
            rot_widths=np.asarray(rot_widths, dtype=np.int32),
            rot_density_offsets=np.asarray(rot_density_offsets, dtype=np.int32),
            densities_flat=densities_flat,
            H=H,
            W=W,
            mode=mode,
        )
        if native is None:
            return None

        first_valid_local, out_ctx_local, out_bin_local, out_rot_local = native
        ctx_indices_arr = np.asarray(ctx_indices, dtype=np.int32)
        first_valid_global = np.full(len(context_info), -1, dtype=np.int32)
        first_valid_global[ctx_indices_arr] = np.asarray(first_valid_local, dtype=np.int32)

        out_ctx_global = ctx_indices_arr[np.asarray(out_ctx_local, dtype=np.int32)]
        return (
            first_valid_global,
            np.asarray(out_ctx_global, dtype=np.int32),
            np.asarray(out_bin_local, dtype=np.int32),
            np.asarray(out_rot_local, dtype=np.int32),
        )

    def _plan_phase56_contexts(self, best_ti_per_ctx, sc_valid, sc_rows, sc_cols):
        native = _plan_phase56_cpp(best_ti_per_ctx, sc_valid, sc_rows, sc_cols)
        if native is not None:
            place_ctx, place_ti, place_rows, place_cols, newbin_ctx = native
            return (
                np.asarray(place_ctx, dtype=np.int64),
                np.asarray(place_ti, dtype=np.int64),
                np.asarray(place_rows, dtype=np.int64),
                np.asarray(place_cols, dtype=np.int64),
                np.asarray(newbin_ctx, dtype=np.int64),
            )

        best = np.asarray(best_ti_per_ctx, dtype=np.int64)
        valid = np.asarray(sc_valid, dtype=bool)
        in_range = (best >= 0) & (best < len(valid))
        place_mask = np.zeros_like(in_range, dtype=bool)
        if len(valid) > 0:
            idx = best[in_range]
            place_mask[in_range] = valid[idx]
        place_ctx = np.nonzero(place_mask)[0].astype(np.int64, copy=False)
        place_ti = best[place_ctx].astype(np.int64, copy=False)
        rows = np.asarray(sc_rows, dtype=np.float64)
        cols = np.asarray(sc_cols, dtype=np.float64)
        place_rows = rows[place_ti].astype(np.int64, copy=False)
        place_cols = cols[place_ti].astype(np.int64, copy=False)
        newbin_ctx = np.nonzero(~place_mask)[0].astype(np.int64, copy=False)
        return place_ctx, place_ti, place_rows, place_cols, newbin_ctx

    def _phase6_cached_row_vacancy(self, part_data, rot, bin_width):
        key = (int(part_data.id), int(rot), int(bin_width))
        cached = self._phase6_row_vacancy_cache.get(key)
        if cached is not None:
            return cached

        part_matrix = part_data.rotations_uint8[rot]
        h, w = part_matrix.shape
        probe = np.zeros((h, bin_width), dtype=np.uint8)
        probe[:, :w] = part_matrix
        row_vacancy = np.empty(h, dtype=np.int32)
        update_vacancy_vector_rows(row_vacancy, probe, 0)
        self._phase6_row_vacancy_cache[key] = row_vacancy
        return row_vacancy
    
    def _process_wave_true_batch(self, contexts, mach_data, grid_states, grid_ffts,
                                 row_idx, col_idx, neg_inf):
        """
        TRUE cross-solution batching: batch FFT across ALL contexts at once.
        
        Algorithm:
        1. For each context, gather metadata about parts/bins/rotations to test
        2. Batch update ALL grid FFTs that are invalid
        3. For each context, try bins in first-fit order with batched validity check
        4. Batch the final placement FFTs for all contexts
        """
        if not contexts:
            return
        
        H, W = mach_data.bin_length, mach_data.bin_width
        
        # Phase 1: Gather part info and check feasibility for each context
        context_info = []  # [(ctx, part_data, mach_part_data), ...]
        
        for ctx in contexts:
            if ctx.current_part_idx >= len(ctx.parts_sequence):
                ctx.is_done = True
                continue
            
            part_id = ctx.parts_sequence[ctx.current_part_idx]
            part_data = self.parts[part_id]
            mach_part_data = mach_data.parts[part_id]
            
            # Check if part fits machine
            shape0 = part_data.shapes[0]
            if ((shape0[0] > H or shape0[1] > W) and (shape0[1] > H or shape0[0] > W)):
                ctx.is_feasible = False
                continue
            
            context_info.append((ctx, part_data, mach_part_data))
        
        if not context_info:
            return
        
        # Phase 2: Update all invalid grid FFTs in one batch
        invalid_grid_indices = []
        invalid_bin_states = []
        for ctx, _, _ in context_info:
            for bin_state in ctx.open_bins:
                if not bin_state.grid_fft_valid:
                    invalid_grid_indices.append(bin_state.grid_state_idx)
                    invalid_bin_states.append(bin_state)
        
        if invalid_grid_indices:
            with torch.inference_mode():
                indices = torch.tensor(invalid_grid_indices, device=self.device)
                batch_grids = grid_states[indices]  # (N, H, W)
                batch_ffts = torch.fft.rfft2(batch_grids)  # ONE batched rFFT!
                grid_ffts[indices] = batch_ffts
                for bs in invalid_bin_states:
                    bs.grid_fft_valid = True
        
        # Phase 3 / Phase 4: Two-pass collection + IFFT.
        #
        # Because bin_idx dominates the composite score (×1e9 multiplier), any valid
        # placement in bin B always beats a valid placement in bin B+1.  We exploit this:
        #   Pass 1 — for each context, test only its FIRST bin that passes area + vacancy.
        #   Pass 2 — only for contexts with no geometric hit in Pass 1, test remaining bins.
        # In the common case (most parts fit into their first open bin), Pass 2 is tiny or
        # empty, halving or better the total IFFT work.
        #
        # Parallel arrays use the same layout as before; Pass-1 and Pass-2 results are
        # concatenated before Phase 5, which is left entirely unchanged.

        n_contexts = len(context_info)
        ctx_first_valid_bin = [-1] * n_contexts   # bin_idx of first vacancy-passing bin

        # ---- Pass 1 collection ---------------------------------------------------
        p1_grid_indices      = []
        p1_part_ffts         = []
        p1_heights           = []
        p1_widths            = []
        p1_bin_indices       = []
        p1_shapes            = []
        p1_bin_states        = []
        p1_rotations         = []
        p1_ctx_indices       = []
        p1_enclosure_lengths = []
        p1_bin_areas         = []
        p1_part_areas        = []
        p1_n_tests = 0

        p1_native_candidates = [
            ctx_idx
            for ctx_idx, (ctx, part_data, _) in enumerate(context_info)
            if (part_data.id not in self._debug_part_ids)
            and part_data.densities_flat is not None
            and part_data.shapes_heights is not None
            and part_data.shapes_widths is not None
            and part_data.density_offsets is not None
            and ctx.open_bins
        ]
        p1_native_handled = set()
        p1_native = self._collect_phase3_tests_native_batch(
            context_info=context_info,
            ctx_indices=p1_native_candidates,
            H=H,
            W=W,
            mode=0,
            ctx_first_valid_bin=ctx_first_valid_bin,
        )
        if p1_native is not None:
            p1_native_handled = set(p1_native_candidates)
            first_valid_global, out_ctx_global, out_bin_local, out_rot_local = p1_native
            for ctx_idx in p1_native_candidates:
                if first_valid_global[ctx_idx] != -1:
                    ctx_first_valid_bin[ctx_idx] = int(first_valid_global[ctx_idx])

            for ctx_idx_i, b_local_i, rot_i in zip(out_ctx_global, out_bin_local, out_rot_local):
                ctx_idx = int(ctx_idx_i)
                b_local = int(b_local_i)
                rot = int(rot_i)
                ctx, part_data, mach_part_data = context_info[ctx_idx]
                bin_state = ctx.open_bins[b_local]
                shape = part_data.shapes[rot]
                p1_grid_indices.append(bin_state.grid_state_idx)
                p1_part_ffts.append(mach_part_data.ffts[rot])
                p1_heights.append(shape[0])
                p1_widths.append(shape[1])
                p1_bin_indices.append(b_local)
                p1_shapes.append(shape)
                p1_bin_states.append(bin_state)
                p1_rotations.append(rot)
                p1_ctx_indices.append(ctx_idx)
                p1_enclosure_lengths.append(bin_state.enclosure_box_length)
                p1_bin_areas.append(bin_state.area)
                p1_part_areas.append(part_data.area)
                p1_n_tests += 1

        for ctx_idx, (ctx, part_data, mach_part_data) in enumerate(context_info):
            if ctx_idx in p1_native_handled:
                continue
            _dbg = part_data.id in self._debug_part_ids

            if _dbg:
                print(f"\n[DEBUG] Part {part_data.id} | Machine {ctx.machine_idx} | "
                      f"Open bins: {len(ctx.open_bins)} | Area: {part_data.area:.1f} | "
                      f"Shapes: {part_data.shapes[:part_data.nrot]}")
            for bin_idx, bin_state in enumerate(ctx.open_bins):
                if bin_state.area + part_data.area > ctx.bin_area:
                    if _dbg:
                        print(f"  Bin {bin_idx}: SKIP area ({bin_state.area:.1f} + "
                              f"{part_data.area:.1f} > {ctx.bin_area:.1f})")
                    continue
                rots_passing = []
                for rot in range(part_data.nrot):
                    shape = part_data.shapes[rot]
                    if shape[0] > H or shape[1] > W:
                        if _dbg:
                            print(f"  Bin {bin_idx} rot {rot}: SKIP dims "
                                  f"({shape[0]}x{shape[1]} vs {H}x{W})")
                        continue
                    vac_pass = check_vacancy_fit_simple(bin_state.vacancy_vector,
                                               part_data.densities[rot])
                    if _dbg:
                        print(f"  Bin {bin_idx} rot {rot}: vacancy={'PASS' if vac_pass else 'FAIL'} "
                              f"(shape {shape[0]}x{shape[1]}, "
                              f"enc_box={bin_state.enclosure_box_length})")
                    if vac_pass:
                        rots_passing.append((rot, shape))
                if rots_passing:
                    ctx_first_valid_bin[ctx_idx] = bin_idx
                    if _dbg:
                        print(f"  → Pass 1 selects Bin {bin_idx} "
                              f"({len(rots_passing)} rotations passing)")
                    for rot, shape in rots_passing:
                        p1_grid_indices.append(bin_state.grid_state_idx)
                        p1_part_ffts.append(mach_part_data.ffts[rot])
                        p1_heights.append(shape[0])
                        p1_widths.append(shape[1])
                        p1_bin_indices.append(bin_idx)
                        p1_shapes.append(shape)
                        p1_bin_states.append(bin_state)
                        p1_rotations.append(rot)
                        p1_ctx_indices.append(ctx_idx)
                        p1_enclosure_lengths.append(bin_state.enclosure_box_length)
                        p1_bin_areas.append(bin_state.area)
                        p1_part_areas.append(part_data.area)
                        p1_n_tests += 1
                    break  # only first valid bin goes into Pass 1
                elif _dbg:
                    print(f"  Bin {bin_idx}: no rotations pass vacancy")

        # ---- Phase 4a: IFFT for Pass 1 ------------------------------------------
        if p1_n_tests:
            p1_placement_results, p1_score_comp = self._batch_fft_all_tests(
                p1_n_tests, p1_grid_indices, p1_part_ffts, p1_heights, p1_widths,
                p1_bin_indices, p1_enclosure_lengths, p1_bin_areas, p1_part_areas,
                grid_ffts, H, W, row_idx, col_idx, neg_inf)
        else:
            _empty_sc = {'bin_indices': np.array([], dtype=np.float64),
                         'densities': np.array([], dtype=np.float64),
                         'rows': np.array([], dtype=np.float64),
                         'cols': np.array([], dtype=np.float64),
                         'valid': np.array([], dtype=bool)}
            p1_placement_results, p1_score_comp = [], _empty_sc

        # Determine which contexts already have a valid geometric placement from Pass 1.
        # Valid = score strictly above -1e18 (the sentinel for "no zero-overlap position").
        ctx_p1_hit = [False] * n_contexts
        for ti, ctx_idx in enumerate(p1_ctx_indices):
            if p1_placement_results[ti] is not None:
                ctx_p1_hit[ctx_idx] = True

        # Debug: report Pass 1 IFFT results for traced parts
        if self._debug_part_ids:
            for ti, ctx_idx in enumerate(p1_ctx_indices):
                _pd = context_info[ctx_idx][1]
                if _pd.id in self._debug_part_ids:
                    _sc_info = ""
                    if p1_score_comp['valid'][ti]:
                        _sc_info = (f"density={p1_score_comp['densities'][ti]:.4f} "
                                    f"row={p1_score_comp['rows'][ti]:.0f} "
                                    f"col={p1_score_comp['cols'][ti]:.0f}")
                    print(f"[DEBUG] Part {_pd.id} Pass1 test {ti}: "
                          f"bin={p1_bin_indices[ti]} rot={p1_rotations[ti]} "
                          f"shape={p1_shapes[ti]} "
                          f"result={p1_placement_results[ti]} "
                          f"{_sc_info} "
                          f"geom={'HIT' if p1_placement_results[ti] is not None else 'MISS'}")
                    if ctx_p1_hit[ctx_idx]:
                        print(f"  → Pass 1 HIT for ctx {ctx_idx}")
                    else:
                        print(f"  → Pass 1 MISS — will go to Pass 2")

        # ---- Pass 2 collection: remaining bins for Pass-1 misses ----------------
        p2_grid_indices      = []
        p2_part_ffts         = []
        p2_heights           = []
        p2_widths            = []
        p2_bin_indices       = []
        p2_shapes            = []
        p2_bin_states        = []
        p2_rotations         = []
        p2_ctx_indices       = []
        p2_enclosure_lengths = []
        p2_bin_areas         = []
        p2_part_areas        = []
        p2_n_tests = 0

        p2_native_candidates = [
            ctx_idx
            for ctx_idx, (ctx, part_data, _) in enumerate(context_info)
            if not ctx_p1_hit[ctx_idx]
            and (part_data.id not in self._debug_part_ids)
            and part_data.densities_flat is not None
            and part_data.shapes_heights is not None
            and part_data.shapes_widths is not None
            and part_data.density_offsets is not None
            and ctx.open_bins
        ]
        p2_native_handled = set()
        p2_native = self._collect_phase3_tests_native_batch(
            context_info=context_info,
            ctx_indices=p2_native_candidates,
            H=H,
            W=W,
            mode=1,
            ctx_first_valid_bin=ctx_first_valid_bin,
        )
        if p2_native is not None:
            p2_native_handled = set(p2_native_candidates)
            _, out_ctx_global, out_bin_local, out_rot_local = p2_native
            for ctx_idx_i, b_local_i, rot_i in zip(out_ctx_global, out_bin_local, out_rot_local):
                ctx_idx = int(ctx_idx_i)
                b_local = int(b_local_i)
                rot = int(rot_i)
                ctx, part_data, mach_part_data = context_info[ctx_idx]
                bin_state = ctx.open_bins[b_local]
                shape = part_data.shapes[rot]
                p2_grid_indices.append(bin_state.grid_state_idx)
                p2_part_ffts.append(mach_part_data.ffts[rot])
                p2_heights.append(shape[0])
                p2_widths.append(shape[1])
                p2_bin_indices.append(b_local)
                p2_shapes.append(shape)
                p2_bin_states.append(bin_state)
                p2_rotations.append(rot)
                p2_ctx_indices.append(ctx_idx)
                p2_enclosure_lengths.append(bin_state.enclosure_box_length)
                p2_bin_areas.append(bin_state.area)
                p2_part_areas.append(part_data.area)
                p2_n_tests += 1

        for ctx_idx, (ctx, part_data, mach_part_data) in enumerate(context_info):
            if ctx_p1_hit[ctx_idx]:
                continue  # Pass 1 already found a valid placement
            if ctx_idx in p2_native_handled:
                continue
            _dbg2 = part_data.id in self._debug_part_ids
            first_valid = ctx_first_valid_bin[ctx_idx]  # skip this bin (already tested)

            if _dbg2:
                print(f"[DEBUG] Part {part_data.id} Pass2: first_valid={first_valid}, "
                      f"open_bins={len(ctx.open_bins)}")
            for bin_idx, bin_state in enumerate(ctx.open_bins):
                if bin_idx == first_valid:
                    if _dbg2:
                        print(f"  Bin {bin_idx}: SKIP (already tested in Pass 1)")
                    continue  # already in Pass 1 (and found no geometric fit)
                if bin_state.area + part_data.area > ctx.bin_area:
                    if _dbg2:
                        print(f"  Bin {bin_idx}: SKIP area ({bin_state.area:.1f} + "
                              f"{part_data.area:.1f} > {ctx.bin_area:.1f})")
                    continue
                for rot in range(part_data.nrot):
                    shape = part_data.shapes[rot]
                    if shape[0] > H or shape[1] > W:
                        continue
                    vac_pass = check_vacancy_fit_simple(bin_state.vacancy_vector,
                                               part_data.densities[rot])
                    if _dbg2:
                        print(f"  Bin {bin_idx} rot {rot}: vacancy={'PASS' if vac_pass else 'FAIL'} "
                              f"(shape {shape[0]}x{shape[1]})")
                    if vac_pass:
                        p2_grid_indices.append(bin_state.grid_state_idx)
                        p2_part_ffts.append(mach_part_data.ffts[rot])
                        p2_heights.append(shape[0])
                        p2_widths.append(shape[1])
                        p2_bin_indices.append(bin_idx)
                        p2_shapes.append(shape)
                        p2_bin_states.append(bin_state)
                        p2_rotations.append(rot)
                        p2_ctx_indices.append(ctx_idx)
                        p2_enclosure_lengths.append(bin_state.enclosure_box_length)
                        p2_bin_areas.append(bin_state.area)
                        p2_part_areas.append(part_data.area)
                        p2_n_tests += 1

        if self._debug_part_ids and p2_n_tests:
            print(f"[DEBUG] Pass 2: {p2_n_tests} tests collected")

        # ---- Phase 4b: IFFT for Pass 2 ------------------------------------------
        if p2_n_tests:
            p2_placement_results, p2_score_comp = self._batch_fft_all_tests(
                p2_n_tests, p2_grid_indices, p2_part_ffts, p2_heights, p2_widths,
                p2_bin_indices, p2_enclosure_lengths, p2_bin_areas, p2_part_areas,
                grid_ffts, H, W, row_idx, col_idx, neg_inf)
        else:
            _empty_sc = {'bin_indices': np.array([], dtype=np.float64),
                         'densities': np.array([], dtype=np.float64),
                         'rows': np.array([], dtype=np.float64),
                         'cols': np.array([], dtype=np.float64),
                         'valid': np.array([], dtype=bool)}
            p2_placement_results, p2_score_comp = [], _empty_sc

        # Merge Pass 1 + Pass 2 into flat arrays that Phase 5 consumes unchanged.
        test_ctx_indices  = p1_ctx_indices  + p2_ctx_indices
        test_bin_states   = p1_bin_states   + p2_bin_states
        test_rotations    = p1_rotations    + p2_rotations
        test_shapes       = p1_shapes       + p2_shapes
        placement_results = p1_placement_results + p2_placement_results
        # Merge score components
        if p2_n_tests:
            sc_bin_indices = np.concatenate([p1_score_comp['bin_indices'], p2_score_comp['bin_indices']])
            sc_densities   = np.concatenate([p1_score_comp['densities'],  p2_score_comp['densities']])
            sc_rows        = np.concatenate([p1_score_comp['rows'],       p2_score_comp['rows']])
            sc_cols        = np.concatenate([p1_score_comp['cols'],       p2_score_comp['cols']])
            sc_valid       = np.concatenate([p1_score_comp['valid'],      p2_score_comp['valid']])
        else:
            sc_bin_indices = p1_score_comp['bin_indices']
            sc_densities   = p1_score_comp['densities']
            sc_rows        = p1_score_comp['rows']
            sc_cols        = p1_score_comp['cols']
            sc_valid       = p1_score_comp['valid']

        # Phase 5: Find best placement per context using proper lexicographic comparison.
        # Matches PP's tie-breaking exactly:
        #   1. Lower bin_idx wins (fewer bins is better)
        #   2. Higher packing density wins (any nonzero difference decides)
        #   3. Larger row wins (bottom-left heuristic)
        #   4. Smaller col wins (bottom-left heuristic)
        n_contexts = len(context_info)
        best_ti_per_ctx = self._select_best_tests_per_context(
            test_ctx_indices=test_ctx_indices,
            sc_bin_indices=sc_bin_indices,
            sc_densities=sc_densities,
            sc_rows=sc_rows,
            sc_cols=sc_cols,
            sc_valid=sc_valid,
            n_contexts=n_contexts,
        )

        # Collect placements; separate GPU and CPU work so the GPU kernel
        # fires in one shot and CPU updates can overlap with its execution.
        contexts_needing_new_bin = []
        _placements = []  # (bin_state, col, row, rot, shape, part_data, mach_part_data)

        if self._debug_part_ids:
            for ctx_idx, (ctx, part_data, mach_part_data) in enumerate(context_info):
                ti = int(best_ti_per_ctx[ctx_idx])
                if part_data.id in self._debug_part_ids:
                    if ti == -1:
                        print(f"[DEBUG] Part {part_data.id} Phase5: NO test found → new bin")
                    elif placement_results[ti] is None:
                        print(f"[DEBUG] Part {part_data.id} Phase5: best test {ti} has None result → new bin")
                    else:
                        print(f"[DEBUG] Part {part_data.id} Phase5: best test {ti}, "
                              f"bin={test_bin_states[ti].bin_idx}, "
                              f"rot={test_rotations[ti]}, "
                              f"pos={placement_results[ti]}, "
                              f"density={sc_densities[ti]:.4f} "
                              f"row={sc_rows[ti]:.0f} col={sc_cols[ti]:.0f}")

                if ti == -1 or placement_results[ti] is None:
                    contexts_needing_new_bin.append((ctx, part_data, mach_part_data))
                    continue

                col, row = placement_results[ti]
                bin_state = test_bin_states[ti]
                rot = test_rotations[ti]
                shape = test_shapes[ti]

                if self._collect_contexts:
                    mlog = self._placement_log.setdefault(ctx.machine_idx, [])
                    mlog.append((part_data.id, bin_state.bin_idx, col, row, rot, shape))

                bin_state.grid_fft_valid = False
                _placements.append((bin_state, col, row, rot, shape, part_data, mach_part_data))
                ctx.current_part_idx += 1
        else:
            place_ctx, place_ti, place_rows, place_cols, newbin_ctx = self._plan_phase56_contexts(
                best_ti_per_ctx=best_ti_per_ctx,
                sc_valid=sc_valid,
                sc_rows=sc_rows,
                sc_cols=sc_cols,
            )

            for ctx_idx in newbin_ctx.tolist():
                ctx, part_data, mach_part_data = context_info[int(ctx_idx)]
                contexts_needing_new_bin.append((ctx, part_data, mach_part_data))

            for ctx_idx, ti, row, col in zip(
                place_ctx.tolist(),
                place_ti.tolist(),
                place_rows.tolist(),
                place_cols.tolist(),
            ):
                ctx, part_data, mach_part_data = context_info[int(ctx_idx)]
                bin_state = test_bin_states[int(ti)]
                rot = test_rotations[int(ti)]
                shape = test_shapes[int(ti)]

                if self._collect_contexts:
                    mlog = self._placement_log.setdefault(ctx.machine_idx, [])
                    mlog.append((part_data.id, bin_state.bin_idx, col, row, rot, shape))

                bin_state.grid_fft_valid = False
                _placements.append((bin_state, col, row, rot, shape, part_data, mach_part_data))
                ctx.current_part_idx += 1

        if _placements:
            if self.flat_parts_gpu is not None:
                # Option A: single CUDA kernel launch for all placements
                _kernel_args = []
                for bin_state, col, row, rot, shape, pd_, _ in _placements:
                    y_start = row - shape[0] + 1
                    flat_offset, ph, pw = self.part_update_meta[(pd_.id, rot)]
                    _kernel_args.append(
                        (bin_state.grid_state_idx, y_start, col, flat_offset, ph, pw))
                _cuda_batch_update(grid_states, self.flat_parts_gpu, _kernel_args, H, W)
            else:
                # Option C fallback: tight GPU loop (no custom kernel)
                for bin_state, col, row, rot, shape, pd_, _ in _placements:
                    y_start = row - shape[0] + 1
                    part_gpu = (pd_.rotations_gpu[rot]
                                if pd_.rotations_gpu and pd_.rotations_gpu[rot] is not None
                                else torch.as_tensor(pd_.rotations[rot],
                                                     dtype=torch.float32, device=self.device))
                    grid_states[bin_state.grid_state_idx, y_start:row+1,
                                col:col+shape[1]] += part_gpu

            # CPU updates — run while GPU kernel executes (async)
            for bin_state, col, row, rot, shape, pd_, mpd in _placements:
                y_start = row - shape[0] + 1
                bin_state.grid[y_start:row+1, col:col+shape[1]] += pd_.rotations_uint8[rot]
                update_vacancy_vector_rows(
                    bin_state.vacancy_vector, bin_state.grid[y_start:row+1, :], y_start)
                bin_state.parts_assigned.append(pd_.id)
                bin_state.area += pd_.area
                bin_state.min_occupied_row = min(bin_state.min_occupied_row, y_start)
                bin_state.max_occupied_row = max(bin_state.max_occupied_row, row)
                bin_state.enclosure_box_length = bin_state.bin_length - bin_state.min_occupied_row
                bin_state.proc_time += mpd.proc_time
                bin_state.proc_time_height = max(bin_state.proc_time_height, mpd.proc_time_height)
        
        # Phase 6: Batch new-bin creation.
        # Previously called _start_new_bin() per context (serial GPU ops).
        # Now: one index_fill_ to zero all new grids + one CUDA kernel for all first placements.
        if contexts_needing_new_bin:
            _new_placements = []  # (new_bin, part_data, mach_part_data, best_rot, shape)
            _new_grid_indices = []

            # Step 1: Create all BinState objects (CPU only)
            for ctx, part_data, mach_part_data in contexts_needing_new_bin:
                grid_idx = ctx.next_grid_idx
                ctx.next_grid_idx += 1
                new_bin = BinState(
                    bin_idx=len(ctx.open_bins),
                    grid=np.zeros((ctx.bin_length, ctx.bin_width), dtype=np.uint8),
                    vacancy_vector=np.full(ctx.bin_length, ctx.bin_width, dtype=np.int32),
                    grid_state_idx=grid_idx,
                    area=0.0, enclosure_box_length=0,
                    min_occupied_row=ctx.bin_length, max_occupied_row=-1,
                    proc_time=0.0, proc_time_height=0.0,
                    grid_fft_valid=False, parts_assigned=[],
                    bin_length=ctx.bin_length, bin_width=ctx.bin_width
                )
                best_rot = part_data.best_rotation
                shape = part_data.shapes[best_rot]
                ctx.open_bins.append(new_bin)
                ctx.current_part_idx += 1
                _new_placements.append((new_bin, part_data, mach_part_data, best_rot, shape))
                _new_grid_indices.append(grid_idx)

                if self._collect_contexts:
                    y_start = ctx.bin_length - shape[0]
                    mlog = self._placement_log.setdefault(ctx.machine_idx, [])
                    mlog.append((part_data.id, new_bin.bin_idx, 0,
                                 ctx.bin_length - 1, best_rot, shape))

            # Step 2: Zero all new GPU grids in one op
            _new_idx_t = torch.tensor(_new_grid_indices, device=self.device, dtype=torch.long)
            grid_states.index_fill_(0, _new_idx_t, 0.0)

            # Step 3: Batch GPU part placements (single CUDA kernel or tight fallback loop)
            if self.flat_parts_gpu is not None:
                _kernel_args = []
                for new_bin, part_data, _, best_rot, shape in _new_placements:
                    y_start = new_bin.bin_length - shape[0]
                    flat_offset, ph, pw = self.part_update_meta[(part_data.id, best_rot)]
                    _kernel_args.append((new_bin.grid_state_idx, y_start, 0, flat_offset, ph, pw))
                _cuda_batch_update(grid_states, self.flat_parts_gpu, _kernel_args, H, W)
            else:
                for new_bin, part_data, _, best_rot, shape in _new_placements:
                    y_start = new_bin.bin_length - shape[0]
                    part_gpu = (part_data.rotations_gpu[best_rot]
                                if part_data.rotations_gpu and part_data.rotations_gpu[best_rot] is not None
                                else torch.as_tensor(part_data.rotations[best_rot],
                                                     dtype=torch.float32, device=self.device))
                    grid_states[new_bin.grid_state_idx, y_start:new_bin.bin_length, 0:shape[1]] += part_gpu

            # Step 4: CPU updates — run while GPU kernel executes (async)
            if self._phase6_structured:
                for new_bin, part_data, mach_part_data, best_rot, shape in _new_placements:
                    y_start = new_bin.bin_length - shape[0]
                    new_bin.grid[y_start:new_bin.bin_length, 0:shape[1]] = part_data.rotations_uint8[best_rot]
                    new_bin.vacancy_vector[y_start:new_bin.bin_length] = self._phase6_cached_row_vacancy(
                        part_data, best_rot, new_bin.bin_width
                    )
                    new_bin.parts_assigned.append(part_data.id)
                    new_bin.area += part_data.area
                    new_bin.min_occupied_row = y_start
                    new_bin.max_occupied_row = new_bin.bin_length - 1
                    new_bin.enclosure_box_length = shape[0]
                    new_bin.proc_time += mach_part_data.proc_time
                    new_bin.proc_time_height = mach_part_data.proc_time_height
                    new_bin.grid_fft_valid = False
            else:
                for new_bin, part_data, mach_part_data, best_rot, shape in _new_placements:
                    y_start = new_bin.bin_length - shape[0]
                    new_bin.grid[y_start:new_bin.bin_length, 0:shape[1]] += part_data.rotations_uint8[best_rot]
                    update_vacancy_vector_rows(
                        new_bin.vacancy_vector, new_bin.grid[y_start:new_bin.bin_length, :], y_start)
                    new_bin.parts_assigned.append(part_data.id)
                    new_bin.area += part_data.area
                    new_bin.min_occupied_row = y_start
                    new_bin.max_occupied_row = new_bin.bin_length - 1
                    new_bin.enclosure_box_length = shape[0]
                    new_bin.proc_time += mach_part_data.proc_time
                    new_bin.proc_time_height = mach_part_data.proc_time_height
                    new_bin.grid_fft_valid = False
    
    def _batch_fft_all_tests(self, n_tests, test_grid_indices, test_part_ffts,
                              test_heights, test_widths,
                              test_bin_indices, test_enclosure_lengths,
                              test_bin_areas, test_part_areas,
                              grid_ffts, H, W, row_idx, col_idx, neg_inf):
        """
        Perform batched IFFT for ALL tests across ALL contexts.
        Chunks to avoid OOM with large batches.

        Returns (all_results, score_components) where score_components is a dict
        with arrays for bin_indices, densities, rows, cols, and valid mask.
        Phase 5 uses these for proper lexicographic comparison matching PP's
        tie-breaking: bin_idx (lower wins) > density (higher) > row (larger) > col (smaller).
        """
        if n_tests == 0:
            return [], {'bin_indices': np.array([], dtype=np.float64),
                        'densities': np.array([], dtype=np.float64),
                        'rows': np.array([], dtype=np.float64),
                        'cols': np.array([], dtype=np.float64),
                        'valid': np.array([], dtype=bool)}

        # Chunk to avoid OOM - empirically tuned for RTX A4000 (16GB)
        # Benchmarked: 250=6.05s, 500=5.90s, 750=5.88s (best), 1000=5.89s, 1500=5.88s, 2000+=OOM risk
        # 750-1500 all within noise; 750 chosen as sweet spot with safe VRAM headroom
        CHUNK_SIZE = 750
        all_results = [None] * n_tests
        all_scores  = np.full(n_tests, -1e18, dtype=np.float64)
        rows_np  = np.zeros(n_tests, dtype=np.float64)
        cols_np  = np.zeros(n_tests, dtype=np.float64)
        valid_np = np.zeros(n_tests, dtype=bool)

        # Pre-build GPU index tensors once; per-chunk slices are zero-cost views.
        # Score data (bin_indices, enclosure_lengths, bin_areas, part_areas) stays on CPU
        # — numpy vectorized ops on ~2000 elements are sub-millisecond and avoid the
        # ~13 extra CUDA kernel launches per chunk that GPU scoring would require.
        with torch.inference_mode():
            all_grid_indices = torch.tensor(test_grid_indices, device=self.device, dtype=torch.long)
            all_heights      = torch.tensor(test_heights,      device=self.device, dtype=torch.long)
            all_widths       = torch.tensor(test_widths,       device=self.device, dtype=torch.long)

        for chunk_start in range(0, n_tests, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n_tests)
            chunk_n = chunk_end - chunk_start

            with torch.inference_mode():
                # Free views into pre-built GPU tensors — no allocation, no CUDA sync
                grid_indices    = all_grid_indices[chunk_start:chunk_end]
                part_heights    = all_heights[chunk_start:chunk_end]
                part_widths     = all_widths[chunk_start:chunk_end]

                batch_grid_ffts = grid_ffts[grid_indices]  # (chunk_n, H, W//2+1)
                batch_part_ffts = torch.stack(test_part_ffts[chunk_start:chunk_end], dim=0)

                # Batched IFFT (real-valued output via irfft2)
                overlap_batch = torch.fft.irfft2(batch_grid_ffts * batch_part_ffts, s=(H, W))

                # Fused post-IFFT scoring via torch.compile (Triton kernel).
                # Replaces ~10 separate elementwise/reduction kernel launches with ~2.
                has_valid, smallest_cols, best_row = _post_ifft_score(
                    overlap_batch, part_heights, part_widths, row_idx, col_idx, neg_inf)

                # Transfer (has_valid, col, row) to CPU — same as original
                results_cpu = torch.stack([has_valid.int(), smallest_cols, best_row],
                                          dim=1).cpu().numpy()

            for i in range(chunk_n):
                if results_cpu[i, 0] == 1:
                    c, r = int(results_cpu[i, 1]), int(results_cpu[i, 2])
                    all_results[chunk_start + i] = (c, r)
                    cols_np[chunk_start + i] = c
                    rows_np[chunk_start + i] = r
                    valid_np[chunk_start + i] = True

        # Compute score components on CPU with numpy (float64).
        # Phase 5 uses proper lexicographic comparison matching PP's tie-breaking:
        #   bin_idx (lower wins) > density (higher wins) > row (larger wins) > col (smaller wins)
        enc_np = np.asarray(test_enclosure_lengths, dtype=np.float64)
        ba_np  = np.asarray(test_bin_areas,          dtype=np.float64)
        pa_np  = np.asarray(test_part_areas,         dtype=np.float64)
        bi_np  = np.asarray(test_bin_indices,        dtype=np.float64)
        ht_np  = np.asarray(test_heights,            dtype=np.float64)

        y_starts  = rows_np - ht_np + 1
        new_lens  = np.maximum(enc_np, H - y_starts)
        densities = (ba_np + pa_np) / (new_lens * W)

        score_components = {
            'bin_indices': bi_np,
            'densities': densities,
            'rows': rows_np,
            'cols': cols_np,
            'valid': valid_np,
        }

        return all_results, score_components
    
    def _place_part_in_bin(self, bin_state, x, y, part_matrix, shape, area, mach_part_data, grid_states,
                           part_gpu_tensor=None):
        y_start = y - shape[0] + 1
        y_end = y + 1

        bin_state.grid[y_start:y_end, x:x+shape[1]] += part_matrix

        if part_gpu_tensor is None:
            part_gpu_tensor = torch.as_tensor(part_matrix, dtype=torch.float32, device=self.device)
        grid_states[bin_state.grid_state_idx, y_start:y_end, x:x+shape[1]] += part_gpu_tensor
        bin_state.grid_fft_valid = False
        
        update_vacancy_vector_rows(bin_state.vacancy_vector, bin_state.grid[y_start:y_end, :], y_start)
        
        bin_state.area += area
        bin_state.min_occupied_row = min(bin_state.min_occupied_row, y_start)
        bin_state.max_occupied_row = max(bin_state.max_occupied_row, y)
        bin_state.enclosure_box_length = bin_state.bin_length - bin_state.min_occupied_row
        bin_state.proc_time += mach_part_data.proc_time
        bin_state.proc_time_height = max(bin_state.proc_time_height, mach_part_data.proc_time_height)
    
    def _start_new_bin(self, ctx, part_data, mach_part_data, mach_data, grid_states):
        grid_idx = ctx.next_grid_idx
        ctx.next_grid_idx += 1
        
        new_bin = BinState(
            bin_idx=len(ctx.open_bins),
            grid=np.zeros((ctx.bin_length, ctx.bin_width), dtype=np.uint8),
            vacancy_vector=np.full(ctx.bin_length, ctx.bin_width, dtype=np.int32),
            grid_state_idx=grid_idx,
            area=0.0, enclosure_box_length=0,
            min_occupied_row=ctx.bin_length, max_occupied_row=-1,
            proc_time=0.0, proc_time_height=0.0,
            grid_fft_valid=False, parts_assigned=[],
            bin_length=ctx.bin_length, bin_width=ctx.bin_width
        )
        
        grid_states[grid_idx].zero_()
        
        best_rot = part_data.best_rotation
        shape = part_data.shapes[best_rot]
        self._place_part_in_bin(new_bin, 0, ctx.bin_length - 1, part_data.rotations_uint8[best_rot],
                                shape, part_data.area, mach_part_data, grid_states,
                                part_gpu_tensor=part_data.rotations_gpu[best_rot])
        ctx.open_bins.append(new_bin)


def evaluate_batch_wave(problem_data, nbParts, nbMachines, thresholds,
                        chromosomes, instance_parts, collision_backend):
    evaluator = WaveBatchEvaluator(
        problem_data, nbParts, nbMachines, thresholds,
        instance_parts, collision_backend
    )
    return evaluator.evaluate_batch(chromosomes)
