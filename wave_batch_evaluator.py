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

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from numba_utils import check_vacancy_fit_simple, update_vacancy_vector_rows

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
        
        sequences = self._decode_sequences(chromosomes, machine_idx)
        contexts = self._init_batch_contexts(sequences, machine_idx, num_solutions, mach_data)
        
        # Allocate GPU tensors for grid states
        max_bins_per_sol = 10
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
    
    def _init_batch_contexts(self, sequences, machine_idx, num_solutions, mach_data):
        contexts = []
        max_bins_per_sol = 10
        
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
        
        # Phase 3: Collect ALL (context, bin, rotation) tests to batch
        # Parallel arrays replace list-of-dicts to avoid per-entry dict allocation and
        # string-keyed lookups in the hot path of _batch_fft_all_tests.
        test_grid_indices      = []   # bin_state.grid_state_idx       (for GPU gather)
        test_part_ffts         = []   # mach_part_data.ffts[rot]       (for torch.stack)
        test_heights           = []   # shape[0]                       (for valid-position mask)
        test_widths            = []   # shape[1]                       (for valid-position mask)
        test_bin_indices       = []   # bin_idx                        (for GPU score)
        test_shapes            = []   # shape tuple                    (for Phase-5 placement)
        test_bin_states        = []   # BinState object                (for Phase-5 placement)
        test_rotations         = []   # rot                            (for Phase-5 placement)
        test_ctx_indices       = []   # ctx_idx                        (for per-context argmax)
        test_enclosure_lengths = []   # bin_state.enclosure_box_length (for GPU density)
        test_bin_areas         = []   # bin_state.area                 (for GPU density)
        test_part_areas        = []   # part_data.area                 (for GPU density)
        n_tests = 0

        for ctx_idx, (ctx, part_data, mach_part_data) in enumerate(context_info):
            for bin_idx, bin_state in enumerate(ctx.open_bins):
                # Area check
                if bin_state.area + part_data.area > ctx.bin_area:
                    continue

                # Vacancy check for each rotation
                for rot in range(part_data.nrot):
                    shape = part_data.shapes[rot]
                    if shape[0] > H or shape[1] > W:
                        continue
                    dens = part_data.densities[rot]
                    if check_vacancy_fit_simple(bin_state.vacancy_vector, dens):
                        test_grid_indices.append(bin_state.grid_state_idx)
                        test_part_ffts.append(mach_part_data.ffts[rot])
                        test_heights.append(shape[0])
                        test_widths.append(shape[1])
                        test_bin_indices.append(bin_idx)
                        test_shapes.append(shape)
                        test_bin_states.append(bin_state)
                        test_rotations.append(rot)
                        test_ctx_indices.append(ctx_idx)
                        test_enclosure_lengths.append(bin_state.enclosure_box_length)
                        test_bin_areas.append(bin_state.area)
                        test_part_areas.append(part_data.area)
                        n_tests += 1
        
        # Phase 4: Batch FFT collision check for ALL tests at once
        if n_tests:
            placement_results, all_scores = self._batch_fft_all_tests(
                n_tests, test_grid_indices, test_part_ffts, test_heights, test_widths,
                test_bin_indices, test_enclosure_lengths, test_bin_areas, test_part_areas,
                grid_ffts, H, W, row_idx, col_idx, neg_inf)
        else:
            placement_results, all_scores = [], np.array([], dtype=np.float32)
        
        # Phase 5: Find best placement per context using GPU-computed scores.
        # Scores encode the full tie-breaking hierarchy (bin_idx > density > row > col)
        # so a single linear scan replaces the nested Python comparison loop.
        n_contexts = len(context_info)
        best_ti_per_ctx = [-1] * n_contexts          # winning test index per context
        best_sc_per_ctx = np.full(n_contexts, -np.inf, dtype=np.float32)

        for ti, ctx_idx in enumerate(test_ctx_indices):
            sc = all_scores[ti]
            if sc > best_sc_per_ctx[ctx_idx]:
                best_sc_per_ctx[ctx_idx] = sc
                best_ti_per_ctx[ctx_idx] = ti

        # Collect placements; separate GPU and CPU work so the GPU kernel
        # fires in one shot and CPU updates can overlap with its execution.
        contexts_needing_new_bin = []
        _placements = []  # (bin_state, col, row, rot, shape, part_data, mach_part_data)

        for ctx_idx, (ctx, part_data, mach_part_data) in enumerate(context_info):
            ti = best_ti_per_ctx[ctx_idx]

            if ti == -1 or placement_results[ti] is None:
                contexts_needing_new_bin.append((ctx, part_data, mach_part_data))
                continue

            col, row  = placement_results[ti]
            bin_state = test_bin_states[ti]
            rot       = test_rotations[ti]
            shape     = test_shapes[ti]

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
                    vacancy_vector=np.zeros(ctx.bin_length, dtype=np.int32) + ctx.bin_width,
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
            for new_bin, part_data, mach_part_data, best_rot, shape in _new_placements:
                y_start = new_bin.bin_length - shape[0]
                new_bin.grid[y_start:new_bin.bin_length, 0:shape[1]] += part_data.rotations_uint8[best_rot]
                update_vacancy_vector_rows(
                    new_bin.vacancy_vector, new_bin.grid[y_start:new_bin.bin_length, :], y_start)
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

        Computes a composite placement score on the GPU encoding the full
        tie-breaking hierarchy: bin_idx > density > row > col.
        Returns (all_results, all_scores) so Phase 5 only needs a linear scan.
        """
        if n_tests == 0:
            return [], np.array([], dtype=np.float32)

        # Chunk to avoid OOM - empirically tuned for RTX A4000 (16GB)
        # Benchmarked: 250=6.05s, 500=5.90s, 750=5.88s (best), 1000=5.89s, 1500=5.88s, 2000+=OOM risk
        # 750-1500 all within noise; 750 chosen as sweet spot with safe VRAM headroom
        CHUNK_SIZE = 750
        all_results = [None] * n_tests
        all_scores  = np.full(n_tests, -1e18, dtype=np.float32)
        rows_np  = np.zeros(n_tests, dtype=np.float32)
        cols_np  = np.zeros(n_tests, dtype=np.float32)
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

        # Compute composite placement scores on CPU with numpy.
        # ~2000-element vectorized ops are sub-millisecond; avoids ~13 CUDA kernel
        # launches per chunk that would be needed to do this on the GPU.
        # Score encodes tie-breaking hierarchy: bin_idx > density > row > col.
        # Multipliers chosen so each level dominates the next for actual value ranges:
        #   bin_idx 0-10, density 0-1, row 0-H (~300), col 0-W (~200).
        enc_np = np.asarray(test_enclosure_lengths, dtype=np.float32)
        ba_np  = np.asarray(test_bin_areas,          dtype=np.float32)
        pa_np  = np.asarray(test_part_areas,         dtype=np.float32)
        bi_np  = np.asarray(test_bin_indices,        dtype=np.float32)
        ht_np  = np.asarray(test_heights,            dtype=np.float32)

        y_starts  = rows_np - ht_np + 1
        new_lens  = np.maximum(enc_np, H - y_starts)
        densities = (ba_np + pa_np) / (new_lens * W)
        all_scores = -bi_np * 1e9 + densities * 1e6 + rows_np * 1e3 - cols_np
        all_scores[~valid_np] = -1e18

        return all_results, all_scores
    
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
            vacancy_vector=np.zeros(ctx.bin_length, dtype=np.int32) + ctx.bin_width,
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
