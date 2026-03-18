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
        grid_ffts = torch.zeros((max_total_bins, H, W), dtype=torch.complex64, device=self.device)
        
        # Process waves
        max_waves = max(len(seq) for seq in sequences) * 3 if sequences else 0
        for wave in range(max_waves):
            active = [c for c in contexts if not c.is_done and c.is_feasible]
            if not active:
                break
            self._process_wave_true_batch(active, mach_data, grid_states, grid_ffts)
        
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
    
    def _process_wave_true_batch(self, contexts, mach_data, grid_states, grid_ffts):
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
                batch_ffts = torch.fft.fft2(batch_grids)  # ONE batched FFT!
                grid_ffts[indices] = batch_ffts
                for bs in invalid_bin_states:
                    bs.grid_fft_valid = True
        
        # Phase 3: Collect ALL (context, bin, rotation) tests to batch
        # Structure: list of (ctx_idx, bin_state, rot, shape, part_fft, part_data, mach_part_data)
        all_tests = []
        ctx_to_tests = {i: [] for i in range(len(context_info))}
        
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
                    dens = part_data.densities[rot].astype(np.int32)
                    if check_vacancy_fit_simple(bin_state.vacancy_vector, dens):
                        test_entry = {
                            'ctx_idx': ctx_idx,
                            'bin_idx': bin_idx,
                            'bin_state': bin_state,
                            'rot': rot,
                            'shape': shape,
                            'part_fft': mach_part_data.ffts[rot],
                            'part_data': part_data,
                            'mach_part_data': mach_part_data
                        }
                        all_tests.append(test_entry)
                        ctx_to_tests[ctx_idx].append(len(all_tests) - 1)
        
        # Phase 4: Batch FFT collision check for ALL tests at once
        if all_tests:
            placement_results = self._batch_fft_all_tests(all_tests, grid_ffts, H, W)
        else:
            placement_results = []
        
        # Phase 5: Process results - find best placement per context
        contexts_needing_new_bin = []
        
        for ctx_idx, (ctx, part_data, mach_part_data) in enumerate(context_info):
            test_indices = ctx_to_tests[ctx_idx]
            
            if not test_indices:
                # No feasible tests - need new bin
                contexts_needing_new_bin.append((ctx, part_data, mach_part_data))
                continue
            
            # Find best placement following first-fit + tie-breaking rules
            best_result = None
            best_bin_idx = float('inf')  # First-fit: prefer lower bin index
            best_density = 0
            best_row = -1
            best_col = float('inf')
            
            for test_idx in test_indices:
                test = all_tests[test_idx]
                result = placement_results[test_idx]
                
                if result is None:
                    continue
                
                col, row = result
                bin_idx = test['bin_idx']
                shape = test['shape']
                bin_state = test['bin_state']
                
                y_start = row - shape[0] + 1
                new_length = max(bin_state.enclosure_box_length, H - y_start)
                potential_area = bin_state.area + part_data.area
                density = potential_area / (new_length * W)
                
                # First-fit: always prefer lower bin index
                if bin_idx < best_bin_idx:
                    better = True
                elif bin_idx == best_bin_idx:
                    # Same bin: use density > row > col tie-breaking
                    if density > best_density:
                        better = True
                    elif density == best_density and row > best_row:
                        better = True
                    elif density == best_density and row == best_row and col < best_col:
                        better = True
                    else:
                        better = False
                else:
                    better = False
                
                if better:
                    best_bin_idx = bin_idx
                    best_density = density
                    best_row = row
                    best_col = col
                    best_result = (test['bin_state'], col, row, test['rot'], shape, 
                                   part_data, mach_part_data)
            
            if best_result is not None:
                bin_state, x, y, rot, shape, pd, mpd = best_result
                self._place_part_in_bin(bin_state, x, y, pd.rotations[rot], 
                                        shape, pd.area, mpd, grid_states)
                ctx.current_part_idx += 1
            else:
                contexts_needing_new_bin.append((ctx, part_data, mach_part_data))
        
        # Phase 6: Handle new bins
        for ctx, part_data, mach_part_data in contexts_needing_new_bin:
            self._start_new_bin(ctx, part_data, mach_part_data, mach_data, grid_states)
            ctx.current_part_idx += 1
    
    def _batch_fft_all_tests(self, all_tests, grid_ffts, H, W):
        """
        Perform ONE batched IFFT for ALL tests across ALL contexts.
        
        Returns list of results: (col, row) or None for each test.
        """
        n_tests = len(all_tests)
        
        with torch.inference_mode():
            # Gather grid FFTs and part FFTs
            grid_indices = torch.tensor([t['bin_state'].grid_state_idx for t in all_tests], 
                                        device=self.device, dtype=torch.long)
            batch_grid_ffts = grid_ffts[grid_indices]  # (n_tests, H, W)
            batch_part_ffts = torch.stack([t['part_fft'] for t in all_tests], dim=0)  # (n_tests, H, W)
            
            # ONE batched IFFT!
            overlap_batch = torch.fft.ifft2(batch_grid_ffts * batch_part_ffts).real
            rounded_batch = torch.round(overlap_batch)
            
            # Find valid positions for all tests
            zero_mask = (rounded_batch == 0)  # (n_tests, H, W)
            
            part_heights = torch.tensor([t['shape'][0] for t in all_tests], device=self.device)
            part_widths = torch.tensor([t['shape'][1] for t in all_tests], device=self.device)
            
            row_idx = torch.arange(H, device=self.device).view(1, H, 1)
            col_idx = torch.arange(W, device=self.device).view(1, 1, W)
            
            valid_row = row_idx >= (part_heights - 1).view(-1, 1, 1)
            valid_col = col_idx >= (part_widths - 1).view(-1, 1, 1)
            valid_mask = valid_row & valid_col
            valid_zeros = zero_mask[:, :H, :W] & valid_mask
            
            # Score: prefer bottom-left (high row, low col)
            score = torch.where(
                valid_zeros,
                row_idx.float() * (W + 1) - col_idx.float(),
                torch.tensor(-1e9, device=self.device)
            )
            
            flat_scores = score.view(n_tests, -1)
            best_flat_idx = flat_scores.argmax(dim=1)
            max_scores = flat_scores.max(dim=1).values
            
            best_row = best_flat_idx // W
            best_col = best_flat_idx % W
            has_valid = max_scores > -1e8
            smallest_cols = best_col - (part_widths - 1)
            
            # Transfer to CPU
            results_cpu = torch.stack([has_valid.int(), smallest_cols, best_row], dim=1).cpu().numpy()
        
        # Convert to list of results
        results = []
        for i in range(n_tests):
            if results_cpu[i, 0] == 1:
                results.append((int(results_cpu[i, 1]), int(results_cpu[i, 2])))
            else:
                results.append(None)
        
        return results
    
    def _place_part_in_bin(self, bin_state, x, y, part_matrix, shape, area, mach_part_data, grid_states):
        y_start = y - shape[0] + 1
        y_end = y + 1
        
        bin_state.grid[y_start:y_end, x:x+shape[1]] += part_matrix.astype(np.uint8)
        
        part_tensor = torch.as_tensor(part_matrix, dtype=torch.float32, device=self.device)
        grid_states[bin_state.grid_state_idx, y_start:y_end, x:x+shape[1]] += part_tensor
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
        self._place_part_in_bin(new_bin, 0, ctx.bin_length - 1, part_data.rotations[best_rot],
                                shape, part_data.area, mach_part_data, grid_states)
        ctx.open_bins.append(new_bin)


def evaluate_batch_wave(problem_data, nbParts, nbMachines, thresholds,
                        chromosomes, instance_parts, collision_backend):
    evaluator = WaveBatchEvaluator(
        problem_data, nbParts, nbMachines, thresholds,
        instance_parts, collision_backend
    )
    return evaluator.evaluate_batch(chromosomes)
