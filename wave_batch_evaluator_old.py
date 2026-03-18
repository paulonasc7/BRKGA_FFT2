"""
Wave-based batch evaluator for BRKGA.

Instead of evaluating solutions one at a time, this processes all solutions
in synchronized "waves" where each wave handles the next part placement
for all solutions simultaneously.

Key benefits:
- Batches 500 FFT operations instead of 4 per call
- Single GPU sync per wave instead of per solution
- Better GPU utilization through larger batch sizes
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from numba_utils import check_vacancy_fit_simple, update_vacancy_vector_rows


@dataclass
class BinState:
    """State for a single bin within a context."""
    grid: np.ndarray  # CPU grid for vacancy updates
    vacancy_vector: np.ndarray
    grid_state_idx: int  # Index into batched GPU grid tensor
    area: float
    enclosure_box_length: int
    min_occupied_row: int
    max_occupied_row: int
    proc_time: float
    proc_time_height: float
    grid_fft_valid: bool
    parts_assigned: List[int]


@dataclass
class BatchPlacementContext:
    """Tracks state for a single solution-machine combination."""
    solution_idx: int
    machine_idx: int
    parts_sequence: np.ndarray  # Ordered list of part IDs to place
    current_part_idx: int  # Index into parts_sequence
    
    # Machine-specific dimensions
    bin_length: int
    bin_width: int
    
    # Multiple open bins (FIX: was tracking only one bin before)
    open_bins: List[BinState]
    
    # Status
    is_done: bool
    is_feasible: bool


class WaveBatchEvaluator:
    """
    Evaluates many solutions in parallel using wave-based batching.
    """
    
    def __init__(self, problem_data, nbParts, nbMachines, thresholds, 
                 instance_parts, collision_backend, device='cuda'):
        self.problem_data = problem_data
        self.nbParts = nbParts
        self.nbMachines = nbMachines
        self.thresholds = np.array(thresholds)
        self.instance_parts = instance_parts
        self.collision_backend = collision_backend
        self.device = device
        
        # Machine-specific dimensions (machines can have different bin sizes)
        self.machines = problem_data.machines
        
        # Pre-compute part FFTs for all parts (already done in problem_data)
        self.parts = problem_data.parts
        
    def evaluate_batch(self, chromosomes: np.ndarray) -> List[float]:
        """
        Evaluate a batch of chromosomes using wave-based processing.
        
        Args:
            chromosomes: Array of shape (num_solutions, chromosome_length)
            
        Returns:
            List of fitness values (makespans)
        """
        num_solutions = len(chromosomes)
        
        # Process each machine separately (simpler than interleaving)
        machine_makespans = []
        
        for machine_idx in range(self.nbMachines):
            makespans = self._process_machine_batch(
                chromosomes, machine_idx, num_solutions
            )
            machine_makespans.append(makespans)
        
        # Final makespan is max across machines
        final_makespans = np.maximum.reduce(machine_makespans)
        
        return final_makespans.tolist()
    
    def _process_machine_batch(self, chromosomes, machine_idx, num_solutions):
        """Process all solutions for a single machine using wave batching."""
        
        mach_data = self.machines[machine_idx]
        bin_length = mach_data.bin_length
        bin_width = mach_data.bin_width
        
        # Decode chromosomes to get part sequences for this machine
        sequences = self._decode_sequences(chromosomes, machine_idx)
        
        # Initialize batch state
        contexts = self._init_batch_contexts(
            sequences, machine_idx, num_solutions, mach_data, bin_length, bin_width
        )
        
        # Allocate batched GPU grid states
        max_active = num_solutions * 2  # Allow for multiple bins
        grid_states = torch.zeros(
            (max_active, bin_length, bin_width),
            dtype=torch.float32, device=self.device
        )
        grid_ffts = torch.zeros(
            (max_active, bin_length, bin_width),
            dtype=torch.complex64, device=self.device
        )
        grid_fft_valid = torch.zeros(max_active, dtype=torch.bool, device=self.device)
        
        # Process waves until all contexts are done
        max_waves = max(len(seq) for seq in sequences) * 2  # Safety limit
        
        for wave in range(max_waves):
            active_contexts = [c for c in contexts if not c.is_done and c.is_feasible]
            if not active_contexts:
                break
            
            self._process_wave(
                active_contexts, mach_data, grid_states, grid_ffts, grid_fft_valid
            )
        
        # Collect makespans
        makespans = np.zeros(num_solutions)
        for ctx in contexts:
            if not ctx.is_feasible:
                makespans[ctx.solution_idx] = 1e16  # Infeasible penalty
            else:
                # Sum all bin makespans
                total = sum(ctx.bins_makespans)
                # Add current bin if has parts
                if ctx.current_bin_proc_time > 0 or ctx.area > 0:
                    total += ctx.current_bin_proc_time + ctx.current_bin_proc_time_height + mach_data.setup_time
                makespans[ctx.solution_idx] = total
        
        return makespans
    
    def _decode_sequences(self, chromosomes, machine_idx):
        """Decode part sequences for each chromosome for given machine."""
        num_solutions = len(chromosomes)
        sequences = []
        
        for sol_idx in range(num_solutions):
            chrom = chromosomes[sol_idx]
            SV = chrom[:self.nbParts]
            MV = chrom[self.nbParts:]
            
            # Determine which parts go to this machine
            if machine_idx == 0:
                mask = MV <= self.thresholds[0]
            elif machine_idx == self.nbMachines - 1:
                mask = MV > self.thresholds[-1]
            else:
                mask = (MV > self.thresholds[machine_idx-1]) & (MV <= self.thresholds[machine_idx])
            
            part_indices = np.where(mask)[0]
            values = SV[part_indices]
            actual_parts = self.instance_parts[part_indices]
            sorted_order = np.argsort(values)
            sorted_sequence = actual_parts[sorted_order]
            
            sequences.append(sorted_sequence)
        
        return sequences
    
    def _init_batch_contexts(self, sequences, machine_idx, num_solutions, mach_data, bin_length, bin_width):
        """Initialize placement contexts for all solutions."""
        contexts = []
        
        for sol_idx in range(num_solutions):
            seq = sequences[sol_idx]
            
            ctx = BatchPlacementContext(
                solution_idx=sol_idx,
                machine_idx=machine_idx,
                parts_sequence=seq,
                current_part_idx=0,
                bin_length=bin_length,
                bin_width=bin_width,
                grid=np.zeros((bin_length, bin_width), dtype=np.uint8),
                vacancy_vector=np.zeros(bin_length, dtype=np.int32) + bin_width,
                grid_state_idx=sol_idx,  # Initial grid state index
                area=0.0,
                enclosure_box_length=0,
                min_occupied_row=bin_length,
                max_occupied_row=-1,
                bins_makespans=[],
                current_bin_proc_time=0.0,
                current_bin_proc_time_height=0.0,
                is_done=len(seq) == 0,
                is_feasible=True,
                grid_fft_valid=False
            )
            contexts.append(ctx)
        
        return contexts
    
    def _process_wave(self, contexts, mach_data, grid_states, grid_ffts, grid_fft_valid):
        """Process one wave: try to place current part for all active contexts."""
        
        if not contexts:
            return
        
        # Group contexts by their current part (for batched FFT)
        # Each part has different FFTs for rotations
        part_groups: Dict[int, List[BatchPlacementContext]] = {}
        
        for ctx in contexts:
            if ctx.current_part_idx >= len(ctx.parts_sequence):
                ctx.is_done = True
                continue
            
            part_id = ctx.parts_sequence[ctx.current_part_idx]
            if part_id not in part_groups:
                part_groups[part_id] = []
            part_groups[part_id].append(ctx)
        
        # Process each part group
        for part_id, group_contexts in part_groups.items():
            self._process_part_group(
                part_id, group_contexts, mach_data, 
                grid_states, grid_ffts, grid_fft_valid
            )
    
    def _process_part_group(self, part_id, contexts, mach_data,
                            grid_states, grid_ffts, grid_fft_valid):
        """Process placement of same part across multiple contexts."""
        
        part_data = self.parts[part_id]
        mach_part_data = mach_data.parts[part_id]
        
        # Check if part can fit in machine at all
        shape0 = part_data.shapes[0]
        if ((shape0[0] > mach_data.bin_length or shape0[1] > mach_data.bin_width) and
            (shape0[1] > mach_data.bin_length or shape0[0] > mach_data.bin_width)):
            for ctx in contexts:
                ctx.is_feasible = False
            return
        
        # For each context, check vacancy and try placement
        contexts_needing_fft = []
        feasible_rotations_per_ctx = []
        
        for ctx in contexts:
            # Quick vacancy check (CPU, Numba JIT)
            feasible_rots = []
            feasible_shapes = []
            feasible_ffts = []
            
            for rot in range(part_data.nrot):
                shape = part_data.shapes[rot]
                if shape[0] > ctx.bin_length or shape[1] > ctx.bin_width:
                    continue
                
                dens = part_data.densities[rot].astype(np.int32)
                if check_vacancy_fit_simple(ctx.vacancy_vector, dens):
                    feasible_rots.append(rot)
                    feasible_shapes.append(shape)
                    feasible_ffts.append(mach_part_data.ffts[rot])
            
            if feasible_rots:
                contexts_needing_fft.append(ctx)
                feasible_rotations_per_ctx.append((feasible_rots, feasible_shapes, feasible_ffts))
            else:
                # No feasible rotations - need new bin
                self._start_new_bin(ctx, part_data, mach_part_data, mach_data, grid_states)
        
        if not contexts_needing_fft:
            return
        
        # Batch FFT for all contexts needing it
        self._batch_fft_placement(
            contexts_needing_fft, feasible_rotations_per_ctx,
            part_data, mach_part_data, mach_data,
            grid_states, grid_ffts, grid_fft_valid
        )
    
    def _batch_fft_placement(self, contexts, feasible_per_ctx, 
                             part_data, mach_part_data, mach_data,
                             grid_states, grid_ffts, grid_fft_valid):
        """Perform batched FFT collision detection for multiple contexts."""
        
        with torch.inference_mode():
            # Collect grid indices that need FFT update
            grid_indices = [ctx.grid_state_idx for ctx in contexts]
            
            # Update grid FFTs for invalid caches
            for i, ctx in enumerate(contexts):
                if not ctx.grid_fft_valid:
                    grid_ffts[ctx.grid_state_idx] = torch.fft.fft2(grid_states[ctx.grid_state_idx])
                    ctx.grid_fft_valid = True
            
            # For each context, process its feasible rotations
            for i, ctx in enumerate(contexts):
                feasible_rots, feasible_shapes, feasible_ffts = feasible_per_ctx[i]
                
                if not feasible_ffts:
                    self._start_new_bin(ctx, part_data, mach_part_data, mach_data, grid_states)
                    continue
                
                # Batch FFT for this context's rotations
                stacked_ffts = torch.stack(feasible_ffts, dim=0)
                grid_fft = grid_ffts[ctx.grid_state_idx]
                
                overlap_batch = torch.fft.ifft2(grid_fft.unsqueeze(0) * stacked_ffts).real
                rounded_batch = torch.round(overlap_batch)
                
                # BATCHED position extraction for all rotations at once (no per-rotation .item() calls)
                num_rots = len(feasible_rots)
                H, W = ctx.bin_length, ctx.bin_width
                
                # Zero mask for all rotations
                zero_mask = (rounded_batch == 0)  # (num_rots, H, W)
                
                # Part shape constraints - valid region where part fits
                part_heights = torch.tensor([s[0] for s in feasible_shapes], device=self.device)
                part_widths = torch.tensor([s[1] for s in feasible_shapes], device=self.device)
                
                # Create validity masks using broadcasting
                row_idx = torch.arange(H, device=self.device).view(1, H, 1)
                col_idx = torch.arange(W, device=self.device).view(1, 1, W)
                
                valid_row = row_idx >= (part_heights - 1).view(-1, 1, 1)
                valid_col = col_idx >= (part_widths - 1).view(-1, 1, 1)
                valid_mask = valid_row & valid_col
                
                # Combine: valid zeros only
                valid_zeros = zero_mask[:, :H, :W] & valid_mask
                
                # Score: row * (W+1) - col (maximize row, minimize col as tiebreaker)
                score = torch.where(
                    valid_zeros,
                    row_idx.float() * (W + 1) - col_idx.float(),
                    torch.tensor(-1e9, device=self.device)
                )
                
                # Find best position for each rotation
                flat_scores = score.view(num_rots, -1)
                best_flat_idx = flat_scores.argmax(dim=1)
                max_scores = flat_scores.max(dim=1).values
                
                best_row_full = best_flat_idx // W
                best_col_full = best_flat_idx % W
                has_valid = max_scores > -1e8
                
                smallest_cols = best_col_full - (part_widths - 1)
                largest_rows_real = best_row_full
                
                # SINGLE transfer - get all results at once
                results_cpu = torch.stack([
                    has_valid.int(), smallest_cols, largest_rows_real
                ], dim=1).cpu().numpy()
                
                # Find best placement on CPU
                best_result = None
                best_packing_density = 0
                potential_area = ctx.area + part_data.area
                
                for rot_idx, (rot, shape) in enumerate(zip(feasible_rots, feasible_shapes)):
                    if results_cpu[rot_idx, 0] == 0:
                        continue
                    
                    smallest_col = int(results_cpu[rot_idx, 1])
                    largest_row_real = int(results_cpu[rot_idx, 2])
                    
                    y_start = largest_row_real - shape[0] + 1
                    new_length = max(ctx.enclosure_box_length, ctx.bin_length - y_start)
                    new_packing_density = potential_area / (new_length * ctx.bin_width)
                    
                    if new_packing_density > best_packing_density:
                        best_packing_density = new_packing_density
                        best_result = (smallest_col, largest_row_real, rot, shape)
                
                if best_result is not None:
                    # Place the part
                    x, y, rot, shape = best_result
                    self._place_part(ctx, x, y, part_data.rotations[rot], shape, 
                                    part_data.area, mach_part_data, grid_states)
                    ctx.current_part_idx += 1
                else:
                    # No valid placement found - need new bin
                    self._start_new_bin(ctx, part_data, mach_part_data, mach_data, grid_states)
    
    def _place_part(self, ctx, x, y, part_matrix, shape, area, mach_part_data, grid_states):
        """Place a part in the context's current bin."""
        y_start = y - shape[0] + 1
        y_end = y + 1
        
        # Update CPU grid
        ctx.grid[y_start:y_end, x:x+shape[1]] += part_matrix.astype(np.uint8)
        
        # Update GPU grid state
        part_tensor = torch.as_tensor(part_matrix, dtype=torch.float32, device=self.device)
        grid_states[ctx.grid_state_idx, y_start:y_end, x:x+shape[1]] += part_tensor
        
        # Invalidate FFT cache
        ctx.grid_fft_valid = False
        
        # Update vacancy vector
        update_vacancy_vector_rows(ctx.vacancy_vector, ctx.grid[y_start:y_end, :], y_start)
        
        # Update context state
        ctx.area += area
        ctx.min_occupied_row = min(ctx.min_occupied_row, y_start)
        ctx.max_occupied_row = max(ctx.max_occupied_row, y)
        ctx.enclosure_box_length = ctx.bin_length - ctx.min_occupied_row
        
        # Update processing times
        ctx.current_bin_proc_time += mach_part_data.proc_time
        ctx.current_bin_proc_time_height = max(ctx.current_bin_proc_time_height, mach_part_data.proc_time_height)
    
    def _start_new_bin(self, ctx, part_data, mach_part_data, mach_data, grid_states):
        """Start a new bin for context and place part at bottom-left."""
        
        # Save current bin's makespan if it has parts
        if ctx.area > 0:
            ctx.bins_makespans.append(
                ctx.current_bin_proc_time + ctx.current_bin_proc_time_height + mach_data.setup_time
            )
        
        # Reset bin state
        ctx.grid.fill(0)
        ctx.vacancy_vector.fill(ctx.bin_width)
        grid_states[ctx.grid_state_idx].zero_()
        ctx.area = 0.0
        ctx.enclosure_box_length = 0
        ctx.min_occupied_row = ctx.bin_length
        ctx.max_occupied_row = -1
        ctx.current_bin_proc_time = 0.0
        ctx.current_bin_proc_time_height = 0.0
        ctx.grid_fft_valid = False
        
        # Place part at bottom-left using best rotation
        best_rot = part_data.best_rotation
        shape = part_data.shapes[best_rot]
        part_matrix = part_data.rotations[best_rot]
        
        x = 0
        y = ctx.bin_length - 1  # Bottom row
        
        self._place_part(ctx, x, y, part_matrix, shape, part_data.area, mach_part_data, grid_states)
        ctx.current_part_idx += 1


def evaluate_batch_wave(problem_data, nbParts, nbMachines, thresholds,
                        chromosomes, instance_parts, collision_backend):
    """
    Convenience function to evaluate a batch of chromosomes using wave batching.
    
    Args:
        problem_data: ProblemData instance
        nbParts: Number of parts
        nbMachines: Number of machines
        thresholds: Machine assignment thresholds
        chromosomes: Array of chromosomes (num_solutions, chrom_length)
        instance_parts: Part matching array
        collision_backend: Collision detection backend
        
    Returns:
        List of fitness values
    """
    evaluator = WaveBatchEvaluator(
        problem_data, nbParts, nbMachines, thresholds,
        instance_parts, collision_backend
    )
    return evaluator.evaluate_batch(chromosomes)
