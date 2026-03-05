import time
import numpy as np
import pandas as pd
from binClassNew import BuildingPlate
import csv
from concurrent.futures import ThreadPoolExecutor


# Penalty value for infeasible solutions
INFEASIBLE_PENALTY = 10000000000000000


def _process_single_machine(args):
    """
    Process all parts assigned to a single machine.
    
    Returns:
        tuple: (machine_idx, makespan, is_feasible, bins_list)
    """
    machine_idx, sorted_sequence, parts, mach_data, collision_backend = args
    
    openBins = []
    
    for partInd in sorted_sequence:
        result = False
        part_data = parts[partInd]
        mach_part_data = mach_data.parts[partInd]
        part_shapes0 = part_data.shapes[0]
        
        # Check if part can fit in this machine at all
        if (
            (part_shapes0[0] > mach_data.bin_length or part_shapes0[1] > mach_data.bin_width)
            and 
            (part_shapes0[1] > mach_data.bin_length or part_shapes0[0] > mach_data.bin_width)
        ):
            # Part cannot fit in any orientation
            return (machine_idx, INFEASIBLE_PENALTY, False, [])
        
        # Try to place in existing bins
        for bin in openBins:
            # Area check first (cheap)
            if bin.area + part_data.area > mach_data.bin_area:
                continue
            
            # Check if placement is possible (expensive FFT check)
            result = bin.can_insert(part_data, mach_part_data)
            if result:
                break
        
        # If no existing bin works, create a new one
        if not result:
            newBin = BuildingPlate(mach_data.bin_width, mach_data.bin_length, collision_backend)
            openBins.append(newBin)
            
            # Use pre-computed best rotation (minimum height)
            best_rotation = part_data.best_rotation
            
            # Insert at bottom-left
            newBin.insert(0, mach_data.bin_length-1, part_data.rotations[best_rotation], 
                         part_data.shapes[best_rotation], part_data.area)
            newBin.calculate_enclosure_box_length()
            
            newBin.processingTime += mach_part_data.proc_time
            newBin.processingTimeHeight = max(newBin.processingTimeHeight, mach_part_data.proc_time_height)
            newBin.partsAssigned.append(part_data.id)
    
    # Calculate makespan for this machine
    makespan = sum(
        batch.processingTime + batch.processingTimeHeight + mach_data.setup_time
        for batch in openBins
    )
    
    return (machine_idx, makespan, True, openBins)


def placementProcedure(problem_data, nbParts, nbMachines, thresholds, chromosome, matching, collision_backend, plot=False, parallel=True):
    """
    Placement procedure using ProblemData dataclass for faster attribute access.
    
    Args:
        problem_data: ProblemData dataclass with parts and machines attributes
        nbParts: Number of parts
        nbMachines: Number of machines
        thresholds: Machine assignment thresholds
        chromosome: BRKGA chromosome
        matching: Part matching array
        collision_backend: FFT collision backend
        plot: Whether to save plate files
        parallel: Whether to process machines in parallel (default True)
    """
    SV = chromosome[:nbParts]
    MV = chromosome[nbParts:]

    # Cache parts and machines for faster access
    parts = problem_data.parts
    machines = problem_data.machines
    
    # Pre-compute which parts go to which machine and their sorted sequence
    machine_tasks = []
    for i in range(nbMachines):
        mach_data = machines[i]
        
        # Determine which parts are assigned to this machine
        if i == 0:
            mask = MV <= thresholds[i]
        elif i == nbMachines - 1:
            mask = MV > thresholds[i-1]
        else:
            mask = (MV > thresholds[i-1]) & (MV <= thresholds[i])
        
        sequence = np.where(mask)[0]
        values = SV[sequence]
        sequence2 = matching[sequence]
        sorted_indices = np.argsort(values)
        sorted_sequence = sequence2[sorted_indices]
        
        machine_tasks.append((i, sorted_sequence, parts, mach_data, collision_backend))
    
    # Process machines (parallel or sequential)
    if parallel and nbMachines > 1:
        # Use ThreadPoolExecutor for parallel processing
        # Note: For GPU backends, threads share the same device which may have some contention
        # but avoids CUDA context issues that ProcessPoolExecutor would cause
        with ThreadPoolExecutor(max_workers=nbMachines) as executor:
            results = list(executor.map(_process_single_machine, machine_tasks))
    else:
        # Sequential processing
        results = [_process_single_machine(task) for task in machine_tasks]
    
    # Check for infeasibility and find worst makespan
    binsPerMachine = [None] * nbMachines
    worstMakespan = 0
    
    for machine_idx, makespan, is_feasible, bins_list in results:
        if not is_feasible:
            return INFEASIBLE_PENALTY
        binsPerMachine[machine_idx] = bins_list
        if makespan > worstMakespan:
            worstMakespan = makespan
    
    # Handle plotting if requested
    if plot:
        for i in range(nbMachines):
            for x, batch in enumerate(binsPerMachine[i]):
                batch.save_plate_to_file(f"Final_Building_Plate_{i+1,x}.txt")
                print("Batch", x, "from machine", i, batch.partsAssigned)
    
    return worstMakespan
