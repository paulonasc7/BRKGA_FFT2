"""Quick profiling script that runs only a few generations to identify bottlenecks."""
import cProfile
import pstats
import io
import sys
import time
import numpy as np
import math
import os
import pickle
import pandas as pd

# Set up environment before imports
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

from BRKGA_alg3 import BRKGA
from collision_backend import create_collision_backend
from data_structures import PartData, MachinePartData, MachineData, ProblemData

def setup_problem(nbParts=50, nbMachines=2, instNumber=0, backend_name="torch_gpu"):
    """Set up the problem data (same as in BRKGA_alg3.py main)."""
    collision_backend = create_collision_backend(backend_name)
    
    with open(f'data/Instances/P{nbParts}M{nbMachines}-{instNumber}.txt', 'r') as file:
        data = file.read()
    instanceParts = np.array([int(x) for x in data.split()])
    instancePartsUnique = np.unique(instanceParts)
    
    # Job specifications
    cache_path = 'data/PartsMachines/cached_specs.pkl'
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            jobSpecAll, machSpec, area, polRotations = pickle.load(f)
    else:
        jobSpecAll = pd.read_excel('data/PartsMachines/part-machine-information.xlsx', sheet_name='part', header=0, index_col=0)
        machSpec = pd.read_excel('data/PartsMachines/part-machine-information.xlsx', sheet_name='machine', header=0, index_col=0)
        area = pd.read_excel('data/PartsMachines/polygon_areas.xlsx', header=0)["Area"].tolist()
        polRotations = pd.read_excel('data/PartsMachines/parts_rotations.xlsx', header=0)["rot"].tolist()
        with open(cache_path, 'wb') as f:
            pickle.dump((jobSpecAll, machSpec, area, polRotations), f)
    
    jobSpec = jobSpecAll.loc[instancePartsUnique]
    
    # Build ProblemData
    parts_dict = {}
    machines_list = []
    
    for part in instancePartsUnique:
        matrix = np.load(f'data/partsMatrices/matrix_{part}.npy')
        matrix = matrix.astype(np.int32)
        matrix = np.ascontiguousarray(matrix)
        
        if np.array_equal(matrix, np.rot90(matrix, 2)):
            nrot = 2
        else:
            nrot = 4
        
        rotations = []
        shapes = []
        densities = []
        
        for rot in range(nrot):
            rotated = np.ascontiguousarray(np.rot90(matrix, rot))
            rotations.append(rotated)
            shapes.append((rotated.shape[0], rotated.shape[1]))
            
            padded = np.pad(rotated, ((0, 0), (1, 1)), constant_values=0)
            diffs = np.diff(padded.astype(np.int8), axis=1)
            start_indices = np.where(diffs == 1)
            end_indices = np.where(diffs == -1)
            run_lengths = end_indices[1] - start_indices[1]
            max_runs = np.zeros(rotated.shape[0], dtype=np.int32)
            if len(start_indices[0]) > 0:
                np.maximum.at(max_runs, start_indices[0], run_lengths)
            densities.append(max_runs)
        
        lengths = [s[0] for s in shapes]
        best_rotation = int(np.argmin(lengths))
        
        rotations_gpu = None
        if hasattr(collision_backend, 'prepare_rotation_tensor'):
            rotations_gpu = [collision_backend.prepare_rotation_tensor(rot) for rot in rotations]
        
        rotations_uint8 = [r.astype(np.uint8) for r in rotations]

        part_data = PartData(
            id=part,
            area=area[part],
            nrot=nrot,
            rotations=rotations,
            shapes=shapes,
            densities=densities,
            best_rotation=best_rotation,
            rotations_gpu=rotations_gpu,
            rotations_uint8=rotations_uint8
        )
        part_data.prepare_jit_data()
        parts_dict[part] = part_data
    
    for m in range(nbMachines):
        binLength = machSpec['L(mm)'].iloc[m]
        binWidth = machSpec['W(mm)'].iloc[m]
        binArea = binLength * binWidth
        setupTime = machSpec['ST(s)'].iloc[m]
        
        machine_parts = {}
        
        for part in instancePartsUnique:
            part_data = parts_dict[part]
            
            ffts = []
            for rot in range(part_data.nrot):
                fft = collision_backend.prepare_part_fft(
                    part_data.rotations[rot],
                    binLength,
                    binWidth,
                )
                ffts.append(fft)
            
            proc_time = (
                jobSpec["volume(mm3)"].loc[part] * machSpec["VT(s/mm3)"].iloc[m] +
                jobSpec["support(mm3)"].loc[part] * machSpec["SPT(s/mm3)"].iloc[m]
            )
            proc_time_height = (
                jobSpec["height(mm)"].loc[part] * machSpec["HT(s/mm3)"].iloc[m]
            )
            
            machine_parts[part] = MachinePartData(
                ffts=ffts,
                proc_time=proc_time,
                proc_time_height=proc_time_height
            )
        
        machines_list.append(MachineData(
            bin_length=binLength,
            bin_width=binWidth,
            bin_area=binArea,
            setup_time=setupTime,
            parts=machine_parts
        ))
    
    problem_data = ProblemData(
        parts=parts_dict,
        machines=machines_list,
        instance_parts=instanceParts,
        instance_parts_unique=instancePartsUnique
    )
    
    thresholds = [t / nbMachines for t in range(1, nbMachines)]
    
    # Simple initial solution (random)
    initial_sol = np.random.uniform(0, 1, 2 * nbParts).astype(np.float32)
    
    return problem_data, nbParts, nbMachines, thresholds, instanceParts, initial_sol, collision_backend


def profile_generations(num_gens=3, backend_name="torch_gpu", mult=10):
    """Profile a few generations."""
    print(f"Setting up problem with backend: {backend_name}...")
    problem_data, nbParts, nbMachines, thresholds, instanceParts, initial_sol, collision_backend = setup_problem(backend_name=backend_name)
    
    # nbParts is len(instanceParts), not len(instancePartsUnique)
    num_individuals = mult * nbParts
    num_elites = max(1, int(num_individuals * 0.1))
    num_mutants = max(1, int(num_individuals * 0.15))
    
    print(f"\nProfiling {num_gens} generations with {num_individuals} individuals (mult={mult}, {nbParts} parts, {nbMachines} machines)...")
    print(f"Backend: {collision_backend.name}")
    
    model = BRKGA(
        problem_data, nbParts, nbMachines, thresholds, instanceParts, initial_sol,
        collision_backend=collision_backend,
        eval_mode="wave_batch", eval_workers=4, eval_chunksize=1,
        num_generations=num_gens, 
        num_individuals=num_individuals,
        num_elites=num_elites,
        num_mutants=num_mutants,
        eliteCProb=0.70
    )
    
    # Profile the fit method
    pr = cProfile.Profile()
    pr.enable()
    
    start = time.time()
    model.fit(verbose=True)
    total_time = time.time() - start
    
    pr.disable()
    
    print(f"\n{'='*60}")
    print(f"Total time for {num_gens} generations: {total_time:.2f}s")
    print(f"Average per generation: {total_time/num_gens:.2f}s")
    print(f"{'='*60}\n")
    
    # Print top functions by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(50)
    print(s.getvalue())
    
    # Also print by total time (self time)
    print("\n" + "="*60)
    print("TOP FUNCTIONS BY SELF TIME (tottime):")
    print("="*60)
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
    ps2.print_stats(30)
    print(s2.getvalue())
    
    model.shutdown()


def time_single_evaluation(backend_name="torch_gpu"):
    """Time a single solution evaluation to understand placement costs."""
    print(f"Setting up problem with backend: {backend_name}...")
    problem_data, nbParts, nbMachines, thresholds, instanceParts, initial_sol, collision_backend = setup_problem(backend_name=backend_name)
    
    print(f"Backend: {collision_backend.name}")
    
    from placement import placementProcedure
    
    # Warm up
    for _ in range(2):
        placementProcedure(problem_data, nbParts, nbMachines, thresholds, initial_sol, instanceParts, collision_backend)
    
    # Time multiple evaluations
    times = []
    for i in range(20):
        sol = np.random.uniform(0, 1, 2 * nbParts).astype(np.float32)
        start = time.time()
        placementProcedure(problem_data, nbParts, nbMachines, thresholds, sol, instanceParts, collision_backend)
        times.append(time.time() - start)
    
    print(f"\nSingle evaluation times ({len(times)} runs):")
    print(f"  Mean: {np.mean(times)*1000:.2f} ms")
    print(f"  Std:  {np.std(times)*1000:.2f} ms")
    print(f"  Min:  {np.min(times)*1000:.2f} ms")
    print(f"  Max:  {np.max(times)*1000:.2f} ms")
    
    # If 500 individuals, estimate per generation
    n_individuals = 500
    eval_per_gen = n_individuals - 50  # Elites not re-evaluated
    print(f"\nEstimated time for {eval_per_gen} evaluations per generation: {np.mean(times) * eval_per_gen:.2f}s")


if __name__ == "__main__":
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Parse arguments: [eval|profile] [backend_name]
    # Examples:
    #   python profile_quick.py eval torch_gpu
    #   python profile_quick.py eval cupy_gpu
    #   python profile_quick.py cupy_gpu
    #   python profile_quick.py
    
    mode = "profile"
    backend_name = "torch_gpu"
    
    for arg in sys.argv[1:]:
        if arg == "eval":
            mode = "eval"
        elif arg in ["torch_gpu", "cupy_gpu", "cupy_gpu_optimized", "torch_cpu", "numpy_cpu"]:
            backend_name = arg
    
    # Check for mult parameter (e.g., mult=4)
    mult = 10  # default
    for arg in sys.argv[1:]:
        if arg.startswith("mult="):
            mult = int(arg.split("=")[1])
    
    print(f"Mode: {mode}, Backend: {backend_name}, Mult: {mult}")
    
    if mode == "eval":
        time_single_evaluation(backend_name=backend_name)
    else:
        profile_generations(num_gens=3, backend_name=backend_name, mult=mult)
