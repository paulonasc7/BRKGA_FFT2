import math
import random
import numpy as np
from placement import placementProcedure
from wave_batch_evaluator import WaveBatchEvaluator
from binClassInitialSol import BuildingPlate
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time 
import sys
import torch
import itertools
import pandas as pd
import pickle
from collision_backend import create_collision_backend
from data_structures import PartData, MachinePartData, MachineData, ProblemData
from full_native_decoder import FullNativeDecoderEvaluator
import os

# Optimize PyTorch settings for inference workloads
torch.set_num_threads(1)  # Avoid over-subscription with multiprocessing
torch.set_grad_enabled(False)  # Disable gradient computation globally

class BRKGA():
    def __init__(self, problem_data, nbParts, nbMachines, thresholds, instanceParts, initialSol,
                 collision_backend, eval_mode="auto", eval_workers=4, eval_chunksize=1,
                 num_generations = 200, num_individuals=100, num_elites = 12, num_mutants = 18, eliteCProb = 0.7):

        # Input
        self.problem_data = problem_data  # ProblemData dataclass with parts and machines
        self.nbMachines = nbMachines
        self.thresholds = thresholds
        self.N = nbParts
        self.instanceParts = instanceParts
        self.initialSol = initialSol
        self.collision_backend = collision_backend
        
        # Auto-detect optimal eval_mode based on backend
        # GPU backends: serial is often faster (avoids GPU contention)
        # CPU backends: thread parallelism helps
        if eval_mode == "auto":
            if "gpu" in collision_backend.name or "cuda" in collision_backend.name:
                self.eval_mode = "serial"  # GPU already parallel internally
            else:
                self.eval_mode = "thread"
        else:
            self.eval_mode = eval_mode
            
        self.eval_workers = int(eval_workers) if eval_workers else 0
        self.eval_chunksize = int(eval_chunksize) if eval_chunksize else 1
        
        # Configuration
        self.num_generations = num_generations
        self.num_individuals = int(num_individuals)
        self.num_gene = 2*self.N
        
        self.num_elites = int(num_elites)
        self.num_mutants = int(num_mutants)
        self.eliteCProb = eliteCProb
        
        # Result
        self.used_bins = -1
        self.solution = None
        self.best_fitness = -1
        self.history = {
            'mean': [],
            'min': [],
            'time': []
        }
        
        # Create executor once at initialization (avoid recreating per generation)
        self._executor = None
        self._wave_evaluator = None
        self._native_decoder = None

        force_native = os.getenv("ABRKGA_FULL_NATIVE_DECODER", "0").strip() not in {"0", "false", "False"}
        if self.eval_mode == "wave_batch" and force_native:
            self.eval_mode = "native_full"
        
        if self.eval_mode == "native_full":
            self._native_decoder = FullNativeDecoderEvaluator(
                problem_data, nbParts, nbMachines, thresholds,
                instanceParts, collision_backend
            )
        elif self.eval_mode == "wave_batch":
            # Initialize wave batch evaluator for GPU batching
            self._wave_evaluator = WaveBatchEvaluator(
                problem_data, nbParts, nbMachines, thresholds,
                instanceParts, collision_backend
            )
        elif self.eval_mode == "thread" and self.eval_workers > 1:
            max_workers = self.eval_workers if self.eval_workers > 0 else min(32, (os.cpu_count() or 4))
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Fitness memoization cache (only for serial/thread/process — native/wave batch chromosomes
        # are continuous floats that never repeat, so the cache would have 0% hit rate)
        if self.eval_mode not in {"wave_batch", "native_full"}:
            self._fitness_cache = {}
            self._cache_hits = 0
            self._cache_misses = 0


    def evaluate_solution(self, solution):
        decoder = placementProcedure(
            self.problem_data,
            self.N,
            self.nbMachines,
            self.thresholds,
            solution,
            self.instanceParts,
            self.collision_backend,
        )
        return decoder

    def _hash_solution(self, solution):
        """Create hashable key from solution (quantize to avoid floating point issues)."""
        # Quantize to 4 decimal places for robust hashing
        return tuple(np.round(solution, 4))
    
    def cal_fitness(self, population):
        if self.eval_mode == "native_full":
            return self._native_decoder.evaluate_batch(np.array(population))
        if self.eval_mode == "wave_batch":
            return self._wave_evaluator.evaluate_batch(np.array(population))

        # Cache-based path for serial/thread/process modes
        results = [None] * len(population)
        to_evaluate = []  # (index, solution) pairs that need evaluation

        for i, sol in enumerate(population):
            key = self._hash_solution(sol)
            if key in self._fitness_cache:
                results[i] = self._fitness_cache[key]
                self._cache_hits += 1
            else:
                to_evaluate.append((i, sol, key))
                self._cache_misses += 1

        # If all cached, return early
        if not to_evaluate:
            return results

        # Evaluate only uncached solutions
        solutions_to_eval = [sol for _, sol, _ in to_evaluate]

        if self.eval_mode == "serial" or self.eval_workers <= 1:
            new_fitness = [self.evaluate_solution(sol) for sol in solutions_to_eval]
        elif self.eval_mode == "thread":
            new_fitness = list(self._executor.map(self.evaluate_solution, solutions_to_eval))
        elif self.eval_mode == "process":
            if "cuda" in self.collision_backend.name:
                raise ValueError("eval_mode=process is not supported with CUDA backends. Use thread or serial.")
            max_workers = self.eval_workers if self.eval_workers > 0 else min(32, (os.cpu_count() or 4))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                new_fitness = list(executor.map(self.evaluate_solution, solutions_to_eval, chunksize=self.eval_chunksize))
        else:
            raise ValueError(f"Unsupported eval_mode: {self.eval_mode}")

        # Store results and update cache
        for (i, _, key), fitness in zip(to_evaluate, new_fitness):
            results[i] = fitness
            self._fitness_cache[key] = fitness

        return results
    
    def shutdown(self):
        """Clean up executor resources and report cache stats."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        
        # Report cache effectiveness (only when cache is active)
        if self.eval_mode not in {"wave_batch", "native_full"}:
            total = self._cache_hits + self._cache_misses
            if total > 0:
                hit_rate = 100.0 * self._cache_hits / total
                print(f"Fitness cache: {self._cache_hits}/{total} hits ({hit_rate:.1f}%), {len(self._fitness_cache)} unique solutions")

    def partition(self, population, fitness_list):
        fitness_arr = np.asarray(fitness_list)
        # O(n) partial sort - only need top num_elites
        partition_indices = np.argpartition(fitness_arr, self.num_elites)
        elite_indices = partition_indices[:self.num_elites]
        non_elite_indices = partition_indices[self.num_elites:]
        # Sort only the elite subset (small array)
        elite_indices = elite_indices[np.argsort(fitness_arr[elite_indices])]
        return population[elite_indices], population[non_elite_indices], fitness_arr[elite_indices]
    
    def mating(self, elites, non_elites):
        # Vectorized mating: generate all offspring at once
        num_offspring = self.num_individuals - self.num_elites - self.num_mutants
        
        # Select parent indices in bulk
        elite_indices = np.random.randint(0, len(elites), size=num_offspring)
        non_elite_indices = np.random.randint(0, len(non_elites), size=num_offspring)
        
        # Get parent arrays
        elite_parents = elites[elite_indices]  # (num_offspring, num_gene)
        non_elite_parents = non_elites[non_elite_indices]  # (num_offspring, num_gene)
        
        # Generate crossover mask for all offspring at once
        crossover_mask = np.random.uniform(0.0, 1.0, size=(num_offspring, self.num_gene)) < self.eliteCProb
        
        # Vectorized crossover
        offspring = np.where(crossover_mask, elite_parents, non_elite_parents).astype(np.float32)
        return offspring
    
    def mutants(self):
        return np.random.uniform(low=0.0, high=1.0, size=(self.num_mutants, self.num_gene)).astype(np.float32)
        
    def fit(self, verbose = False):
        startfit = time.time()
        # Initial population & fitness
        population = np.random.uniform(low=0.0, high=1.0, size=(self.num_individuals, self.num_gene)).astype(np.float32)
        population[0] = self.initialSol
        fitness_list = self.cal_fitness(population)
        
        if verbose:
            print('\nInitial Population:')
            print('  ->  shape:',population.shape)
            print('  ->  Best Fitness:',min(fitness_list))
            
        # best    
        best_fitness = np.min(fitness_list)
        best_solution = population[np.argmin(fitness_list)]
        self.history['min'].append(np.min(fitness_list))
        self.history['mean'].append(np.mean(fitness_list))
        self.history['time'].append(time.time()-startfit)
        
        
        # Repeat generations
        best_iter = 0
        
        for g in range(self.num_generations):
            startTime = time.time()
            # Select elite group
            elites, non_elites, elite_fitness_list = self.partition(population, fitness_list)
            if verbose:
                print(f"Elite avg: {np.average(elite_fitness_list):.4f}")
            # Biased Mating & Crossover
            offsprings = self.mating(elites, non_elites)
            
            # Generate mutants
            mutants = self.mutants()

            # New Population & fitness
            offspring = np.concatenate((mutants,offsprings), axis=0)
            offspring_fitness_list = self.cal_fitness(offspring)
            
            population = np.concatenate((elites, mutants, offsprings), axis = 0)
            fitness_list = list(elite_fitness_list) + offspring_fitness_list

            # Update Best Fitness (compute argmin once, not per iteration)
            min_idx = np.argmin(fitness_list)
            min_fitness = fitness_list[min_idx]
            if min_fitness < best_fitness:
                best_iter = g
                best_fitness = min_fitness
                best_solution = population[min_idx]
            
            self.history['min'].append(min_fitness)
            self.history['mean'].append(np.mean(fitness_list))
            self.history['time'].append(time.time()-startfit)
            
            if verbose:
                print(f"Generation {g}: Best={best_fitness:.4f}, Time={time.time()-startTime:.2f}s")

        self.used_bins = math.floor(best_fitness)
        self.best_fitness = best_fitness
        self.solution = best_solution

        # Print CPU hotspot profile if available
        if self.eval_mode == "native_full" and self._native_decoder is not None:
            try:
                from full_native_decoder import FullNativeDecoderEvaluator
                print(FullNativeDecoderEvaluator.get_profile_summary())
            except Exception:
                pass

        # Clean up executor after fit completes
        self.shutdown()
        return 'feasible'
    
if __name__ == "__main__":
    '''INITIAL AND KNOWN DATA'''
    nbParts = int(sys.argv[1])
    #nbParts = 25
    nbMachines = int(sys.argv[2])
    #nbMachines = 2
    instNumber = int(sys.argv[3])
    #instNumber = 0
    backend_name = sys.argv[4] if len(sys.argv) > 4 else "torch_gpu"
    eval_mode = sys.argv[5] if len(sys.argv) > 5 else "auto"  # "auto" selects serial for GPU, thread for CPU
    eval_workers = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    eval_chunksize = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    num_generations = int(sys.argv[8]) if len(sys.argv) > 8 else 30
    collision_backend = create_collision_backend(backend_name)

    '''DEFINE DATA'''
    with open(f'data/Instances/P{nbParts}M{nbMachines}-{instNumber}.txt', 'r') as file:
        data = file.read()
    # read instance to know which parts are in it
    instanceParts = np.array([int(x) for x in data.split()])
    #instanceParts = np.array([38,38])
    instancePartsUnique = np.unique(instanceParts)
    
    
    # Job specifications (with caching for faster startup)
    cache_path = 'data/PartsMachines/cached_specs.pkl'
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            jobSpecAll, machSpec, area, polRotations = pickle.load(f)
    else:
        jobSpecAll = pd.read_excel('data/PartsMachines/part-machine-information.xlsx', sheet_name='part', header=0, index_col=0)
        machSpec = pd.read_excel('data/PartsMachines/part-machine-information.xlsx', sheet_name='machine', header=0, index_col=0)
        area = pd.read_excel('data/PartsMachines/polygon_areas.xlsx', header=0)["Area"].tolist()
        polRotations = pd.read_excel('data/PartsMachines/parts_rotations.xlsx', header=0)["rot"].tolist()
        # Cache for future runs
        with open(cache_path, 'wb') as f:
            pickle.dump((jobSpecAll, machSpec, area, polRotations), f)
    
    jobSpec = jobSpecAll.loc[instancePartsUnique]

    data = {}
    # Load the binary matrices from .npy files
    startLoad = time.time()
    
    # ========== Create ProblemData using dataclasses ==========
    parts_dict: dict[int, PartData] = {}
    machines_list: list[MachineData] = []
    
    # PHASE 1: Load parts ONCE (independent of machines)
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
            
            # Vectorized density calculation (max consecutive 1s per row)
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
        
        # Pre-compute GPU tensors for rotations (avoids CPU->GPU transfer per insert)
        rotations_gpu = None
        if hasattr(collision_backend, 'prepare_rotation_tensor'):
            rotations_gpu = [collision_backend.prepare_rotation_tensor(rot) for rot in rotations]

        # Pre-cast uint8 versions (avoids .astype(np.uint8) on every grid insert)
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
        # Pre-compute JIT-optimized data structures for batched vacancy check
        part_data.prepare_jit_data()
        parts_dict[part] = part_data
    
    # PHASE 2: Compute machine-specific data (FFTs, processing times)
    for m in range(nbMachines):
        binLength = machSpec['L(mm)'].iloc[m]
        binWidth = machSpec['W(mm)'].iloc[m]
        binArea = binLength * binWidth
        setupTime = machSpec['ST(s)'].iloc[m]
        
        machine_parts: dict[int, MachinePartData] = {}
        
        for part in instancePartsUnique:
            part_data = parts_dict[part]
            
            # Compute machine-specific FFTs for each rotation
            ffts = []
            for rot in range(part_data.nrot):
                fft = collision_backend.prepare_part_fft(
                    part_data.rotations[rot],
                    binLength,
                    binWidth,
                )
                ffts.append(fft)
            
            # Machine-specific processing times
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
    
    # Create the ProblemData container
    problem_data = ProblemData(
        parts=parts_dict,
        machines=machines_list,
        instance_parts=instanceParts,
        instance_parts_unique=instancePartsUnique
    )

    #print(time.time()-startLoad)
    thresholds = [t / nbMachines for t in range(1, nbMachines)] # define the the thresholds for the random keys of the BRKGA for machine assignment


    ''' CREATE INITIAL SOLUTION '''
    # Create dictionary that will hold info about which parts are assgined to which machines and their sequence
    machines_dict = {f'machine_{i}': {'makespan': 0, 'parts':[], 'batches':[]} for i in range(nbMachines)}
    
    # Pre-compute sorted parts order (by decreasing height, then decreasing area)
    # This is cached once and reused
    partsInfo = jobSpec.loc[instanceParts]["height(mm)"]
    partsAR = pd.read_excel(f'data/PartsMachines/polygon_areas.xlsx', header = 0, index_col = 0).loc[instanceParts]["Area"]
    conc = pd.concat([partsInfo, partsAR], axis = 1)
    sorted_df = conc.sort_values(by=['height(mm)', 'Area'], ascending=[False, False])
    part_sortedSequence = sorted_df.index.to_list()
    
    # Cache current worst makespan across machines (avoid recomputing in inner loops)
    current_worst_makespan = 0
    # Pre-compute machine makespans list for faster access
    machine_makespans = [0] * nbMachines

    for part in part_sortedSequence:
        # Variable to hold makespan of the system
        best_makespan = float('inf')
        bestBatch = []
        
        # Get part data from dataclass (fast attribute access)
        part_data = problem_data.parts[part]
        part_shapes0 = part_data.shapes[0]
        
        for mach in range(nbMachines):
            mach_data = problem_data.machines[mach]
            mach_part_data = mach_data.parts[part]
            
            placedInExist = False
            if (
                (part_shapes0[0] > mach_data.bin_length or part_shapes0[1] > mach_data.bin_width)
                and 
                (part_shapes0[1] > mach_data.bin_length or part_shapes0[0] > mach_data.bin_width)
                ):
                continue
            
            machineMakespan = machine_makespans[mach] + mach_data.setup_time + mach_part_data.proc_time + mach_part_data.proc_time_height
            newMakespan = max(current_worst_makespan, machineMakespan)
            #print("Place part ", part, " in a new batch in machine ", mach, "leads to makespan ", newMakespan)


            if newMakespan <= best_makespan:
                best_makespan = newMakespan
                # Use pre-computed best rotation (minimum height)
                best_rotation = part_data.best_rotation
                bestBatch = ['new', mach, [0, mach_data.bin_length-1], best_rotation, machineMakespan]
            
            for x, batch in enumerate(machines_dict[f'machine_{mach}']['batches']):
                res = batch.can_insert(part_data, mach_part_data)
                if res[0]:
                    if batch.processingTimeHeight < mach_part_data.proc_time_height:
                        machineMakespan = machine_makespans[mach] - batch.processingTimeHeight + mach_part_data.proc_time_height + mach_part_data.proc_time
                    else:
                        machineMakespan = machine_makespans[mach] + mach_part_data.proc_time

                    newMakespan = max(current_worst_makespan, machineMakespan)
                    #print("Place part ", part, " in batch", x, "in machine ", mach, "leads to makespan ", newMakespan)
                    if newMakespan <= best_makespan:
                        if placedInExist and newMakespan == best_makespan:
                            continue
                        else:
                            placedInExist = True
                            best_makespan = newMakespan
                            bestBatch = ['exist', mach, res[1], res[2], machineMakespan, batch]

        if bestBatch[0] == 'new':
            #print("Placed part ",part," in a new batch in machine ", bestBatch[1]," and current makespan is ", bestBatch[4])
            mach_idx = bestBatch[1]
            mach_data = problem_data.machines[mach_idx]
            machines_dict[f'machine_{mach_idx}']['parts'].append(part)
            machine_makespans[mach_idx] = bestBatch[4]
            machines_dict[f'machine_{mach_idx}']['makespan'] = bestBatch[4]
            newBin = BuildingPlate(mach_data.bin_width, mach_data.bin_length, collision_backend)
            machines_dict[f'machine_{mach_idx}']['batches'].append(newBin)
            
            # Insert the part in the bottomest-leftest position (use GPU tensor if available)
            rot_idx = bestBatch[3]
            gpu_tensor = part_data.rotations_gpu[rot_idx] if part_data.rotations_gpu else None
            uint8_matrix = part_data.rotations_uint8[rot_idx] if part_data.rotations_uint8 else part_data.rotations[rot_idx]
            newBin.insert(0, mach_data.bin_length-1, uint8_matrix, part_data.shapes[rot_idx], part_data.area, gpu_tensor=gpu_tensor)
            
            # Update batch current state
            newBin.calculate_enclosure_box_length()  # Update box length
            #newBin.calculate_enclosure_box_width()  # Update box width
            
            mach_part_data = mach_data.parts[part]
            newBin.processingTime += mach_part_data.proc_time
            newBin.processingTimeHeight = max(newBin.processingTimeHeight, mach_part_data.proc_time_height)
            newBin.partsAssigned.append(part_data.id)
            
            # Update cached worst makespan
            current_worst_makespan = max(current_worst_makespan, bestBatch[4])
            
        elif bestBatch[0] == 'exist':
            #print("Placed part ",part," in an existing batch in machine ", bestBatch[1]," and current makespan is ", bestBatch[4])
            mach_idx = bestBatch[1]
            mach_data = problem_data.machines[mach_idx]
            machines_dict[f'machine_{mach_idx}']['parts'].append(part)
            machine_makespans[mach_idx] = bestBatch[4]
            machines_dict[f'machine_{mach_idx}']['makespan'] = bestBatch[4]
            newBin = bestBatch[5]
            
            # Use pre-computed best rotation (minimum height)
            best_rotation = part_data.best_rotation
            
            # Insert the part in the bottomest-leftest position (use GPU tensor if available)
            rot_idx = bestBatch[3]
            gpu_tensor = part_data.rotations_gpu[rot_idx] if part_data.rotations_gpu else None
            uint8_matrix = part_data.rotations_uint8[rot_idx] if part_data.rotations_uint8 else part_data.rotations[rot_idx]
            newBin.insert(bestBatch[2][0], bestBatch[2][1], uint8_matrix, part_data.shapes[rot_idx], part_data.area, gpu_tensor=gpu_tensor)
            
            # Update batch current state
            newBin.calculate_enclosure_box_length()  # Update box length
            #newBin.calculate_enclosure_box_width()  # Update box width
            
            mach_part_data = mach_data.parts[part]
            newBin.processingTime += mach_part_data.proc_time
            newBin.processingTimeHeight = max(newBin.processingTimeHeight, mach_part_data.proc_time_height)
            newBin.partsAssigned.append(part_data.id)
            
            # Update cached worst makespan
            current_worst_makespan = max(current_worst_makespan, bestBatch[4])
    
    array = np.zeros(2*nbParts)

    used_indices = set()
    for m in range(nbMachines):
        positions = []
        
        partsMachine = np.concatenate([batch.partsAssigned for batch in machines_dict[f'machine_{m}']['batches']])
        for value in partsMachine:
            for idx, val in enumerate(instanceParts):
                if val == value and idx not in used_indices:
                    positions.append(idx)
                    used_indices.add(idx)
                    break
        
        positions_array = np.array(positions)

        if m == 0: ## Might have to reorganize these "if" statements because if I have more than 3 machines, it is more likely to fall under the "else"
            array[positions_array+nbParts] = random.uniform(0,thresholds[m])    
            #mask = MV <= thresholds[i]
        elif m == nbMachines - 1:
            array[positions_array+nbParts] = random.uniform(thresholds[m-1]+0.0001,1-0.0001)    
            #mask = MV > thresholds[i-1]
        else:
            array[positions_array+nbParts] = random.uniform(thresholds[m-1]+0.0001,thresholds[m]-0.0001)
            #mask = (MV > thresholds[i-1]) & (MV <= thresholds[i])

        # Generate strictly increasing values between 0 and 1
        values = np.linspace(0, 1, len(positions_array), endpoint=False)  # Equally spaced values

        # Assign these values to the specified positions in arrayZeros
        for i, pos in enumerate(positions_array):
            array[pos] = values[i]
    
    print("Makespan of initial solution: ",best_makespan)

    ''' CALL BRKGA'''
    # Possible values for p
    #prob = [10,15,20,30]
    prob = [10]
    
    for mult in prob:
        # Initialize the Excel writer object
        name = f'P{nbParts}M{nbMachines}-{instNumber}'
        writer = pd.ExcelWriter(f'OriginalInitialSol_{name}_prob_{mult}.xlsx', engine='openpyxl')
        for i in range(1):
            model = BRKGA(problem_data, nbParts, nbMachines, thresholds, instanceParts, array,
                  collision_backend=collision_backend,
                  eval_mode=eval_mode, eval_workers=eval_workers, eval_chunksize=eval_chunksize,
                  num_generations=num_generations, num_individuals=mult*nbParts, num_elites = math.ceil(mult*nbParts*0.1), num_mutants = math.ceil(mult*nbParts*0.15), eliteCProb = 0.70)
            model.fit(verbose = True)

            # Convert dictionary to DataFrame
            df = pd.DataFrame(model.history)
            # Export DataFrame to Excel file
            df.to_excel(writer, sheet_name = f'Iteration{i+1}', index=False)
            
            del model

        # Save the Excel file
        writer.close()
