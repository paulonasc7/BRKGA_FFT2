"""
inspect_chromosome.py
---------------------
Run one generation of BRKGA (wave_batch, torch_gpu) for P50M2-0
exactly as BRKGA_alg3.py would (including the greedy initial solution),
pick a random feasible chromosome from the resulting population, and report:
  - chromosome values
  - makespan from wave_batch
  - per-machine, per-bin breakdown of which parts were assigned where
  - comparison against placementProcedure (original sequential decoder)

Usage:
    python inspect_chromosome.py
"""

import numpy as np
import pickle, os, sys, time, random, math
import pandas as pd
import torch

torch.set_num_threads(1)
torch.set_grad_enabled(False)

from collision_backend import create_collision_backend
from data_structures import PartData, MachinePartData, MachineData, ProblemData
from wave_batch_evaluator import WaveBatchEvaluator
from binClassInitialSol import BuildingPlate
from binClassNew import BuildingPlate as BinPlate
from placement import _process_single_machine

# ── instance parameters ────────────────────────────────────────────────────
NB_PARTS    = 50
NB_MACHINES = 2
INST_NUMBER = 0
BACKEND     = "torch_gpu"
MULT        = 10

# ── load instance ──────────────────────────────────────────────────────────
with open(f'data/Instances/P{NB_PARTS}M{NB_MACHINES}-{INST_NUMBER}.txt', 'r') as f:
    instance_parts = np.array([int(x) for x in f.read().split()])
instance_parts_unique = np.unique(instance_parts)

cache_path = 'data/PartsMachines/cached_specs.pkl'
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        jobSpecAll, machSpec, area, polRotations = pickle.load(f)
else:
    jobSpecAll   = pd.read_excel('data/PartsMachines/part-machine-information.xlsx', sheet_name='part',    header=0, index_col=0)
    machSpec     = pd.read_excel('data/PartsMachines/part-machine-information.xlsx', sheet_name='machine', header=0, index_col=0)
    area         = pd.read_excel('data/PartsMachines/polygon_areas.xlsx',            header=0)["Area"].tolist()
    polRotations = pd.read_excel('data/PartsMachines/parts_rotations.xlsx',          header=0)["rot"].tolist()
    with open(cache_path, 'wb') as f:
        pickle.dump((jobSpecAll, machSpec, area, polRotations), f)

jobSpec = jobSpecAll.loc[instance_parts_unique]

# ── build problem data (identical to BRKGA_alg3.py) ───────────────────────
collision_backend = create_collision_backend(BACKEND)

parts_dict = {}
for part in instance_parts_unique:
    matrix = np.load(f'data/partsMatrices/matrix_{part}.npy').astype(np.int32)
    matrix = np.ascontiguousarray(matrix)
    nrot = 2 if np.array_equal(matrix, np.rot90(matrix, 2)) else 4
    rotations, shapes, densities = [], [], []
    for rot in range(nrot):
        rotated = np.ascontiguousarray(np.rot90(matrix, rot))
        rotations.append(rotated)
        shapes.append((rotated.shape[0], rotated.shape[1]))
        padded = np.pad(rotated, ((0, 0), (1, 1)), constant_values=0)
        diffs  = np.diff(padded.astype(np.int8), axis=1)
        starts = np.where(diffs == 1)
        ends   = np.where(diffs == -1)
        runs   = ends[1] - starts[1]
        maxr   = np.zeros(rotated.shape[0], dtype=np.int32)
        if len(starts[0]) > 0:
            np.maximum.at(maxr, starts[0], runs)
        densities.append(maxr)
    best_rotation   = int(np.argmin([s[0] for s in shapes]))
    rotations_gpu   = [collision_backend.prepare_rotation_tensor(r) for r in rotations] \
                      if hasattr(collision_backend, 'prepare_rotation_tensor') else None
    rotations_uint8 = [r.astype(np.uint8) for r in rotations]
    pd_ = PartData(id=part, area=area[part], nrot=nrot,
                   rotations=rotations, shapes=shapes, densities=densities,
                   best_rotation=best_rotation,
                   rotations_gpu=rotations_gpu, rotations_uint8=rotations_uint8)
    pd_.prepare_jit_data()
    parts_dict[part] = pd_

machines_list = []
for m in range(NB_MACHINES):
    bin_length = machSpec['L(mm)'].iloc[m]
    bin_width  = machSpec['W(mm)'].iloc[m]
    machine_parts = {}
    for part in instance_parts_unique:
        pd_ = parts_dict[part]
        ffts = [collision_backend.prepare_part_fft(pd_.rotations[r], bin_length, bin_width)
                for r in range(pd_.nrot)]
        proc_time = (jobSpec["volume(mm3)"].loc[part]  * machSpec["VT(s/mm3)"].iloc[m] +
                     jobSpec["support(mm3)"].loc[part] * machSpec["SPT(s/mm3)"].iloc[m])
        proc_time_height = jobSpec["height(mm)"].loc[part] * machSpec["HT(s/mm3)"].iloc[m]
        machine_parts[part] = MachinePartData(ffts=ffts, proc_time=proc_time,
                                               proc_time_height=proc_time_height)
    machines_list.append(MachineData(bin_length=bin_length, bin_width=bin_width,
                                     bin_area=bin_length * bin_width,
                                     setup_time=machSpec['ST(s)'].iloc[m],
                                     parts=machine_parts))

problem_data = ProblemData(parts=parts_dict, machines=machines_list,
                           instance_parts=instance_parts,
                           instance_parts_unique=instance_parts_unique)
thresholds = [t / NB_MACHINES for t in range(1, NB_MACHINES)]

# ── build greedy initial solution (identical to BRKGA_alg3.py) ────────────
partsInfo = jobSpec.loc[instance_parts]["height(mm)"]
partsAR   = pd.read_excel('data/PartsMachines/polygon_areas.xlsx', header=0, index_col=0).loc[instance_parts]["Area"]
conc      = pd.concat([partsInfo, partsAR], axis=1)
part_sortedSequence = conc.sort_values(by=['height(mm)', 'Area'], ascending=[False, False]).index.tolist()

machines_dict  = {f'machine_{i}': {'makespan': 0, 'parts': [], 'batches': []} for i in range(NB_MACHINES)}
current_worst  = 0
machine_makespans_init = [0] * NB_MACHINES

for part in part_sortedSequence:
    best_makespan = float('inf')
    bestBatch = []
    part_data = problem_data.parts[part]
    part_shapes0 = part_data.shapes[0]
    for mach in range(NB_MACHINES):
        mach_data = problem_data.machines[mach]
        mach_part_data = mach_data.parts[part]
        if ((part_shapes0[0] > mach_data.bin_length or part_shapes0[1] > mach_data.bin_width) and
                (part_shapes0[1] > mach_data.bin_length or part_shapes0[0] > mach_data.bin_width)):
            continue
        placedInExist = False
        for batch in machines_dict[f'machine_{mach}']['batches']:
            if batch.area + part_data.area > mach_data.bin_area:
                continue
            result = batch.can_insert(part_data, mach_part_data)
            if result:
                new_ms = machine_makespans_init[mach] + mach_part_data.proc_time
                worst  = max(new_ms, current_worst)
                if worst < best_makespan:
                    best_makespan = worst
                    bestBatch = [batch, mach, new_ms, worst, new_ms]
                placedInExist = True
                break
        if not placedInExist:
            newBin = BuildingPlate(mach_data.bin_width, mach_data.bin_length, collision_backend)
            best_rot = part_data.best_rotation
            gpu_t    = part_data.rotations_gpu[best_rot] if part_data.rotations_gpu else None
            uint8_m  = part_data.rotations_uint8[best_rot]
            newBin.insert(0, mach_data.bin_length - 1, uint8_m,
                          part_data.shapes[best_rot], part_data.area, gpu_tensor=gpu_t)
            newBin.calculate_enclosure_box_length()
            new_ms = machine_makespans_init[mach] + mach_part_data.proc_time
            worst  = max(new_ms + mach_data.setup_time, current_worst)
            if worst < best_makespan:
                best_makespan = worst
                bestBatch = [newBin, mach, new_ms, worst, new_ms, True]
    if not bestBatch:
        continue
    if len(bestBatch) == 6 and bestBatch[5]:
        machines_dict[f'machine_{bestBatch[1]}']['batches'].append(bestBatch[0])
    mach_idx = bestBatch[1]
    machine_makespans_init[mach_idx] = bestBatch[2]
    current_worst = bestBatch[3]
    bestBatch[0].processingTime += problem_data.machines[mach_idx].parts[part].proc_time
    bestBatch[0].processingTimeHeight = max(
        bestBatch[0].processingTimeHeight,
        problem_data.machines[mach_idx].parts[part].proc_time_height)
    bestBatch[0].partsAssigned.append(part_data.id)

initial_sol = np.zeros(2 * NB_PARTS)
used_indices = set()
for m in range(NB_MACHINES):
    partsMachine = np.concatenate([b.partsAssigned for b in machines_dict[f'machine_{m}']['batches']])
    positions = []
    for value in partsMachine:
        for idx, val in enumerate(instance_parts):
            if val == value and idx not in used_indices:
                positions.append(idx)
                used_indices.add(idx)
                break
    positions_array = np.array(positions)
    if m == 0:
        initial_sol[positions_array + NB_PARTS] = random.uniform(0, thresholds[0])
    elif m == NB_MACHINES - 1:
        initial_sol[positions_array + NB_PARTS] = random.uniform(thresholds[-1] + 0.0001, 1 - 0.0001)
    else:
        initial_sol[positions_array + NB_PARTS] = random.uniform(thresholds[m-1] + 0.0001, thresholds[m] - 0.0001)
    sv_vals = np.linspace(0.01, 0.99, len(positions_array))
    initial_sol[positions_array] = sv_vals

# ── build evaluator & population ──────────────────────────────────────────
evaluator = WaveBatchEvaluator(problem_data, NB_PARTS, NB_MACHINES, thresholds,
                               instance_parts, collision_backend, device='cuda')

num_individuals = MULT * NB_PARTS
num_genes       = 2 * NB_PARTS

np.random.seed(42)
population = np.random.uniform(0.0, 1.0, size=(num_individuals, num_genes)).astype(np.float32)
population[0] = initial_sol.astype(np.float32)

# ── gen-0 evaluation ───────────────────────────────────────────────────────
print(f"Evaluating {num_individuals} chromosomes via wave_batch (gen 0)...")
t0 = time.time()
fitness_list = evaluator.evaluate_batch(population)
print(f"Done in {time.time()-t0:.2f}s\n")

# ── pick random feasible chromosome (not the best) ────────────────────────
best_idx   = int(np.argmin(fitness_list))
rng        = np.random.default_rng(seed=7)
candidates = [i for i in range(num_individuals)
              if i != best_idx and fitness_list[i] < 1e15]
chosen_idx = int(rng.choice(candidates))

chosen_chromosome = population[chosen_idx].copy()
chosen_makespan   = fitness_list[chosen_idx]
print(f"Chosen chromosome index : {chosen_idx}")
print(f"Best makespan in population : {fitness_list[best_idx]:.4f}")
print(f"Chosen makespan (wave_batch): {chosen_makespan:.4f}\n")

# ── re-run that single chromosome with context collection + debug tracing ──
print("Re-running chosen chromosome alone to collect bin assignments...")
print("(Debug tracing enabled for part 23)\n")
evaluator._collect_contexts = True
evaluator._last_contexts    = {}
evaluator._debug_part_ids   = {81}  # trace part 81 on M1
t0 = time.time()
single_fitness = evaluator.evaluate_batch(np.array([chosen_chromosome]))
evaluator._collect_contexts = False
evaluator._debug_part_ids   = set()
print(f"Done in {time.time()-t0:.2f}s")
print(f"Makespan from single re-run: {single_fitness[0]:.4f}\n")

if abs(single_fitness[0] - chosen_makespan) > 1.0:
    print("WARNING: makespan differs between batch run and single re-run — "
          "this may indicate non-determinism or a state issue.\n")
else:
    print("Makespan matches batch run. Bin assignments are consistent.\n")

# ── report bin assignments ─────────────────────────────────────────────────
print("=" * 70)
print("BIN ASSIGNMENT DETAILS")
print("=" * 70)

for machine_idx in range(NB_MACHINES):
    mach_data = problem_data.machines[machine_idx]
    contexts  = evaluator._last_contexts[machine_idx]
    ctx       = contexts[0]  # only one solution

    print(f"\nMachine {machine_idx}  "
          f"(bin size: {mach_data.bin_length}×{mach_data.bin_width}, "
          f"setup_time: {mach_data.setup_time:.1f}s)")

    active_bins = [b for b in ctx.open_bins if b.area > 0]
    if not active_bins:
        print("  No parts assigned to this machine.")
        continue

    machine_total = 0.0
    for b in active_bins:
        contribution = b.proc_time + b.proc_time_height + mach_data.setup_time
        machine_total += contribution
        print(f"  Bin {b.bin_idx}: {len(b.parts_assigned)} parts  "
              f"area_used={b.area:.1f}  "
              f"proc_time={b.proc_time:.2f}s  "
              f"proc_time_height={b.proc_time_height:.2f}s  "
              f"contribution={contribution:.2f}s")
        print(f"    Parts (in placement order): {b.parts_assigned}")

    print(f"  → Machine makespan: {machine_total:.4f}s  ({len(active_bins)} bins)")

print("\n" + "=" * 70)
print(f"Overall makespan (worst machine): {single_fitness[0]:.4f}s")
print("=" * 70)

np.save("chosen_chromosome.npy", chosen_chromosome)
print("\nChromosome saved to: chosen_chromosome.npy")

# ── run the SAME chromosome through placementProcedure ─────────────────────
print("\n" + "=" * 70)
print("RUNNING SAME CHROMOSOME THROUGH placementProcedure (original decoder)")
print("=" * 70)

SV = chosen_chromosome[:NB_PARTS]
MV = chosen_chromosome[NB_PARTS:]

pp_results = {}
pp_placement_log = {}  # machine_idx -> [(part_id, bin_idx, col, row, rot, shape)]
pp_overall_makespan = 0.0

for m_idx in range(NB_MACHINES):
    mach_data = problem_data.machines[m_idx]

    if m_idx == 0:
        mask = MV <= thresholds[0]
    elif m_idx == NB_MACHINES - 1:
        mask = MV > thresholds[-1]
    else:
        mask = (MV > thresholds[m_idx - 1]) & (MV <= thresholds[m_idx])

    part_indices = np.where(mask)[0]
    values = SV[part_indices]
    actual_parts = instance_parts[part_indices]
    sorted_sequence = actual_parts[np.argsort(values)]

    # Run PP with position tracking via _last_placement
    H_m, W_m = mach_data.bin_length, mach_data.bin_width
    openBins = []
    pp_log = []
    is_feasible = True

    for partInd in sorted_sequence:
        result_found = False
        part_data = problem_data.parts[partInd]
        mach_part_data = mach_data.parts[partInd]
        ps0 = part_data.shapes[0]

        if ((ps0[0] > H_m or ps0[1] > W_m) and (ps0[1] > H_m or ps0[0] > W_m)):
            is_feasible = False
            break

        for b_idx, b_obj in enumerate(openBins):
            if b_obj.area + part_data.area > mach_data.bin_area:
                continue
            ok = b_obj.can_insert(part_data, mach_part_data)
            if ok:
                lp = getattr(b_obj, '_last_placement', None)
                if lp:
                    pp_log.append((lp[0], b_idx, lp[1], lp[2], lp[3], lp[4]))
                result_found = True
                break

        if not result_found:
            newBin = BinPlate(mach_data.bin_width, mach_data.bin_length, collision_backend)
            best_rot = part_data.best_rotation
            shape = part_data.shapes[best_rot]
            gpu_t = part_data.rotations_gpu[best_rot] if part_data.rotations_gpu else None
            uint8_m = part_data.rotations_uint8[best_rot]
            newBin.insert(0, H_m - 1, uint8_m, shape, part_data.area, gpu_tensor=gpu_t)
            newBin.calculate_enclosure_box_length()
            newBin.processingTime += mach_part_data.proc_time
            newBin.processingTimeHeight = max(newBin.processingTimeHeight,
                                              mach_part_data.proc_time_height)
            newBin.partsAssigned.append(part_data.id)
            b_idx = len(openBins)
            openBins.append(newBin)
            pp_log.append((partInd, b_idx, 0, H_m - 1, best_rot, shape))

    makespan_r = sum(b.processingTime + b.processingTimeHeight + mach_data.setup_time
                     for b in openBins) if is_feasible else 1e16

    pp_results[m_idx] = (makespan_r, openBins, is_feasible)
    pp_placement_log[m_idx] = pp_log
    if makespan_r > pp_overall_makespan:
        pp_overall_makespan = makespan_r
    print(f"Machine {m_idx}: placementProcedure makespan = {makespan_r:.4f}s  "
          f"({len(openBins)} bins)")

print(f"\nplacementProcedure overall makespan: {pp_overall_makespan:.4f}s")
print(f"wave_batch overall makespan:         {single_fitness[0]:.4f}s")
diff = abs(pp_overall_makespan - single_fitness[0])
print(f"Absolute difference:                 {diff:.6f}s")
if diff < 1.0:
    print("RESULT: MATCH — both decoders agree.")
else:
    print("RESULT: MISMATCH — decoders disagree!")

# ── per-placement comparison using wave_batch placement log ────────────────
print("\n" + "=" * 70)
print("PER-PLACEMENT COMPARISON (wave_batch log vs PP)")
print("=" * 70)

wb_log = evaluator._placement_log  # machine_idx -> [(part_id, bin_idx, col, row, rot, shape)]

for m_idx in range(NB_MACHINES):
    mach_data = problem_data.machines[m_idx]
    wb_entries = wb_log.get(m_idx, [])
    pp_bins = pp_results[m_idx][1]

    # Build PP placement log by replaying: for each part in order, determine
    # which bin it ended up in and compare.
    # We already have pp_bins[bin_idx].partsAssigned for bin membership.
    # For grid comparison: compare WB and PP grids for bins with the same parts.

    wb_ctx = evaluator._last_contexts[m_idx][0]
    wb_bins = [b for b in wb_ctx.open_bins if b.area > 0]

    print(f"\nMachine {m_idx}: WB has {len(wb_bins)} bins, PP has {len(pp_bins)} bins")

    # Compare grids for bins that have the same parts
    max_bins = max(len(wb_bins), len(pp_bins))
    for b_idx in range(max_bins):
        wb_parts = wb_bins[b_idx].parts_assigned if b_idx < len(wb_bins) else []
        pp_parts = pp_bins[b_idx].partsAssigned if b_idx < len(pp_bins) else []
        parts_match = (wb_parts == pp_parts)

        if b_idx < len(wb_bins) and b_idx < len(pp_bins) and parts_match:
            # Same parts — compare grids
            wb_grid = wb_bins[b_idx].grid
            pp_grid = pp_bins[b_idx].grid
            grid_match = np.array_equal(wb_grid, pp_grid)
            if grid_match:
                print(f"  Bin {b_idx}: parts MATCH, grid MATCH")
            else:
                diff_cells = np.sum(wb_grid != pp_grid)
                print(f"  Bin {b_idx}: parts MATCH, grid DIFFER ({diff_cells} cells)")
                # Find first differing row/col
                diff_locs = np.argwhere(wb_grid != pp_grid)
                if len(diff_locs) > 0:
                    r, c = diff_locs[0]
                    print(f"    First diff at row={r}, col={c}: "
                          f"WB={wb_grid[r,c]}, PP={pp_grid[r,c]}")
        else:
            print(f"  Bin {b_idx}: parts DIFFER — WB={wb_parts}, PP={pp_parts}")

    # Compare placement positions between WB and PP
    wb_entries = wb_log.get(m_idx, [])
    pp_entries = pp_placement_log.get(m_idx, [])

    min_len = min(len(wb_entries), len(pp_entries))
    print(f"\n  Per-placement comparison ({len(wb_entries)} WB vs {len(pp_entries)} PP):")
    first_diff_found = False
    for i in range(min_len):
        wb_pid, wb_bidx, wb_col, wb_row, wb_rot, wb_shape = wb_entries[i]
        pp_pid, pp_bidx, pp_col, pp_row, pp_rot, pp_shape = pp_entries[i]
        if wb_pid != pp_pid:
            print(f"    {i:3d}: DIFFERENT PART — WB=part {wb_pid}, PP=part {pp_pid}")
            if not first_diff_found:
                print(f"    *** FIRST DIVERGENCE at placement {i} ***")
                first_diff_found = True
            break
        same_pos = (wb_bidx == pp_bidx and wb_col == pp_col and
                    wb_row == pp_row and wb_rot == pp_rot)
        if same_pos:
            print(f"    {i:3d}: part={wb_pid:3d} bin={wb_bidx} "
                  f"col={wb_col} row={wb_row} rot={wb_rot} ✓")
        else:
            print(f"    {i:3d}: part={wb_pid:3d} DIFFER:")
            print(f"         WB: bin={wb_bidx} col={wb_col} row={wb_row} "
                  f"rot={wb_rot} shape={wb_shape}")
            print(f"         PP: bin={pp_bidx} col={pp_col} row={pp_row} "
                  f"rot={pp_rot} shape={pp_shape}")
            if not first_diff_found:
                print(f"    *** FIRST DIVERGENCE at placement {i} ***")
                first_diff_found = True

# ── save output files ──────────────────────────────────────────────────────
import os
out_dir = "inspection_output"
os.makedirs(out_dir, exist_ok=True)

summary_path = os.path.join(out_dir, "solution_summary.txt")
with open(summary_path, 'w') as f:
    f.write(f"Instance: P{NB_PARTS}M{NB_MACHINES}-{INST_NUMBER}\n")
    f.write(f"Chromosome index in population: {chosen_idx}\n")
    f.write(f"Makespan (wave_batch, batch run):  {chosen_makespan:.4f}s\n")
    f.write(f"Makespan (wave_batch, single rerun): {single_fitness[0]:.4f}s\n")
    f.write(f"Makespan (placementProcedure):       {pp_overall_makespan:.4f}s\n")
    f.write(f"wave_batch match (batch vs single): {'YES' if abs(single_fitness[0] - chosen_makespan) <= 1.0 else 'NO'}\n")
    f.write(f"wave_batch vs placementProcedure:    {'MATCH' if diff < 1.0 else 'MISMATCH'} (diff={diff:.6f}s)\n\n")

    # ── wave_batch details ──
    f.write("=" * 60 + "\n")
    f.write("WAVE_BATCH DECODER\n")
    f.write("=" * 60 + "\n\n")

    for machine_idx in range(NB_MACHINES):
        mach_data = problem_data.machines[machine_idx]
        ctx       = evaluator._last_contexts[machine_idx][0]
        active    = [b for b in ctx.open_bins if b.area > 0]

        f.write(f"Machine {machine_idx}  "
                f"(bin size: {mach_data.bin_length}×{mach_data.bin_width}, "
                f"setup_time: {mach_data.setup_time:.1f}s)\n")
        f.write(f"Bins used: {len(active)}\n")
        f.write(f"Part processing order: {list(ctx.parts_sequence)}\n\n")

        machine_total = 0.0
        for b in active:
            contrib = b.proc_time + b.proc_time_height + mach_data.setup_time
            machine_total += contrib
            f.write(f"  Bin {b.bin_idx}:\n")
            f.write(f"    Parts (placement order): {b.parts_assigned}\n")
            f.write(f"    Number of parts : {len(b.parts_assigned)}\n")
            f.write(f"    Area used       : {b.area:.1f} / {mach_data.bin_area:.0f} "
                    f"({100*b.area/mach_data.bin_area:.1f}%)\n")
            f.write(f"    proc_time       : {b.proc_time:.2f}s\n")
            f.write(f"    proc_time_height: {b.proc_time_height:.2f}s\n")
            f.write(f"    Makespan contrib: {contrib:.2f}s\n")
            f.write(f"    Grid file       : wb_bin_m{machine_idx}_b{b.bin_idx}.txt\n\n")

        f.write(f"  Machine {machine_idx} total makespan: {machine_total:.4f}s\n\n")

    f.write(f"Overall makespan (worst machine): {single_fitness[0]:.4f}s\n\n")

    # ── placementProcedure details ──
    f.write("=" * 60 + "\n")
    f.write("PLACEMENTPROCEDURE DECODER\n")
    f.write("=" * 60 + "\n\n")

    for machine_idx in range(NB_MACHINES):
        mach_data = problem_data.machines[machine_idx]
        makespan_r, bins_list_r, is_feasible_r = pp_results[machine_idx]

        f.write(f"Machine {machine_idx}  "
                f"(bin size: {mach_data.bin_length}×{mach_data.bin_width}, "
                f"setup_time: {mach_data.setup_time:.1f}s)\n")
        f.write(f"Bins used: {len(bins_list_r)}\n\n")

        machine_total = 0.0
        for b_idx, b in enumerate(bins_list_r):
            contrib = b.processingTime + b.processingTimeHeight + mach_data.setup_time
            machine_total += contrib
            f.write(f"  Bin {b_idx}:\n")
            f.write(f"    Parts (placement order): {b.partsAssigned}\n")
            f.write(f"    Number of parts : {len(b.partsAssigned)}\n")
            f.write(f"    Area used       : {b.area:.1f} / {mach_data.bin_area:.0f} "
                    f"({100*b.area/mach_data.bin_area:.1f}%)\n")
            f.write(f"    proc_time       : {b.processingTime:.2f}s\n")
            f.write(f"    proc_time_height: {b.processingTimeHeight:.2f}s\n")
            f.write(f"    Makespan contrib: {contrib:.2f}s\n")
            f.write(f"    Grid file       : pp_bin_m{machine_idx}_b{b_idx}.txt\n\n")

        f.write(f"  Machine {machine_idx} total makespan: {machine_total:.4f}s\n\n")

    f.write(f"Overall makespan (worst machine): {pp_overall_makespan:.4f}s\n")

    # ── per-bin comparison ──
    f.write("\n" + "=" * 60 + "\n")
    f.write("BIN-BY-BIN COMPARISON\n")
    f.write("=" * 60 + "\n\n")

    for machine_idx in range(NB_MACHINES):
        ctx = evaluator._last_contexts[machine_idx][0]
        wb_bins = [b for b in ctx.open_bins if b.area > 0]
        pp_bins = pp_results[machine_idx][1]

        f.write(f"Machine {machine_idx}:\n")
        f.write(f"  wave_batch bins: {len(wb_bins)}   placementProcedure bins: {len(pp_bins)}\n")

        max_bins = max(len(wb_bins), len(pp_bins))
        for b_idx in range(max_bins):
            wb_parts = wb_bins[b_idx].parts_assigned if b_idx < len(wb_bins) else []
            pp_parts = pp_bins[b_idx].partsAssigned if b_idx < len(pp_bins) else []
            match = (wb_parts == pp_parts)
            f.write(f"  Bin {b_idx}: WB={wb_parts}  PP={pp_parts}  {'MATCH' if match else 'DIFFER'}\n")
        f.write("\n")

print(f"Summary saved to: {summary_path}")

# Grid files — wave_batch
grid_files = []
for machine_idx in range(NB_MACHINES):
    ctx = evaluator._last_contexts[machine_idx][0]
    for b in ctx.open_bins:
        if b.area == 0:
            continue
        grid_path = os.path.join(out_dir, f"wb_bin_m{machine_idx}_b{b.bin_idx}.txt")
        with open(grid_path, 'w') as f:
            for row in b.grid:
                f.write(' '.join(f'{val:2d}' for val in row) + '\n')
        grid_files.append(grid_path)

# Grid files — placementProcedure
for machine_idx in range(NB_MACHINES):
    for b_idx, b in enumerate(pp_results[machine_idx][1]):
        grid_path = os.path.join(out_dir, f"pp_bin_m{machine_idx}_b{b_idx}.txt")
        with open(grid_path, 'w') as f:
            for row in b.grid:
                f.write(' '.join(f'{val:2d}' for val in row) + '\n')
        grid_files.append(grid_path)

print(f"Grid files saved ({len(grid_files)} total)")
print(f"All output in: {out_dir}/")
