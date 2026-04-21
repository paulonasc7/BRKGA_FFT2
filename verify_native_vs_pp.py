"""
verify_native_vs_pp.py
----------------------
Build the exact same population as inspect_chromosome.py (np.random.seed(42),
500 individuals of 100 genes each for P50M2-0), extract chromosome index 462,
and evaluate it through the full native C++/CUDA decoder.

The golden answer from placementProcedure is makespan = 448166.8336 (see
results/inspection_output/solution_summary.txt).

Usage:
    python verify_native_vs_pp.py
    (or via remote.py run)
"""

import os, pickle, sys
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
torch.set_grad_enabled(False)

from collision_backend import create_collision_backend
from data_structures import PartData, MachinePartData, MachineData, ProblemData
from full_native_decoder import FullNativeDecoderEvaluator

NB_PARTS, NB_MACHINES, INST_NUMBER = 50, 2, 0
BACKEND = "torch_gpu"
MULT = 10
GOLDEN_MAKESPAN = 448166.8336
GOLDEN_CHROM_IDX = 462

# ── load instance (identical to inspect_chromosome.py) ─────────────────────
with open(f'data/Instances/P{NB_PARTS}M{NB_MACHINES}-{INST_NUMBER}.txt') as f:
    instance_parts = np.array([int(x) for x in f.read().split()])
instance_parts_unique = np.unique(instance_parts)

cache_path = 'data/PartsMachines/cached_specs.pkl'
with open(cache_path, 'rb') as f:
    jobSpecAll, machSpec, area, polRotations = pickle.load(f)
jobSpec = jobSpecAll.loc[instance_parts_unique]

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
        diffs = np.diff(padded.astype(np.int8), axis=1)
        starts, ends = np.where(diffs == 1), np.where(diffs == -1)
        runs = ends[1] - starts[1]
        maxr = np.zeros(rotated.shape[0], dtype=np.int32)
        if len(starts[0]) > 0:
            np.maximum.at(maxr, starts[0], runs)
        densities.append(maxr)
    best_rotation = int(np.argmin([s[0] for s in shapes]))
    rotations_gpu = [collision_backend.prepare_rotation_tensor(r) for r in rotations] \
                    if hasattr(collision_backend, 'prepare_rotation_tensor') else None
    rotations_uint8 = [r.astype(np.uint8) for r in rotations]
    pd_ = PartData(id=part, area=area[part], nrot=nrot, rotations=rotations,
                   shapes=shapes, densities=densities, best_rotation=best_rotation,
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

# ── rebuild population (seed 42) ───────────────────────────────────────────
num_individuals = MULT * NB_PARTS
num_genes       = 2 * NB_PARTS
np.random.seed(42)
population = np.random.uniform(0.0, 1.0, size=(num_individuals, num_genes)).astype(np.float32)
# Note: inspect_chromosome.py overwrites population[0] with a greedy initial_sol
# built from `random.uniform` (unseeded).  That's non-deterministic across
# processes, but chromosome 462 comes from the np.random stream so it IS
# reproducible.  We don't touch population[0].

chosen = population[GOLDEN_CHROM_IDX].astype(np.float32).copy()
print(f"Chromosome index {GOLDEN_CHROM_IDX}: first 5 genes = {chosen[:5]}")

# ── evaluate ONLY this chromosome (single row batch) through native decoder ──
evaluator = FullNativeDecoderEvaluator(
    problem_data, NB_PARTS, NB_MACHINES, thresholds,
    instance_parts, collision_backend, device='cuda'
)

# Warm-up (compiles extension, allocates workspaces).
_ = evaluator.evaluate_batch(np.array([chosen]))

# Part A: sweep batch size N and place chosen at row (N-1).
print()
print("=" * 72)
print("Part A: batch size sweep, chosen at row (N-1)")
print(f"Golden: {GOLDEN_MAKESPAN:.4f}")
print(f"{'N':>5}  {'row':>5}  {'makespan':>14}  {'diff':>14}")
for N in [1, 100, 200, 250, 300, 400, 463, 500]:
    if N == 1:
        batch = np.array([chosen], dtype=np.float32)
        target_row = 0
    else:
        batch = np.empty((N, num_genes), dtype=np.float32)
        batch[:N - 1] = population[:N - 1]
        batch[N - 1]  = chosen
        target_row = N - 1
    f = evaluator.evaluate_batch(batch)
    ms = float(f[target_row])
    diff = ms - GOLDEN_MAKESPAN
    marker = "OK" if abs(diff) < 1.0 else "DIVERGE"
    print(f"{N:>5}  {target_row:>5}  {ms:>14.4f}  {diff:>+14.4f}  {marker}")

# Part B: full 500-batch with chosen at ORIGINAL row 462 vs at row 0.
print()
print("=" * 72)
print("Part B: full 500-batch — effect of chosen's row position")
print(f"{'position':>10}  {'makespan':>14}  {'diff':>14}")
# Variant 1: original population, row 462 is chosen naturally
f1 = evaluator.evaluate_batch(population)
ms1 = float(f1[GOLDEN_CHROM_IDX])
print(f"{'462 (orig)':>10}  {ms1:>14.4f}  {ms1 - GOLDEN_MAKESPAN:>+14.4f}  "
      f"{'OK' if abs(ms1 - GOLDEN_MAKESPAN) < 1 else 'DIVERGE'}")

# Variant 2: swap row 0 and row 462, so chosen is at row 0 now.
swapped = population.copy()
swapped[[0, GOLDEN_CHROM_IDX]] = swapped[[GOLDEN_CHROM_IDX, 0]]
f2 = evaluator.evaluate_batch(swapped)
ms2 = float(f2[0])
print(f"{'0 (swap)':>10}  {ms2:>14.4f}  {ms2 - GOLDEN_MAKESPAN:>+14.4f}  "
      f"{'OK' if abs(ms2 - GOLDEN_MAKESPAN) < 1 else 'DIVERGE'}")

# Variant 3: only chosen (at row 0) + 499 copies of chosen (same gene).
same = np.tile(chosen, (500, 1))
f3 = evaluator.evaluate_batch(same)
ms3 = float(f3[0])
ms3_last = float(f3[499])
print(f"{'tile[0]':>10}  {ms3:>14.4f}  {ms3 - GOLDEN_MAKESPAN:>+14.4f}  "
      f"{'OK' if abs(ms3 - GOLDEN_MAKESPAN) < 1 else 'DIVERGE'}")
print(f"{'tile[499]':>10}  {ms3_last:>14.4f}  {ms3_last - GOLDEN_MAKESPAN:>+14.4f}  "
      f"{'OK' if abs(ms3_last - GOLDEN_MAKESPAN) < 1 else 'DIVERGE'}")
print("=" * 72)

# Part C: repeat full-batch twice, check determinism.
print()
print("Part C: determinism within same batch content")
f_a = evaluator.evaluate_batch(population)
f_b = evaluator.evaluate_batch(population)
ms_a = float(f_a[GOLDEN_CHROM_IDX])
ms_b = float(f_b[GOLDEN_CHROM_IDX])
print(f"  run 1 row 462: {ms_a:.4f}")
print(f"  run 2 row 462: {ms_b:.4f}")
print(f"  identical: {ms_a == ms_b}")

# Part D: isolate which other rows poison row 462.
# Start with tile[462] (all chosen, all correct), then replace one row at a
# time with population[k] and see if row 462 breaks.
print()
print("Part D: which single row contaminates row 462?")
base = np.tile(chosen, (500, 1))
poisoners = []
for k in [0, 1, 50, 100, 200, 300, 400, 461, 463]:
    if k == GOLDEN_CHROM_IDX:
        continue
    batch = base.copy()
    batch[k] = population[k]
    f = evaluator.evaluate_batch(batch)
    ms = float(f[GOLDEN_CHROM_IDX])
    diff = ms - GOLDEN_MAKESPAN
    marker = "OK" if abs(diff) < 1.0 else "DIVERGE"
    print(f"  poison row {k:>3} with population[{k}] → row 462 = {ms:.4f} ({diff:+.4f}) {marker}")
    if marker == "DIVERGE":
        poisoners.append(k)
print(f"  poisoners: {poisoners}")

sys.exit(0)
