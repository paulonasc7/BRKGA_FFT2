"""
verify_full_population_vs_pp.py
-------------------------------
Run placementProcedure (pp) on every chromosome in the np.random.seed(42)
population for P50M2-0 and compare element-wise with the native C++/CUDA
decoder's full-batch output.  Exits 0 iff every chromosome matches (|diff|<1s).
"""

import os, pickle, sys, time
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
torch.set_grad_enabled(False)

from collision_backend import create_collision_backend
from data_structures import PartData, MachinePartData, MachineData, ProblemData
from full_native_decoder import FullNativeDecoderEvaluator
from binClassNew import BuildingPlate as BinPlate

NB_PARTS, NB_MACHINES, INST_NUMBER = 50, 2, 0
BACKEND = "torch_gpu"
MULT = 10

with open(f'data/Instances/P{NB_PARTS}M{NB_MACHINES}-{INST_NUMBER}.txt') as f:
    instance_parts = np.array([int(x) for x in f.read().split()])
instance_parts_unique = np.unique(instance_parts)

with open('data/PartsMachines/cached_specs.pkl', 'rb') as f:
    jobSpecAll, machSpec, area, polRotations = pickle.load(f)
jobSpec = jobSpecAll.loc[instance_parts_unique]

collision_backend = create_collision_backend(BACKEND)

parts_dict = {}
for part in instance_parts_unique:
    matrix = np.ascontiguousarray(np.load(f'data/partsMatrices/matrix_{part}.npy').astype(np.int32))
    nrot = 2 if np.array_equal(matrix, np.rot90(matrix, 2)) else 4
    rotations, shapes, densities = [], [], []
    for rot in range(nrot):
        rotated = np.ascontiguousarray(np.rot90(matrix, rot))
        rotations.append(rotated)
        shapes.append((rotated.shape[0], rotated.shape[1]))
        padded = np.pad(rotated, ((0, 0), (1, 1)), constant_values=0)
        diffs = np.diff(padded.astype(np.int8), axis=1)
        starts = np.where(diffs == 1)
        ends   = np.where(diffs == -1)
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

# Build population.
num_individuals = MULT * NB_PARTS
num_genes       = 2 * NB_PARTS
np.random.seed(42)
population = np.random.uniform(0.0, 1.0, size=(num_individuals, num_genes)).astype(np.float32)

# ── native full-batch eval ─────────────────────────────────────────────────
print("Building native evaluator (first run compiles CUDA, 2-3 min)...", flush=True)
t0 = time.time()
evaluator = FullNativeDecoderEvaluator(problem_data, NB_PARTS, NB_MACHINES, thresholds,
                                       instance_parts, collision_backend, device='cuda')
_ = evaluator.evaluate_batch(population)  # warm-up + caches
native_fitness = np.array(evaluator.evaluate_batch(population), dtype=np.float64)
print(f"  native full-batch done in {time.time() - t0:.1f}s", flush=True)

# ── pp eval for every chromosome ───────────────────────────────────────────
def pp_one(chrom):
    SV = chrom[:NB_PARTS]
    MV = chrom[NB_PARTS:]
    overall = 0.0
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

        H_m, W_m = mach_data.bin_length, mach_data.bin_width
        openBins = []
        is_feasible = True
        for partInd in sorted_sequence:
            part_data = problem_data.parts[partInd]
            mach_part_data = mach_data.parts[partInd]
            ps0 = part_data.shapes[0]
            if ((ps0[0] > H_m or ps0[1] > W_m) and (ps0[1] > H_m or ps0[0] > W_m)):
                is_feasible = False
                break
            placed = False
            for b_obj in openBins:
                if b_obj.area + part_data.area > mach_data.bin_area:
                    continue
                if b_obj.can_insert(part_data, mach_part_data):
                    placed = True
                    break
            if not placed:
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
                openBins.append(newBin)
        makespan = sum(b.processingTime + b.processingTimeHeight + mach_data.setup_time
                       for b in openBins) if is_feasible else 1e16
        overall = max(overall, makespan)
    return overall

print(f"\nRunning pp on all {num_individuals} chromosomes sequentially...", flush=True)
t0 = time.time()
pp_fitness = np.zeros(num_individuals, dtype=np.float64)
for i in range(num_individuals):
    pp_fitness[i] = pp_one(population[i])
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        remaining = elapsed / (i + 1) * (num_individuals - i - 1)
        print(f"  {i + 1}/{num_individuals}  ({elapsed:.0f}s elapsed, ~{remaining:.0f}s left)",
              flush=True)
print(f"  pp done in {time.time() - t0:.1f}s", flush=True)

# ── element-wise comparison ────────────────────────────────────────────────
diff = native_fitness - pp_fitness
abs_diff = np.abs(diff)
n_match = int((abs_diff < 1.0).sum())
n_mismatch = int((abs_diff >= 1.0).sum())

print()
print("=" * 72)
print(f"Element-wise comparison: native full-batch vs pp (N={num_individuals})")
print(f"  match  (|diff|<1.0): {n_match}")
print(f"  mismatch             : {n_mismatch}")
print(f"  max |diff|           : {abs_diff.max():.6f}  at row {int(abs_diff.argmax())}")
print(f"  mean native fitness  : {native_fitness.mean():.4f}")
print(f"  mean pp fitness      : {pp_fitness.mean():.4f}")
print(f"  |mean diff|          : {abs(native_fitness.mean() - pp_fitness.mean()):.6f}")
print("=" * 72)

if n_mismatch > 0:
    print("\nFirst 20 mismatches (row, native, pp, diff):")
    mismatch_rows = np.where(abs_diff >= 1.0)[0][:20]
    for r in mismatch_rows:
        print(f"  row {int(r):3d}: native={native_fitness[r]:14.4f}  "
              f"pp={pp_fitness[r]:14.4f}  diff={diff[r]:+14.4f}")
    sys.exit(1)
else:
    print("\nRESULT: ALL MATCH — native full-batch agrees with pp element-wise.")
    sys.exit(0)
