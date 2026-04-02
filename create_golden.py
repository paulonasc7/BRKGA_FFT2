"""
create_golden.py
----------------
Creates a golden reference file for correctness verification of the
wave_batch evaluator.  Run this once with the current correct code;
verify_correctness.py then checks that any refactored code produces
identical per-placement output.

Picks a fixed chromosome from a seeded population (same seeds used by
inspect_chromosome.py) so the golden is fully deterministic.

Saves:  results/golden/P{n}M{m}-{i}/golden.json
  - chromosome (as float32 list)
  - overall makespan
  - per-machine makespan
  - flat per-placement log: [{part_id, bin_idx, col, row, rot}]

Usage:
    python create_golden.py 50 2 0 torch_gpu
    python create_golden.py 75 2 0 torch_gpu
"""

import sys, os, json, pickle, random
import numpy as np
import torch
import pandas as pd

torch.set_num_threads(1)
torch.set_grad_enabled(False)

from collision_backend import create_collision_backend
from data_structures import PartData, MachinePartData, MachineData, ProblemData
from wave_batch_evaluator import WaveBatchEvaluator

# ── Args ───────────────────────────────────────────────────────────────────────
nbParts      = int(sys.argv[1]) if len(sys.argv) > 1 else 50
nbMachines   = int(sys.argv[2]) if len(sys.argv) > 2 else 2
instNumber   = int(sys.argv[3]) if len(sys.argv) > 3 else 0
backend_name = sys.argv[4]      if len(sys.argv) > 4 else 'torch_gpu'

MULT         = 10
POPULATION_SEED  = 42   # same as inspect_chromosome.py
SELECTION_SEED   = 7    # same as inspect_chromosome.py

instance_label = f"P{nbParts}M{nbMachines}-{instNumber}"
print(f"Creating golden for {instance_label}  backend={backend_name}")

# ── Load instance ──────────────────────────────────────────────────────────────
with open(f'data/Instances/P{nbParts}M{nbMachines}-{instNumber}.txt') as f:
    instance_parts = np.array([int(x) for x in f.read().split()])
instance_parts_unique = np.unique(instance_parts)

cache_path = 'data/PartsMachines/cached_specs.pkl'
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as fh:
        jobSpecAll, machSpec, area, polRotations = pickle.load(fh)
else:
    jobSpecAll   = pd.read_excel('data/PartsMachines/part-machine-information.xlsx',
                                 sheet_name='part',    header=0, index_col=0)
    machSpec     = pd.read_excel('data/PartsMachines/part-machine-information.xlsx',
                                 sheet_name='machine', header=0, index_col=0)
    area         = pd.read_excel('data/PartsMachines/polygon_areas.xlsx',
                                 header=0)["Area"].tolist()
    polRotations = pd.read_excel('data/PartsMachines/parts_rotations.xlsx',
                                 header=0)["rot"].tolist()
    with open(cache_path, 'wb') as fh:
        pickle.dump((jobSpecAll, machSpec, area, polRotations), fh)

jobSpec = jobSpecAll.loc[instance_parts_unique]

# ── Build problem data ─────────────────────────────────────────────────────────
collision_backend = create_collision_backend(backend_name)

parts_dict = {}
for part in instance_parts_unique:
    matrix = np.ascontiguousarray(
        np.load(f'data/partsMatrices/matrix_{part}.npy').astype(np.int32))
    nrot = 2 if np.array_equal(matrix, np.rot90(matrix, 2)) else 4
    rotations, shapes, densities = [], [], []
    for rot in range(nrot):
        r = np.ascontiguousarray(np.rot90(matrix, rot))
        rotations.append(r)
        shapes.append((r.shape[0], r.shape[1]))
        padded = np.pad(r, ((0, 0), (1, 1)), constant_values=0)
        diffs  = np.diff(padded.astype(np.int8), axis=1)
        si, ei = np.where(diffs == 1), np.where(diffs == -1)
        rl     = ei[1] - si[1]
        mx     = np.zeros(r.shape[0], dtype=np.int32)
        if len(si[0]):
            np.maximum.at(mx, si[0], rl)
        densities.append(mx)
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
for m in range(nbMachines):
    bL = machSpec['L(mm)'].iloc[m]
    bW = machSpec['W(mm)'].iloc[m]
    machine_parts = {}
    for part in instance_parts_unique:
        pd_  = parts_dict[part]
        ffts = [collision_backend.prepare_part_fft(pd_.rotations[r], bL, bW)
                for r in range(pd_.nrot)]
        proc_time = (jobSpec["volume(mm3)"].loc[part]  * machSpec["VT(s/mm3)"].iloc[m] +
                     jobSpec["support(mm3)"].loc[part] * machSpec["SPT(s/mm3)"].iloc[m])
        proc_time_height = jobSpec["height(mm)"].loc[part] * machSpec["HT(s/mm3)"].iloc[m]
        machine_parts[part] = MachinePartData(ffts=ffts, proc_time=proc_time,
                                              proc_time_height=proc_time_height)
    machines_list.append(MachineData(bin_length=bL, bin_width=bW, bin_area=bL * bW,
                                     setup_time=machSpec['ST(s)'].iloc[m],
                                     parts=machine_parts))

problem_data = ProblemData(parts=parts_dict, machines=machines_list,
                           instance_parts=instance_parts,
                           instance_parts_unique=instance_parts_unique)
thresholds = [t / nbMachines for t in range(1, nbMachines)]

# ── Build evaluator ────────────────────────────────────────────────────────────
evaluator = WaveBatchEvaluator(problem_data, nbParts, nbMachines, thresholds,
                               instance_parts, collision_backend, device='cuda')

# ── Create seeded population and pick chromosome ───────────────────────────────
num_individuals = MULT * nbParts
num_genes       = 2 * nbParts

np.random.seed(POPULATION_SEED)
population = np.random.uniform(0.0, 1.0,
                               size=(num_individuals, num_genes)).astype(np.float32)

print(f"Evaluating {num_individuals} chromosomes (warm-up + batch)...")
fitness_list = evaluator.evaluate_batch(population)

best_idx   = int(np.argmin(fitness_list))
rng        = np.random.default_rng(seed=SELECTION_SEED)
candidates = [i for i in range(num_individuals)
              if i != best_idx and fitness_list[i] < 1e15]
chosen_idx = int(rng.choice(candidates))
chrom      = population[chosen_idx].copy()

print(f"  Best makespan in population : {fitness_list[best_idx]:.4f}")
print(f"  Chosen chromosome index     : {chosen_idx}  "
      f"makespan={fitness_list[chosen_idx]:.4f}")

# ── Re-run single chromosome with placement logging enabled ───────────────────
evaluator._collect_contexts = True
evaluator._last_contexts    = {}
evaluator._placement_log    = {}
single_fitness = evaluator.evaluate_batch(np.array([chrom]))
evaluator._collect_contexts = False

makespan = float(single_fitness[0])
print(f"  Single re-run makespan      : {makespan:.4f}")

if abs(makespan - fitness_list[chosen_idx]) > 1.0:
    print("WARNING: makespan differs between batch run and single re-run "
          "— possible non-determinism.  Golden may be unreliable.")

# ── Extract per-machine data ───────────────────────────────────────────────────
machines_data = {}
for m_idx in range(nbMachines):
    mach_data   = problem_data.machines[m_idx]
    ctx         = evaluator._last_contexts[m_idx][0]
    active_bins = [b for b in ctx.open_bins if b.area > 0]

    machine_makespan = sum(
        b.proc_time + b.proc_time_height + mach_data.setup_time
        for b in active_bins)

    # Flat placement log from _placement_log: (part_id, bin_idx, col, row, rot, shape)
    raw_log = evaluator._placement_log.get(m_idx, [])
    placements = [
        {"part_id": int(pid), "bin_idx": int(bidx),
         "col": int(col), "row": int(row), "rot": int(rot)}
        for pid, bidx, col, row, rot, _shape in raw_log
    ]

    machines_data[str(m_idx)] = {
        "makespan":   round(machine_makespan, 6),
        "n_bins":     len(active_bins),
        "placements": placements,
    }
    print(f"  Machine {m_idx}: makespan={machine_makespan:.4f}  "
          f"bins={len(active_bins)}  placements={len(placements)}")

# ── Save golden JSON ───────────────────────────────────────────────────────────
golden = {
    "instance":    instance_label,
    "nbParts":     nbParts,
    "nbMachines":  nbMachines,
    "instNumber":  instNumber,
    "chromosome_idx": chosen_idx,
    "chromosome":  chrom.tolist(),   # float32 list — reload with np.array(..., dtype=np.float32)
    "makespan":    round(makespan, 6),
    "machines":    machines_data,
}

out_dir = os.path.join("results", "golden", instance_label)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "golden.json")

with open(out_path, 'w') as f:
    json.dump(golden, f, indent=2)

total_placements = sum(len(machines_data[k]["placements"]) for k in machines_data)
print(f"\nGolden saved → {out_path}")
print(f"  {total_placements} total placements across {nbMachines} machines")
