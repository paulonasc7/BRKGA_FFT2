"""
verify_correctness.py
---------------------
Verifies that the wave_batch evaluator produces per-placement output that
is bit-for-bit identical to the golden reference created by create_golden.py.

Run this after any refactoring of wave_batch_evaluator.py (e.g. C++/pybind11
CPU phase extension) to confirm that no placements changed.

Exit code: 0 on PASS, 1 on FAIL.

Usage:
    python verify_correctness.py 50 2 0 torch_gpu
    python verify_correctness.py 75 2 0 torch_gpu
"""

import sys, os, json, pickle
import numpy as np
import torch
import math

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

import pandas as pd

instance_label = f"P{nbParts}M{nbMachines}-{instNumber}"
golden_path    = os.path.join("results", "golden", instance_label, "golden.json")

print(f"Verifying {instance_label}  backend={backend_name}")

# ── Load golden ────────────────────────────────────────────────────────────────
if not os.path.exists(golden_path):
    print(f"FAIL: golden file not found: {golden_path}")
    print("      Run create_golden.py first.")
    sys.exit(1)

with open(golden_path) as f:
    golden = json.load(f)

chrom = np.array(golden["chromosome"], dtype=np.float32)
print(f"  Loaded golden  chromosome_idx={golden['chromosome_idx']}  "
      f"makespan={golden['makespan']:.4f}")

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

# ── Run evaluator on golden chromosome ────────────────────────────────────────
evaluator = WaveBatchEvaluator(problem_data, nbParts, nbMachines, thresholds,
                               instance_parts, collision_backend, device='cuda')

evaluator._collect_contexts = True
evaluator._last_contexts    = {}
evaluator._placement_log    = {}
fitness = evaluator.evaluate_batch(np.array([chrom]))
evaluator._collect_contexts = False

got_makespan = float(fitness[0])
print(f"  Got makespan  {got_makespan:.4f}  "
      f"(expected {golden['makespan']:.4f})")

# ── Compare ────────────────────────────────────────────────────────────────────
failures = []
MAKESPAN_ABS_TOL = 1e-6

# 1. Overall makespan
if not math.isclose(got_makespan, golden["makespan"], rel_tol=0.0, abs_tol=MAKESPAN_ABS_TOL):
    failures.append(
        f"Overall makespan: got {got_makespan:.6f}, "
        f"expected {golden['makespan']:.6f}  "
        f"(diff={abs(got_makespan - golden['makespan']):.6f}, "
        f"tol={MAKESPAN_ABS_TOL:.1e})")

# 2. Per-machine placements
for m_idx in range(nbMachines):
    key = str(m_idx)
    exp_placements = golden["machines"][key]["placements"]
    raw_log        = evaluator._placement_log.get(m_idx, [])
    got_placements = [
        {"part_id": int(pid), "bin_idx": int(bidx),
         "col": int(col), "row": int(row), "rot": int(rot)}
        for pid, bidx, col, row, rot, _shape in raw_log
    ]

    if len(got_placements) != len(exp_placements):
        failures.append(
            f"Machine {m_idx}: placement count "
            f"got={len(got_placements)} expected={len(exp_placements)}")
        continue  # no point comparing element-by-element

    for i, (got, exp) in enumerate(zip(got_placements, exp_placements)):
        if got != exp:
            failures.append(
                f"Machine {m_idx} placement {i}:\n"
                f"  got      {got}\n"
                f"  expected {exp}")
            # Report first 5 differences per machine, then stop
            if sum(1 for f in failures if f"Machine {m_idx}" in f) >= 5:
                failures.append(f"  Machine {m_idx}: stopping after 5 diffs")
                break

# ── Report ─────────────────────────────────────────────────────────────────────
print()
if not failures:
    total = sum(len(golden["machines"][str(m)]["placements"])
                for m in range(nbMachines))
    print(f"PASS  — {total} placements across {nbMachines} machines, "
          f"all identical to golden.")
    sys.exit(0)
else:
    print(f"FAIL  — {len(failures)} issue(s):")
    for msg in failures:
        print(f"  • {msg}")
    sys.exit(1)
