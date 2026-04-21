"""
Profile CPU hotspots inside FullNativeDecoderEvaluator.

Loads P50M2-0 (or a configurable instance), builds the native decoder, runs a
warmup, resets the internal profiler, then runs N_REPS timed generations and
prints the accumulated hotspot breakdown.

Usage:
    python profile_cpu_hotspots.py                # P50M2-0, 5 reps
    python profile_cpu_hotspots.py 75 2 0 torch_gpu 3

This script is a local, uncommitted diagnostic tool (see
CPU_OPTIMIZATION_CANDIDATES.md).
"""

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

from collision_backend import create_collision_backend
from data_structures import PartData, MachinePartData, MachineData, ProblemData

SEEDS = [123]          # single seed is enough to get proportional breakdown
N_REPS_DEFAULT = 5
MULT = 10

nb_parts     = int(sys.argv[1]) if len(sys.argv) > 1 else 50
nb_machines  = int(sys.argv[2]) if len(sys.argv) > 2 else 2
inst_idx     = int(sys.argv[3]) if len(sys.argv) > 3 else 0
backend_name = sys.argv[4]      if len(sys.argv) > 4 else "torch_gpu"
n_reps       = int(sys.argv[5]) if len(sys.argv) > 5 else N_REPS_DEFAULT
device       = "cuda" if backend_name == "torch_gpu" else "cpu"

num_individuals = MULT * nb_parts
num_genes       = 2 * nb_parts
instance_label  = f"P{nb_parts}M{nb_machines}-{inst_idx}"

print(f"CPU hotspot profile: {instance_label}  pop={num_individuals}  device={device}")
print(f"Reps: {n_reps}  (seed={SEEDS[0]})")
print()

# ── load problem data (same logic as benchmark_evaluators.py) ────────────────
collision_backend = create_collision_backend(backend_name)

with open(f"data/Instances/P{nb_parts}M{nb_machines}-{inst_idx}.txt", "r") as f:
    instance_parts = np.array([int(x) for x in f.read().split()])
instance_parts_unique = np.unique(instance_parts)

cache_path = "data/PartsMachines/cached_specs.pkl"
if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
        job_spec_all, mach_spec, area, pol_rotations = pickle.load(f)
else:
    job_spec_all = pd.read_excel("data/PartsMachines/part-machine-information.xlsx",
                                 sheet_name="part", header=0, index_col=0)
    mach_spec    = pd.read_excel("data/PartsMachines/part-machine-information.xlsx",
                                 sheet_name="machine", header=0, index_col=0)
    area          = pd.read_excel("data/PartsMachines/polygon_areas.xlsx",
                                  header=0)["Area"].tolist()
    pol_rotations = pd.read_excel("data/PartsMachines/parts_rotations.xlsx",
                                  header=0)["rot"].tolist()
    with open(cache_path, "wb") as f:
        pickle.dump((job_spec_all, mach_spec, area, pol_rotations), f)

job_spec = job_spec_all.loc[instance_parts_unique]

print("Loading parts and machine data...", flush=True)
t0 = time.time()
parts_dict: dict = {}
for part in instance_parts_unique:
    matrix = np.ascontiguousarray(np.load(f"data/partsMatrices/matrix_{part}.npy").astype(np.int32))
    nrot = 2 if np.array_equal(matrix, np.rot90(matrix, 2)) else 4
    rotations, shapes, densities = [], [], []
    for rot in range(nrot):
        rotated = np.ascontiguousarray(np.rot90(matrix, rot))
        rotations.append(rotated)
        shapes.append(rotated.shape)
        padded = np.pad(rotated, ((0, 0), (1, 1)), constant_values=0)
        diffs = np.diff(padded.astype(np.int8), axis=1)
        run_lengths = np.where(diffs == -1)[1] - np.where(diffs == 1)[1]
        max_runs = np.zeros(rotated.shape[0], dtype=np.int32)
        si = np.where(diffs == 1)
        if len(si[0]):
            np.maximum.at(max_runs, si[0], run_lengths)
        densities.append(max_runs)
    best_rotation = int(np.argmin([s[0] for s in shapes]))
    rotations_gpu = ([collision_backend.prepare_rotation_tensor(r) for r in rotations]
                     if hasattr(collision_backend, "prepare_rotation_tensor") else None)
    rotations_uint8 = [r.astype(np.uint8) for r in rotations]
    pd_obj = PartData(id=part, area=area[part], nrot=nrot, rotations=rotations,
                      shapes=shapes, densities=densities, best_rotation=best_rotation,
                      rotations_gpu=rotations_gpu, rotations_uint8=rotations_uint8)
    pd_obj.prepare_jit_data()
    parts_dict[part] = pd_obj

machines_list = []
for m in range(nb_machines):
    blen = mach_spec["L(mm)"].iloc[m]
    bwid = mach_spec["W(mm)"].iloc[m]
    setup = mach_spec["ST(s)"].iloc[m]
    mp = {}
    for part in instance_parts_unique:
        pd_obj = parts_dict[part]
        ffts = [collision_backend.prepare_part_fft(pd_obj.rotations[rot], blen, bwid)
                for rot in range(pd_obj.nrot)]
        pt  = (job_spec["volume(mm3)"].loc[part]  * mach_spec["VT(s/mm3)"].iloc[m] +
               job_spec["support(mm3)"].loc[part] * mach_spec["SPT(s/mm3)"].iloc[m])
        pth = job_spec["height(mm)"].loc[part] * mach_spec["HT(s/mm3)"].iloc[m]
        mp[part] = MachinePartData(ffts=ffts, proc_time=pt, proc_time_height=pth)
    machines_list.append(MachineData(bin_length=blen, bin_width=bwid,
                                     bin_area=blen * bwid, setup_time=setup, parts=mp))

problem_data = ProblemData(parts=parts_dict, machines=machines_list,
                           instance_parts=instance_parts,
                           instance_parts_unique=instance_parts_unique)
thresholds = [t / nb_machines for t in range(1, nb_machines)]
print(f"  data loaded in {time.time() - t0:.1f}s", flush=True)

# ── build native evaluator ────────────────────────────────────────────────────
print("\nBuilding FullNativeDecoderEvaluator (first run compiles CUDA — may take 2-3 min)...", flush=True)
t0 = time.time()
from full_native_decoder import FullNativeDecoderEvaluator
native_eval = FullNativeDecoderEvaluator(problem_data, nb_parts, nb_machines, thresholds,
                                          instance_parts, collision_backend, device=device)
print(f"  ready in {time.time() - t0:.1f}s", flush=True)

# ── warmup ───────────────────────────────────────────────────────────────────
print("\nWarmup (1 untimed call)...", flush=True)
rng = np.random.default_rng(SEEDS[0])
warmup_pop = rng.random((num_individuals, num_genes), dtype=np.float64).astype(np.float32)
native_eval.evaluate_batch(warmup_pop)
if device == "cuda":
    torch.cuda.synchronize()
print("  warmup done", flush=True)

# ── reset profiler and run timed reps ────────────────────────────────────────
FullNativeDecoderEvaluator.reset_profile()
if hasattr(native_eval._decoder, "reset_bin_stats"):
    native_eval._decoder.reset_bin_stats()

print(f"\nRunning {n_reps} timed generations...", flush=True)
gen_times = []
last_makespans = None
for rep in range(n_reps):
    pop_rng = np.random.default_rng(SEEDS[0] + rep)
    pop = pop_rng.random((num_individuals, num_genes), dtype=np.float64).astype(np.float32)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    last_makespans = native_eval.evaluate_batch(pop)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gen_times.append(dt)
    print(f"  rep {rep + 1}: {dt:.3f}s", flush=True)

# Correctness fingerprint: first 5 *feasible* makespans of the last generation
# (filters out 1e16 infeasibles, which carry no discriminative information).
# Deterministic given seed + problem data, so any code change that preserves
# correctness must reproduce these exact values.
feasible = [m for m in last_makespans if m < 1e15]
n_feasible = len(feasible)
n_infeasible = len(last_makespans) - n_feasible
print(f"\nMakespan fingerprint (last rep, first 5 feasible of {n_feasible}): "
      f"{[f'{m:.6f}' for m in feasible[:5]]}", flush=True)
print(f"  (infeasible {n_infeasible} / {len(last_makespans)})", flush=True)

mean_s = float(np.mean(gen_times))
std_s  = float(np.std(gen_times))
total_s = float(np.sum(gen_times))

print(f"\nWall clock: mean {mean_s:.3f}s  std {std_s:.3f}s  total {total_s:.3f}s  "
      f"({n_reps} reps)", flush=True)

# ── print accumulated hotspot summary ────────────────────────────────────────
print(FullNativeDecoderEvaluator.get_profile_summary())

# Normalise to per-rep numbers and to % of total wall clock
print(f"\nPer-generation averages (divide above by {n_reps}):")
print(f"  Total wall clock per gen: {mean_s * 1000:.1f} ms")
print(f"  Total accumulated ns in instrumented blocks: see summary above")

# Sub-stage 2a: per-machine bins-per-solution high-water across all reps.
if hasattr(native_eval._decoder, "get_bin_stats"):
    stats = native_eval._decoder.get_bin_stats()
    print("\n═══ Bins-per-solution high-water (sub-stage 2a) ═══")
    print(f"  {'machine':>7}  {'peak':>6}  {'avg':>8}  {'total_bins':>10}  {'sols':>8}  {'batches':>7}  {'peak_global':>11}  {'growths':>7}")
    for m, (peak, total_bins, total_sols, num_batches, peak_global, growths) in enumerate(stats):
        avg = total_bins / max(total_sols, 1)
        print(f"  {m:>7}  {peak:>6}  {avg:>8.3f}  {total_bins:>10}  {total_sols:>8}  {num_batches:>7}  {peak_global:>11}  {growths:>7}")
