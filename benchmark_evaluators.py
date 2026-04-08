"""
Head-to-head benchmark: WaveBatchEvaluator vs FullNativeDecoderEvaluator.

Usage:
    python benchmark_evaluators.py <nb_parts> <nb_machines> <inst_idx> <backend>

Example:
    python benchmark_evaluators.py 50 2 0 torch_gpu
    python benchmark_evaluators.py 75 2 0 torch_gpu

For each of three fixed seeds the script:
  1. Generates a random population (same chromosomes for both evaluators).
  2. Runs one warmup call (not timed).
  3. Runs N_REPS timed calls.
  4. Reports per-call mean ± std and the speedup ratio.

N_REPS = 5 per seed → 15 measurements per evaluator.
Results are saved to results/benchmarks/benchmark_<instance>_<date>.json.
"""

import json
import math
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from collision_backend import create_collision_backend
from data_structures import PartData, MachinePartData, MachineData, ProblemData

# ── configuration ─────────────────────────────────────────────────────────────

SEEDS = [123, 321, 777]
N_REPS = 5          # timed calls per seed
MULT = 10           # population = MULT * nb_parts (matches BRKGA_alg3.py default)

# ── argument parsing ──────────────────────────────────────────────────────────

nb_parts    = int(sys.argv[1]) if len(sys.argv) > 1 else 50
nb_machines = int(sys.argv[2]) if len(sys.argv) > 2 else 2
inst_idx    = int(sys.argv[3]) if len(sys.argv) > 3 else 0
backend_name = sys.argv[4]    if len(sys.argv) > 4 else "torch_gpu"
device      = "cuda" if backend_name == "torch_gpu" else "cpu"

num_individuals = MULT * nb_parts
num_genes       = 2 * nb_parts

instance_label = f"P{nb_parts}M{nb_machines}-{inst_idx}"
print(f"Benchmark: {instance_label}  pop={num_individuals}  device={device}")
print(f"Seeds: {SEEDS}  reps/seed: {N_REPS}")
print()

# ── load problem data (same logic as BRKGA_alg3.py) ──────────────────────────

collision_backend = create_collision_backend(backend_name)

with open(f'data/Instances/P{nb_parts}M{nb_machines}-{inst_idx}.txt', 'r') as f:
    instance_parts = np.array([int(x) for x in f.read().split()])
instance_parts_unique = np.unique(instance_parts)

cache_path = 'data/PartsMachines/cached_specs.pkl'
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        job_spec_all, mach_spec, area, pol_rotations = pickle.load(f)
else:
    job_spec_all    = pd.read_excel('data/PartsMachines/part-machine-information.xlsx', sheet_name='part',    header=0, index_col=0)
    mach_spec       = pd.read_excel('data/PartsMachines/part-machine-information.xlsx', sheet_name='machine', header=0, index_col=0)
    area            = pd.read_excel('data/PartsMachines/polygon_areas.xlsx',             header=0)["Area"].tolist()
    pol_rotations   = pd.read_excel('data/PartsMachines/parts_rotations.xlsx',           header=0)["rot"].tolist()
    with open(cache_path, 'wb') as f:
        pickle.dump((job_spec_all, mach_spec, area, pol_rotations), f)

job_spec = job_spec_all.loc[instance_parts_unique]

print("Loading parts and machine data...", flush=True)
t0 = time.time()

parts_dict: dict = {}
for part in instance_parts_unique:
    matrix = np.ascontiguousarray(np.load(f'data/partsMatrices/matrix_{part}.npy').astype(np.int32))
    nrot = 2 if np.array_equal(matrix, np.rot90(matrix, 2)) else 4
    rotations, shapes, densities = [], [], []
    for rot in range(nrot):
        rotated = np.ascontiguousarray(np.rot90(matrix, rot))
        rotations.append(rotated)
        shapes.append(rotated.shape)
        padded = np.pad(rotated, ((0,0),(1,1)), constant_values=0)
        diffs = np.diff(padded.astype(np.int8), axis=1)
        run_lengths = np.where(diffs == -1)[1] - np.where(diffs == 1)[1]
        max_runs = np.zeros(rotated.shape[0], dtype=np.int32)
        si = np.where(diffs == 1)
        if len(si[0]):
            np.maximum.at(max_runs, si[0], run_lengths)
        densities.append(max_runs)
    best_rotation = int(np.argmin([s[0] for s in shapes]))
    rotations_gpu = ([collision_backend.prepare_rotation_tensor(r) for r in rotations]
                     if hasattr(collision_backend, 'prepare_rotation_tensor') else None)
    rotations_uint8 = [r.astype(np.uint8) for r in rotations]
    pd_obj = PartData(id=part, area=area[part], nrot=nrot, rotations=rotations,
                      shapes=shapes, densities=densities, best_rotation=best_rotation,
                      rotations_gpu=rotations_gpu, rotations_uint8=rotations_uint8)
    pd_obj.prepare_jit_data()
    parts_dict[part] = pd_obj

machines_list = []
for m in range(nb_machines):
    blen = mach_spec['L(mm)'].iloc[m]
    bwid = mach_spec['W(mm)'].iloc[m]
    setup = mach_spec['ST(s)'].iloc[m]
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
                                     bin_area=blen*bwid, setup_time=setup, parts=mp))

problem_data = ProblemData(parts=parts_dict, machines=machines_list,
                           instance_parts=instance_parts,
                           instance_parts_unique=instance_parts_unique)
thresholds = [t / nb_machines for t in range(1, nb_machines)]
print(f"  data loaded in {time.time()-t0:.1f}s", flush=True)

# ── build evaluators ──────────────────────────────────────────────────────────

print("\nBuilding WaveBatchEvaluator...", flush=True)
t0 = time.time()
from wave_batch_evaluator import WaveBatchEvaluator
wave_eval = WaveBatchEvaluator(problem_data, nb_parts, nb_machines, thresholds,
                                instance_parts, collision_backend)
print(f"  ready in {time.time()-t0:.1f}s", flush=True)

print("\nBuilding FullNativeDecoderEvaluator (first run compiles CUDA — may take 2-3 min)...", flush=True)
t0 = time.time()
from full_native_decoder import FullNativeDecoderEvaluator
native_eval = FullNativeDecoderEvaluator(problem_data, nb_parts, nb_machines, thresholds,
                                          instance_parts, collision_backend, device=device)
print(f"  ready in {time.time()-t0:.1f}s", flush=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def make_population(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((num_individuals, num_genes), dtype=np.float64).astype(np.float32)

def timed_call(evaluator, pop: np.ndarray) -> float:
    """Single timed evaluate_batch call. Returns wall time in seconds."""
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    evaluator.evaluate_batch(pop)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0

# ── warmup — one untimed call per evaluator per seed to avoid cold-start bias ─

print("\nWarming up...", flush=True)
warmup_pop = make_population(SEEDS[0])
_ = timed_call(wave_eval,   warmup_pop)
_ = timed_call(native_eval, warmup_pop)
print("  warmup done", flush=True)

# ── benchmark ─────────────────────────────────────────────────────────────────

results = {
    "instance": instance_label,
    "num_individuals": num_individuals,
    "num_genes": num_genes,
    "device": device,
    "seeds": SEEDS,
    "n_reps": N_REPS,
    "wave_batch": {},
    "native_full": {},
}

print()
print(f"{'Seed':>6}  {'Rep':>4}  {'wave_batch':>12}  {'native_full':>12}  {'speedup':>8}")
print("-" * 52)

wave_all, native_all = [], []

for seed in SEEDS:
    pop = make_population(seed)
    wave_times, native_times = [], []

    for rep in range(N_REPS):
        wt = timed_call(wave_eval,   pop)
        nt = timed_call(native_eval, pop)
        wave_times.append(wt)
        native_times.append(nt)
        speedup = wt / nt if nt > 0 else float('nan')
        print(f"{seed:>6}  {rep+1:>4}  {wt:>11.3f}s  {nt:>11.3f}s  {speedup:>7.2f}×")

    wave_all.extend(wave_times)
    native_all.extend(native_times)
    results["wave_batch"][str(seed)]   = wave_times
    results["native_full"][str(seed)] = native_times

print()
print("=" * 52)
wm  = np.mean(wave_all);   ws  = np.std(wave_all)
nm  = np.mean(native_all); ns  = np.std(native_all)
overall_speedup = wm / nm if nm > 0 else float('nan')
print(f"  wave_batch   mean: {wm:.3f}s  std: {ws:.3f}s")
print(f"  native_full  mean: {nm:.3f}s  std: {ns:.3f}s")
print(f"  Overall speedup:   {overall_speedup:.2f}×")
print("=" * 52)

results["summary"] = {
    "wave_batch_mean":   wm,  "wave_batch_std":   ws,
    "native_full_mean":  nm,  "native_full_std":  ns,
    "overall_speedup":   overall_speedup,
}

# ── save ──────────────────────────────────────────────────────────────────────

os.makedirs("results/benchmarks", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"results/benchmarks/benchmark_{instance_label}_{ts}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {out_path}")
