"""
torch.profiler-based profiler for FullNativeDecoderEvaluator.

Usage:
    python profile_native.py <nb_parts> <nb_machines> <inst_idx> <backend>

Example:
    python profile_native.py 50 2 0 torch_gpu
    python profile_native.py 75 2 0 torch_gpu

For each of three fixed seeds the script runs one profiled evaluate_batch call.
Reports:
  - Top CUDA ops by total time  (GPU bottlenecks)
  - Top CPU ops by total time   (CPU / PCIe-transfer overhead)
  - Per-seed wall time for stability check
  - Chrome trace saved to results/benchmarks/native_trace_<instance>_<ts>.json
    (open in chrome://tracing or https://ui.perfetto.dev for visual inspection)
"""

import json
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.profiler

from collision_backend import create_collision_backend
from data_structures import PartData, MachinePartData, MachineData, ProblemData

# ── config ────────────────────────────────────────────────────────────────────

SEEDS      = [123, 321, 777]
N_WARMUP   = 2   # untimed calls before profiling starts
MULT       = 10  # population = MULT * nb_parts

# ── args ──────────────────────────────────────────────────────────────────────

nb_parts     = int(sys.argv[1]) if len(sys.argv) > 1 else 50
nb_machines  = int(sys.argv[2]) if len(sys.argv) > 2 else 2
inst_idx     = int(sys.argv[3]) if len(sys.argv) > 3 else 0
backend_name = sys.argv[4]      if len(sys.argv) > 4 else "torch_gpu"
device       = "cuda" if backend_name == "torch_gpu" else "cpu"

num_individuals = MULT * nb_parts
num_genes       = 2 * nb_parts
instance_label  = f"P{nb_parts}M{nb_machines}-{inst_idx}"

print(f"profile_native: {instance_label}  pop={num_individuals}  device={device}")
print(f"Seeds: {SEEDS}  warmup calls: {N_WARMUP}")
print()

# ── load problem data ─────────────────────────────────────────────────────────

collision_backend = create_collision_backend(backend_name)

with open(f'data/Instances/P{nb_parts}M{nb_machines}-{inst_idx}.txt') as f:
    instance_parts = np.array([int(x) for x in f.read().split()])
instance_parts_unique = np.unique(instance_parts)

cache_path = 'data/PartsMachines/cached_specs.pkl'
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        job_spec_all, mach_spec, area, pol_rotations = pickle.load(f)
else:
    job_spec_all  = pd.read_excel('data/PartsMachines/part-machine-information.xlsx', sheet_name='part',    header=0, index_col=0)
    mach_spec     = pd.read_excel('data/PartsMachines/part-machine-information.xlsx', sheet_name='machine', header=0, index_col=0)
    area          = pd.read_excel('data/PartsMachines/polygon_areas.xlsx',             header=0)["Area"].tolist()
    pol_rotations = pd.read_excel('data/PartsMachines/parts_rotations.xlsx',           header=0)["rot"].tolist()
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
        padded = np.pad(rotated, ((0, 0), (1, 1)), constant_values=0)
        diffs = np.diff(padded.astype(np.int8), axis=1)
        si = np.where(diffs == 1)
        run_lengths = np.where(diffs == -1)[1] - si[1]
        max_runs = np.zeros(rotated.shape[0], dtype=np.int32)
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
    blen  = mach_spec['L(mm)'].iloc[m]
    bwid  = mach_spec['W(mm)'].iloc[m]
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
                                     bin_area=blen * bwid, setup_time=setup, parts=mp))

problem_data = ProblemData(parts=parts_dict, machines=machines_list,
                           instance_parts=instance_parts,
                           instance_parts_unique=instance_parts_unique)
thresholds = [t / nb_machines for t in range(1, nb_machines)]
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

# ── build evaluator ───────────────────────────────────────────────────────────

print("\nBuilding FullNativeDecoderEvaluator...", flush=True)
t0 = time.time()
from full_native_decoder import FullNativeDecoderEvaluator
native_eval = FullNativeDecoderEvaluator(problem_data, nb_parts, nb_machines,
                                          thresholds, instance_parts,
                                          collision_backend, device=device)
print(f"  ready in {time.time()-t0:.1f}s", flush=True)

# ── fixed-seed population helper ──────────────────────────────────────────────

def make_population(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(
        (num_individuals, num_genes), dtype=np.float64
    ).astype(np.float32)

# ── warmup ────────────────────────────────────────────────────────────────────

print(f"\nWarmup ({N_WARMUP} calls)...", flush=True)
for i in range(N_WARMUP):
    pop = make_population(SEEDS[i % len(SEEDS)])
    native_eval.evaluate_batch(pop)
if device == "cuda":
    torch.cuda.synchronize()
print("  done", flush=True)

# ── profiling loop ────────────────────────────────────────────────────────────
# One profiler context per seed so traces are separate and seed-level wall
# times are easy to read.  All three are merged for the aggregate tables.

os.makedirs("results/benchmarks", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
trace_path = f"results/benchmarks/native_trace_{instance_label}_{ts}.json"

print(f"\nProfiling ({len(SEEDS)} seeds × 1 profiled call each)...", flush=True)

activities = [torch.profiler.ProfilerActivity.CPU]
if device == "cuda":
    activities.append(torch.profiler.ProfilerActivity.CUDA)

seed_wall_times = {}
all_key_avgs = []   # collect KeyAverages objects across seeds

for seed in SEEDS:
    pop = make_population(seed)
    if device == "cuda":
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        t0 = time.perf_counter()
        native_eval.evaluate_batch(pop)
        if device == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

    seed_wall_times[seed] = wall
    all_key_avgs.append(prof.key_averages())
    print(f"  seed {seed}: {wall:.3f}s", flush=True)

# ── export chrome trace (last seed only — representative) ─────────────────────

prof.export_chrome_trace(trace_path)
print(f"\nChrome trace saved → {trace_path}")
print("  (open in chrome://tracing or https://ui.perfetto.dev)")

# ── aggregate across seeds ────────────────────────────────────────────────────
# Build a unified view: sum self_cuda_time and self_cpu_time across seeds for
# each operator key.  This gives stable averages without per-seed noise.

from collections import defaultdict

agg_cuda = defaultdict(float)   # key → total self CUDA µs
agg_cpu  = defaultdict(float)   # key → total self CPU  µs
agg_count = defaultdict(int)    # key → number of calls

for ka in all_key_avgs:
    for evt in ka:
        agg_cuda[evt.key]  += evt.self_cuda_time_total   # µs
        agg_cpu[evt.key]   += evt.self_cpu_time_total    # µs
        agg_count[evt.key] += evt.count

total_cuda_us = sum(agg_cuda.values())
total_cpu_us  = sum(agg_cpu.values())

# ── print CUDA table ──────────────────────────────────────────────────────────

print()
print("=" * 72)
print(f"TOP CUDA OPS  (aggregated over {len(SEEDS)} seeds)")
print(f"Total CUDA time: {total_cuda_us/1e6:.3f}s")
print("=" * 72)
print(f"{'Op':<42}  {'CUDA ms':>9}  {'% total':>8}  {'calls':>7}")
print("-" * 72)

sorted_cuda = sorted(agg_cuda.items(), key=lambda x: x[1], reverse=True)
for key, us in sorted_cuda[:20]:
    if us < 100:   # skip ops under 0.1ms total
        continue
    pct = 100.0 * us / total_cuda_us if total_cuda_us > 0 else 0.0
    ms  = us / 1e3
    cnt = agg_count[key]
    print(f"{key:<42}  {ms:>9.2f}  {pct:>7.1f}%  {cnt:>7}")

# ── print CPU table ───────────────────────────────────────────────────────────

print()
print("=" * 72)
print(f"TOP CPU OPS  (aggregated over {len(SEEDS)} seeds)")
print(f"Total CPU time: {total_cpu_us/1e6:.3f}s")
print("=" * 72)
print(f"{'Op':<42}  {'CPU ms':>9}  {'% total':>8}  {'calls':>7}")
print("-" * 72)

sorted_cpu = sorted(agg_cpu.items(), key=lambda x: x[1], reverse=True)
for key, us in sorted_cpu[:20]:
    if us < 100:
        continue
    pct = 100.0 * us / total_cpu_us if total_cpu_us > 0 else 0.0
    ms  = us / 1e3
    cnt = agg_count[key]
    print(f"{key:<42}  {ms:>9.2f}  {pct:>7.1f}%  {cnt:>7}")

# ── summary ratios ────────────────────────────────────────────────────────────

print()
print("=" * 72)
print("SUMMARY")
print("=" * 72)

irfft_us = agg_cuda.get("fft_irfft2", 0) + agg_cuda.get("aten::fft_irfft2", 0)
rfft_us  = agg_cuda.get("fft_rfft2",  0) + agg_cuda.get("aten::fft_rfft2",  0)
mul_us   = agg_cuda.get("aten::mul",  0)
copy_us  = agg_cpu.get("aten::copy_", 0)

def pct_cuda(us): return f"{100*us/total_cuda_us:.1f}%" if total_cuda_us > 0 else "n/a"

print(f"  irfft2 (Phase 4 IFFT):     {irfft_us/1e3:>8.1f} ms  {pct_cuda(irfft_us)} of CUDA")
print(f"  rfft2  (Phase 2 grid FFT): {rfft_us/1e3:>8.1f} ms  {pct_cuda(rfft_us)} of CUDA")
print(f"  mul    (FFT pointwise):    {mul_us/1e3:>8.1f} ms  {pct_cuda(mul_us)} of CUDA")
print(f"  copy_  (CPU→GPU transfers):{copy_us/1e3:>8.1f} ms  (CPU time)")
print()
print(f"  Per-seed wall times: { {s: f'{t:.3f}s' for s, t in seed_wall_times.items()} }")
print()

# ── save summary JSON ─────────────────────────────────────────────────────────

summary = {
    "instance": instance_label,
    "num_individuals": num_individuals,
    "device": device,
    "seeds": SEEDS,
    "seed_wall_times": seed_wall_times,
    "total_cuda_us": total_cuda_us,
    "total_cpu_us": total_cpu_us,
    "top_cuda_ops": [
        {"op": k, "cuda_ms": v/1e3, "pct": 100*v/total_cuda_us if total_cuda_us else 0,
         "calls": agg_count[k]}
        for k, v in sorted_cuda[:20] if v >= 100
    ],
    "top_cpu_ops": [
        {"op": k, "cpu_ms": v/1e3, "pct": 100*v/total_cpu_us if total_cpu_us else 0,
         "calls": agg_count[k]}
        for k, v in sorted_cpu[:20] if v >= 100
    ],
}
summary_path = f"results/benchmarks/native_profile_{instance_label}_{ts}.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary JSON saved → {summary_path}")
