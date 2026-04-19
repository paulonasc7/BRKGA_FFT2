"""
Deep profiler: CPU active/stalled + GPU active/idle breakdown.

Combines:
  1. C++ WaveProfile instrumentation (wall-clock per phase)
  2. torch.profiler with CUDA activity tracing (actual GPU kernel time)
  3. Fixed seeds for full reproducibility

Produces a comprehensive timing breakdown showing exactly where CPU and GPU
time is spent, including stalls.

Usage:
    python profile_deep.py [nb_parts] [nb_machines] [inst_idx] [backend] [n_reps]
    python profile_deep.py 50 2 0 torch_gpu 5
"""

import os
import pickle
import sys
import time
import json

import numpy as np
import pandas as pd
import torch
import torch.profiler

from collision_backend import create_collision_backend
from data_structures import PartData, MachinePartData, MachineData, ProblemData

# ── config ──────────────────────────────────────────────────────────────────
SEED = 42
N_WARMUP = 2

nb_parts     = int(sys.argv[1]) if len(sys.argv) > 1 else 50
nb_machines  = int(sys.argv[2]) if len(sys.argv) > 2 else 2
inst_idx     = int(sys.argv[3]) if len(sys.argv) > 3 else 0
backend_name = sys.argv[4]      if len(sys.argv) > 4 else "torch_gpu"
n_reps       = int(sys.argv[5]) if len(sys.argv) > 5 else 5
device       = "cuda" if backend_name == "torch_gpu" else "cpu"

MULT = 10
num_individuals = MULT * nb_parts
num_genes       = 2 * nb_parts
instance_label  = f"P{nb_parts}M{nb_machines}-{inst_idx}"

print(f"Deep profile: {instance_label}  pop={num_individuals}  device={device}")
print(f"Seed: {SEED}  warmup: {N_WARMUP}  reps: {n_reps}")
print()

# ── load problem data ───────────────────────────────────────────────────────
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
parts_dict = {}
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

# ── build native evaluator ──────────────────────────────────────────────────
print("\nBuilding FullNativeDecoderEvaluator...", flush=True)
t0 = time.time()
from full_native_decoder import FullNativeDecoderEvaluator
native_eval = FullNativeDecoderEvaluator(problem_data, nb_parts, nb_machines, thresholds,
                                          instance_parts, collision_backend, device=device)
print(f"  ready in {time.time() - t0:.1f}s", flush=True)

# ── deterministic population generator ──────────────────────────────────────
def make_population(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(
        (num_individuals, num_genes), dtype=np.float64
    ).astype(np.float32)

# ── warmup ──────────────────────────────────────────────────────────────────
print(f"\nWarmup ({N_WARMUP} calls)...", flush=True)
for i in range(N_WARMUP):
    pop = make_population(SEED + 1000 + i)
    native_eval.evaluate_batch(pop)
    if device == "cuda":
        torch.cuda.synchronize()
print("  done", flush=True)

# ── Phase A: Wall-clock + C++ instrumentation (no torch.profiler overhead) ──
print(f"\n{'='*70}")
print(f"Phase A: C++ instrumented wall-clock timing ({n_reps} reps, seed={SEED})")
print(f"{'='*70}")

FullNativeDecoderEvaluator.reset_profile()

gen_times = []
fingerprint = None
for rep in range(n_reps):
    pop = make_population(SEED + rep)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = native_eval.evaluate_batch(pop)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gen_times.append(dt)
    fingerprint = [f"{m:.6f}" for m in result[:5]]
    print(f"  rep {rep}: {dt*1000:.1f}ms", flush=True)

print(f"\nMakespan fingerprint (rep {n_reps-1}, first 5): {fingerprint}")
mean_ms = np.mean(gen_times) * 1000
std_ms = np.std(gen_times) * 1000
print(f"Wall clock: {mean_ms:.1f} ± {std_ms:.1f} ms/gen  (n={n_reps})")
print(FullNativeDecoderEvaluator.get_profile_summary())

# ── Phase B: torch.profiler with CUDA activity tracing ──────────────────────
print(f"\n{'='*70}")
print(f"Phase B: torch.profiler CUDA activity trace (1 rep, seed={SEED})")
print(f"{'='*70}")

pop = make_population(SEED)
if device == "cuda":
    torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=False,
    with_stack=False,
    with_flops=False,
) as prof:
    t0 = time.perf_counter()
    native_eval.evaluate_batch(pop)
    if device == "cuda":
        torch.cuda.synchronize()
    wall_profiled = time.perf_counter() - t0

print(f"  Profiled wall time: {wall_profiled*1000:.1f}ms")
print(f"  (profiler adds ~5-15% overhead, so this is slower than Phase A)")

# Aggregate CUDA and CPU times by operator
key_averages = prof.key_averages()

# ── CUDA kernel time breakdown ──────────────────────────────────────────────
print(f"\n{'─'*70}")
print("TOP CUDA KERNELS (by self_cuda_time)")
print(f"{'─'*70}")
print(f"{'Operator':<45} {'CUDA ms':>10} {'CPU ms':>10} {'Calls':>7}")
print(f"{'─'*45} {'─'*10} {'─'*10} {'─'*7}")

cuda_entries = [(e.key, e.self_cuda_time_total / 1000, e.self_cpu_time_total / 1000, e.count)
                for e in key_averages if e.self_cuda_time_total > 0]
cuda_entries.sort(key=lambda x: -x[1])

total_cuda_ms = sum(x[1] for x in cuda_entries)
for name, cuda_ms, cpu_ms, count in cuda_entries[:20]:
    pct = 100 * cuda_ms / total_cuda_ms if total_cuda_ms > 0 else 0
    print(f"  {name:<43} {cuda_ms:>9.1f} {cpu_ms:>9.1f} {count:>7}  ({pct:>5.1f}%)")

print(f"  {'TOTAL':<43} {total_cuda_ms:>9.1f}")

# ── CPU time breakdown ──────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("TOP CPU OPERATORS (by self_cpu_time)")
print(f"{'─'*70}")
print(f"{'Operator':<45} {'CPU ms':>10} {'CUDA ms':>10} {'Calls':>7}")
print(f"{'─'*45} {'─'*10} {'─'*10} {'─'*7}")

cpu_entries = [(e.key, e.self_cpu_time_total / 1000, e.self_cuda_time_total / 1000, e.count)
               for e in key_averages]
cpu_entries.sort(key=lambda x: -x[1])

total_cpu_ms = sum(x[1] for x in cpu_entries)
for name, cpu_ms, cuda_ms, count in cpu_entries[:20]:
    pct = 100 * cpu_ms / total_cpu_ms if total_cpu_ms > 0 else 0
    print(f"  {name:<43} {cpu_ms:>9.1f} {cuda_ms:>9.1f} {count:>7}  ({pct:>5.1f}%)")

print(f"  {'TOTAL':<43} {total_cpu_ms:>9.1f}")

# ── Categorized summary ─────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("CATEGORIZED BREAKDOWN")
print(f"{'─'*70}")

# Categorize CUDA ops
fft_cuda = 0.0
fft_cpu = 0.0
copy_cuda = 0.0
copy_cpu = 0.0
custom_kernel_cuda = 0.0
custom_kernel_cpu = 0.0
other_cuda = 0.0
other_cpu = 0.0

for e in key_averages:
    name = e.key.lower()
    cuda_ms = e.self_cuda_time_total / 1000
    cpu_ms = e.self_cpu_time_total / 1000

    if 'fft' in name or 'cufft' in name or 'irfft' in name or 'rfft' in name:
        fft_cuda += cuda_ms
        fft_cpu += cpu_ms
    elif 'copy' in name or 'memcpy' in name or 'to(' in name:
        copy_cuda += cuda_ms
        copy_cpu += cpu_ms
    elif ('native_' in name or '_kernel' in name or 'fused_gather' in name
          or 'vacancy' in name or 'grid_update' in name or 'select_best' in name):
        custom_kernel_cuda += cuda_ms
        custom_kernel_cpu += cpu_ms
    else:
        other_cuda += cuda_ms
        other_cpu += cpu_ms

print(f"  {'Category':<25} {'CUDA ms':>10} {'CPU ms':>10} {'CUDA %':>8}")
print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*8}")
for cat, cuda_ms, cpu_ms in [
    ("FFT (rfft2+irfft2)", fft_cuda, fft_cpu),
    ("Copy/Transfer", copy_cuda, copy_cpu),
    ("Custom CUDA kernels", custom_kernel_cuda, custom_kernel_cpu),
    ("Other GPU ops", other_cuda, other_cpu),
]:
    pct = 100 * cuda_ms / total_cuda_ms if total_cuda_ms > 0 else 0
    print(f"  {cat:<25} {cuda_ms:>9.1f} {cpu_ms:>9.1f} {pct:>7.1f}%")

print(f"  {'TOTAL':<25} {total_cuda_ms:>9.1f} {total_cpu_ms:>9.1f}")

# ── CPU vs GPU utilization analysis ─────────────────────────────────────────
print(f"\n{'─'*70}")
print("CPU vs GPU TIME ANALYSIS")
print(f"{'─'*70}")

wall_ms = wall_profiled * 1000
print(f"  Wall clock (profiled):   {wall_ms:>9.1f} ms")
print(f"  Total CUDA kernel time:  {total_cuda_ms:>9.1f} ms  ({100*total_cuda_ms/wall_ms:.1f}% of wall)")
print(f"  Total CPU operator time: {total_cpu_ms:>9.1f} ms  ({100*total_cpu_ms/wall_ms:.1f}% of wall)")
print()
# Note: total_cpu_ms includes time spent launching GPU ops and waiting for results.
# GPU utilization = CUDA kernel time / wall time.
# CPU "active" time ≈ total_cpu_ms - sync_stall_time
# But sync stalls are hard to separate from torch.profiler alone.

# Look for synchronization ops
sync_cpu_ms = 0.0
for e in key_averages:
    name = e.key.lower()
    # .to(kCPU), copy_ with sync, cudaStreamSynchronize, etc.
    if ('synchronize' in name or 'cudadevicesynchronize' in name
        or 'cudastreamsynchronize' in name):
        sync_cpu_ms += e.self_cpu_time_total / 1000

# The biggest sync stalls are in copy_ calls that do GPU→CPU with non_blocking=false.
# In the profiler, these show up as CPU time in aten::copy_ that includes the device sync.
copy_cpu_total = 0.0
for e in key_averages:
    if 'copy_' in e.key:
        copy_cpu_total += e.self_cpu_time_total / 1000

print(f"  Explicit sync calls:     {sync_cpu_ms:>9.1f} ms")
print(f"  copy_ CPU time (incl sync): {copy_cpu_total:>9.1f} ms")
print(f"  GPU utilization:         {100*total_cuda_ms/wall_ms:>8.1f}%")

# ── Phase C: CUDA event timing for per-phase GPU time ───────────────────────
# This gives us the actual GPU compute time for the major operations,
# separate from CPU launch overhead.
print(f"\n{'='*70}")
print(f"Phase C: CUDA event GPU timing (1 rep, seed={SEED})")
print(f"{'='*70}")

if device == "cuda":
    pop = make_population(SEED)
    torch.cuda.synchronize()

    # We'll measure the total GPU time for one evaluate_batch using CUDA events.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    native_eval.evaluate_batch(pop)
    end_event.record()
    torch.cuda.synchronize()

    gpu_total_ms = start_event.elapsed_time(end_event)
    print(f"  Total GPU time (CUDA events): {gpu_total_ms:.1f}ms")
    print(f"  Wall clock (Phase A mean):    {mean_ms:.1f}ms")
    print(f"  GPU utilization:              {100*gpu_total_ms/mean_ms:.1f}%")
    print(f"  CPU overhead + stalls:        {mean_ms - gpu_total_ms:.1f}ms ({100*(mean_ms - gpu_total_ms)/mean_ms:.1f}%)")

# ── Save trace for detailed analysis ────────────────────────────────────────
trace_path = f"profile_deep_{instance_label}.json"
prof.export_chrome_trace(trace_path)
print(f"\nChrome trace saved to: {trace_path}")
print("  Open in chrome://tracing or https://ui.perfetto.dev/")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Instance:     {instance_label}")
print(f"Population:   {num_individuals} × {num_genes}")
print(f"Seed:         {SEED}")
print(f"Wall clock:   {mean_ms:.1f} ± {std_ms:.1f} ms/gen  (n={n_reps}, unprofiled)")
if device == "cuda":
    print(f"GPU time:     {gpu_total_ms:.1f} ms  ({100*gpu_total_ms/mean_ms:.1f}% utilization)")
    print(f"CPU overhead: {mean_ms - gpu_total_ms:.1f} ms  ({100*(mean_ms - gpu_total_ms)/mean_ms:.1f}%)")
print(f"FFT CUDA:     {fft_cuda:.1f} ms  ({100*fft_cuda/total_cuda_ms:.1f}% of GPU)")
print(f"Transfers:    {copy_cuda:.1f} ms  ({100*copy_cuda/total_cuda_ms:.1f}% of GPU)")
print(f"Custom kern:  {custom_kernel_cuda:.1f} ms  ({100*custom_kernel_cuda/total_cuda_ms:.1f}% of GPU)")
