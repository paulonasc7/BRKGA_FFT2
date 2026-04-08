"""
Per-phase wall-clock profiler for FullNativeDecoderEvaluator.

Instruments the C++ process_wave to emit per-phase timings. Since we can't
modify the C++ source at runtime, we wrap the Python-level call and measure
the total evaluate_batch, then cross-reference with torch.profiler data
to understand phase-level breakdown.

This script also counts key metrics per wave (n_tests p1/p2, n_placements,
n_new_bins) by comparing grid_states before/after.

Usage:
    python profile_phases_native.py 50 2 0 torch_gpu
"""

import json
import os
import pickle
import sys
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.profiler

from collision_backend import create_collision_backend
from data_structures import PartData, MachinePartData, MachineData, ProblemData

# ── config ───────────────────────────────────────────────────────────────────

SEEDS = [123, 321, 777]
N_WARMUP = 2
MULT = 10

# ── args ─────────────────────────────────────────────────────────────────────

nb_parts     = int(sys.argv[1]) if len(sys.argv) > 1 else 50
nb_machines  = int(sys.argv[2]) if len(sys.argv) > 2 else 2
inst_idx     = int(sys.argv[3]) if len(sys.argv) > 3 else 0
backend_name = sys.argv[4]      if len(sys.argv) > 4 else "torch_gpu"
device       = "cuda" if backend_name == "torch_gpu" else "cpu"

num_individuals = MULT * nb_parts
num_genes       = 2 * nb_parts
instance_label  = f"P{nb_parts}M{nb_machines}-{inst_idx}"

print(f"profile_phases_native: {instance_label}  pop={num_individuals}  device={device}")
print(f"Seeds: {SEEDS}  warmup calls: {N_WARMUP}")
print()

# ── load problem data ────────────────────────────────────────────────────────

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

# ── build evaluator ──────────────────────────────────────────────────────────

print("\nBuilding FullNativeDecoderEvaluator...", flush=True)
t0 = time.time()
from full_native_decoder import FullNativeDecoderEvaluator
native_eval = FullNativeDecoderEvaluator(problem_data, nb_parts, nb_machines,
                                          thresholds, instance_parts,
                                          collision_backend, device=device)
print(f"  ready in {time.time()-t0:.1f}s", flush=True)

# ── fixed-seed population helper ─────────────────────────────────────────────

def make_population(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(
        (num_individuals, num_genes), dtype=np.float64
    ).astype(np.float32)

# ── warmup ───────────────────────────────────────────────────────────────────

print(f"\nWarmup ({N_WARMUP} calls)...", flush=True)
for i in range(N_WARMUP):
    pop = make_population(SEEDS[i % len(SEEDS)])
    native_eval.evaluate_batch(pop)
if device == "cuda":
    torch.cuda.synchronize()
print("  done", flush=True)

# ── profiling with trace ─────────────────────────────────────────────────────

print(f"\nProfiling with trace ({len(SEEDS)} seeds)...", flush=True)

activities = [torch.profiler.ProfilerActivity.CPU]
if device == "cuda":
    activities.append(torch.profiler.ProfilerActivity.CUDA)

os.makedirs("results/benchmarks", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Aggregate ops across seeds
agg_cuda = defaultdict(float)
agg_cpu = defaultdict(float)
agg_count = defaultdict(int)
seed_times = {}

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

    seed_times[seed] = wall
    ka = prof.key_averages()
    for evt in ka:
        agg_cuda[evt.key] += evt.self_cuda_time_total
        agg_cpu[evt.key] += evt.self_cpu_time_total
        agg_count[evt.key] += evt.count

    print(f"  seed {seed}: {wall:.3f}s", flush=True)

# Save trace for last seed
trace_path = f"results/benchmarks/native_phases_trace_{instance_label}_{ts}.json"
prof.export_chrome_trace(trace_path)

total_cuda_us = sum(agg_cuda.values())
total_cpu_us = sum(agg_cpu.values())
total_wall = sum(seed_times.values())

# ── Analysis: group ops by phase ─────────────────────────────────────────────

print()
print("=" * 80)
print("PHASE-LEVEL BREAKDOWN (estimated from op grouping)")
print("=" * 80)

# Phase 2: rfft2 (grid FFT recomputation)
rfft2_cuda = agg_cuda.get("aten::_fft_r2c", 0)
# Phase 4: irfft2 + mul (batch FFT collision)
irfft2_cuda = agg_cuda.get("aten::_fft_c2r", 0)
mul_cuda = agg_cuda.get("aten::mul", 0)
# index_select: used in Phase 2 (grid_ffts.index_select) and Phase 4 (batch_grid_ffts, batch_part_ffts)
idx_sel_cuda = agg_cuda.get("aten::index_select", 0)
# CUDA selector kernel
selector_cuda = sum(v for k, v in agg_cuda.items() if 'select_best_positions' in k)
# index_copy_ and index_fill_: Phase 2 (grid_ffts.index_copy_) and Phase 6 (grid_states.index_fill_)
idx_copy_cuda = agg_cuda.get("aten::index_copy_", 0)
idx_fill_cuda = agg_cuda.get("aten::fill_", 0)
# copy_ (CPU→GPU transfers)
copy_cuda = agg_cuda.get("aten::copy_", 0)
# batch_grid_update kernel
grid_update_cuda = sum(v for k, v in agg_cuda.items() if 'batch_grid_update' in k)
# DtoD memcpy (internal to FFT ops)
dtod_cuda = sum(v for k, v in agg_cuda.items() if 'Memcpy DtoD' in k)

# Compute per-seed averages
n_seeds = len(SEEDS)

print(f"\n{'Component':<45} {'CUDA ms':>10} {'% total':>8}")
print("-" * 65)
print(f"{'irfft2 (Phase 4 inverse FFT)':<45} {irfft2_cuda/1e3:>10.1f} {100*irfft2_cuda/total_cuda_us:>7.1f}%")
print(f"{'mul (Phase 4 pointwise FFT multiply)':<45} {mul_cuda/1e3:>10.1f} {100*mul_cuda/total_cuda_us:>7.1f}%")
print(f"{'index_select (gather grids + part FFTs)':<45} {idx_sel_cuda/1e3:>10.1f} {100*idx_sel_cuda/total_cuda_us:>7.1f}%")
print(f"{'DtoD memcpy (FFT internal)':<45} {dtod_cuda/1e3:>10.1f} {100*dtod_cuda/total_cuda_us:>7.1f}%")
print(f"{'rfft2 (Phase 2 grid FFT recompute)':<45} {rfft2_cuda/1e3:>10.1f} {100*rfft2_cuda/total_cuda_us:>7.1f}%")
print(f"{'CUDA selector kernel':<45} {selector_cuda/1e3:>10.1f} {100*selector_cuda/total_cuda_us:>7.1f}%")
print(f"{'copy_ (CPU→GPU transfers)':<45} {copy_cuda/1e3:>10.1f} {100*copy_cuda/total_cuda_us:>7.1f}%")
print(f"{'index_copy_ (Phase 2 grid FFT writeback)':<45} {idx_copy_cuda/1e3:>10.1f} {100*idx_copy_cuda/total_cuda_us:>7.1f}%")
print(f"{'fill_ (Phase 6 grid_states.index_fill_)':<45} {idx_fill_cuda/1e3:>10.1f} {100*idx_fill_cuda/total_cuda_us:>7.1f}%")
print(f"{'batch_grid_update kernel':<45} {grid_update_cuda/1e3:>10.1f} {100*grid_update_cuda/total_cuda_us:>7.1f}%")
print(f"{'─ subtotal accounted':<45} {(irfft2_cuda+mul_cuda+idx_sel_cuda+dtod_cuda+rfft2_cuda+selector_cuda+copy_cuda+idx_copy_cuda+idx_fill_cuda+grid_update_cuda)/1e3:>10.1f} {100*(irfft2_cuda+mul_cuda+idx_sel_cuda+dtod_cuda+rfft2_cuda+selector_cuda+copy_cuda+idx_copy_cuda+idx_fill_cuda+grid_update_cuda)/total_cuda_us:>7.1f}%")
print(f"{'─ other/unaccounted':<45} {(total_cuda_us - irfft2_cuda-mul_cuda-idx_sel_cuda-dtod_cuda-rfft2_cuda-selector_cuda-copy_cuda-idx_copy_cuda-idx_fill_cuda-grid_update_cuda)/1e3:>10.1f}")
print(f"{'TOTAL CUDA':<45} {total_cuda_us/1e3:>10.1f}")

# Grouped phase estimates
phase4_cuda = irfft2_cuda + mul_cuda + dtod_cuda + selector_cuda
phase2_cuda = rfft2_cuda + idx_copy_cuda
phase_transfer = copy_cuda
phase_indexsel = idx_sel_cuda

print(f"\n--- Grouped phase estimates ---")
print(f"{'Phase 4 (IFFT+mul+selector+DtoD)':<45} {phase4_cuda/1e3:>10.1f} {100*phase4_cuda/total_cuda_us:>7.1f}%")
print(f"{'Phase 2 (rfft2+index_copy_)':<45} {phase2_cuda/1e3:>10.1f} {100*phase2_cuda/total_cuda_us:>7.1f}%")
print(f"{'index_select (shared Ph2+Ph4)':<45} {phase_indexsel/1e3:>10.1f} {100*phase_indexsel/total_cuda_us:>7.1f}%")
print(f"{'CPU→GPU transfers':<45} {phase_transfer/1e3:>10.1f} {100*phase_transfer/total_cuda_us:>7.1f}%")

# CPU-side breakdown
print(f"\n--- CPU time breakdown ---")
print(f"{'Component':<45} {'CPU ms':>10} {'% total':>8}")
print("-" * 65)
sorted_cpu = sorted(agg_cpu.items(), key=lambda x: x[1], reverse=True)
for key, us in sorted_cpu[:15]:
    if us < 50:
        continue
    pct = 100.0 * us / total_cpu_us
    print(f"{key:<45} {us/1e3:>10.1f} {pct:>7.1f}%")

# Count call counts to estimate wave structure
print(f"\n--- Call counts (for wave structure estimation) ---")
irfft2_count = agg_count.get("aten::_fft_c2r", 0)
rfft2_count = agg_count.get("aten::_fft_r2c", 0)
idx_sel_count = agg_count.get("aten::index_select", 0)
copy_count = agg_count.get("aten::copy_", 0)
print(f"irfft2 calls: {irfft2_count} (= {irfft2_count//n_seeds}/seed, ≈ IFFT chunks)")
print(f"rfft2 calls: {rfft2_count} (= {rfft2_count//n_seeds}/seed, ≈ grid FFT recomputes)")
print(f"index_select calls: {idx_sel_count} (= {idx_sel_count//n_seeds}/seed)")
print(f"copy_ calls: {copy_count} (= {copy_count//n_seeds}/seed)")
print(f"mul calls: {agg_count.get('aten::mul', 0)} (= {agg_count.get('aten::mul', 0)//n_seeds}/seed)")

# ── Wall time vs CUDA time ───────────────────────────────────────────────────

print(f"\n--- Wall time vs CUDA time ---")
print(f"Total wall time ({n_seeds} seeds): {total_wall:.3f}s")
print(f"Mean wall time per seed: {total_wall/n_seeds:.3f}s")
print(f"Total CUDA time ({n_seeds} seeds): {total_cuda_us/1e6:.3f}s")
print(f"Mean CUDA time per seed: {total_cuda_us/1e6/n_seeds:.3f}s")
cpu_only_est = total_wall - total_cuda_us/1e6/n_seeds  # rough: wall - CUDA/seed overlap
print(f"\nCPU-only time estimate (wall - CUDA/seed): ~{total_wall/n_seeds - total_cuda_us/1e6/n_seeds:.3f}s/seed")
print(f"  → This is roughly Phase 1 (decode) + Phase 3 (vacancy) + Phase 5 (selection)")
print(f"     + Phase 6 (new bin creation) + Python overhead")

# ── save summary ─────────────────────────────────────────────────────────────

summary = {
    "instance": instance_label,
    "num_individuals": num_individuals,
    "device": device,
    "seeds": SEEDS,
    "seed_wall_times": seed_times,
    "total_cuda_us": total_cuda_us,
    "total_cpu_us": total_cpu_us,
    "phase_breakdown": {
        "phase4_cuda_ms": phase4_cuda / 1e3,
        "phase2_cuda_ms": phase2_cuda / 1e3,
        "index_select_cuda_ms": phase_indexsel / 1e3,
        "transfer_cuda_ms": phase_transfer / 1e3,
        "irfft2_count_per_seed": irfft2_count // n_seeds,
        "rfft2_count_per_seed": rfft2_count // n_seeds,
    },
}
summary_path = f"results/benchmarks/native_phases_{instance_label}_{ts}.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved → {summary_path}")
