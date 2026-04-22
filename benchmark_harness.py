"""
benchmark_harness.py
--------------------
Reproducible wall-clock benchmark for the native C++/CUDA decoder.

Runs a fixed suite of instances with a fixed seed, sweeps
ABRKGA_NATIVE_NUM_WORKERS (1 and 2 on CUDA; just 1 on CPU), and writes
a structured JSON report to benchmarks/{timestamp}_{gpu}.json.

Designed to be run on whichever GPU host you care about (A4000 today,
L40S tomorrow) so before/after comparisons are unambiguous.  Use
compare_benchmarks.py to diff two reports.

Usage:
    python benchmark_harness.py                # default suite + sweep
    python benchmark_harness.py --quick        # skip P100M4 (fast mode)
    python benchmark_harness.py --tag stage3   # label the run

Output:
    benchmarks/{YYYYMMDD_HHMMSS}_{gpu}_{tag}.json
"""

import argparse
import datetime as _dt
import json
import os
import pickle
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Tuple

# Enable expandable_segments BEFORE torch is imported, so that the 2-worker
# configuration on P100M4-class instances doesn't OOM from caching-allocator
# fragmentation.  (See PARALLEL_MACHINES_PLAN.md sub-stage 3c findings.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch

from collision_backend import create_collision_backend
from data_structures import PartData, MachinePartData, MachineData, ProblemData

# ── Benchmark suite (fixed — do not change casually) ─────────────────────────
# 2 instances per class, picked to mix IDs.  Adding/dropping entries makes
# historical JSONs incomparable; add *new* entries rather than editing.
SUITE: List[Tuple[int, int, int]] = [
    (25, 2, 2),   # P25M2-2  — small sanity
    (25, 2, 4),   # P25M2-4
    (50, 2, 1),   # P50M2-1
    (50, 2, 3),   # P50M2-3
    (75, 2, 0),   # P75M2-0
    (75, 2, 4),   # P75M2-4
    (100, 4, 0),  # P100M4-0 — parallel target
    (100, 4, 3),  # P100M4-3
]

# Trim suite in --quick mode (skip the slow P100M4 instances).
QUICK_EXCLUDE = {(100, 4, 0), (100, 4, 3)}

SEED = 123
N_REPS = 5
MULT = 10            # num_individuals = MULT * nb_parts
BACKEND = "torch_gpu"


# ─────────────────────────────────────────────────────────────────────────────
# Problem-data loading (adapted from profile_cpu_hotspots.py; kept in-file so
# the harness has zero non-repo dependencies).
# ─────────────────────────────────────────────────────────────────────────────
def _load_problem(nb_parts: int, nb_machines: int, inst_idx: int, device: str,
                  backend):
    path = f"data/Instances/P{nb_parts}M{nb_machines}-{inst_idx}.txt"
    with open(path, "r") as f:
        instance_parts = np.array([int(x) for x in f.read().split()])
    instance_parts_unique = np.unique(instance_parts)

    cache_path = "data/PartsMachines/cached_specs.pkl"
    with open(cache_path, "rb") as f:
        job_spec_all, mach_spec, area, pol_rotations = pickle.load(f)
    job_spec = job_spec_all.loc[instance_parts_unique]

    parts_dict: Dict[int, PartData] = {}
    for part in instance_parts_unique:
        matrix = np.ascontiguousarray(
            np.load(f"data/partsMatrices/matrix_{part}.npy").astype(np.int32)
        )
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
        rotations_gpu = ([backend.prepare_rotation_tensor(r) for r in rotations]
                         if hasattr(backend, "prepare_rotation_tensor") else None)
        rotations_uint8 = [r.astype(np.uint8) for r in rotations]
        pd_obj = PartData(
            id=part, area=area[part], nrot=nrot, rotations=rotations,
            shapes=shapes, densities=densities, best_rotation=best_rotation,
            rotations_gpu=rotations_gpu, rotations_uint8=rotations_uint8,
        )
        pd_obj.prepare_jit_data()
        parts_dict[part] = pd_obj

    machines_list = []
    for m in range(nb_machines):
        blen = mach_spec["L(mm)"].iloc[m]
        bwid = mach_spec["W(mm)"].iloc[m]
        setup = mach_spec["ST(s)"].iloc[m]
        mp: Dict[int, MachinePartData] = {}
        for part in instance_parts_unique:
            pd_obj = parts_dict[part]
            ffts = [backend.prepare_part_fft(pd_obj.rotations[rot], blen, bwid)
                    for rot in range(pd_obj.nrot)]
            pt = (job_spec["volume(mm3)"].loc[part] * mach_spec["VT(s/mm3)"].iloc[m]
                  + job_spec["support(mm3)"].loc[part] * mach_spec["SPT(s/mm3)"].iloc[m])
            pth = job_spec["height(mm)"].loc[part] * mach_spec["HT(s/mm3)"].iloc[m]
            mp[part] = MachinePartData(ffts=ffts, proc_time=pt, proc_time_height=pth)
        machines_list.append(MachineData(
            bin_length=blen, bin_width=bwid, bin_area=blen * bwid,
            setup_time=setup, parts=mp,
        ))

    problem_data = ProblemData(
        parts=parts_dict, machines=machines_list,
        instance_parts=instance_parts,
        instance_parts_unique=instance_parts_unique,
    )
    thresholds = [t / nb_machines for t in range(1, nb_machines)]
    return problem_data, thresholds, instance_parts


def _run_one(nb_parts: int, nb_machines: int, inst_idx: int,
             num_workers: int, device: str) -> Dict:
    """Build evaluator + run N_REPS timed generations.  Returns result dict."""
    # num_workers is baked into the evaluator at construction via env var —
    # must be set *before* FullNativeDecoderEvaluator is built.
    os.environ["ABRKGA_NATIVE_NUM_WORKERS"] = str(num_workers)

    # Import lazily so the env var is picked up at _pack_problem_data time.
    from full_native_decoder import FullNativeDecoderEvaluator

    label = f"P{nb_parts}M{nb_machines}-{inst_idx}"
    backend = create_collision_backend(BACKEND)
    problem_data, thresholds, instance_parts = _load_problem(
        nb_parts, nb_machines, inst_idx, device, backend
    )

    evaluator = FullNativeDecoderEvaluator(
        problem_data, nb_parts, nb_machines, thresholds,
        instance_parts, backend, device=device,
    )

    num_individuals = MULT * nb_parts
    num_genes = 2 * nb_parts

    # Warmup (untimed).
    warmup_rng = np.random.default_rng(SEED)
    warmup_pop = warmup_rng.random((num_individuals, num_genes),
                                   dtype=np.float64).astype(np.float32)
    evaluator.evaluate_batch(warmup_pop)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Timed reps — exactly the same population scheme as profile_cpu_hotspots.py
    # so fingerprints are directly comparable.
    gen_times: List[float] = []
    last_makespans = None
    for rep in range(N_REPS):
        pop_rng = np.random.default_rng(SEED + rep)
        pop = pop_rng.random((num_individuals, num_genes),
                             dtype=np.float64).astype(np.float32)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        last_makespans = evaluator.evaluate_batch(pop)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        gen_times.append(time.perf_counter() - t0)

    feasible = [float(m) for m in last_makespans if m < 1e15]
    fingerprint = [f"{m:.6f}" for m in feasible[:5]]

    return {
        "instance": label,
        "nb_parts": nb_parts,
        "nb_machines": nb_machines,
        "inst_idx": inst_idx,
        "num_workers": num_workers,
        "num_individuals": num_individuals,
        "n_reps": N_REPS,
        "mean_s": statistics.mean(gen_times),
        "std_s": statistics.pstdev(gen_times) if len(gen_times) > 1 else 0.0,
        "min_s": min(gen_times),
        "max_s": max(gen_times),
        "rep_times_s": gen_times,
        "fingerprint": fingerprint,
        "n_feasible": len(feasible),
        "n_infeasible": len(last_makespans) - len(feasible),
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


def _gpu_name(device: str) -> str:
    if not device.startswith("cuda"):
        return "cpu"
    idx = int(device.split(":")[-1]) if ":" in device else 0
    try:
        return torch.cuda.get_device_name(idx)
    except Exception:
        return "unknown-cuda"


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="skip P100M4 instances (shorter run)")
    parser.add_argument("--tag", default="",
                        help="free-form label to include in output filename")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu / cuda:N (default: auto)")
    parser.add_argument("--workers", default=None,
                        help="comma-separated list of num_workers values "
                             "(default: 1,2 on CUDA; 1 on CPU)")
    parser.add_argument("--out-dir", default="benchmarks",
                        help="directory for JSON output")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.workers:
        workers_sweep = [int(x) for x in args.workers.split(",") if x.strip()]
    else:
        workers_sweep = [1, 2] if device.startswith("cuda") else [1]

    suite = [s for s in SUITE if not (args.quick and s in QUICK_EXCLUDE)]

    print(f"Benchmark harness — device={device}, workers_sweep={workers_sweep}")
    print(f"Suite: {[f'P{p}M{m}-{i}' for p, m, i in suite]}")
    print(f"Reps per config: {N_REPS}   seed={SEED}")
    print()

    results: List[Dict] = []
    t_start = time.time()
    total_runs = len(suite) * len(workers_sweep)
    run_idx = 0
    for nb_parts, nb_machines, inst_idx in suite:
        for nw in workers_sweep:
            run_idx += 1
            label = f"P{nb_parts}M{nb_machines}-{inst_idx}"
            print(f"[{run_idx}/{total_runs}] {label}  workers={nw} ...",
                  flush=True)
            try:
                r = _run_one(nb_parts, nb_machines, inst_idx, nw, device)
                results.append(r)
                print(f"    mean={r['mean_s']:.3f}s  std={r['std_s']:.3f}s  "
                      f"min={r['min_s']:.3f}s  feasible={r['n_feasible']}/"
                      f"{r['n_feasible'] + r['n_infeasible']}",
                      flush=True)
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
                results.append({
                    "instance": label,
                    "nb_parts": nb_parts,
                    "nb_machines": nb_machines,
                    "inst_idx": inst_idx,
                    "num_workers": nw,
                    "error": f"{type(e).__name__}: {e}",
                })

    elapsed = time.time() - t_start
    gpu = _gpu_name(device)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": timestamp,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "tag": args.tag,
        "device": device,
        "gpu_name": gpu,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": SEED,
        "n_reps": N_REPS,
        "mult": MULT,
        "workers_sweep": workers_sweep,
        "pytorch_cuda_alloc_conf": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", ""
        ),
        "total_elapsed_s": elapsed,
        "results": results,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    tag_part = f"_{_sanitize(args.tag)}" if args.tag else ""
    fname = (f"{timestamp}_{_sanitize(gpu)}"
             f"_w{'-'.join(str(w) for w in workers_sweep)}{tag_part}.json")
    out_path = os.path.join(args.out_dir, fname)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print()
    print(f"Wrote {out_path}  (harness total {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
