# Claude Context — BRKGA_FFT2

This file is automatically read by Claude Code at the start of every conversation. It gives you the full picture of what this project is, how to work on it, and how to run experiments autonomously on the remote GPU.

---

## What this project is

A BRKGA (Biased Random-Key Genetic Algorithm) optimizer for a **2D Nesting + Scheduling** problem in additive manufacturing. The goal is to assign 3D-printed parts to machines and pack them into batches (bins) to minimize the worst-case makespan across all machines. FFT-based collision detection is used for GPU-accelerated placement evaluation.

**Full technical reference → [PROJECT_DEEP_ANALYSIS.md](PROJECT_DEEP_ANALYSIS.md)**

This document covers: problem domain, chromosome encoding, decoder pipeline, collision detection math, evaluation modes, data structures, performance bottlenecks, and optimization history. Read it before proposing any non-trivial changes.

---

## How to run experiments (remote GPU)

All experiments run on a **Paperspace Gradient notebook** (NVIDIA RTX A4000, 16 GB VRAM). You have full autonomous access to it via `remote.py` — a local CLI tool that can upload files, execute commands on the remote GPU, and download results.

**Full remote execution reference → [REMOTE_EXECUTION.md](REMOTE_EXECUTION.md)**

This document covers: all `remote.py` commands, the full autonomous loop, run command format, output files, and gotchas (numba install, token expiry, timeouts).

### Quick reference

```bash
# Push local changes to Paperspace
python remote.py sync . BRKGA_FFT2 --ext .py

# Run an experiment (streams output in real time)
python remote.py run "python BRKGA_alg3.py 50 2 0 torch_gpu wave_batch 1 1 3" \
    --cwd /notebooks/BRKGA_FFT2 --timeout 900

# Download result file
python remote.py download BRKGA_FFT2/OriginalInitialSol_P50M2-0_prob_10.xlsx results/

# Check GPU / environment
python remote.py run "nvidia-smi" --cwd /notebooks/BRKGA_FFT2
```

`remote.py` is **gitignored** (contains the Paperspace token). It lives only on the local machine. If it is missing, recreate it from the template in [REMOTE_EXECUTION.md](REMOTE_EXECUTION.md) and fill in the URL and token from Paperspace.

---

## Key files

| File | Role |
|------|------|
| `BRKGA_alg3.py` | Main algorithm — population init, evolution loop, fitness tracking |
| `placement.py` | Chromosome decoder — assigns parts to machines, packs bins |
| `binClassNew.py` | `BuildingPlate` class — bin packing with FFT collision detection |
| `collision_backend.py` | GPU FFT backend (torch) for collision checking |
| `wave_batch_evaluator.py` | Batches FFT ops across multiple solutions simultaneously (latest approach) |
| `cuda_batch_update.py` | Custom CUDA kernel for batched GPU grid updates in Phase 5 (JIT-compiled via `load_inline`) |
| `data_structures.py` | Dataclasses for parts, machines, problem data |
| `numba_utils.py` | JIT-compiled vacancy checking (called ~100K×/run) |
| `profile_phases.py` | Per-phase wall-clock profiling of wave_batch_evaluator |
| `data/Instances/` | Problem instance files (`P50M2-0.txt`, etc.) |
| `data/partsMatrices/` | Binary part shape matrices (`.npy`) |
| `remote.py` | Autonomous remote runner — **gitignored** |

---

## Workflow

1. Read `PROJECT_DEEP_ANALYSIS.md` to understand the current state and any open questions.
2. Make code changes locally.
3. Push with `python remote.py sync . BRKGA_FFT2 --ext .py`.
4. Run on the GPU and read the streamed output.
5. Download result files if needed.
6. Reason about results, iterate.

The user's intent is for you to drive this loop autonomously — proposing changes, running experiments, reading results, and iterating — with minimal manual intervention on their part.

---

## Performance optimization summary

**Also see → [PHASE5_OPTIMIZATION.md](PHASE5_OPTIMIZATION.md)** for detailed analysis of Phase 5 optimization ideas (status of each).

### Current performance (P50M2-0, 500 individuals, wave_batch, torch_gpu)

| Metric | Value |
|--------|-------|
| Mean gen time | **~3.80s** |
| Original baseline (pre-optimization) | ~40s/gen |
| Total speedup | **~10.5x** |

### Key optimizations applied (most recent session)

| Change | Impact | File(s) |
|--------|--------|---------|
| `rfft2`/`irfft2` instead of `fft2`/`ifft2` | 5.74s → 4.53s | `wave_batch_evaluator.py`, `collision_backend.py` |
| NumPy-side composite scoring (moved density calc out of Python loop) | 4.53s → 4.23s | `wave_batch_evaluator.py` |
| Custom CUDA kernel for batched GPU grid updates | 4.23s → 3.80s | `cuda_batch_update.py`, `wave_batch_evaluator.py` |

### Phase breakdown (5 gens, 360 waves, profiled with `profile_phases.py`)

| Phase | Time(s) | % | Description |
|-------|---------|---|-------------|
| Phase 1 | 0.074 | 0.4% | Gather context info |
| Phase 2 | 0.900 | 4.8% | Batch grid FFTs |
| Phase 3 | 1.418 | 7.6% | Vacancy check + collect tests |
| **Phase 4** | **12.398** | **66.1%** | **Batch IFFT (dominant)** |
| Phase 5 | 2.629 | 14.0% | Find best placements + grid updates |
| Phase 6 | 1.347 | 7.2% | Open new bins |

### Important notes for the CUDA kernel

- **First-run compilation**: `cuda_batch_update.py` uses `torch.utils.cpp_extension.load_inline` to JIT-compile a CUDA kernel. First run takes 2-3 minutes. The compiled `.so` is cached in `~/.cache/torch_extensions/` for subsequent runs.
- **Cache invalidation**: If you change the CUDA kernel source, delete the cache: `rm -rf /root/.cache/torch_extensions/py311_cu121/_cuda_batch_update_ext/`
- **Fallback**: If CUDA compilation fails, the evaluator falls back to sequential GPU slice updates (Option C) automatically.
