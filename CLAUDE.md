# Claude Context — BRKGA_FFT2

## What this project is

A BRKGA (Biased Random-Key Genetic Algorithm) optimizer for a **2D Nesting + Scheduling** problem in additive manufacturing. Parts are assigned to machines and packed into bins (building plates) to minimize worst-case makespan. FFT-based collision detection runs on GPU for placement evaluation.

**Technical reference:** [PROJECT_DEEP_ANALYSIS.md](PROJECT_DEEP_ANALYSIS.md) covers the problem domain, chromosome encoding, BRKGA loop, and decoder pipeline. Note: it was written during the wave_batch era — the decoder architecture section is outdated, but the problem/algorithm sections (§1–§2) remain accurate.

---

## Current architecture (as of April 2026)

The decoder is a **single C++/CUDA extension** compiled at runtime via `torch.utils.cpp_extension.load_inline`. All decoder phases (1–6) run in compiled C++ with custom CUDA kernels. Python only handles the BRKGA outer loop (crossover, mutation, selection).

| Component | File | Role |
|-----------|------|------|
| BRKGA outer loop | `BRKGA_alg3.py` | Population init, evolution, fitness tracking |
| **Full native decoder** | **`full_native_decoder.py`** | **C++/CUDA extension — entire decoder (Phases 1–6)** |
| Data structures | `data_structures.py` | Dataclasses for parts, machines, problem data |
| Collision backend | `collision_backend.py` | GPU FFT backend — used for part FFT pre-computation at init |
| Instance data | `data/Instances/`, `data/partsMatrices/` | Problem files (`.txt`) and part shape matrices (`.npy`) |

**Legacy files (not used in native_full mode but still importable):** `wave_batch_evaluator.py`, `placement.py`, `binClassNew.py`, `cuda_batch_update.py`, `numba_utils.py`, `profile_phases.py`. These were the Python-era decoder. The `wave_batch` eval_mode still works but is 2–4× slower than `native_full`.

### Decoder phases (inside full_native_decoder.py)

| Phase | What it does | Bottleneck? |
|-------|-------------|-------------|
| 1 | Gather context info (which parts, which bins) | No (~0.4%) |
| 2 | Batch rfft2 of all active grids | No (~5%) |
| 3 | GPU vacancy check + collect FFT test pairs (p1/p2 two-pass) | No (~8%) |
| **4** | **Batch irfft2 — collision detection** | **Yes (~52%)** |
| 5 | CUDA selector kernel finds best positions + CPU grid updates | Moderate (~20%) |
| 6 | Open new bins for parts that didn't fit (bin pool reuse) | No (~7%) |

### Key implementation details

- **First-valid-bin (p1/p2):** For each part, only the first bin passing vacancy is tested in p1. Only parts failing FFT collision go to p2 (remaining bins). >99% resolve in p1.
- **Custom CUDA kernels:** fused gather-multiply, batch grid update, selector (best position), vacancy check, vacancy recompute.
- **AVX2 SIMD:** vacancy row updates, grid writes, vacancy fitting — all with runtime CPU feature detection.
- **cuFFT plan cache:** Set to 4096 entries. Plans cached after first generation.
- **irfft2 "forward" norm:** Avoids normalization multiply; part FFTs pre-divided at init.
- **Bin pool:** Keyed by (H,W), recycles BinStateNative structs to avoid re-allocation.
- **First-run compilation:** Takes 2–3 min. Cached in `~/.cache/torch_extensions/`. If you change CUDA kernel source in `full_native_decoder.py`, delete cache: `rm -rf /root/.cache/torch_extensions/py*/`

---

## Performance

| Instance | Gen time | Baseline (Python serial) | Speedup |
|----------|----------|-------------------------|---------|
| P50M2-0 | **0.974s** | ~40s | **41×** |
| P75M2-0 | **1.742s** | — | — |

GPU utilization is ~43% (nsys-confirmed). The remaining 57% is structural CPU-GPU sync overhead (vacancy readbacks, selector readbacks, inter-phase CPU work). Single CUDA stream — no overlap possible without PyTorch allocator issues.

**Optimization history:** [OPTIMIZATION_CATALOG.md](OPTIMIZATION_CATALOG.md) has every optimization idea ever considered, categorized as tested-positive, tested-negative, untested-worth-testing, untested-not-worth-testing.

---

## How to run experiments (remote GPU)

All experiments run on **Paperspace Gradient** (NVIDIA RTX A4000, 16 GB VRAM). Use `remote.py` (gitignored, contains token) for autonomous access.

```bash
# Push code to Paperspace
python remote.py sync . BRKGA_FFT2 --ext .py

# Run with native decoder (default when ABRKGA_FULL_NATIVE_DECODER=1 or eval_mode=native_full)
python remote.py run "ABRKGA_FULL_NATIVE_DECODER=1 python BRKGA_alg3.py 50 2 0 torch_gpu wave_batch 1 1 3" \
    --cwd /notebooks/BRKGA_FFT2 --timeout 900

# Or use native_full eval_mode directly
python remote.py run "python BRKGA_alg3.py 50 2 0 torch_gpu native_full 1 1 3" \
    --cwd /notebooks/BRKGA_FFT2 --timeout 900

# Download results
python remote.py download BRKGA_FFT2/OriginalInitialSol_P50M2-0_prob_10.xlsx results/

# Check GPU
python remote.py run "nvidia-smi" --cwd /notebooks/BRKGA_FFT2
```

**CLI args:** `python BRKGA_alg3.py <nbParts> <nbMachines> <instNumber> [backend] [eval_mode] [workers] [chunksize] [generations]`

Full remote reference: [REMOTE_EXECUTION.md](REMOTE_EXECUTION.md). If `remote.py` is missing, recreate from template there.

---

## Workflow

1. Make code changes locally.
2. `python remote.py sync . BRKGA_FFT2 --ext .py`
3. Run on GPU, read streamed output.
4. Download result files if needed.
5. Reason about results, iterate.

Drive this loop autonomously — propose changes, run experiments, read results, iterate — with minimal user intervention.
