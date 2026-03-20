# Remote Execution on Paperspace Gradient

This document explains how to control the Paperspace GPU environment autonomously — uploading code, triggering runs, and retrieving results — using `remote.py`.

---

## Overview

The Paperspace Gradient notebook exposes a **Jupyter server** with a REST API and WebSocket kernel protocol. `remote.py` wraps both into a simple CLI that can be called from Bash, giving an AI assistant full autonomous control without any manual steps from the user.

**What this enables:**
- Push local code changes directly to Paperspace (no git required for iteration)
- Execute shell commands or Python code on the remote GPU and stream output back
- Download result files (Excel outputs, logs, etc.) for local inspection
- Full autonomous research loop: edit → sync → run → read results → reason → iterate

---

## Infrastructure

| Item | Value |
|------|-------|
| Paperspace URL | `https://n066qklbsx.clg07azjl.paperspacegradient.com` |
| Token location | Hardcoded in `remote.py` (env var `PAPERSPACE_TOKEN` overrides) |
| Remote project path | `/notebooks/BRKGA_FFT2/` |
| Remote working dir | `/notebooks/BRKGA_FFT2/` (use `--cwd` flag) |
| GPU | NVIDIA RTX A4000, 16 GB VRAM |
| CUDA | 12.4 / Driver 550.144.03 |
| PyTorch | 2.1.1+cu121 |
| Python | System Python 3 (at `/usr/bin/python`) |

**Note:** `remote.py` is in `.gitignore` because it contains the Paperspace token. It lives only on the local machine and must be recreated manually if lost (just copy the template and fill in the URL and token from Paperspace).

---

## Setup Requirements (local machine)

Both packages are already available:
- `requests` — REST API for file operations
- `aiohttp` — WebSocket for kernel execution

No install needed. If missing: `pip install requests aiohttp`.

---

## remote.py Command Reference

### List files on Paperspace
```bash
python remote.py ls [remote_path]
```
```bash
python remote.py ls                      # root (shows BRKGA_FFT2/)
python remote.py ls BRKGA_FFT2           # project directory
python remote.py ls BRKGA_FFT2/data      # data subdirectory
```

### Upload a single file
```bash
python remote.py upload <local_path> [remote_path]
```
```bash
python remote.py upload BRKGA_alg3.py BRKGA_FFT2/BRKGA_alg3.py
```
If `remote_path` is omitted, uses the filename in the root.

### Sync all Python files (bulk push after local edits)
```bash
python remote.py sync [local_dir] [remote_dir] [--ext .py .txt ...]
```
```bash
# Push all .py files from local root to BRKGA_FFT2/ on Paperspace
python remote.py sync . BRKGA_FFT2 --ext .py

# Push everything (all file types)
python remote.py sync . BRKGA_FFT2
```
Skips: `.git/`, `__pycache__/`, `.venv/`, `.ipynb_checkpoints/`, `results/`.

### Run a shell command and stream output
```bash
python remote.py run "<shell command>" [--cwd /path] [--timeout 600]
```
```bash
# Run the BRKGA algorithm
python remote.py run "python BRKGA_alg3.py 50 2 0 torch_gpu wave_batch 1 1 3" \
    --cwd /notebooks/BRKGA_FFT2 --timeout 900

# Check GPU status
python remote.py run "nvidia-smi" --cwd /notebooks/BRKGA_FFT2

# Install a missing package
python remote.py run "pip install numba -q" --cwd /notebooks/BRKGA_FFT2

# Run profiling
python remote.py run "python profile_quick.py" --cwd /notebooks/BRKGA_FFT2
```

### Execute arbitrary Python code
```bash
python remote.py exec "<python code>" [--timeout 600]
```
```bash
python remote.py exec "import torch; print(torch.cuda.get_device_name(0))"
```

### Download a result file
```bash
python remote.py download <remote_path> [local_path]
```
```bash
# Download an Excel result file
python remote.py download BRKGA_FFT2/OriginalInitialSol_P50M2-0_prob_10.xlsx results/run1.xlsx

# Print a text file to stdout (omit local_path)
python remote.py download BRKGA_FFT2/wave_batch_results.txt
```

### Kernel management
```bash
python remote.py kernels          # list all running kernels
python remote.py kill <kernel_id> # stop a specific kernel
```
Kernels are automatically killed after each `run` or `exec` call. Use these if something gets stuck.

---

## The Autonomous Research Loop

A complete cycle looks like this:

```
1. Make code changes locally (edit files in VSCode / via Edit tool)

2. Push changes to Paperspace:
   python remote.py sync . BRKGA_FFT2 --ext .py

3. Run experiment on GPU:
   python remote.py run "python BRKGA_alg3.py 50 2 0 torch_gpu wave_batch 1 1 3" \
       --cwd /notebooks/BRKGA_FFT2 --timeout 900

4. Read stdout output directly (it streams back in real time)

5. If result files exist, download them:
   python remote.py download BRKGA_FFT2/OriginalInitialSol_P50M2-0_prob_10.xlsx results/

6. Reason about results → go to step 1
```

This entire loop can be executed autonomously via the `Bash` tool without any user intervention.

---

## BRKGA Run Command Format

```bash
python BRKGA_alg3.py <nbParts> <nbMachines> <instNumber> [backend] [eval_mode] [eval_workers] [num_individuals] [num_generations]
```

| Argument | Values | Notes |
|----------|--------|-------|
| `nbParts` | 25, 50, 75, 100, 150, 200 | Number of parts |
| `nbMachines` | 2, 4 | Number of machines |
| `instNumber` | 0–4 | Problem variant |
| `backend` | `torch_gpu`, `torch_cpu`, `numpy_cpu` | Use `torch_gpu` on Paperspace |
| `eval_mode` | `wave_batch`, `thread`, `serial` | `wave_batch` is the latest GPU batch mode |
| `eval_workers` | integer | Workers for parallel evaluation |
| `num_individuals` | integer | Population size (default 500) |
| `num_generations` | integer | Number of generations to run |

**Quick test (3 generations):**
```bash
python BRKGA_alg3.py 50 2 0 torch_gpu wave_batch 1 1 3
```

**Full run:**
```bash
python BRKGA_alg3.py 50 2 0 torch_gpu wave_batch 4 500 200
```

---

## Output Files

After a run, the algorithm writes:
```
OriginalInitialSol_P{nbParts}M{nbMachines}-{instNumber}_prob_{mult}.xlsx
```
Located in `/notebooks/BRKGA_FFT2/` on Paperspace. Download with:
```bash
python remote.py download BRKGA_FFT2/OriginalInitialSol_P50M2-0_prob_10.xlsx results/
```

---

## Notes & Gotchas

- **`numba` may not be installed** on a fresh Paperspace session. Run `pip install numba -q` before the first experiment.
- **Token expiry:** If the Paperspace URL/token changes, update `PAPERSPACE_URL` and `PAPERSPACE_TOKEN` in `remote.py` (or set them as environment variables).
- **Timeout:** Long runs (200 generations, large instances) can take 10–30+ minutes. Set `--timeout` accordingly (in seconds).
- **Kernel lifecycle:** Each `run` or `exec` call starts a fresh kernel and kills it when done. State does not persist between calls.
- **Paperspace sleep:** If the Gradient notebook goes idle, the Jupyter server may stop. The user needs to wake it manually from the Paperspace dashboard.
