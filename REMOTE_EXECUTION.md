# Remote Execution on Paperspace Gradient

How to control the Paperspace GPU environment autonomously using `remote.py`.

---

## Infrastructure

| Item | Value |
|------|-------|
| Paperspace URL | Hardcoded in `remote.py` (changes when notebook restarts) |
| Token location | Hardcoded in `remote.py` (env var `PAPERSPACE_TOKEN` overrides) |
| Remote project path | `/notebooks/BRKGA_FFT2/` |
| GPU | NVIDIA RTX A4000, 16 GB VRAM |
| CUDA | 12.4 / Driver 550.144.03 |
| PyTorch | 2.1.1+cu121 |

`remote.py` is **gitignored** (contains the Paperspace token). It lives only on the local machine.

---

## Command Reference

```bash
# List remote files
python remote.py ls [remote_path]

# Upload a single file
python remote.py upload <local_path> [remote_path]

# Sync all Python files (bulk push)
python remote.py sync . BRKGA_FFT2 --ext .py

# Run a shell command (streams output)
python remote.py run "<command>" --cwd /notebooks/BRKGA_FFT2 [--timeout 900]

# Execute arbitrary Python code
python remote.py exec "<python code>" [--timeout 600]

# Download a result file
python remote.py download BRKGA_FFT2/<file> [local_path]

# Kernel management
python remote.py kernels          # list running kernels
python remote.py kill <kernel_id> # stop a kernel
```

Skipped dirs during sync: `.git/`, `__pycache__/`, `.venv/`, `.ipynb_checkpoints/`, `results/`.

---

## BRKGA Run Command

```bash
python BRKGA_alg3.py <nbParts> <nbMachines> <instNumber> [backend] [eval_mode] [workers] [chunksize] [generations]
```

| Argument | Values | Default |
|----------|--------|---------|
| `nbParts` | 25, 50, 75, 100, 150, 200 | required |
| `nbMachines` | 2, 4 | required |
| `instNumber` | 0–4 | required |
| `backend` | `torch_gpu`, `torch_cpu`, `numpy_cpu` | `torch_gpu` |
| `eval_mode` | `native_full`, `wave_batch`, `thread`, `serial` | `auto` |
| `workers` | integer | 4 |
| `chunksize` | integer | 1 |
| `generations` | integer | 30 |

**Recommended: use `native_full` eval_mode** (full C++ decoder, ~4× faster than `wave_batch`).

Alternative: set `eval_mode=wave_batch` + env var `ABRKGA_FULL_NATIVE_DECODER=1` to force native decoder while keeping `wave_batch` in the CLI args.

### Examples

```bash
# Quick test (3 generations, native decoder)
python remote.py run "python BRKGA_alg3.py 50 2 0 torch_gpu native_full 1 1 3" \
    --cwd /notebooks/BRKGA_FFT2 --timeout 900

# Full run (30 generations)
python remote.py run "python BRKGA_alg3.py 50 2 0 torch_gpu native_full 1 1 30" \
    --cwd /notebooks/BRKGA_FFT2 --timeout 1800

# Larger instance
python remote.py run "python BRKGA_alg3.py 75 2 0 torch_gpu native_full 1 1 10" \
    --cwd /notebooks/BRKGA_FFT2 --timeout 1800

# Check GPU
python remote.py run "nvidia-smi" --cwd /notebooks/BRKGA_FFT2
```

---

## The Autonomous Research Loop

```
1. Make code changes locally
2. python remote.py sync . BRKGA_FFT2 --ext .py
3. python remote.py run "python BRKGA_alg3.py 50 2 0 torch_gpu native_full 1 1 3" \
       --cwd /notebooks/BRKGA_FFT2 --timeout 900
4. Read streamed stdout output
5. python remote.py download BRKGA_FFT2/OriginalInitialSol_P50M2-0_prob_10.xlsx results/
6. Reason about results → go to step 1
```

---

## Output Files

```
OriginalInitialSol_P{nbParts}M{nbMachines}-{instNumber}_prob_{mult}.xlsx
```
Located in `/notebooks/BRKGA_FFT2/` on Paperspace.

---

## Notes & Gotchas

- **First-run C++ compilation:** `full_native_decoder.py` JIT-compiles a C++/CUDA extension on first run (~2–3 min). Cached in `~/.cache/torch_extensions/`. If you change CUDA kernel source, clear cache: `rm -rf /root/.cache/torch_extensions/py*/`
- **Token/URL expiry:** The Paperspace URL changes when the notebook restarts. Update `PAPERSPACE_URL` and `PAPERSPACE_TOKEN` in `remote.py`.
- **Timeout:** Large instances (P200, 200 generations) can take 30+ min. Set `--timeout` accordingly.
- **Kernel lifecycle:** Each `run`/`exec` starts a fresh kernel and kills it when done. No state persists.
- **Paperspace sleep:** If idle, the Jupyter server may stop. User must wake it from the Paperspace dashboard.
