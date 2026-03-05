# Running FFT_Placement on Paperspace Gradient

This guide helps you set up and run the FFT_Placement BRKGA optimization algorithm on Paperspace Gradient notebooks.

---

## Quick Start

### 1. Upload Files to Paperspace

Upload all the files from your `FFT_Placement` folder to your Paperspace Gradient notebook environment. Make sure you include:

```
FFT_Placement/
├── BRKGA_alg3.py
├── placement.py
├── binClassNew.py
├── binClassInitialSol.py
├── collision_backend.py
├── data_structures.py
├── setup_paperspace.sh
└── data/
    ├── Instances/
    │   └── *.txt files
    ├── PartsMachines/
    │   ├── part-machine-information.xlsx
    │   ├── polygon_areas.xlsx
    │   └── parts_rotations.xlsx
    └── partsMatrices/
        └── matrix_*.npy files
```

### 2. Run Setup Script

Open a terminal in your Paperspace notebook and run:

```bash
cd /notebooks  # or wherever you uploaded files
chmod +x setup_paperspace.sh
bash setup_paperspace.sh
```

This will install:
- PyTorch with CUDA support
- NumPy
- Pandas
- openpyxl

### 3. Run the Algorithm

**Basic command format:**
```bash
python BRKGA_alg3.py <nbParts> <nbMachines> <instNumber> [backend] [eval_mode] [eval_workers]
```

**Required arguments:**
- `nbParts`: Number of parts (25, 50, 75, 100, 150, or 200)
- `nbMachines`: Number of machines (2 or 4)
- `instNumber`: Instance number (0, 1, 2, 3, or 4)

**Optional arguments:**
- `backend`: Collision backend (default: `torch_gpu`)
  - `torch_gpu` - GPU-accelerated (recommended for Paperspace)
  - `torch_cpu` - CPU-only PyTorch
  - `numpy_cpu` - NumPy fallback
- `eval_mode`: Evaluation parallelization (default: `thread`)
  - `thread` - Multi-threaded (recommended)
  - `serial` - Single-threaded
  - `process` - Multi-process (not compatible with GPU)
- `eval_workers`: Number of parallel workers (default: 2)

---

## Example Commands

### Small instance (fast test):
```bash
python BRKGA_alg3.py 25 2 0 torch_gpu thread 4
```
Runs instance P25M2-0 (25 parts, 2 machines) with GPU and 4 worker threads.

### Medium instance:
```bash
python BRKGA_alg3.py 100 4 0 torch_gpu thread 8
```
Runs instance P100M4-0 (100 parts, 4 machines) with GPU and 8 worker threads.

### Large instance:
```bash
python BRKGA_alg3.py 200 4 2 torch_gpu thread 12
```
Runs instance P200M4-2 (200 parts, 4 machines) with GPU and 12 worker threads.

### CPU fallback (if GPU issues):
```bash
python BRKGA_alg3.py 50 2 1 torch_cpu thread 4
```

---

## Running in a Notebook Cell

If you prefer to run from a Jupyter/IPython notebook cell instead of terminal:

```python
!cd /notebooks && python BRKGA_alg3.py 100 4 0 torch_gpu thread 8
```

---

## Performance Tips

1. **Use GPU backend** (`torch_gpu`) - Paperspace provides GPU by default
2. **Increase workers** for multi-machine instances - set to number of machines or more
3. **Monitor GPU usage**: 
   ```bash
   nvidia-smi
   ```
4. **Check CUDA availability**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0)}")
   ```

---

## Output

The algorithm will output:
- Excel file: `OriginalInitialSol_P{nbParts}M{nbMachines}-{instNumber}_prob_{mult}.xlsx`
- Console output showing generation progress and fitness values

---

## Troubleshooting

**CUDA out of memory:**
- Reduce `eval_workers` to 2 or switch to `torch_cpu`

**Import errors:**
- Re-run `setup_paperspace.sh`
- Verify all files uploaded correctly

**File not found errors:**
- Check that `data/` folder structure is preserved
- Ensure you're in the correct directory (`cd /notebooks/FFT_Placement`)

---

## Available Test Instances

| Instance | Parts | Machines | Files |
|----------|-------|----------|-------|
| P25M2 | 25 | 2 | P25M2-{0,1,2,3,4}.txt |
| P50M2 | 50 | 2 | P50M2-{0,1,2,3,4}.txt |
| P75M2 | 75 | 2 | P75M2-{0,1,2,3,4}.txt |
| P100M4 | 100 | 4 | P100M4-{0,1,2,3,4}.txt |
| P150M4 | 150 | 4 | P150M4-{0,1,2,3,4}.txt |
| P200M4 | 200 | 4 | P200M4-{0,1,2,3,4}.txt |

Each instance has 5 different problem variants (0-4).
