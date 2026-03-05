#!/bin/bash
# Paperspace Gradient Setup Script for FFT_Placement
# This script installs all required dependencies for running the BRKGA optimization

echo "=================================================="
echo "FFT_Placement Environment Setup for Paperspace"
echo "=================================================="
echo ""

# Update pip to latest version
echo "[1/5] Updating pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (Paperspace typically has CUDA available)
echo ""
echo "[2/5] Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install NumPy and Pandas
echo ""
echo "[3/5] Installing NumPy and Pandas..."
pip install numpy pandas

# Install openpyxl for Excel file handling
echo ""
echo "[4/5] Installing openpyxl for Excel support..."
pip install openpyxl

# Verify installations
echo ""
echo "[5/5] Verifying installations..."
python3 << EOF
import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} installed")
except ImportError:
    print("✗ NumPy installation failed")

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__} installed")
except ImportError:
    print("✗ Pandas installation failed")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} installed")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("✗ PyTorch installation failed")

try:
    import openpyxl
    print(f"✓ openpyxl {openpyxl.__version__} installed")
except ImportError:
    print("✗ openpyxl installation failed")
EOF

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "To run the BRKGA algorithm, use:"
echo "  python BRKGA_alg3.py <nbParts> <nbMachines> <instNumber> [backend] [eval_mode] [eval_workers]"
echo ""
echo "Example:"
echo "  python BRKGA_alg3.py 100 4 0 torch_gpu thread 4"
echo ""
echo "Available instances:"
echo "  P25M2-{0,1,2,3,4}   (25 parts, 2 machines)"
echo "  P50M2-{0,1,2,3,4}   (50 parts, 2 machines)"
echo "  P75M2-{0,1,2,3,4}   (75 parts, 2 machines)"
echo "  P100M4-{0,1,2,3,4}  (100 parts, 4 machines)"
echo "  P150M4-{0,1,2,3,4}  (150 parts, 4 machines)"
echo "  P200M4-{0,1,2,3,4}  (200 parts, 4 machines)"
echo ""
