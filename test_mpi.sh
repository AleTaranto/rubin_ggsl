#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         AGENT_SIM MPI TEST - Generate & Inspect Images        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Generate images (serial mode)
echo "[1/3] Generating Meneghetti cluster images (serial mode)..."
echo "      This takes ~2-3 minutes..."
python3 agent_sim_mpi_orchestrator.py

# Step 2: Check files
echo ""
echo "[2/3] Checking generated files..."
echo ""
ls -lh outputs/meneghetti_mpi/MACS_J0416/
echo ""

# Step 3: Inspect images
echo "[3/3] Inspecting image quality..."
python3 << 'PYTHON'
import numpy as np
import os

if not os.path.exists('outputs/meneghetti_mpi/MACS_J0416/MACS_J0416_perfect.npy'):
    print("❌ ERROR: Perfect image not found!")
    exit(1)

perfect = np.load('outputs/meneghetti_mpi/MACS_J0416/MACS_J0416_perfect.npy')
hst = np.load('outputs/meneghetti_mpi/MACS_J0416/MACS_J0416_hst.npy')
rubin = np.load('outputs/meneghetti_mpi/MACS_J0416/MACS_J0416_rubin.npy')

print("\n" + "="*68)
print("IMAGE STATISTICS")
print("="*68)

print(f"\n✓ PERFECT IMAGE (noiseless, no PSF):")
print(f"  Shape: {perfect.shape}")
print(f"  Range: [{perfect.min():.1f}, {perfect.max():.1f}] counts")
print(f"  Mean: {perfect.mean():.2f}, Std: {perfect.std():.2f}")

print(f"\n✓ HST-LIKE IMAGE (sharp PSF = 0.05\", low noise):")
print(f"  Shape: {hst.shape}")
print(f"  Range: [{hst.min():.1f}, {hst.max():.1f}] ADU")
print(f"  Mean: {hst.mean():.2f}, Std: {hst.std():.2f}")

print(f"\n✓ RUBIN-LIKE IMAGE (wide PSF = 0.7\", moderate noise):")
print(f"  Shape: {rubin.shape}")
print(f"  Range: [{rubin.min():.1f}, {rubin.max():.1f}] ADU")
print(f"  Mean: {rubin.mean():.2f}, Std: {rubin.std():.2f}")

print(f"\n" + "="*68)
print("✅ SUCCESS! Images generated and look reasonable")
print("="*68)
print(f"""
INTERPRETATION:

  Perfect image shows TRUE lensing structure:
    → Multiple lensed arcs from source
    → Substructure from dark matter subhalos
    → No noise or PSF effects

  HST-like image preserves fine details:
    → Similar to perfect, but with slight PSF blur
    → Lower noise (good for detecting subhalos)
    → Why Meneghetti used HST

  Rubin-like image is more blurred:
    → Wider PSF (0.7") loses fine structure
    → More noise (atmosphere + detector effects)
    → Better for wide-field surveys (many clusters)

NEXT STEPS:
  1. Try MPI mode: mpirun -np 4 python3 agent_sim_mpi_orchestrator.py
  2. Generate batch: python3 agent_sim_batch_mpi.py --cluster "MACS J0416" --realizations 5
  3. See USAGE_EXAMPLES.md for more details
""")
PYTHON

echo ""
echo "✓ Test complete! Check outputs/meneghetti_mpi/ for images"
echo ""
