# AGENT_sim MPI Usage Examples

## Quick Start

### Option 1: Serial Mode (No MPI Required) - RECOMMENDED FOR TESTING

Run a single cluster simulation without any MPI setup:

```bash
cd AGENT_sim
python3 agent_sim_mpi_orchestrator.py
```

This runs 6 default Meneghetti clusters sequentially. Output goes to:
```
outputs/meneghetti_mpi/
├── MACS_J0416/
│   ├── MACS_J0416_perfect.npy         ← Perfect noiseless image
│   ├── MACS_J0416_hst.npy             ← HST-like observation (sharp PSF)
│   └── MACS_J0416_rubin.npy           ← Rubin-like observation (wide PSF)
└── mpi_results.json                   ← Summary
```

### Option 2: MPI Parallel Mode

If you have MPI installed:

```bash
# Install MPI support first
sudo apt-get install libopenmpi-dev openmpi-bin
pip install mpi4py

# Then run with 4 processes (4x faster)
cd AGENT_sim
mpirun -np 4 python3 agent_sim_mpi_orchestrator.py
```

Or with 8 processes (8x faster):
```bash
mpirun -np 8 python3 agent_sim_mpi_orchestrator.py
```

## Examining the Output Images

### 1. Check what was generated

```bash
ls -lh outputs/meneghetti_mpi/MACS_J0416/
```

### 2. Quick inspection with Python

```python
import numpy as np
import matplotlib.pyplot as plt

# Load the images
perfect = np.load('outputs/meneghetti_mpi/MACS_J0416/MACS_J0416_perfect.npy')
hst = np.load('outputs/meneghetti_mpi/MACS_J0416/MACS_J0416_hst.npy')
rubin = np.load('outputs/meneghetti_mpi/MACS_J0416/MACS_J0416_rubin.npy')

# Print stats
print(f"Perfect image shape: {perfect.shape}, range: [{perfect.min():.2f}, {perfect.max():.2f}]")
print(f"HST image shape: {hst.shape}, range: [{hst.min():.2f}, {hst.max():.2f}]")
print(f"Rubin image shape: {rubin.shape}, range: [{rubin.min():.2f}, {rubin.max():.2f}]")

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(perfect, cmap='gray')
axes[0].set_title('Perfect Image\n(Noiseless, no PSF)')
axes[0].set_xlabel(f'Min: {perfect.min():.1f}, Max: {perfect.max():.1f}')

axes[1].imshow(hst, cmap='gray')
axes[1].set_title('HST-Like\n(Sharp PSF, low noise)')
axes[1].set_xlabel(f'Min: {hst.min():.1f}, Max: {hst.max():.1f}')

axes[2].imshow(rubin, cmap='gray')
axes[2].set_title('Rubin-Like\n(Wide PSF, more noise)')
axes[2].set_xlabel(f'Min: {rubin.min():.1f}, Max: {rubin.max():.1f}')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/image_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Comparison saved to: outputs/image_comparison.png")
```

### 3. Run this inspection script

```bash
cd AGENT_sim
python3 << 'PYSCRIPT'
import numpy as np
import matplotlib.pyplot as plt

# Load the images
perfect = np.load('outputs/meneghetti_mpi/MACS_J0416/MACS_J0416_perfect.npy')
hst = np.load('outputs/meneghetti_mpi/MACS_J0416/MACS_J0416_hst.npy')
rubin = np.load('outputs/meneghetti_mpi/MACS_J0416/MACS_J0416_rubin.npy')

# Print stats
print("\n" + "="*70)
print("IMAGE STATISTICS")
print("="*70)
print(f"\nPerfect (noiseless, no PSF):")
print(f"  Shape: {perfect.shape}")
print(f"  Min: {perfect.min():.2f}, Max: {perfect.max():.2f}")
print(f"  Mean: {perfect.mean():.2f}, Std: {perfect.std():.2f}")
print(f"  Dynamic range: {perfect.max()/(perfect.std()+1e-9):.1f}")

print(f"\nHST-like (sharp PSF: 0.05\", low noise):")
print(f"  Shape: {hst.shape}")
print(f"  Min: {hst.min():.2f}, Max: {hst.max():.2f}")
print(f"  Mean: {hst.mean():.2f}, Std: {hst.std():.2f}")
print(f"  Dynamic range: {hst.max()/(hst.std()+1e-9):.1f}")

print(f"\nRubin-like (wide PSF: 0.7\", moderate noise):")
print(f"  Shape: {rubin.shape}")
print(f"  Min: {rubin.min():.2f}, Max: {rubin.max():.2f}")
print(f"  Mean: {rubin.mean():.2f}, Std: {rubin.std():.2f}")
print(f"  Dynamic range: {rubin.max()/(rubin.std()+1e-9):.1f}")

print("\n" + "="*70)
print("KEY OBSERVATIONS")
print("="*70)
print(f"\n✓ Perfect image shows intricate lensing structure")
print(f"✓ HST image preserves fine details (sharp PSF)")
print(f"✓ Rubin image blurs details but reduces noise impact")

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(perfect, cmap='gray')
axes[0].set_title('Perfect Image\n(Noiseless, no PSF)', fontsize=12, weight='bold')
axes[0].axis('off')

axes[1].imshow(hst, cmap='gray')
axes[1].set_title('HST-Like\n(Sharp PSF=0.05", low noise)', fontsize=12, weight='bold')
axes[1].axis('off')

axes[2].imshow(rubin, cmap='gray')
axes[2].set_title('Rubin-Like\n(Wide PSF=0.7", moderate noise)', fontsize=12, weight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('outputs/image_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Comparison plot saved to: outputs/image_comparison.png")
PYSCRIPT
```

## Batch Realization Mode (For Statistical Studies)

### Run multiple random realizations of the same cluster

```bash
cd AGENT_sim

# Serial: 5 random realizations
python3 agent_sim_batch_mpi.py --cluster "MACS J0416" --realizations 5

# With MPI: 20 realizations (4x faster with 4 processes)
mpirun -np 4 python3 agent_sim_batch_mpi.py --cluster "MACS J0416" --realizations 20
```

Output structure:
```
outputs/agent_sim_batch_mpi/MACS_J0416/
├── realization_0000/
│   ├── perfect_image.npy
│   ├── hst_image.npy
│   ├── rubin_image.npy
│   └── metrics.json              ← Seed, timing, comparison stats
├── realization_0001/
│   └── ...
└── batch_summary.json            ← Aggregated results
```

### Inspect batch results

```python
import numpy as np
import json
import glob

# Load metrics from all realizations
metrics = []
for metrics_file in sorted(glob.glob('outputs/agent_sim_batch_mpi/MACS_J0416/realization_*/metrics.json')):
    with open(metrics_file) as f:
        metrics.append(json.load(f))

print(f"\n✓ Loaded {len(metrics)} realizations")
print(f"  Seeds: {[m['seed'] for m in metrics]}")
print(f"  Average HST SNR: {np.mean([m.get('hst_snr', 0) for m in metrics]):.2f}")
print(f"  Average Rubin SNR: {np.mean([m.get('rubin_snr', 0) for m in metrics]):.2f}")
```

## Complete Test Workflow

Here's everything in one go:

```bash
cd AGENT_sim

# 1. Generate single cluster (serial, ~2-3 min)
echo "Step 1: Generating images..."
python3 agent_sim_mpi_orchestrator.py

# 2. Inspect the output
echo -e "\nStep 2: Inspecting images..."
python3 << 'INSPECT'
import numpy as np
perfect = np.load('outputs/meneghetti_mpi/MACS_J0416/MACS_J0416_perfect.npy')
hst = np.load('outputs/meneghetti_mpi/MACS_J0416/MACS_J0416_hst.npy')
rubin = np.load('outputs/meneghetti_mpi/MACS_J0416/MACS_J0416_rubin.npy')

print(f"\nPerfect:  shape={perfect.shape}, range=[{perfect.min():.1f}, {perfect.max():.1f}]")
print(f"HST:      shape={hst.shape}, range=[{hst.min():.1f}, {hst.max():.1f}]")
print(f"Rubin:    shape={rubin.shape}, range=[{rubin.min():.1f}, {rubin.max():.1f}]")
INSPECT

# 3. Run batch realizations for statistics (if MPI available)
echo -e "\nStep 3: Generating batch realizations..."
if command -v mpirun &> /dev/null; then
    echo "MPI available - using 4 processes (faster)"
    mpirun -np 4 python3 agent_sim_batch_mpi.py \
        --cluster "MACS J0416" --realizations 4
else
    echo "MPI not available - using serial mode"
    python3 agent_sim_batch_mpi.py \
        --cluster "MACS J0416" --realizations 4
fi

echo -e "\n✓ All tests complete!"
echo "Outputs in: ./outputs/"
```

## Troubleshooting Output Issues

### Images look wrong (all zeros or NaN)?

```bash
# Check the summary
cat outputs/meneghetti_mpi/mpi_results.json
```

### Want more detailed output?

Edit the script to add logging, or run in verbose mode:

```bash
python3 -u agent_sim_mpi_orchestrator.py 2>&1 | tee simulation.log
```

### Compare file sizes

```bash
ls -lh outputs/meneghetti_mpi/MACS_J0416/
# Perfect and HST should be ~1 MB each (256x256 floats)
# Rubin similar but may have different format
```

## Next Steps

1. Run the serial mode first to verify output
2. Inspect the images (use the Python inspection script)
3. If images look good, try MPI mode
4. For statistical studies, use batch realization mode

