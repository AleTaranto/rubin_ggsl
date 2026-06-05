# GGSL Simulation Cases

Three different approaches to GGSL event simulation, each with different tools and configurations.

## Overview

| Case | Tool Stack | Use Case |
|------|-----------|----------|
| **Case 1** | PyLensLib only | Pure lensing computations, debugging, full control over pipeline |
| **Case 2** | SLSim only | End-to-end simulation, built-in Rubin effects |
| **Case 3** | PyLensLib + ImSim | Hybrid approach: PyLensLib for lensing, ImSim for realistic images |

## Quick Start

### Setup

```bash
# First time only - setup sitcom environment
ssh rubin-devl
source /opt/sitcom/rubin_env/bin/activate.sh
setup_sitcom
```

### Running a Simulation

Each case uses the same command structure:

```bash
# Serial mode (single core)
cd cases/case1_pylenslib_only
python run.py config.yaml

# MPI mode with auto-core detection
cd cases/case1_pylenslib_only
mpirun -np auto python run.py config.yaml

# MPI mode with specific core count
cd cases/case1_pylenslib_only
mpirun -np 4 python run.py config.yaml
```

## Case Directories

### Case 1: PyLensLib-only (`case1_pylenslib_only/`)

**Best for:** Pure lensing computations, step-by-step control, debugging

**Features:**
- Full PyLensLib lensing engine
- Mock image generation
- Critical line and caustic detection
- Customizable grids and source populations

**Configuration:** `config.yaml`
**Examples:** 
- `config_examples/small_cluster.yaml` - Quick test
- `config_examples/large_cluster.yaml` - Full-featured example

**Output:**
- `combined_image_*.npz` - Lensed mock images
- `lensing_maps_*.npz` - Deflection and magnification fields
- `summary_*.json` - Event statistics

### Case 2: SLSim-only (`case2_slsim_only/`)

**Best for:** Complete end-to-end simulation, Rubin-like effects built-in

**Features:**
- SLSim for unified simulation
- Built-in Rubinization (PSF, noise, detector effects)
- Parallel processing support
- Consistent interface

**Configuration:** `config.yaml`
- Includes Rubinization parameters (PSF, read noise, sky background)

**Output:**
- `mock_image_*.npz` - Perfect mock images
- `rubinized_image_*.npz` - Images with realistic effects applied
- `lensing_maps_*.npz` - Magnification fields
- `summary_*.json` - Event statistics

### Case 3: PyLensLib + ImSim (`case3_pylenslib_imsim/`)

**Best for:** Combining best of both worlds: fine lensing control + realistic imaging

**Features:**
- PyLensLib for precise lensing computations
- ImSim for realistic Rubin detector effects
- Separate perfect and ImSim-processed images
- Critical line detection

**Configuration:** `config.yaml`
- Separate sections for PyLensLib and ImSim parameters
- Control over which tool handles each step

**Output:**
- `perfect_image_*.npz` - Mock images without realistic effects
- `imsim_image_*.npz` - Same images with ImSim effects applied
- `lensing_maps_*.npz` - Full lensing field data
- `summary_*.json` - Event statistics

## Configuration Files

All cases use YAML configuration format. Key sections:

```yaml
cluster:
  name: ClusterName
  main_halo: { M200, concentration, redshift }
  subhalos: [{ M200, concentration, redshift, position }]

simulation:
  grid_size: grid spacing (arcsec)
  x_range, y_range: coordinate ranges
  n_pixels: resolution

sources:
  redshift: source redshift
  n_sources: number of background sources
  magnitude_min/max: source brightness range
  radius_min/max: source effective radius range

images:
  size: image pixels per side
  pixel_scale: arcsec/pixel

output:
  format: npz or fits
  save_lensing_maps: boolean
  save_critical_lines: boolean
```

## MPI Usage

All cases support MPI parallelization:

```bash
# Auto-detect available cores
mpirun -np auto python run.py config.yaml

# Fixed number of processes
mpirun -np 4 python run.py config.yaml

# Serial fallback (if MPI unavailable)
python run.py config.yaml
```

Master rank (0) handles I/O and coordination. Other ranks process independent simulations.

## Output Directory

Results are saved to `--output` directory (default: `outputs/case{N}/`):

```
outputs/
├── case1/
│   ├── combined_image_*.npz
│   ├── lensing_maps_*.npz
│   └── summary_*.json
├── case2/
│   ├── mock_image_*.npz
│   ├── rubinized_image_*.npz
│   └── ...
└── case3/
    ├── perfect_image_*.npz
    ├── imsim_image_*.npz
    └── ...
```

## Loading Results

```python
import numpy as np

# Load image data
data = np.load('outputs/case1/combined_image_ExampleCluster.npz')
image = data['image']
flux = np.sum(image)

# Load lensing maps
maps = np.load('outputs/case1/lensing_maps_ExampleCluster.npz')
magnification = maps['magnification']
deflection_x = maps['deflection_x']
```

## Troubleshooting

### Empty Images

If images have no flux (all zeros):
1. Check source magnitudes - should be 0-25 (astronomical convention) or adjusted appropriately
2. Verify cluster mass - M200 should be >1e14
3. Increase `n_sources` in config
4. Check grid range matches source positions

### MPI Issues

If MPI fails to start:
1. Verify `module load` or `source setup_sitcom` was run
2. Check `mpirun` availability: `which mpirun`
3. Fall back to serial: `python run.py config.yaml` (no mpirun)

### Performance

For faster runs:
- Reduce `n_pixels` in simulation config
- Reduce `n_sources`
- Use smaller `image_size`
- Increase `pixel_scale` (coarser resolution)

For more accurate results:
- Increase all of the above!
- Use case 1 (PyLensLib) for most control

## Examples

### Quick Test (2 minutes)

```bash
cd cases/case1_pylenslib_only
python run.py config_examples/small_cluster.yaml
```

### Full Simulation (10-30 minutes)

```bash
cd cases/case1_pylenslib_only
mpirun -np 4 python run.py config_examples/large_cluster.yaml
```

### Compare All Cases

```bash
# Case 1
cd cases/case1_pylenslib_only
python run.py config.yaml --output outputs/comparison_case1

# Case 2
cd ../case2_slsim_only
python run.py config.yaml --output outputs/comparison_case2

# Case 3
cd ../case3_pylenslib_imsim
python run.py config.yaml --output outputs/comparison_case3

# Compare outputs in outputs/comparison_case{1,2,3}/
```

## Adding Custom Configurations

Create a new YAML file based on `config.yaml`:

```yaml
cluster:
  name: MyCluster
  main_halo:
    M200: 1.5e15  # Customize mass
    concentration: 4.5  # Customize concentration
    redshift: 0.35  # Different redshift

# ... rest of configuration
```

Then run:

```bash
python run.py path/to/my_config.yaml
```

## Performance Notes

- **Single core (serial):** ~5-60 minutes depending on grid size and n_sources
- **4 cores (MPI):** ~2-20 minutes (near-linear scaling)
- **8+ cores:** Scaling plateaus around 50-75% efficiency

Bottleneck is typically lensing map computation for high-resolution grids.
