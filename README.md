# rubin_ggsl (AGENT_sim)

**rubin_ggsl** (also known as **AGENT_sim**) is a Python framework designed to simulate expected **Galaxy-Galaxy Strong Lensing (GGSL)** and **Cluster Strong Lensing** events under various dark matter profile conditions and realistic observational effects matching the **Vera C. Rubin Observatory (LSST)**.

---

## 🇮🇹 Documentazione in Italiano

### Che cos'è `rubin_ggsl`?
`rubin_ggsl` è una suite di simulazione scientifica per il lensing gravitazionale forte. Permette di:
1. **Modellare ammassi di galassie e profili di materia oscura**: Supporta profili NFW, CoreNFW, SIS, lenti ellittiche (SIE, PIEMD) e popolazioni di sub-oni / galassie satellite.
2. **Calcolare mappe di lensing**: Genera campi di deflessione, mappe di magnificazione $\mu$, convergenza $\kappa$, shear $\gamma$ e linee critiche/caustiche.
3. **Simulare immagini mock perfette e realistiche**: Traccia i raggi della luce da sorgenti a redshift cosmologici differenti per produrre archi ed anelli di Einstein.
4. **Applicare la "Rubinizzazione" (Rubinization)**: Aggiunge effetti osservativi realistici specifici per la survey Vera C. Rubin LSST (Point Spread Function PSF, rumore di Poisson/fotoni, rumore di lettura dei sensori CCD, fondo del cielo, raggi cosmici e saturazione dei pixel).
5. **Esecuzione Parallela tramite MPI**: Supporta la generazione massiva di cataloghi di simulazioni su cluster di calcolo parallelo.

---

## 🇬🇧 Overview & Key Features

- **Dual Simulation Architecture**:
  - **`PyLensLib + ImSim` Branch**: Custom fine-grained physics ray-tracing engine integrated with ImSim-like detector noise and PSF models.
  - **`SLSim` Branch**: Unified interface for external SLSim lensing engine and SLSim Rubinization pipelines.
- **Physical Models Supported**:
  - **Main Halos**: NFW (Navarro-Frenk-White), CoreNFW, SIS (Singular Isothermal Sphere), SIE, PIEMD.
  - **Subhalos & Galaxies**: Subhalo mass functions (SHMF), spatial distributions, and galaxy light profiles (Sérsic).
- **Rubinization Engine**:
  - **PSF Convolution**: Gaussian/custom atmospheric and instrumental PSF.
  - **Sky Background & Photon Noise**: Poisson fluctuations based on sky brightness magnitude.
  - **Readout Noise & Gain**: Sensor electronics noise and ADU conversion.
  - **Cosmic Rays & Saturation**: High-energy particle hits and full-well detector clipping.
- **Parallel Computing**: MPI-orchestrated batch execution across cluster realizations (`mpi4py`).

---

## 📁 Repository Structure

```
rubin_ggsl/
├── agent_sim/                        # Core Python Package
│   ├── core/                         # Physics & Lensing Engines
│   │   ├── models/                   # DM profiles (NFW, CoreNFW, SIS, Cluster)
│   │   ├── lensing/                  # Deflection fields, magnification, critical lines
│   │   ├── observables/              # GGSL event definitions & counters
│   │   └── utils/                    # Physical constants & I/O
│   ├── pylenslib_imsim_branch/       # PyLensLib physics + ImSim Rubinization
│   │   ├── imaging/                  # Source placement, ray tracing, Rubinization
│   │   ├── simulator.py              # PyLenslibSimulator pipeline
│   │   └── examples/                 # Branch usage examples
│   ├── slsim_branch/                 # SLSim physics + SLSim Rubinization
│   │   ├── imaging/                  # SLSim interfaces
│   │   ├── simulator.py              # SlsimSimulator pipeline
│   │   └── examples/                 # Branch usage examples
│   ├── meneghetti_configurations.py  # Reference cluster profiles (e.g. MACS J0416)
│   └── run_meneghetti_simulation.py  # Single-cluster test script
├── pylenslib_sim/                    # Standalone Orchestrator & Configuration Runner
│   ├── main.py                       # CLI catalog generator driver
│   ├── config_examples/             # YAML configurations (e.g. Rubin realistic, MACS1206)
│   └── plot_perturbed_ring.py        # Diagnostic visualization script
├── cases/                            # Workflow case studies (Standalone, SLSim, ImSim)
├── tests/                            # Automated test suite (pytest)
├── agent_sim_mpi_orchestrator.py     # MPI parallel simulation orchestrator
├── mpi_utils.py                      # MPI helper utilities
├── pyproject.toml / setup.py         # Package installation configuration
├── RUBINIZATION_GUIDE.md             # Detailed comparison guide (PyLensLib vs SLSim)
├── RUBINIZATION_README.md            # In-depth Rubinization parameter reference
└── USAGE_EXAMPLES.md                 # Detailed MPI & batch realization usage guide
```

---

## 🛠️ Installation

### 1. Requirements
- Python $\ge 3.9$
- Core dependencies: `numpy`, `scipy`, `astropy`, `matplotlib`, `pyyaml`
- Optional parallel execution: `mpi4py` (requires OpenMPI/MPICH)

### 2. Install Package in Editable Mode
```bash
git clone https://github.com/username/rubin_ggsl.git
cd rubin_ggsl
pip install -e .
```

---

## 🚀 Quick Start & Usage

### 1. Basic Usage in Python

#### PyLensLib + ImSim Branch
```python
import numpy as np
from agent_sim.core.models import NFW, Cluster
from agent_sim.pylenslib_imsim_branch import PyLenslibSimulator
from agent_sim.pylenslib_imsim_branch.imaging import SourcePlacement, ImSimRubinizer

# 1. Define lens cluster
halo = NFW(M200=1e15, c=4.0, redshift=0.3)
cluster = Cluster(halo, name="MyCluster")

# 2. Initialize simulator & compute lensing maps
sim = PyLenslibSimulator(cluster)
maps = sim.compute_lensing_maps(x_range=(-50, 50), y_range=(-50, 50), n_pixels=200)

# 3. Generate mock noiseless image
sources = [
    SourcePlacement(source_id=0, source_position=(0.2, -0.1), redshift=1.0, magnitude=22.0)
]
_, perfect_image = sim.generate_mock_images(sources, image_size=200, pixel_scale=0.2)

# 4. Apply Rubinization (Rubin LSST parameters)
rubinizer = ImSimRubinizer(psf_sigma=0.7, read_noise_e=10, sky_brightness_mag=22, gain=3.0)
rubinized_image, psf = rubinizer.rubinize(perfect_image, pixel_scale=0.2)

print("Perfect image shape:", perfect_image.shape)
print("Rubinized image max ADU:", rubinized_image.max())
```

#### SLSim Branch
```python
from agent_sim.core.models import NFW, Cluster
from agent_sim.slsim_branch import SlsimSimulator

cluster = Cluster(NFW(M200=1e15, c=4.0, redshift=0.3))
sim = SlsimSimulator(cluster)

# Run SLSim workflow
lensing_maps = sim.compute_lensing_maps()
```

---

### 2. Running Simulations via Config (YAML)

You can run full catalog simulations using `pylenslib_sim`:

```bash
python -m pylenslib_sim.main pylenslib_sim/config_examples/ggsl_rubin_realistic.yaml
```

Outputs (FITS images, PNG plots, JSON summaries) will be saved in `pylenslib_outputs/`.

---

### 3. Parallel Execution with MPI

To run parallel batch simulations across multiple CPU cores:

```bash
# Run 6 cluster simulations across 4 processes
mpirun -np 4 python3 agent_sim_mpi_orchestrator.py
```

Or for batch realizations of a single cluster:
```bash
mpirun -np 4 python3 agent_sim_batch_mpi.py --cluster "MACS J0416" --realizations 20
```

---

## 📷 Rubinization Observational Parameters

Default parameters match Vera C. Rubin Observatory (LSST) baseline specifications:

| Parameter | Default Value | Description |
|---|---|---|
| `psf_sigma` | $0.7''$ | Point Spread Function standard deviation |
| `read_noise_e` | $10\ e^-$ | Detector readout noise |
| `sky_brightness_mag` | $22.0\ \text{mag}/\text{arcsec}^2$ | Sky background level |
| `gain` | $3.0\ e^-/\text{ADU}$ | Detector conversion gain |
| Saturation | $65,000\ \text{ADU}$ | Full-well pixel limit |
| Cosmic Ray Rate | $0.1\%$ | Fraction of affected pixels per exposure |

---

## 🧪 Testing

Run the automated pytest suite:

```bash
pytest -v
```

All 9 integration and unit tests cover profile calculations, lensing potentials, simulator pipelines, and Rubinization modules.

---

## 📖 Further Reading

- **`RUBINIZATION_GUIDE.md`**: Comparison between PyLensLib+ImSim and SLSim branches.
- **`RUBINIZATION_README.md`**: Detailed technical breakdown of detector noise and PSF convolution math.
- **`USAGE_EXAMPLES.md`**: Guide for MPI orchestrations and image analysis scripts.

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.
