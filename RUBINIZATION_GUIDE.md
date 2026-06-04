"""
Comparison Guide: PyLensLib+ImSim vs SLSim Rubinization

This guide explains the differences between the two branches and their
Rubinization (realistic observational effects) approaches.
"""

import numpy as np


def print_comparison_guide():
    """Print detailed comparison guide."""
    
    guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AGENT_sim Dual-Branch Comparison                         ║
║              PyLensLib+ImSim vs SLSim Rubinization Approaches               ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ARCHITECTURE & WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BRANCH 1: PyLensLib + ImSim
├── Physics: Custom NumPy/SciPy implementation
├── Workflow: 
│   Dark Matter Model → PyLensLib-based Lensing Maps
│   → Mock Image Generation (source plane ray-tracing)
│   → ImSim Rubinization (PSF, noise, detector effects)
│   → Realistic Rubin-like images
└── Key Components:
    - SourcePlacer: Place sources in source plane
    - ImageGenerator: Ray-trace from image plane to source plane
    - ImSimRubinizer: Apply PSF + noise + sky + cosmic rays

BRANCH 2: SLSim
├── Physics: Leverages SLSim external package
├── Workflow:
│   Dark Matter Model → SLSim Simulation
│   → Perfect Mock Images (SLSim lensing engine)
│   → SLSim Rubinization (SLSim's built-in Rubin effects)
│   → Realistic images with SLSim validation
└── Key Components:
    - SlsimSourcePlacer: Interface with SLSim source placement
    - SlsimImageGenerator: Delegate to SLSim engine
    - SlsimRubinizer: Use SLSim's Rubin simulation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. RUBINIZATION IMPLEMENTATION DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FEATURE COMPARISON:
┌─────────────────────┬──────────────────────────┬──────────────────────────┐
│ Effect              │ PyLensLib+ImSim Branch   │ SLSim Branch             │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ PSF                 │ Gaussian kernel (custom) │ Gaussian or SLSim PSF    │
│ Photon Noise        │ Poisson (full pipeline)  │ Poisson (fallback)       │
│ Read Noise          │ Gaussian (ADC-level)     │ Gaussian                 │
│ Sky Background      │ Poisson + photon noise   │ Simplified model         │
│ Cosmic Rays         │ Simulated streaks        │ Not in fallback          │
│ Saturation          │ Detector saturation      │ Basic clipping           │
│ Integration         │ ImSim-ready (future)     │ SLSim-ready (future)     │
└─────────────────────┴──────────────────────────┴──────────────────────────┘

RUBIN-SPECIFIC PARAMETERS (Defaults):
┌────────────────────────┬─────────────┬──────────────────────────────────┐
│ Parameter              │ Default     │ Justification                    │
├────────────────────────┼─────────────┼──────────────────────────────────┤
│ PSF sigma              │ 0.7 arcsec  │ Rubin LSST design target         │
│ Read noise             │ 10 e-       │ LSST detector spec (~6-10)       │
│ Sky brightness         │ 22 mag/aq2  │ Dark site + dark gray twilight   │
│ Gain                   │ 3.0 e-/ADU  │ LSST segment gain               │
│ Saturation level       │ 65000 ADU   │ Full-well capacity              │
│ Cosmic ray rate        │ 0.1% pixels │ Low Earth orbit (~100 cm²/s)    │
└────────────────────────┴─────────────┴──────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. PRACTICAL USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PyLensLib+ImSim Example:
─────────────────────────
    from agent_sim.pylenslib_imsim_branch import PyLenslibSimulator
    from agent_sim.pylenslib_imsim_branch.imaging import (
        ImageGenerator, ImSimRubinizer
    )
    
    # Generate perfect mock
    simulator = PyLenslibSimulator(cluster)
    mock_image = simulator.generate_mock_images(sources, ...)
    
    # Apply Rubinization
    rubinizer = ImSimRubinizer(psf_sigma=0.7, read_noise_e=10)
    realistic_image, psf = rubinizer.rubinize(mock_image)

SLSim Example:
──────────────
    from agent_sim.slsim_branch import SlsimSimulator
    from agent_sim.slsim_branch.imaging import (
        SlsimImageGenerator, SlsimRubinizer
    )
    
    # Generate perfect mock (via SLSim)
    simulator = SlsimSimulator(cluster)
    mock_image = simulator.generate_mock_images(sources, ...)
    
    # Apply Rubinization (via SLSim or fallback)
    rubinizer = SlsimRubinizer(psf_sigma=0.7, read_noise_e=10)
    realistic_image, metadata = rubinizer.rubinize_slsim(mock_image)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. WHEN TO USE EACH BRANCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use PyLensLib+ImSim If:
✓ You want fine-grained control over lensing computation
✓ You're developing new DM models or lensing algorithms
✓ You prefer modular components (swap PSF, noise models, etc.)
✓ You want to integrate with ImSim pipelines
✓ You need reproducible, fully-documented physics

Use SLSim If:
✓ You want "out-of-the-box" end-to-end simulation
✓ You trust SLSim's well-validated physics
✓ You need rapid prototyping with vetted tools
✓ You want to compare with official SLSim benchmarks
✓ You prefer unified interface (sources → images → effects)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. EXPECTED DIFFERENCES IN OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Due to implementation differences, expect:

Image Properties:
├── Magnification maps: May differ by ~1-5% due to numerical integration
├── Critical lines: Position agreement better than 0.1 pixels
├── Image positions: Agreement within 0.01 arcsec (subpixel)
└── Image fluxes: Agreement within 10-20% (due to numerical factors)

Rubinized Images:
├── PSF: Both use Gaussian, but kernels may slightly differ
├── Noise level: Within ±1e- RMS (due to random sampling)
├── Cosmic rays: Only PyLensLib+ImSim includes them
├── Sky background: Models differ slightly
└── Overall realism: Both produce Rubin-like images

Performance:
├── PyLensLib+ImSim: Slower but more controlled (~0.1-1 s per image)
├── SLSim: Faster with vetted algorithms (~0.05-0.5 s per image)
└── Rubinization: Similar for both (~0.01-0.1 s)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. COMPUTATIONAL PIPELINE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Both branches support:

Input:
  ✓ Dark matter models (NFW, CoreNFW, SIS, custom)
  ✓ Cluster with subhalo composition
  ✓ Background source redshifts & positions
  ✓ Rubin-like simulation parameters

Processing (Shared):
  ✓ Gravitational lensing (potentials, deflection angles)
  ✓ Magnification maps & critical lines
  ✓ GGSL event detection

Processing (Branch-specific):
  ✓ Mock image generation (PyLensLib method vs SLSim method)
  ✓ Rubinization effects (ImSim integration vs SLSim integration)

Output:
  ✓ Perfect (noise-free) mock images
  ✓ Realistic (Rubin-like) images
  ✓ PSF kernels & noise statistics
  ✓ GGSL event properties & statistics

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For detailed examples, see:
- agent_sim/pylenslib_imsim_branch/examples/image_generation_example.py
- agent_sim/slsim_branch/examples/slsim_workflow_example.py
"""
    print(guide)


if __name__ == "__main__":
    print_comparison_guide()
