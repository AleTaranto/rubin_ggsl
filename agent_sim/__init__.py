"""
AGENT_sim: Strong Gravitational Lensing Simulations for Dark Matter Studies

A framework for simulating strong gravitational lensing events with multiple
simulation backends (PyLensLib + ImSim, and SLSim) to probe dark matter subhalo structure.
"""

__version__ = "0.1.0"
__author__ = "AGENT Team"

from agent_sim.core import models, lensing, observables, utils

__all__ = [
    "models",
    "lensing",
    "observables",
    "utils",
]
