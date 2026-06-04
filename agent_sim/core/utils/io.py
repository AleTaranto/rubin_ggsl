"""Utilities for I/O and serialization."""

import json
import numpy as np
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return super().default(obj)


def save_simulation_state(state, output_path):
    """
    Save simulation state to JSON.
    
    Parameters
    ----------
    state : dict
        Simulation state dictionary
    output_path : str or Path
        Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(state, f, cls=NumpyEncoder, indent=2)


def load_simulation_state(input_path):
    """
    Load simulation state from JSON.
    
    Parameters
    ----------
    input_path : str or Path
        Input file path
    
    Returns
    -------
    state : dict
        Simulation state dictionary
    """
    with open(input_path, 'r') as f:
        return json.load(f)


def save_observables(observables, output_path):
    """
    Save list of GGSLObservable objects.
    
    Parameters
    ----------
    observables : list of GGSLObservable
        Detected GGSL events
    output_path : str or Path
        Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = []
    for obs in observables:
        data.append({
            'event_id': obs.event_id,
            'source_id': obs.source_id,
            'n_images': obs.n_images,
            'magnifications': obs.magnifications.tolist() if obs.magnifications is not None else None,
            'total_magnification': float(obs.total_magnification),
            'arc_length_arcsec': float(obs.arc_length_arcsec),
            'arc_width_arcsec': float(obs.arc_width_arcsec),
            'source_redshift': float(obs.source_redshift),
            'lens_redshift': float(obs.lens_redshift),
            'Einstein_radius_arcsec': float(obs.Einstein_radius_arcsec) if obs.Einstein_radius_arcsec else None,
        })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)


def save_grid_data(grid_data, x_grid, y_grid, output_path):
    """
    Save grid data (e.g., magnification map) to binary format.
    
    Parameters
    ----------
    grid_data : ndarray
        2D grid of data
    x_grid, y_grid : ndarray
        1D coordinate arrays
    output_path : str or Path
        Output file path (will be .npz)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(output_path, 
             grid_data=grid_data, 
             x_grid=x_grid, 
             y_grid=y_grid)


def load_grid_data(input_path):
    """Load grid data from npz file."""
    data = np.load(input_path)
    return data['grid_data'], data['x_grid'], data['y_grid']
