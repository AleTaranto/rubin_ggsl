"""
Case 1: PyLensLib-only GGSL Simulation

This case uses PyLensLib for everything:
- Computing lensing maps from density profiles
- Finding critical lines and caustics
- Detecting GGSL events
- Generating mock lensed images

Configuration:
    Edit config.yaml to define:
    - Cluster mass, concentration, redshift
    - Subhalo properties and positions
    - Background source population
    - Simulation grid parameters
    - Output settings

Usage:
    # Serial mode (single core)
    python run.py config.yaml
    
    # MPI mode (auto-detect cores)
    mpirun -np auto python run.py config.yaml
    
    # MPI mode (4 processes)
    mpirun -np 4 python run.py config.yaml
"""

import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mpi_utils import get_mpi_info, mpi_print, distribute_range, gather_results, mpi_barrier
from agent_sim.core.models import NFW, CoreNFW, Cluster, SourceDistribution
from agent_sim.pylenslib_imsim_branch import PyLenslibSimulator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_cluster_from_config(config: dict) -> Cluster:
    """Build cluster model from configuration."""
    cluster_cfg = config['cluster']
    
    # Create main halo
    main_halo = NFW(
        M200=cluster_cfg['main_halo']['M200'],
        c=cluster_cfg['main_halo']['concentration'],
        redshift=cluster_cfg['main_halo']['redshift']
    )
    
    cluster = Cluster(main_halo, name=cluster_cfg.get('name', 'SimCluster'))
    
    # Add subhalos
    np.random.seed(cluster_cfg.get('random_seed', 42))
    for subhalo_cfg in cluster_cfg.get('subhalos', []):
        subhalo = NFW(
            M200=subhalo_cfg['M200'],
            c=subhalo_cfg.get('concentration', 4.0),
            redshift=subhalo_cfg.get('redshift', cluster_cfg['main_halo']['redshift'])
        )
        position = tuple(subhalo_cfg['position'])
        cluster.add_subhalo(subhalo, position)
    
    return cluster


def simulate_case(config: dict, case_index: int = None) -> dict:
    """Run simulation for one configuration."""
    mpi_info = get_mpi_info()
    
    if case_index is not None:
        mpi_print(f"[Rank {mpi_info['rank']}/{mpi_info['size']}] Processing case {case_index}...")
    
    # Build cluster
    cluster = build_cluster_from_config(config)
    
    # Initialize simulator
    grid_size = config['simulation'].get('grid_size', 0.05)
    simulator = PyLenslibSimulator(cluster, grid_size=grid_size)
    
    # Get grid parameters
    sim_cfg = config['simulation']
    x_range = tuple(sim_cfg['x_range'])
    y_range = tuple(sim_cfg['y_range'])
    n_pixels = sim_cfg.get('n_pixels', 100)
    
    # Compute lensing maps
    lensing_maps = simulator.compute_lensing_maps(
        x_range=x_range,
        y_range=y_range,
        n_pixels=n_pixels
    )
    
    # Find critical lines
    critical_info = simulator.find_critical_lines(
        x_range=x_range,
        y_range=y_range,
        n_pixels=n_pixels
    )
    
    # Place sources
    src_cfg = config['sources']
    source_dist = SourceDistribution(
        redshift=src_cfg['redshift'],
        density_per_arcmin2=src_cfg.get('density_per_arcmin2', 50)
    )
    np.random.seed(src_cfg.get('random_seed', 42))
    source_positions = source_dist.sample_positions(
        n_sources=src_cfg['n_sources'],
        region_size=src_cfg.get('region_size', 5)
    )
    
    # Create source placement objects
    from agent_sim.pylenslib_imsim_branch.imaging import SourcePlacement
    sources_list = []
    for i, pos in enumerate(source_positions):
        source = SourcePlacement(
            source_id=i,
            source_position=tuple(pos),
            redshift=src_cfg['redshift'],
            magnitude=np.random.uniform(
                src_cfg.get('magnitude_min', 18),
                src_cfg.get('magnitude_max', 24)
            ),
            radius_arcsec=np.random.uniform(
                src_cfg.get('radius_min', 0.05),
                src_cfg.get('radius_max', 0.3)
            )
        )
        sources_list.append(source)
    
    # Generate images
    img_cfg = config['images']
    lensed_images, combined_image = simulator.generate_mock_images(
        sources_list,
        x_range=x_range,
        y_range=y_range,
        image_size=img_cfg.get('size', 256),
        pixel_scale=img_cfg.get('pixel_scale', 0.2)
    )
    
    # Detect GGSL events
    events, summary = simulator.detect_ggsl_events(
        source_positions,
        x_range=x_range,
        y_range=y_range,
        n_pixels=n_pixels,
        source_z=src_cfg['redshift'],
        lens_z=config['cluster']['main_halo']['redshift']
    )
    
    results = {
        'cluster_name': config['cluster']['name'],
        'n_sources': len(sources_list),
        'n_ggsl_events': summary.get('n_events', 0),
        'mean_magnification': summary.get('mean_magnification', 0),
        'combined_image': combined_image,
        'lensing_maps': lensing_maps,
        'events': events,
        'critical_info': critical_info
    }
    
    return results


def save_results(results: dict, output_dir: Path) -> None:
    """Save simulation results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined image as NPZ
    image_path = output_dir / f"combined_image_{results['cluster_name']}.npz"
    np.savez(image_path, image=results['combined_image'])
    mpi_print(f"Saved image to {image_path}")
    
    # Save lensing maps
    maps_path = output_dir / f"lensing_maps_{results['cluster_name']}.npz"
    np.savez(
        maps_path,
        x_grid=results['lensing_maps']['x_grid'],
        y_grid=results['lensing_maps']['y_grid'],
        deflection_x=results['lensing_maps']['deflection_x'],
        deflection_y=results['lensing_maps']['deflection_y'],
        magnification=results['lensing_maps']['magnification']
    )
    mpi_print(f"Saved lensing maps to {maps_path}")
    
    # Save summary statistics
    summary_path = output_dir / f"summary_{results['cluster_name']}.json"
    summary = {
        'cluster_name': results['cluster_name'],
        'n_sources': results['n_sources'],
        'n_ggsl_events': results['n_ggsl_events'],
        'mean_magnification': float(results['mean_magnification']),
        'timestamp': datetime.now().isoformat()
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    mpi_print(f"Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Case 1: PyLensLib-only GGSL Simulation'
    )
    parser.add_argument('config', help='Path to config.yaml')
    parser.add_argument('--output', default='outputs/case1',
                       help='Output directory (default: outputs/case1)')
    
    args = parser.parse_args()
    
    mpi_info = get_mpi_info()
    
    if mpi_info['is_master']:
        print("=" * 70)
        print("Case 1: PyLensLib-only GGSL Simulation")
        print("=" * 70)
        print(f"MPI available: {mpi_info['has_mpi']}")
        print(f"Running with {mpi_info['size']} process(es)")
        print()
    
    # Load configuration
    config = load_config(args.config)
    
    if mpi_info['is_master']:
        print(f"Configuration loaded from: {args.config}")
        print(f"Cluster: {config['cluster']['name']}")
        print()
    
    # Run simulation
    results = simulate_case(config)
    
    mpi_barrier()
    
    # Save results (master rank only)
    if mpi_info['is_master']:
        save_results(results, args.output)
        
        print("\n" + "=" * 70)
        print("Simulation completed successfully!")
        print(f"Results saved to: {args.output}")
        print("=" * 70)


if __name__ == '__main__':
    main()
