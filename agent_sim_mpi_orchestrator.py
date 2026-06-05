#!/usr/bin/env python3
"""MPI-parallel AGENT_sim simulation orchestrator.

Parallelizes Meneghetti cluster simulations across multiple processes.
Each MPI rank processes a subset of clusters or random realizations.

Usage:
    # Serial mode
    python3 agent_sim_mpi_orchestrator.py

    # MPI mode (4 processes)
    mpirun -np 4 python3 agent_sim_mpi_orchestrator.py

    # Custom clusters
    mpirun -np 8 python3 agent_sim_mpi_orchestrator.py --clusters MACS_J0416 Abell2744 RXJ1347
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime
import numpy as np
from astropy.io import fits
import yaml
import matplotlib.pyplot as plt

from mpi_utils import get_mpi_info, mpi_print, mpi_barrier

# MPI info
MPI_INFO = get_mpi_info()
HAS_MPI = MPI_INFO['has_mpi']
RANK = MPI_INFO['rank']
SIZE = MPI_INFO['size']


# local helper kept for compatibility
def _mpi_print(msg: str, rank: int = 0) -> None:
    mpi_print(msg, rank=rank)


def get_default_clusters() -> List[str]:
    """Return list of Meneghetti clusters to simulate."""
    return [
        "MACS J0416",
        "MACS J0717",
        "Abell 2744",
        "Abell S1063",
        "Abell 370",
        "MACS J1149",
    ]


def distribute_clusters(clusters: List[str], rank: int, size: int) -> List[str]:
    """Distribute clusters across MPI ranks (round-robin)."""
    return [clusters[i] for i in range(len(clusters)) if i % size == rank]


def save_fits_with_metadata(image: np.ndarray, output_path: Path, cluster_name: str,
                             image_type: str, script_name: str = "agent_sim_mpi_orchestrator.py") -> None:
    """Save image as FITS file with metadata header.
    
    Parameters
    ----------
    image : np.ndarray
        2D image array to save
    output_path : Path
        Output file path (will use .fits extension)
    cluster_name : str
        Name of the cluster simulated
    image_type : str
        Type of image: 'perfect', 'hst', or 'rubin'
    script_name : str
        Name of the script that generated this image
    """
    # Convert path to .fits extension
    output_path = output_path.with_suffix('.fits')
    
    # Create primary HDU
    hdu = fits.PrimaryHDU(data=image.astype(np.float32))
    
    # Add metadata to header
    hdu.header['DATE'] = (datetime.utcnow().isoformat(), 'UTC creation date')
    hdu.header['CLUSTER'] = (cluster_name, 'Lensing cluster name')
    hdu.header['IMGTYPE'] = (image_type, 'Image type: perfect/hst/rubin')
    hdu.header['SCRIPT'] = (script_name, 'Script used to generate this image')
    hdu.header['ORIGIN'] = ('AGENT_sim MPI', 'Origin of data')
    
    # Add image description
    if image_type == 'perfect':
        hdu.header['DESCRIP'] = 'Perfect noiseless image without PSF convolution'
    elif image_type == 'hst':
        hdu.header['DESCRIP'] = 'HST-like image with PSF=0.08 arcsec'
        hdu.header['PSF'] = (0.08, 'PSF sigma in arcsec')
    elif image_type == 'rubin':
        hdu.header['DESCRIP'] = 'Rubin/LSST-like image with PSF=0.70 arcsec'
        hdu.header['PSF'] = (0.70, 'PSF sigma in arcsec')
    
    # Create HDU list and write
    hdul = fits.HDUList([hdu])
    hdul.writeto(output_path, overwrite=True)


def save_png(image: np.ndarray, output_path: Path, cmap: str = "viridis") -> None:
    """Save an image array as PNG for quick inspection."""
    output_path = output_path.with_suffix('.png')
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(image, origin='lower', cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def run_cluster_simulation(cluster_name: str, output_dir: Path, rank: int,
                           realization_id: int = None, seed: int = None) -> Dict[str, Any]:
    """Run single cluster simulation on this MPI rank.
    
    Returns metadata about the completed simulation.
    """
    sys.path.insert(0, str(Path(__file__).parent / "AGENT_sim"))
    
    try:
        from agent_sim.run_meneghetti_simulation import MeneghettiSimulationPipeline
        
        output_dir.mkdir(parents=True, exist_ok=True)
        cluster_output = output_dir / cluster_name.replace(" ", "_")
        cluster_output.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        mpi_print(f"[Rank {rank}] Starting: {cluster_name}", rank=rank)
        
        # Optionally set a deterministic seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        pipeline = MeneghettiSimulationPipeline(cluster_name)
        results = pipeline.run()
        
        elapsed = time.time() - start_time
        
        if results:
            # Save results as FITS with metadata
            # Save FITS and PNG for each image type
            # Use realization-aware filenames when provided
            suf = f"_real_{realization_id:04d}" if realization_id is not None else ""

            save_fits_with_metadata(
                results['perfect_image'],
                cluster_output / f"{cluster_name}_perfect{suf}.fits",
                cluster_name,
                "perfect",
                "agent_sim_mpi_orchestrator.py"
            )
            save_png(results['perfect_image'], cluster_output / f"{cluster_name}_perfect{suf}.png")

            save_fits_with_metadata(
                results['hst_image'],
                cluster_output / f"{cluster_name}_hst{suf}.fits",
                cluster_name,
                "hst",
                "agent_sim_mpi_orchestrator.py"
            )
            save_png(results['hst_image'], cluster_output / f"{cluster_name}_hst{suf}.png")

            save_fits_with_metadata(
                results['rubin_image'],
                cluster_output / f"{cluster_name}_rubin{suf}.fits",
                cluster_name,
                "rubin",
                "agent_sim_mpi_orchestrator.py"
            )
            save_png(results['rubin_image'], cluster_output / f"{cluster_name}_rubin{suf}.png")
            
            mpi_print(
                f"[Rank {rank}] ✓ Completed: {cluster_name} ({elapsed:.1f}s)",
                rank=rank
            )
            
            return {
                "cluster_name": cluster_name,
                "status": "success",
                "elapsed_seconds": elapsed,
                "output_dir": str(cluster_output),
            }
        else:
            mpi_print(f"[Rank {rank}] ✗ Failed: {cluster_name}", rank=rank)
            return {
                "cluster_name": cluster_name,
                "status": "failed",
                "elapsed_seconds": elapsed,
            }
    
    except Exception as e:
        mpi_print(f"[Rank {rank}] ✗ Error in {cluster_name}: {e}", rank=rank)
        import traceback
        traceback.print_exc()
        return {
            "cluster_name": cluster_name,
            "status": "error",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="MPI-parallel AGENT_sim orchestrator for Meneghetti simulations"
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional YAML config file describing clusters and options")
    parser.add_argument(
        "--clusters",
        nargs="+",
        default=get_default_clusters(),
        help="Cluster names to simulate"
    )
    parser.add_argument(
        "--realizations",
        type=int,
        default=1,
        help="Number of realizations per cluster (batch mode)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/meneghetti_mpi"),
        help="Output directory for results"
    )
    args = parser.parse_args()

    if RANK == 0:
        print("="*70)
        print("AGENT_SIM MPI-PARALLEL ORCHESTRATOR")
        print("Meneghetti GGSL Simulations: HST vs Rubin Comparison")
        print("="*70)
        print(f"[MPI] Running with {SIZE} processes")
        print(f"[MPI] Clusters to simulate: {args.clusters}")
        print()

    # If a config YAML is provided, override clusters/realizations
    if args.config is not None and args.config.exists():
        with args.config.open() as f:
            cfg = yaml.safe_load(f)
        if 'clusters' in cfg:
            args.clusters = cfg['clusters']
        if 'realizations' in cfg:
            args.realizations = int(cfg['realizations'])

    # Distribute work among MPI ranks
    my_clusters = distribute_clusters(args.clusters, RANK, SIZE)
    mpi_print(f"[Rank {RANK}] Assigned {len(my_clusters)} clusters: {my_clusters}", rank=RANK)

    # Run simulations (supporting multiple realizations)
    local_results = []
    for cluster_name in my_clusters:
        for real in range(args.realizations):
            # Create a per-realization seed to ensure different draws
            seed = None
            if args.realizations > 1:
                seed = RANK * 100000 + real

            result = run_cluster_simulation(
                cluster_name=cluster_name,
                output_dir=args.output_dir,
                rank=RANK,
                realization_id=real if args.realizations > 1 else None,
                seed=seed
            )
            local_results.append(result)

    # Gather results on rank 0
    if HAS_MPI:
        all_results = COMM.gather(local_results, root=0)
        COMM.Barrier()
    else:
        all_results = [local_results]

    if RANK == 0:
        print("\n" + "="*70)
        print("SIMULATION RESULTS")
        print("="*70)
        
        combined_results = [r for results in all_results if results for r in results]
        
        successes = [r for r in combined_results if r["status"] == "success"]
        failures = [r for r in combined_results if r["status"] == "failed"]
        errors = [r for r in combined_results if r["status"] == "error"]
        
        print(f"\n✓ Successful: {len(successes)}")
        for r in successes:
            print(f"  - {r['cluster_name']}: {r['elapsed_seconds']:.1f}s")
        
        if failures:
            print(f"\n✗ Failed: {len(failures)}")
            for r in failures:
                print(f"  - {r['cluster_name']}")
        
        if errors:
            print(f"\n✗ Errors: {len(errors)}")
            for r in errors:
                print(f"  - {r['cluster_name']}: {r.get('error', 'unknown')}")
        
        total_time = sum(r.get("elapsed_seconds", 0) for r in successes)
        print(f"\nTotal computation time: {total_time:.1f}s")
        print(f"Output directory: {args.output_dir}")
        
        # Write results to JSON
        results_file = args.output_dir / "mpi_results.json"
        with results_file.open("w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mpi_processes": SIZE,
                "total_clusters": len(args.clusters),
                "successful": len(successes),
                "failed": len(failures),
                "errors": len(errors),
                "results": combined_results,
            }, f, indent=2)
        
        print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
