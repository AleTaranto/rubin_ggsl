#!/usr/bin/env python3
"""MPI-parallel batch simulation for AGENT_sim.

Runs multiple random realizations of cluster simulations in parallel.
Each MPI rank handles independent random seeds for statistical sampling.

Usage:
    # Serial: 10 realizations
    python3 agent_sim_batch_mpi.py --cluster "MACS J0416" --realizations 10

    # MPI: 10 realizations across 4 processes (2-3 per process)
    mpirun -np 4 python3 agent_sim_batch_mpi.py --cluster "MACS J0416" --realizations 10

    # Multiple clusters, 5 realizations each
    mpirun -np 8 python3 agent_sim_batch_mpi.py \
        --clusters "MACS J0416" "Abell 2744" "RXJ 1347" \
        --realizations 5
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import logging

try:
    from mpi4py import MPI
    HAS_MPI = True
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    HAS_MPI = False
    RANK = 0
    SIZE = 1


logging.basicConfig(
    level=logging.INFO,
    format=f"[Rank {RANK}] %(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def mpi_print(msg: str, rank: int = 0) -> None:
    """Print only from specified rank."""
    if RANK == rank:
        print(msg)


def distribute_work(
    clusters: List[str],
    realizations_per_cluster: int,
    rank: int,
    size: int
) -> List[Tuple[str, int]]:
    """Distribute (cluster, realization_id) pairs across ranks via round-robin."""
    work = [
        (cluster, real_id)
        for cluster in clusters
        for real_id in range(realizations_per_cluster)
    ]
    return [work[i] for i in range(len(work)) if i % size == rank]


def run_realization(
    cluster_name: str,
    realization_id: int,
    output_dir: Path,
    rank: int,
    seed_offset: int = 0
) -> Dict[str, Any]:
    """Run single realization of cluster simulation.
    
    Parameters
    ----------
    cluster_name : str
        Cluster name
    realization_id : int
        Realization index (for seeding and output organization)
    output_dir : Path
        Base output directory
    rank : int
        MPI rank
    seed_offset : int
        Offset added to rank for RNG seed
    
    Returns
    -------
    Dict with simulation metadata
    """
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from agent_sim.run_meneghetti_simulation import MeneghettiSimulationPipeline
        
        # Create deterministic seed from rank and realization
        seed = rank * 10000 + realization_id + seed_offset
        np.random.seed(seed)
        
        # Setup output
        cluster_dir = output_dir / cluster_name.replace(" ", "_")
        cluster_dir.mkdir(parents=True, exist_ok=True)
        
        realization_dir = cluster_dir / f"realization_{realization_id:04d}"
        realization_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Starting {cluster_name} realization {realization_id} (seed={seed})"
        )
        start_time = time.time()
        
        # Run simulation
        pipeline = MeneghettiSimulationPipeline(cluster_name)
        results = pipeline.run()
        
        elapsed = time.time() - start_time
        
        if results:
            # Save arrays
            np.save(
                realization_dir / "perfect_image.npy",
                results['perfect_image']
            )
            np.save(
                realization_dir / "hst_image.npy",
                results['hst_image']
            )
            np.save(
                realization_dir / "rubin_image.npy",
                results['rubin_image']
            )
            
            # Save comparison metadata
            comparison = results['comparison']
            with (realization_dir / "metrics.json").open("w") as f:
                json.dump({
                    "cluster": cluster_name,
                    "realization_id": realization_id,
                    "seed": seed,
                    "elapsed_seconds": elapsed,
                    **comparison
                }, f, indent=2)
            
            logger.info(
                f"✓ Completed {cluster_name} realization {realization_id} ({elapsed:.1f}s)"
            )
            
            return {
                "cluster": cluster_name,
                "realization_id": realization_id,
                "status": "success",
                "seed": seed,
                "elapsed_seconds": elapsed,
                "output_dir": str(realization_dir),
            }
        else:
            logger.error(f"✗ Failed {cluster_name} realization {realization_id}")
            return {
                "cluster": cluster_name,
                "realization_id": realization_id,
                "status": "failed",
                "seed": seed,
            }
    
    except Exception as e:
        logger.error(f"✗ Error in {cluster_name} realization {realization_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "cluster": cluster_name,
            "realization_id": realization_id,
            "status": "error",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="MPI-parallel batch simulations for AGENT_sim"
    )
    parser.add_argument(
        "--cluster",
        type=str,
        help="Single cluster to simulate (overrides --clusters)"
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        default=["MACS J0416"],
        help="List of clusters to simulate"
    )
    parser.add_argument(
        "--realizations",
        type=int,
        default=5,
        help="Number of realizations per cluster"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/agent_sim_batch_mpi"),
        help="Output directory"
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Seed offset for reproducibility across runs"
    )
    args = parser.parse_args()

    # Handle single cluster
    if args.cluster:
        args.clusters = [args.cluster]

    if RANK == 0:
        print("="*70)
        print("AGENT_SIM MPI-PARALLEL BATCH SIMULATOR")
        print("="*70)
        print(f"[MPI] Processes: {SIZE}")
        print(f"[MPI] Clusters: {args.clusters}")
        print(f"[MPI] Realizations per cluster: {args.realizations}")
        print(f"[MPI] Total simulations: {len(args.clusters) * args.realizations}")
        print()

    # Distribute work
    work_assignments = distribute_work(
        args.clusters,
        args.realizations,
        RANK,
        SIZE
    )
    
    logger.info(f"Assigned {len(work_assignments)} simulations")

    # Run simulations
    local_results = []
    for cluster_name, realization_id in work_assignments:
        result = run_realization(
            cluster_name=cluster_name,
            realization_id=realization_id,
            output_dir=args.output_dir,
            rank=RANK,
            seed_offset=args.seed_offset
        )
        local_results.append(result)

    # Gather and summarize
    if HAS_MPI:
        all_results = COMM.gather(local_results, root=0)
        COMM.Barrier()
    else:
        all_results = [local_results]

    if RANK == 0:
        print("\n" + "="*70)
        print("BATCH RESULTS SUMMARY")
        print("="*70)
        
        combined = [r for results in all_results if results for r in results]
        
        successes = [r for r in combined if r["status"] == "success"]
        failures = [r for r in combined if r["status"] == "failed"]
        errors = [r for r in combined if r["status"] == "error"]
        
        print(f"\n✓ Successful: {len(successes)}/{len(combined)}")
        
        total_time = sum(r.get("elapsed_seconds", 0) for r in successes)
        avg_time = total_time / len(successes) if successes else 0
        
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average per simulation: {avg_time:.1f}s")
        
        if failures:
            print(f"\n✗ Failed: {len(failures)}")
        if errors:
            print(f"\n✗ Errors: {len(errors)}")
        
        # Summary by cluster
        print(f"\nBy Cluster:")
        for cluster in args.clusters:
            cluster_results = [r for r in successes if r["cluster"] == cluster]
            print(f"  {cluster}: {len(cluster_results)}/{args.realizations} ✓")
        
        # Save summary
        summary_file = args.output_dir / "batch_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        with summary_file.open("w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mpi_processes": SIZE,
                "total_simulations": len(combined),
                "successful": len(successes),
                "failed": len(failures),
                "errors": len(errors),
                "total_computation_time_seconds": total_time,
                "average_per_simulation_seconds": avg_time,
                "clusters": args.clusters,
                "realizations_per_cluster": args.realizations,
            }, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
        print(f"Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
