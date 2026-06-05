"""
Unified MPI utilities for GGSL simulation cases.

Provides consistent MPI interface across all simulation cases:
- Auto-detection of MPI availability
- Graceful fallback to serial execution
- Rank-aware output and task distribution
"""

from typing import List, Dict, Any
import sys

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


def get_mpi_info() -> Dict[str, Any]:
    """
    Get current MPI configuration.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'has_mpi': bool - Whether MPI is available
        - 'rank': int - Current rank (0 if serial)
        - 'size': int - Total number of ranks (1 if serial)
        - 'is_master': bool - True if rank 0
    """
    return {
        'has_mpi': HAS_MPI,
        'rank': RANK,
        'size': SIZE,
        'is_master': RANK == 0,
    }


def mpi_print(msg: str, rank: int = 0) -> None:
    """
    Print only from specified rank.
    
    Parameters
    ----------
    msg : str
        Message to print
    rank : int
        Only print if current rank matches this value (default: 0, master rank)
    """
    if RANK == rank:
        print(msg, flush=True)


def mpi_barrier() -> None:
    """
    Synchronize all ranks.
    
    No-op if MPI not available.
    """
    if HAS_MPI:
        COMM.Barrier()


def distribute_range(total_items: int, rank: int = None, size: int = None) -> tuple:
    """
    Distribute items across MPI ranks.
    
    Parameters
    ----------
    total_items : int
        Total number of items to distribute
    rank : int, optional
        Rank index (uses global RANK if not provided)
    size : int, optional
        Total number of ranks (uses global SIZE if not provided)
    
    Returns
    -------
    tuple
        (start_idx, end_idx) for this rank's items
    """
    if rank is None:
        rank = RANK
    if size is None:
        size = SIZE
    
    # Distribute as evenly as possible
    items_per_rank = total_items // size
    remainder = total_items % size
    
    # Ranks with index < remainder get one extra item
    if rank < remainder:
        start_idx = rank * (items_per_rank + 1)
        end_idx = start_idx + items_per_rank + 1
    else:
        start_idx = remainder * (items_per_rank + 1) + (rank - remainder) * items_per_rank
        end_idx = start_idx + items_per_rank
    
    return start_idx, end_idx


def distribute_list(items: List[Any], rank: int = None, size: int = None) -> List[Any]:
    """
    Distribute list items across MPI ranks.
    
    Parameters
    ----------
    items : list
        Items to distribute
    rank : int, optional
        Rank index (uses global RANK if not provided)
    size : int, optional
        Total number of ranks (uses global SIZE if not provided)
    
    Returns
    -------
    list
        Items assigned to this rank
    """
    start_idx, end_idx = distribute_range(len(items), rank, size)
    return items[start_idx:end_idx]


def gather_results(local_results: Any) -> List[Any]:
    """
    Gather results from all ranks to master (rank 0).
    
    Parameters
    ----------
    local_results : any
        Results from this rank
    
    Returns
    -------
    list or None
        List of all results (only on rank 0), None on other ranks
    """
    if not HAS_MPI:
        return [local_results]
    
    all_results = COMM.gather(local_results, root=0)
    if RANK == 0:
        return all_results
    return None


def broadcast_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Broadcast configuration from master rank to all ranks.
    
    Parameters
    ----------
    config : dict
        Configuration dict (only used on rank 0)
    
    Returns
    -------
    dict
        Configuration on all ranks
    """
    if not HAS_MPI:
        return config
    
    config = COMM.bcast(config, root=0)
    return config


def suggest_process_count() -> int:
    """
    Suggest optimal number of processes for this system.
    
    Returns
    -------
    int
        Suggested process count (1 if single-core)
    """
    import os
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        # Fallback for systems without sched_getaffinity
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
    
    return max(1, cpu_count)
