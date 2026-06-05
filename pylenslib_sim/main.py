"""Orchestrator for P_GGSL mock catalog generation using PyLenslib simulator.

Usage:
    python -m pylenslib_sim.main config.yaml

The YAML config controls clusters, subhalo sampling, source distribution,
and output locations. See `config_examples/pggsl_config.yaml` for an example.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
import os
from astropy.io import fits
import matplotlib.pyplot as plt

from agent_sim.core.models.cluster import ClusterGenerator
from agent_sim.core.models.galaxy import SourceDistribution
from agent_sim.pylenslib_imsim_branch.simulator import PyLenslibSimulator


def save_fits(image, path: Path, header: dict = None):
    path = path.with_suffix('.fits')
    hdu = fits.PrimaryHDU(data=image.astype('float32'))
    if header:
        for k, v in header.items():
            try:
                hdu.header[k] = v
            except Exception:
                pass
    hdu.writeto(path, overwrite=True)


def save_png(image, path: Path, cmap='viridis'):
    path = path.with_suffix('.png')
    plt.figure(figsize=(6,6), dpi=150)
    plt.imshow(image, origin='lower', cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def run_realization(cfg: dict, cluster_cfg: dict, realization_idx: int, outdir: Path):
    """Run one realization and return summary dict."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Create cluster
    n_sub = int(cluster_cfg.get('n_subhalos', 10))
    M200_cluster = float(cluster_cfg.get('M200', 1e15))
    c_cluster = float(cluster_cfg.get('c', 4.0))
    sub_range = cluster_cfg.get('subhalo_mass_range', [1e11, 1e12])
    sub_range = (float(sub_range[0]), float(sub_range[1]))
    seed = cfg.get('seed', None)
    cluster = ClusterGenerator.simple_cluster(M200_cluster, c_cluster,
                                              n_subhalos=n_sub,
                                              M200_sub_range=tuple(sub_range),
                                              redshift=cluster_cfg.get('z', 0.3),
                                              seed=(seed + realization_idx) if seed is not None else None)

    # Initialize simulator
    sim = PyLenslibSimulator(cluster, grid_size=cfg.get('grid_size', 0.05))

    # Prepare source distribution
    src_cfg = cfg.get('sources', {})
    n_sources = int(src_cfg.get('n_sources', 500))
    region_size = float(src_cfg.get('region_size_arcmin', 5.0))
    sd = SourceDistribution(redshift=src_cfg.get('redshift', 1.0), density_per_arcmin2=src_cfg.get('density_per_arcmin2', 50))
    src_positions = sd.sample_positions(n_sources, region_size=region_size)

    # detect GGSL events using simulator helper
    x_range = tuple(cfg.get('x_range', (-150, 150)))
    y_range = tuple(cfg.get('y_range', (-150, 150)))
    n_pixels = int(cfg.get('n_pixels', 200))

    events, summary = sim.detect_ggsl_events(src_positions, x_range=x_range, y_range=y_range, n_pixels=n_pixels,
                                            source_z=src_cfg.get('redshift', 1.0), lens_z=cluster_cfg.get('z', 0.3))

    # Save outputs: catalog and quick maps
    summary_out = {
        'cluster_cfg': cluster_cfg,
        'n_sources': n_sources,
        'n_events': summary.get('n_events', 0),
        'mean_magnification': summary.get('mean_magnification', 0),
        'median_magnification': summary.get('median_magnification', 0),
        'mean_arc_length': summary.get('mean_arc_length', 0),
        'median_arc_length': summary.get('median_arc_length', 0),
    }

    # Save events to JSON
    with open(outdir / f'events_realization_{realization_idx:04d}.json', 'w') as fh:
        json.dump({'events': [e.__dict__ for e in events], 'summary': summary_out}, fh, indent=2)

    # Save caustic / magnification maps for inspection
    maps = sim.compute_lensing_maps(x_range=x_range, y_range=y_range, n_pixels=n_pixels)
    # Save magnification map
    save_fits(maps['magnification'], outdir / f'magnification_real_{realization_idx:04d}.fits')
    save_png(maps['magnification'], outdir / f'magnification_real_{realization_idx:04d}.png')

    # Return summary
    return summary_out


def main(argv=None):
    parser = argparse.ArgumentParser(description='Pylenslib-based P_GGSL catalog generator')
    parser.add_argument('config', help='YAML configuration file')
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(open(args.config))
    outdir = Path(cfg.get('output_dir', 'pylenslib_outputs'))
    outdir.mkdir(parents=True, exist_ok=True)

    clusters = cfg.get('clusters', [{'name': 'generated', 'M200': 1e15, 'c': 4.0, 'n_subhalos': 10}])
    n_real = int(cfg.get('n_realizations', 1))

    all_results = []
    for c_idx, cluster_cfg in enumerate(clusters):
        cluster_name = cluster_cfg.get('name', f'cluster_{c_idx}')
        cluster_out = outdir / cluster_name.replace(' ', '_')
        cluster_out.mkdir(parents=True, exist_ok=True)

        for r in range(n_real):
            real_out = cluster_out / f'realization_{r:04d}'
            real_out.mkdir(parents=True, exist_ok=True)
            summary = run_realization(cfg, cluster_cfg, r, real_out)
            all_results.append({'cluster': cluster_name, 'realization': r, **summary})

    # Save overall catalog
    with open(outdir / 'catalog_summary.json', 'w') as fh:
        json.dump({'results': all_results}, fh, indent=2)


if __name__ == '__main__':
    main()
