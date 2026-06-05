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
from scipy import ndimage

import mpi_utils
from agent_sim.core.models.cluster import ClusterGenerator
from agent_sim.core.models.galaxy import SourceDistribution
from agent_sim.pylenslib_imsim_branch.simulator import PyLenslibSimulator
from mpi_utils import get_mpi_info, distribute_list, mpi_barrier


MPI_INFO = get_mpi_info()
HAS_MPI = MPI_INFO['has_mpi']
RANK = MPI_INFO['rank']
SIZE = MPI_INFO['size']


def _rank_tag() -> str:
    return f"[rank {RANK}/{SIZE}]" if HAS_MPI and SIZE > 1 else "[serial]"


def _log(msg: str) -> None:
    print(f"{_rank_tag()} {msg}", flush=True)


def build_pylenslib_models(cluster_cfg: dict, source_z: float, seed: int = None):
    """Build explicit PyLensLib models from config (PIEMD/SIE/NFW)."""
    lens_cfg = cluster_cfg.get('pylenslib_models', None)
    if not lens_cfg:
        return None

    from astropy.cosmology import Planck18 as cosmo
    from pyLensLib.piemd import piemd
    from pyLensLib.sie import sie
    from pyLensLib.nfwell import nfwell

    zl = float(cluster_cfg.get('z', 0.3))
    rng = np.random.default_rng(seed)
    models = []

    # Main lens
    main = lens_cfg.get('main', None)
    if main:
        profile = str(main.get('profile', 'piemd')).lower()
        if profile == 'piemd':
            m = piemd(
                co=cosmo,
                zl=zl,
                zs=float(source_z),
                sigma0=float(main.get('sigma0', 240.0)),
                q=float(main.get('q', 0.75)),
                pa=float(main.get('pa', np.pi / 6.0)),
                theta_c=float(main.get('theta_c', 0.01)),
                theta_t=float(main.get('theta_t', 4.0)),
                x1=float(main.get('x1', 0.0)),
                x2=float(main.get('x2', 0.0)),
            )
            models.append(m)
        elif profile == 'nfwell':
            m = nfwell(
                co=cosmo,
                zl=zl,
                zs=float(source_z),
                mass=float(main.get('mass', cluster_cfg.get('M200', 1e15))),
                conc=float(main.get('conc', cluster_cfg.get('c', 4.0))),
                q=float(main.get('q', 1.0)),
                pa=float(main.get('pa', 0.0)),
                x1=float(main.get('x1', 0.0)),
                x2=float(main.get('x2', 0.0)),
            )
            models.append(m)

    # Satellites
    sats = lens_cfg.get('satellites', {})
    if sats:
        profile = str(sats.get('profile', 'sie')).lower()
        n_sat = int(sats.get('n', 0))
        r_sat = float(sats.get('radius_arcsec', 1.5))
        phase = float(sats.get('phase_offset', 0.35))
        angles = np.linspace(0, 2 * np.pi, n_sat, endpoint=False) + phase
        sigma0 = float(sats.get('sigma0', 60.0))
        q_min = float(sats.get('q_min', 0.85))
        q_max = float(sats.get('q_max', 0.95))
        for phi in angles:
            x1 = r_sat * np.cos(phi)
            x2 = r_sat * np.sin(phi)
            if profile == 'sie':
                models.append(
                    sie(
                        co=cosmo,
                        zl=zl,
                        zs=float(source_z),
                        sigma0=sigma0,
                        q=float(rng.uniform(q_min, q_max)),
                        pa=float(rng.uniform(0.0, np.pi)),
                        theta_c=float(sats.get('theta_c', 0.001)),
                        x1=float(x1),
                        x2=float(x2),
                    )
                )

    # DM subhalos
    dm = lens_cfg.get('dm_subhalos', {})
    if dm:
        n_dm = int(dm.get('n', 0))
        m_min = float(dm.get('mass_min', 5e8))
        m_max = float(dm.get('mass_max', 5e10))
        slope = float(dm.get('shmf_slope', -0.9))
        conc_min = float(dm.get('conc_min', 10.0))
        conc_max = float(dm.get('conc_max', 25.0))
        r_min = float(dm.get('r_min_arcsec', 0.2))
        r_max = float(dm.get('r_max_arcsec', float(cfg_safe(lens_cfg, 'fov_half_arcsec', 4.5))))

        shmf_s = slope + 1.0
        # Guard against shmf_s == 0 (which would cause division by zero).
        if abs(shmf_s) < 1e-6:
            import warnings
            warnings.warn('shmf_s computed as zero — substituting small epsilon to avoid division by zero')
            shmf_s = 1e-3
        u = rng.random(n_dm)
        masses = (u * (m_max**shmf_s - m_min**shmf_s) + m_min**shmf_s) ** (1.0 / shmf_s)
        masses = np.sort(masses)[::-1]
        concs = rng.uniform(conc_min, conc_max, n_dm)
        r = rng.uniform(r_min, r_max, n_dm)
        phi = rng.uniform(0.0, 2.0 * np.pi, n_dm)
        x1 = r * np.cos(phi)
        x2 = r * np.sin(phi)

        for i in range(n_dm):
            models.append(
                nfwell(
                    co=cosmo,
                    zl=zl,
                    zs=float(source_z),
                    mass=float(masses[i]),
                    conc=float(concs[i]),
                    q=1.0,
                    pa=0.0,
                    x1=float(x1[i]),
                    x2=float(x2[i]),
                )
            )

    return models


def cfg_safe(d: dict, key: str, default):
    val = d.get(key, default)
    return default if val is None else val


def detect_image_candidates(image, pixel_scale=0.2, threshold_rel=0.12, min_pixels=20, elongation_thresh=3.0):
    """Simple image-based arc candidate detector.

    Returns a tuple (masks, objects) where masks is a list of boolean arrays
    and objects is a list of dicts with properties and the mask.
    """
    objects = []
    masks = []
    if np.max(image) <= 0:
        return masks, objects

    thr = np.max(image) * threshold_rel
    binary = image > thr

    labeled, ncomp = ndimage.label(binary)
    for lab in range(1, ncomp + 1):
        comp = (labeled == lab)
        area = int(comp.sum())
        if area < min_pixels:
            continue

        ys, xs = np.nonzero(comp)
        if xs.size < 3:
            continue

        x_mean = xs.mean()
        y_mean = ys.mean()
        x_cent = xs - x_mean
        y_cent = ys - y_mean
        cov_xx = (x_cent * x_cent).mean()
        cov_yy = (y_cent * y_cent).mean()
        cov_xy = (x_cent * y_cent).mean()
        trace = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy * cov_xy
        eig1 = trace / 2.0 + np.sqrt(max(0.0, (trace * trace / 4.0 - det)))
        eig2 = trace / 2.0 - np.sqrt(max(0.0, (trace * trace / 4.0 - det)))
        if eig2 <= 0:
            elongation = 1.0
        else:
            elongation = np.sqrt(eig1 / eig2)

        flux = float(image[comp].sum())

        mask = comp
        masks.append(mask)
        objects.append({
            'mask': mask,
            'area_pixels': area,
            'elongation': float(elongation),
            'flux': flux,
        })

    # mark GGSL-like candidates
    for obj in objects:
        obj['is_candidate'] = obj['elongation'] >= elongation_thresh

    return masks, objects


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


def _build_noiseless_pylenslib_image(sim, sources_list, cfg: dict, cluster_cfg: dict, x_range, y_range,
                                     include_lens_light: bool = True):
    """Render the noiseless combined image before PSF/noise/cropping."""
    from pyLensLib.sersic import sersic
    from pyLensLib.observation import observation

    render_cfg = cfg.get('rendering', {})
    obs_cfg = render_cfg.get('observation', {})

    # Match test_perturbed_ring approach: high-resolution internal grid.
    npix_hires = int(render_cfg.get('npix_hires', cfg.get('n_pixels', 500)))
    sz = float(x_range[1] - x_range[0])
    pixel_scale_arcsec = sz / float(npix_hires)
    theta = np.linspace(x_range[0], x_range[1], npix_hires)

    composite = sim._build_pylenslib_composite()
    composite.setGrid(theta)

    zp = float(obs_cfg.get('zp', 23.9))
    texp = float(obs_cfg.get('texp', 565 * 4))
    mag_sky = float(obs_cfg.get('mag_sky', 22.5))
    ob = observation(size=sz, Npix=npix_hires, zp=zp, texp=texp, bkg=mag_sky)

    combined = np.zeros((npix_hires, npix_hires), dtype=float)

    src_n = float(render_cfg.get('source_n', 2.0))
    src_q = float(render_cfg.get('source_q', 0.9))
    src_pa = float(render_cfg.get('source_pa', 0.5))

    step = max(1, len(sources_list) // 5) if sources_list else 1
    for idx, source in enumerate(sources_list):
        if idx % step == 0 or idx == len(sources_list) - 1:
            _log(f"rendering source {idx + 1}/{len(sources_list)}")
        flux = ob.mag2counts(float(source.magnitude))
        kwargs_source = {
            'n': src_n,
            're': float(source.radius_arcsec),
            'q': src_q,
            'pa': src_pa,
            'ys1': float(source.source_position[0]),
            'ys2': float(source.source_position[1]),
            'flux': flux,
            'zs': float(source.redshift),
        }
        arc_image = sersic(size=sz, Npix=npix_hires, gl=composite, **kwargs_source)
        combined += arc_image.image

    if include_lens_light and bool(render_cfg.get('include_lens_light', True)):
        main_cfg = cluster_cfg.get('pylenslib_models', {}).get('main', {})
        lens_n = float(render_cfg.get('lens_light_n', 4.0))
        lens_re = float(render_cfg.get('lens_light_re', 1.5))
        lens_q = float(main_cfg.get('q', render_cfg.get('lens_light_q', 0.75)))
        lens_pa = float(main_cfg.get('pa', render_cfg.get('lens_light_pa', np.pi / 6.0)))
        lens_x = float(main_cfg.get('x1', 0.0))
        lens_y = float(main_cfg.get('x2', 0.0))
        lens_z = float(cluster_cfg.get('z', 0.3))
        kwargs_lens_light = {
            'n': lens_n,
            're': lens_re,
            'q': lens_q,
            'pa': lens_pa,
            'ys1': lens_x,
            'ys2': lens_y,
            'zs': lens_z,
        }
        lens_light_image = sersic(size=sz, Npix=npix_hires, gl=None, **kwargs_lens_light)
        combined += lens_light_image.image

    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    return combined, pixel_scale_arcsec, ob


def _apply_detector_and_crop(image: np.ndarray, cfg: dict, pixel_scale_arcsec: float, x_range, y_range):
    """Apply PSF, detector noise, and optional zoom crop."""
    render_cfg = cfg.get('rendering', {})
    obs_cfg = render_cfg.get('observation', {})

    # Apply PSF convolution if requested (psf_fwhm in arcsec)
    psf_fwhm = float(render_cfg.get('psf_fwhm', obs_cfg.get('psf_fwhm', 0.0)))
    if psf_fwhm and psf_fwhm > 0.0:
        sigma_pix = psf_fwhm / (2.355 * pixel_scale_arcsec)
        psf_img = ndimage.gaussian_filter(image, sigma=sigma_pix)
    else:
        psf_img = image.copy()

    # Realistic detector noise: sky background, Poisson, and read noise + gain
    from pyLensLib.observation import observation
    zp = float(obs_cfg.get('zp', 23.9))
    texp = float(obs_cfg.get('texp', 565 * 4))
    mag_sky = float(obs_cfg.get('mag_sky', 22.5))
    ob = observation(size=float(x_range[1] - x_range[0]), Npix=max(1, int(image.shape[0])), zp=zp, texp=texp, bkg=mag_sky)
    sky_counts_per_arcsec2 = ob.mag2counts(mag_sky)
    pixel_area = pixel_scale_arcsec * pixel_scale_arcsec
    sky_counts_per_pixel = sky_counts_per_arcsec2 * pixel_area

    readnoise_e = float(obs_cfg.get('readnoise', render_cfg.get('readnoise', 6.0)))
    gain = float(obs_cfg.get('gain', render_cfg.get('gain', 1.0)))
    readnoise_adu = readnoise_e / gain if gain > 0 else readnoise_e

    lam = np.clip(psf_img + sky_counts_per_pixel, 0.0, None)
    poisson = np.random.poisson(lam).astype(float)
    gauss = np.random.normal(0.0, readnoise_adu, size=psf_img.shape)
    noisy = poisson + gauss

    zoom_arcsec = float(render_cfg.get('zoom_arcsec', render_cfg.get('crop_arcsec', 0.0)))
    if zoom_arcsec and zoom_arcsec > 0.0:
        half = zoom_arcsec / 2.0
        center = image.shape[0] // 2
        half_pix = max(1, int((half / pixel_scale_arcsec)))
        imin = max(0, center - half_pix)
        imax = min(image.shape[0], center + half_pix)
        image_c = image[imin:imax, imin:imax]
        noisy_c = noisy[imin:imax, imin:imax]
        extent = (-half, half, -half, half)
    else:
        image_c = image
        noisy_c = noisy
        extent = (x_range[0], x_range[1], y_range[0], y_range[1])

    return image_c, noisy_c, extent


def _render_source_split(sim, sources_list, cfg: dict, cluster_cfg: dict, x_range, y_range):
    render_cfg = cfg.get('rendering', {})
    npix_hires = int(render_cfg.get('npix_hires', cfg.get('n_pixels', 500)))
    sz = float(x_range[1] - x_range[0])
    pixel_scale_arcsec = sz / float(npix_hires)

    local_indices = distribute_list(list(range(len(sources_list))))
    _log(f"source split active: {len(local_indices)}/{len(sources_list)} sources assigned")
    local_sources = [sources_list[i] for i in local_indices]

    local_image, _, _ = _build_noiseless_pylenslib_image(sim, local_sources, cfg, cluster_cfg, x_range, y_range,
                                                         include_lens_light=False)
    if HAS_MPI and SIZE > 1:
        combined = mpi_utils.COMM.allreduce(local_image)
    else:
        combined = local_image

    return combined, pixel_scale_arcsec


def _render_image_split(sim, sources_list, cfg: dict, cluster_cfg: dict, x_range, y_range):
    render_cfg = cfg.get('rendering', {})
    npix_hires = int(render_cfg.get('npix_hires', cfg.get('n_pixels', 500)))
    if not (HAS_MPI and SIZE > 1):
        _log("image split requested but MPI is unavailable; falling back to serial mode")
        return None, float((x_range[1] - x_range[0]) / float(npix_hires)), False

    side = int(round(np.sqrt(SIZE)))
    if side * side != SIZE:
        _log(f"image split requires a square MPI size; got {SIZE}")
        return None, float((x_range[1] - x_range[0]) / float(npix_hires)), False
    if npix_hires % side != 0:
        _log(f"image split requires npix_hires divisible by sqrt(size)={side}; got {npix_hires}")
        return None, float((x_range[1] - x_range[0]) / float(npix_hires)), False

    tile_npix = npix_hires // side
    row = RANK // side
    col = RANK % side
    x_span = float(x_range[1] - x_range[0])
    y_span = float(y_range[1] - y_range[0])
    x0 = x_range[0] + col * (x_span / side)
    x1 = x_range[0] + (col + 1) * (x_span / side)
    y0 = y_range[0] + row * (y_span / side)
    y1 = y_range[0] + (row + 1) * (y_span / side)
    _log(f"image split active: tile {RANK + 1}/{SIZE} -> row {row}, col {col}")

    # Render the tile at the same scale as the full image by using a matching
    # high-resolution grid per tile.
    tile_cfg = dict(cfg)
    tile_render = dict(render_cfg)
    tile_render['npix_hires'] = tile_npix
    tile_render['zoom_arcsec'] = 0.0
    tile_cfg['rendering'] = tile_render

    tile_image, _, _ = _build_noiseless_pylenslib_image(sim, sources_list, tile_cfg, cluster_cfg, (x0, x1), (y0, y1),
                                                        include_lens_light=True)
    payload = {'row': row, 'col': col, 'image': tile_image}
    gathered = mpi_utils.COMM.gather(payload, root=0)

    if RANK != 0:
        return None, float(x_span / float(npix_hires)), True

    full = np.zeros((npix_hires, npix_hires), dtype=float)
    for item in gathered:
        if not item:
            continue
        r = int(item['row'])
        c = int(item['col'])
        img = np.asarray(item['image'])
        y_start = r * tile_npix
        y_end = y_start + img.shape[0]
        x_start = c * tile_npix
        x_end = x_start + img.shape[1]
        full[y_start:y_end, x_start:x_end] = img

    return full, float(x_span / float(npix_hires)), False


def direct_pylenslib_render(sim, sources_list, cfg: dict, cluster_cfg: dict, x_range, y_range, split_mode: str = 'none'):
    """Render lensed image directly with PyLensLib sersic + observation.

    Returns: (lensed_images, combined_noiseless, combined_noisy, extent)
    """
    render_cfg = cfg.get('rendering', {})
    split_mode = str(split_mode or render_cfg.get('mpi_split', 'none')).lower()

    if split_mode in ('image', 'tile', 'region'):
        combined, pixel_scale_arcsec, worker_only = _render_image_split(sim, sources_list, cfg, cluster_cfg, x_range, y_range)
        if worker_only:
            return [], None, None, None
        if combined is not None:
            combined_c, noisy_c, extent = _apply_detector_and_crop(combined, cfg, pixel_scale_arcsec, x_range, y_range)
            return [], combined_c, noisy_c if RANK == 0 or not HAS_MPI or SIZE == 1 else None, extent

    if split_mode in ('source', 'sources'):
        combined, pixel_scale_arcsec = _render_source_split(sim, sources_list, cfg, cluster_cfg, x_range, y_range)
    else:
        combined, pixel_scale_arcsec, _ = _build_noiseless_pylenslib_image(sim, sources_list, cfg, cluster_cfg, x_range, y_range,
                                          include_lens_light=True)

    combined_c, noisy_c, extent = _apply_detector_and_crop(combined, cfg, pixel_scale_arcsec, x_range, y_range)
    if HAS_MPI and SIZE > 1 and split_mode in ('source', 'sources') and RANK != 0:
        noisy_c = None
    return [], combined_c, noisy_c, extent

    # Apply PSF convolution if requested (psf_fwhm in arcsec)
    psf_fwhm = float(render_cfg.get('psf_fwhm', obs_cfg.get('psf_fwhm', 0.0)))
    pixel_scale_arcsec = sz / float(npix_hires)
    if psf_fwhm and psf_fwhm > 0.0:
        sigma_pix = psf_fwhm / (2.355 * pixel_scale_arcsec)
        psf_img = ndimage.gaussian_filter(combined, sigma=sigma_pix)
    else:
        psf_img = combined.copy()

    # Realistic detector noise: sky background, Poisson, and read noise + gain
    mag_sky = float(obs_cfg.get('mag_sky', 22.5))
    sky_counts_per_arcsec2 = ob.mag2counts(mag_sky)
    pixel_area = pixel_scale_arcsec * pixel_scale_arcsec
    sky_counts_per_pixel = sky_counts_per_arcsec2 * pixel_area

    readnoise_e = float(obs_cfg.get('readnoise', render_cfg.get('readnoise', 6.0)))
    gain = float(obs_cfg.get('gain', render_cfg.get('gain', 1.0)))
    readnoise_adu = readnoise_e / gain if gain > 0 else readnoise_e

    # Poisson noise on signal+sky (ensure non-negative)
    lam = np.clip(psf_img + sky_counts_per_pixel, 0.0, None)
    poisson = np.random.poisson(lam).astype(float)
    gauss = np.random.normal(0.0, readnoise_adu, size=psf_img.shape)
    noisy = poisson + gauss

    # Optionally zoom (crop) around center in arcsec
    zoom_arcsec = float(render_cfg.get('zoom_arcsec', render_cfg.get('crop_arcsec', 0.0)))
    if zoom_arcsec and zoom_arcsec > 0.0:
        half = zoom_arcsec / 2.0
        # Compute pixel indices for crop
        center = npix_hires // 2
        half_pix = max(1, int((half / pixel_scale_arcsec)))
        imin = max(0, center - half_pix)
        imax = min(npix_hires, center + half_pix)
        combined_c = combined[imin:imax, imin:imax]
        noisy_c = noisy[imin:imax, imin:imax]
        extent = (-half, half, -half, half)
    else:
        combined_c = combined
        noisy_c = noisy
        extent = (x_range[0], x_range[1], y_range[0], y_range[1])

    return [], combined_c, noisy_c, extent


def run_realization(cfg: dict, cluster_cfg: dict, realization_idx: int, outdir: Path):
    """Run one realization and return summary dict."""
    outdir.mkdir(parents=True, exist_ok=True)
    mpi_cfg = cfg.get('mpi', {})
    split_mode = str(mpi_cfg.get('split_mode', cfg.get('rendering', {}).get('mpi_split', 'none'))).lower()

    _log(f"starting realization {realization_idx:04d} for cluster {cluster_cfg.get('name', 'unknown')} (split_mode={split_mode})")

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

    # Set simulator redshifts for consistent lensing calculations
    sim.default_source_z = float(src_cfg.get('redshift', 1.0))
    sim.default_lens_z = float(cluster_cfg.get('z', 0.3))

    # Optional explicit PyLensLib model stack (PIEMD/SIE/NFW)
    explicit_models = build_pylenslib_models(
        cluster_cfg,
        source_z=float(src_cfg.get('redshift', 1.0)),
        seed=(seed + realization_idx) if seed is not None else None,
    )
    if explicit_models:
        sim.set_pylenslib_models(
            explicit_models,
            lens_z=float(cluster_cfg.get('z', 0.3)),
            source_z=float(src_cfg.get('redshift', 1.0)),
        )
        # Explicit PyLensLib models are defined in arcsec in config.
        sim.cluster_positions_in_kpc = False

    # Generate mock images (perfect noiseless combined image + per-source images)
    x_range = tuple(cfg.get('x_range', (-150, 150)))
    y_range = tuple(cfg.get('y_range', (-150, 150)))
    n_pixels = int(cfg.get('n_pixels', 200))
    image_size = int(cfg.get('image_size', 256))
    pixel_scale = float(cfg.get('pixel_scale', 0.2))

    # Create SourcePlacement objects compatible with the simulator's imaging API
    from agent_sim.pylenslib_imsim_branch.imaging import SourcePlacement
    sources_list = []
    explicit_sources = src_cfg.get('explicit', [])
    if explicit_sources:
        src_positions = []
        for i, s_cfg in enumerate(explicit_sources):
            pos = (float(s_cfg.get('x', 0.0)), float(s_cfg.get('y', 0.0)))
            src_positions.append(pos)
            s = SourcePlacement(
                source_id=i,
                source_position=pos,
                redshift=float(src_cfg.get('redshift', 1.0)),
                magnitude=float(s_cfg.get('magnitude', src_cfg.get('default_magnitude', 23.5))),
                radius_arcsec=float(s_cfg.get('radius_arcsec', src_cfg.get('default_radius_arcsec', 0.08))),
            )
            sources_list.append(s)
        n_sources = len(sources_list)
    else:
        for i, pos in enumerate(src_positions):
            s = SourcePlacement(
                source_id=i,
                source_position=tuple(pos),
                redshift=src_cfg.get('redshift', 1.0),
                magnitude=float(sd.sample_luminosities(1)[0]),
                radius_arcsec=float(sd.sample_sizes(1)[0])
            )
            sources_list.append(s)

    render_backend = str(cfg.get('rendering', {}).get('backend', 'simulator')).lower()
    if render_backend == 'pylenslib_direct':
        lensed_images, combined, noisy, extent = direct_pylenslib_render(
            sim, sources_list, cfg, cluster_cfg, x_range=x_range, y_range=y_range, split_mode=split_mode
        )
    else:
        lensed_images, combined = sim.generate_mock_images(
            sources_list,
            x_range=x_range,
            y_range=y_range,
            image_size=image_size,
            pixel_scale=pixel_scale
        )

    if HAS_MPI and SIZE > 1:
        mpi_utils.mpi_barrier()
        if RANK != 0:
            return {
                'cluster_cfg': cluster_cfg,
                'n_sources': n_sources,
                'status': 'mpi_worker',
                'mpi_rank': RANK,
            }

    _log(f"render complete for realization {realization_idx:04d}; proceeding with detection and output")

    # Run the simulator's GGSL detector on the source catalog (grid-based ray tracing)
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

    with open(outdir / f'events_realization_{realization_idx:04d}.json', 'w') as fh:
        json.dump({'events': [e.__dict__ for e in events], 'summary': summary_out}, fh, indent=2)

    # Save magnification and convergence maps
    maps = sim.compute_lensing_maps(x_range=x_range, y_range=y_range, n_pixels=n_pixels)
    save_fits(maps['magnification'], outdir / f'magnification_real_{realization_idx:04d}.fits')
    save_png(maps['magnification'], outdir / f'magnification_real_{realization_idx:04d}.png')
    # Convergence (kappa) is useful for diagnostics; save if present
    if 'convergence' in maps:
        try:
            save_fits(maps['convergence'], outdir / f'convergence_real_{realization_idx:04d}.fits')
        except Exception:
            pass

    # Post-processing: image-based detection and diagnostic plot overlaying detections
    detection_cfg = cfg.get('detection', {})
    thr_rel = float(detection_cfg.get('threshold_rel', 0.12))
    min_pixels = int(detection_cfg.get('min_pixels', 20))
    elongation_thresh = float(detection_cfg.get('elongation', 3.0))

    img_candidates, img_objects = detect_image_candidates(combined, pixel_scale=pixel_scale,
                                                          threshold_rel=thr_rel,
                                                          min_pixels=min_pixels,
                                                          elongation_thresh=elongation_thresh)

    # Create diagnostic figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # extent returned by renderer may reflect zoom/crop; use it and label units (arcsec)
    try:
        extent
    except NameError:
        extent = (x_range[0], x_range[1], y_range[0], y_range[1])

    im0 = axes[0, 0].imshow(combined, origin='lower', cmap='inferno', extent=extent)
    axes[0, 0].set_title('Lensed image (noiseless; counts / pixel)')
    axes[0, 0].set_xlabel('arcsec')
    axes[0, 0].set_ylabel('arcsec')
    fig.colorbar(im0, ax=axes[0,0], fraction=0.046, label='counts / pixel')

    # Arcs only (segmented mask overlay)
    arcs_mask = np.zeros_like(combined)
    for obj in img_objects:
        arcs_mask[obj['mask']] = 1
    im1 = axes[0, 1].imshow(combined * arcs_mask, origin='lower', cmap='inferno', extent=extent)
    axes[0, 1].set_title('Lensed arcs only (counts / pixel)')
    axes[0, 1].set_xlabel('arcsec')
    axes[0, 1].set_ylabel('arcsec')
    fig.colorbar(im1, ax=axes[0,1], fraction=0.046, label='counts / pixel')

    # Plot log10 of the convergence (kappa). Mask non-positive values before log10.
    try:
        conv = maps.get('convergence')
        conv_log10 = np.where(conv > 0, np.log10(conv), np.nan)
    except Exception:
        conv_log10 = np.full_like(combined, np.nan)
    im2 = axes[0, 2].imshow(conv_log10, origin='lower', cmap='viridis', extent=extent)
    axes[0, 2].set_title('Convergence (log10 kappa)')
    axes[0, 2].set_xlabel('arcsec')
    axes[0, 2].set_ylabel('arcsec')
    cbar2 = fig.colorbar(im2, ax=axes[0,2], fraction=0.046)
    cbar2.set_label('log10(Convergence, kappa)')

    # Noisy image (from renderer if available)
    try:
        noisy_img = noisy
    except NameError:
        noise = np.random.normal(0, 0.02 * np.std(combined), combined.shape)
        noisy_img = combined + noise
    im3 = axes[1, 0].imshow(noisy_img, origin='lower', cmap='inferno', extent=extent)
    axes[1, 0].set_title('Final image (noise; counts)')
    axes[1, 0].set_xlabel('arcsec')
    axes[1, 0].set_ylabel('arcsec')
    fig.colorbar(im3, ax=axes[1,0], fraction=0.046, label='counts')

    # PSF-convolved + noise (renderer already applied PSF before producing noisy)
    im4 = axes[1, 1].imshow(noisy_img, origin='lower', cmap='inferno', extent=extent)
    axes[1, 1].set_title('PSF-convolved + noise (counts)')
    axes[1, 1].set_xlabel('arcsec')
    axes[1, 1].set_ylabel('arcsec')
    fig.colorbar(im4, ax=axes[1,1], fraction=0.046, label='counts')

    # DM subhalo positions
    submap = np.zeros_like(combined)
    # Cluster generator may return positions in physical kpc; convert to arcsec for plotting
    try:
        from astropy.cosmology import Planck18 as cosmo
        import astropy.units as u
        zl = float(cluster_cfg.get('z', 0.5))
        DA_kpc = cosmo.angular_diameter_distance(zl).to(u.kpc).value
        for (sx, sy) in cluster.subhalo_positions:
            if getattr(sim, 'cluster_positions_in_kpc', True):
                sx_arc = (sx / DA_kpc) * 206265.0
                sy_arc = (sy / DA_kpc) * 206265.0
            else:
                sx_arc = sx
                sy_arc = sy
            axes[1, 2].scatter(sx_arc, sy_arc, c='white', marker='+')
    except Exception:
        for (sx, sy) in cluster.subhalo_positions:
            axes[1, 2].scatter(sx, sy, c='white', marker='+')
    axes[1, 2].set_xlabel('arcsec')
    axes[1, 2].set_ylabel('arcsec')
    im5 = axes[1, 2].imshow(submap, origin='lower', cmap='plasma', extent=extent)
    axes[1, 2].set_title('DM subhalo positions')
    fig.colorbar(im5, ax=axes[1,2], fraction=0.046, label='subhalo marker map')

    # Overlay detected image-based candidate contours (green) and simulator events (blue)
    for obj in img_objects:
        mask = obj['mask']
        # find contours by marching squares via ndimage.find_objects or use binary_dilation to get edges
        edges = ndimage.binary_dilation(mask) ^ mask
        ys, xs = np.nonzero(edges)
        if len(xs) > 0:
            # compute fov from extent (supports zoom/crop)
            xmin, xmax, ymin, ymax = extent
            fov_x = float(xmax - xmin)
            fov_y = float(ymax - ymin)
            xvals = xmin + (xs / combined.shape[1]) * fov_x
            yvals = ymin + (ys / combined.shape[0]) * fov_y
            axes[0, 0].plot(xvals, yvals, '.', color='lime', markersize=0.5)

    # Overlay simulator-detected event image positions
    for ev in events:
        # events may have image positions in ev.__dict__['caustic_properties'] or 'image_positions'
        img_pos = getattr(ev, 'caustic_properties', None)
        if hasattr(ev, 'arc_length_arcsec'):
            # mark source id at center (if available)
            axes[0, 0].scatter([], [])
        # If event has magnifications / image positions, attempt to plot
        try:
            if ev.__dict__.get('n_images', 0) > 0 and ev.__dict__.get('caustic_properties'):
                pass
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(outdir / f'diagnostic_real_{realization_idx:04d}.png', dpi=150)
    plt.close()

    # Save rendered images (perfect + noisy) when available
    try:
        # `combined` is the noiseless perfect image; `noisy_img` is the final noisy image
        save_fits(combined if 'combined' in locals() else np.zeros((1,1)), outdir / f'image_perfect_real_{realization_idx:04d}.fits')
    except Exception:
        pass
    try:
        save_fits(noisy_img if 'noisy_img' in locals() else np.zeros((1,1)), outdir / f'image_noisy_real_{realization_idx:04d}.fits')
    except Exception:
        pass
    plt.close()

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

    if HAS_MPI and SIZE > 1 and RANK != 0:
        return

    # Save overall catalog
    with open(outdir / 'catalog_summary.json', 'w') as fh:
        json.dump({'results': all_results}, fh, indent=2)


if __name__ == '__main__':
    main()
