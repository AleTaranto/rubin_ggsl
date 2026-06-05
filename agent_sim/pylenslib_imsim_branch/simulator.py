"""
PyLensLib + ImSim Simulation Branch

This branch uses:
- PyLensLib for computing lensing maps from density profiles
- ImSim for applying Rubin-like observational effects (PSF, noise, detector)

Workflow:
1. Define cluster + subhalo configuration
2. Use PyLensLib to compute deflection, magnification, critical lines
3. Generate perfect lensed images of background sources
4. Use ImSim to Rubinize (apply realistic effects)
5. Detect and measure GGSL events
"""

import numpy as np
from agent_sim.core import models, lensing, observables, utils


class PyLenslibSimulator:
    """
    Simulator using PyLensLib for lensing computations.
    """
    
    def __init__(self, cluster, grid_size=None):
        """
        Initialize PyLensLib simulator.
        
        Parameters
        ----------
        cluster : Cluster
            Cluster + subhalo configuration
        grid_size : float, optional
            Grid spacing for lensing computations (arcsec or kpc)
        """
        self.cluster = cluster
        self.grid_size = grid_size or 0.05
        
        # Initialize lensing calculations
        self.deflection = lensing.DeflectionField(cluster, grid_size=grid_size)
        self.magnification = lensing.Magnification(cluster, grid_size=grid_size)
        self.critical_lines = lensing.CriticalLines(self.magnification)
        
        # GGSL observables
        self.ggsl_defs = observables.GGSLDefinitions(
            min_magnification=2.0,
            min_arc_length_arcsec=1.0
        )
        self.ggsl_counter = observables.GGSLCounter(
            self.deflection, self.magnification, self.ggsl_defs
        )

        # Optional direct PyLensLib model override (e.g., PIEMD+SIE+NFW)
        self.pylenslib_models = None

    def set_pylenslib_models(self, models, lens_z, source_z):
        """Set explicit PyLensLib models to use for maps and imaging."""
        self.pylenslib_models = {
            'models': list(models),
            'lens_z': float(lens_z),
            'source_z': float(source_z),
        }

    def _build_pylenslib_composite(self, source_z=None, lens_z=None):
        """Build a PyLensLib composite model from explicit models or cluster."""
        from astropy.cosmology import Planck18 as cosmo
        from pyLensLib.compositeModel import compositeModel
        from pyLensLib.nfwell import nfwell

        # Explicit config-driven model path
        if self.pylenslib_models is not None:
            # Clone model instances because some PyLensLib models cache
            # internal arrays tied to the first evaluated grid shape.
            import copy
            models = copy.deepcopy(self.pylenslib_models['models'])
            lens_z_eff = self.pylenslib_models['lens_z']
            source_z_eff = self.pylenslib_models['source_z']
        else:
            # Determine lens and source redshifts
            source_z_eff = source_z if source_z is not None else getattr(self, 'default_source_z', None)
            lens_z_eff = lens_z if lens_z is not None else getattr(self, 'default_lens_z', None)
            if lens_z_eff is None:
                lens_z_eff = getattr(self.cluster.cluster_halo, 'redshift', None) or 0.3
            if source_z_eff is None:
                source_z_eff = lens_z_eff + 0.5

            # conversion: cluster positions may be in kpc (ClusterGenerator) or arcsec
            arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(lens_z_eff).value
            positions_in_kpc = getattr(self, 'cluster_positions_in_kpc', None)
            if positions_in_kpc is None:
                max_coord = 0.0
                try:
                    for p in getattr(self.cluster, 'subhalo_positions', []):
                        max_coord = max(max_coord, abs(p[0]), abs(p[1]))
                    for p in getattr(self.cluster, 'galaxy_positions', []):
                        max_coord = max(max_coord, abs(p[0]), abs(p[1]))
                except Exception:
                    max_coord = 0.0
                positions_in_kpc = (max_coord > 50.0)

            models = []

            # Main halo
            main = self.cluster.cluster_halo
            main_mass = getattr(main, 'M200', getattr(main, 'mass', None))
            main_conc = getattr(main, 'c', getattr(main, 'conc', None))
            main_x = getattr(main, 'x', 0.0)
            main_y = getattr(main, 'y', 0.0)
            if positions_in_kpc:
                main_x_as = float(main_x) * arcsec_per_kpc
                main_y_as = float(main_y) * arcsec_per_kpc
            else:
                main_x_as = float(main_x)
                main_y_as = float(main_y)

            mkw = dict(
                co=cosmo,
                mass=main_mass if main_mass is not None else 1e15,
                conc=main_conc if main_conc is not None else 4.0,
                x1=main_x_as,
                x2=main_y_as,
                zl=lens_z_eff,
                zs=source_z_eff,
            )
            models.append(nfwell(**mkw))

            # Subhalos
            for i, sub in enumerate(self.cluster.subhalos):
                try:
                    x_kpc, y_kpc = self.cluster.subhalo_positions[i]
                except Exception:
                    x_kpc, y_kpc = 0.0, 0.0
                if positions_in_kpc:
                    x_as = float(x_kpc) * arcsec_per_kpc
                    y_as = float(y_kpc) * arcsec_per_kpc
                else:
                    x_as = float(x_kpc)
                    y_as = float(y_kpc)
                sub_mass = getattr(sub, 'M200', getattr(sub, 'mass', None))
                sub_conc = getattr(sub, 'c', getattr(sub, 'conc', None))
                skw = dict(
                    co=cosmo,
                    mass=sub_mass if sub_mass is not None else 1e10,
                    conc=sub_conc if sub_conc is not None else 4.0,
                    x1=x_as,
                    x2=y_as,
                    zl=lens_z_eff,
                    zs=source_z_eff,
                )
                models.append(nfwell(**skw))

            # Optional galaxies approximated as compact NFW halos
            for i, gal in enumerate(self.cluster.galaxies):
                try:
                    x_g, y_g = self.cluster.galaxy_positions[i]
                except Exception:
                    x_g, y_g = 0.0, 0.0
                if positions_in_kpc:
                    x_g_as = float(x_g) * arcsec_per_kpc
                    y_g_as = float(y_g) * arcsec_per_kpc
                else:
                    x_g_as = float(x_g)
                    y_g_as = float(y_g)
                gal_mass = gal.get('mass', 1e10)
                gkw = dict(
                    co=cosmo,
                    mass=gal_mass,
                    conc=10.0,
                    x1=x_g_as,
                    x2=y_g_as,
                    zl=lens_z_eff,
                    zs=source_z_eff,
                )
                models.append(nfwell(**gkw))

        composite = compositeModel(co=cosmo, models=models)
        composite.zl = float(lens_z_eff)
        composite.zs = float(source_z_eff)
        composite.dl = cosmo.angular_diameter_distance(composite.zl)
        composite.ds = cosmo.angular_diameter_distance(composite.zs)
        composite.dls = cosmo.angular_diameter_distance_z1z2(composite.zl, composite.zs)
        return composite
    
    def compute_lensing_maps(self, x_range=None, y_range=None, n_pixels=None):
        """
        Compute lensing maps (deflection, magnification) on a grid.
        
        Parameters
        ----------
        x_range : tuple, optional
            (x_min, x_max) in arcsec or kpc
        y_range : tuple, optional
            (y_min, y_max)
        n_pixels : int, optional
            Number of pixels per side (default: 100)
        
        Returns
        -------
        lensing_maps : dict
            Dictionary containing:
            - 'x_grid', 'y_grid': coordinate arrays
            - 'deflection_x', 'deflection_y': deflection field
            - 'magnification': magnification map
            - 'convergence': convergence map
        """
        if x_range is None:
            x_range = (-100, 100)
        if y_range is None:
            y_range = (-100, 100)
        if n_pixels is None:
            n_pixels = 100
        
        # Create grid
        x_grid = np.linspace(x_range[0], x_range[1], n_pixels)
        y_grid = np.linspace(y_range[0], y_range[1], n_pixels)

        # Build a PyLensLib composite model and use its kappa/gamma/angle
        # methods to compute lensing maps (see test_perturbed_ring.py).
        try:
            composite = self._build_pylenslib_composite()
        except Exception:
            # If PyLensLib / astropy missing, fall back to previous simple result
            return {
                'x_grid': x_grid,
                'y_grid': y_grid,
                'deflection_x': np.zeros((n_pixels, n_pixels)),
                'deflection_y': np.zeros((n_pixels, n_pixels)),
                'magnification': np.ones((n_pixels, n_pixels)),
                'convergence': np.zeros((n_pixels, n_pixels)),
            }

        # Create meshgrid in angular coordinates (arcsec)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Compute convergence kappa and shear (gamma1, gamma2)
        with np.errstate(all='ignore'):
            kappa_grid = composite.kappa(X, Y)
            gamma1_grid, gamma2_grid = composite.gamma(X, Y)

            # Compute magnification: mu = 1 / ((1 - kappa)^2 - |gamma|^2)
            denom = (1.0 - kappa_grid)**2 - (gamma1_grid**2 + gamma2_grid**2)
            mu_grid = np.ones_like(denom)
            mask = (denom != 0)
            mu_grid[mask] = 1.0 / denom[mask]
            mu_grid[~mask] = np.nan

            # Deflection angles (alpha) in arcsec from composite.angle
            alpha_x, alpha_y = composite.angle(X, Y)

        return {
            'x_grid': x_grid,
            'y_grid': y_grid,
            'deflection_x': alpha_x,
            'deflection_y': alpha_y,
            'magnification': mu_grid,
            'convergence': kappa_grid,
        }
    
    def find_critical_lines(self, x_range=None, y_range=None, n_pixels=None):
        """
        Find critical lines (where magnification diverges).
        
        Returns
        -------
        critical_info : dict
            - 'critical_mask': boolean mask of critical regions
            - 'caustics': list of caustic curves
            - 'properties': properties of each caustic
        """
        if x_range is None:
            x_range = (-100, 100)
        if y_range is None:
            y_range = (-100, 100)
        if n_pixels is None:
            n_pixels = 100
        
        x_grid = np.linspace(x_range[0], x_range[1], n_pixels)
        y_grid = np.linspace(y_range[0], y_range[1], n_pixels)
        
        # Find critical lines
        critical_mask, mu_grid = self.critical_lines.find_critical_lines(
            x_grid, y_grid
        )
        
        # Find caustics
        caustics = self.critical_lines.find_caustics(x_grid, y_grid, source_plane=True)
        
        # Characterize each caustic
        caustic_props = [
            self.critical_lines.characterize_critical_line(c) 
            for c in caustics
        ]
        
        return {
            'critical_mask': critical_mask,
            'caustics': caustics,
            'properties': caustic_props,
            'x_grid': x_grid,
            'y_grid': y_grid,
            'magnification_grid': mu_grid,
        }
    
    def detect_ggsl_events(self, sources, x_range=None, y_range=None, n_pixels=None,
                          source_z=1.0, lens_z=0.3):
        """
        Detect GGSL events for a source catalog.
        
        Parameters
        ----------
        sources : ndarray of shape (n_sources, 2)
            Source positions
        x_range, y_range : tuple, optional
            Coordinate ranges
        n_pixels : int, optional
            Number of grid pixels
        source_z : float
            Source redshift
        lens_z : float
            Lens redshift
        
        Returns
        -------
        events : list of GGSLObservable
            Detected GGSL events
        summary : dict
            Summary statistics
        """
        if x_range is None:
            x_range = (-100, 100)
        if y_range is None:
            y_range = (-100, 100)
        if n_pixels is None:
            n_pixels = 100
        
        x_grid = np.linspace(x_range[0], x_range[1], n_pixels)
        y_grid = np.linspace(y_range[0], y_range[1], n_pixels)
        
        # Detect events
        events = self.ggsl_counter.detect_events(
            sources, x_grid, y_grid, source_z=source_z, lens_z=lens_z
        )
        
        # Summary
        summary = self.ggsl_counter.summary()
        
        return events, summary
    
    def generate_mock_images(self, sources_list, x_range=None, y_range=None, 
                            image_size=256, pixel_scale=0.2):
        """
        Generate mock lensed images from sources.
        
        Parameters
        ----------
        sources_list : list of SourcePlacement
            Placed sources
        x_range, y_range : tuple, optional
            Image coordinate ranges
        image_size : int
            Image size (pixels)
        pixel_scale : float
            Pixel scale (arcsec/pixel)
        
        Returns
        -------
        lensed_images : list of LensedImage
            Generated images
        combined_image : ndarray
            Combined image array
        """
        from agent_sim.pylenslib_imsim_branch.imaging import (
            SourcePlacer, ImageGenerator
        )
        
        if x_range is None:
            x_range = (-150, 150)
        if y_range is None:
            y_range = (-150, 150)
        
        # Prefer a PyLensLib-backed adapter so image generation uses the same
        # lens model as compute_lensing_maps.
        class _PyLensLibDeflectionAdapter:
            def __init__(self, composite):
                self.composite = composite

            def deflection_angle(self, X, Y):
                return self.composite.angle(X, Y)

            def compute_deflection_grid(self, x_grid, y_grid):
                X, Y = np.meshgrid(x_grid, y_grid)
                return self.composite.angle(X, Y)

        class _PyLensLibMagnificationAdapter:
            def __init__(self, composite):
                self.composite = composite

            def magnification(self, X, Y, critical_density=1.0):
                kappa = self.composite.kappa(X, Y)
                g1, g2 = self.composite.gamma(X, Y)
                denom = (1.0 - kappa)**2 - (g1**2 + g2**2)
                denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
                return 1.0 / denom

            def compute_magnification_grid(self, x_grid, y_grid, critical_density=1.0):
                X, Y = np.meshgrid(x_grid, y_grid)
                return self.magnification(X, Y)

        try:
            composite = self._build_pylenslib_composite()
            deflection_field = _PyLensLibDeflectionAdapter(composite)
            magnification_calc = _PyLensLibMagnificationAdapter(composite)
        except Exception:
            deflection_field = self.deflection
            magnification_calc = self.magnification

        # Initialize generators
        source_placer = SourcePlacer(deflection_field, magnification_calc)
        image_gen = ImageGenerator(deflection_field, magnification_calc, source_placer)
        
        # Generate images
        lensed_images, combined = image_gen.generate_mock_images(
            sources_list,
            x_range=x_range,
            y_range=y_range,
            image_size=image_size,
            pixel_scale=pixel_scale
        )
        
        return lensed_images, combined
