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
        
        # Compute maps
        alpha_x, alpha_y = self.deflection.compute_deflection_grid(x_grid, y_grid)
        mu_grid = self.magnification.compute_magnification_grid(x_grid, y_grid)
        
        # Create coordinate mesh for convergence
        X, Y = np.meshgrid(x_grid, y_grid)
        kappa_grid = self.magnification.convergence(X, Y)
        
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
        
        # Initialize generators
        source_placer = SourcePlacer(self.deflection, self.magnification)
        image_gen = ImageGenerator(self.deflection, self.magnification, source_placer)
        
        # Generate images
        lensed_images, combined = image_gen.generate_mock_images(
            sources_list,
            x_range=x_range,
            y_range=y_range,
            image_size=image_size,
            pixel_scale=pixel_scale
        )
        
        return lensed_images, combined


# Import numpy for grid operations
import numpy as np
from agent_sim.core import models, lensing, observables, utils
