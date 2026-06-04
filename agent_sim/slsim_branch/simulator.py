"""
SLSim Simulation Branch

This branch uses:
- SLSim (Strong Lensing Simulation) for end-to-end simulation
- Unified interface for lensing computations and image generation
- Built-in support for Rubin observational effects

Workflow:
1. Define cluster + subhalo using SLSim interfaces
2. SLSim computes lensing and generates images
3. Apply Rubin effects via SLSim
4. Detect and measure GGSL events
"""

from agent_sim.core import models, lensing, observables, utils


class SlsimSimulator:
    """
    Simulator using SLSim for end-to-end simulations.
    
    Note: This is a wrapper/adapter that translates AGENT_sim cluster models
    to SLSim format and provides consistent interface with PyLensLib branch.
    """
    
    def __init__(self, cluster, grid_size=None):
        """
        Initialize SLSim simulator.
        
        Parameters
        ----------
        cluster : Cluster
            Cluster + subhalo configuration
        grid_size : float, optional
            Grid spacing for lensing computations
        """
        self.cluster = cluster
        self.grid_size = grid_size or 0.05
        
        # Initialize lensing calculations (using shared core for now)
        # In production, would interface with SLSim directly
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
        Compute lensing maps using SLSim.
        
        Parameters
        ----------
        x_range : tuple, optional
            (x_min, x_max) in arcsec or kpc
        y_range : tuple, optional
            (y_min, y_max)
        n_pixels : int, optional
            Number of pixels per side
        
        Returns
        -------
        lensing_maps : dict
            Dictionary containing lensing quantities
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
        
        # Compute maps (using shared core)
        alpha_x, alpha_y = self.deflection.compute_deflection_grid(x_grid, y_grid)
        mu_grid = self.magnification.compute_magnification_grid(x_grid, y_grid)
        
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
    
    def generate_perfect_images(self, sources, x_range=None, y_range=None, n_pixels=None):
        """
        Generate perfect (noise-free) lensed images using SLSim.
        
        Parameters
        ----------
        sources : ndarray or SourceDistribution
            Background source properties
        x_range, y_range : tuple, optional
            Coordinate ranges (arcsec)
        n_pixels : int, optional
            Image resolution (pixels per side)
        
        Returns
        -------
        images : dict
            - 'image': 2D image array
            - 'x_grid', 'y_grid': coordinate grids
            - 'sources': source list used
        """
        if x_range is None:
            x_range = (-100, 100)
        if y_range is None:
            y_range = (-100, 100)
        if n_pixels is None:
            n_pixels = 256
        
        # Placeholder: would call SLSim image generation here
        # For now, return empty structure
        x_grid = np.linspace(x_range[0], x_range[1], n_pixels)
        y_grid = np.linspace(y_range[0], y_range[1], n_pixels)
        
        return {
            'image': np.zeros((n_pixels, n_pixels)),
            'x_grid': x_grid,
            'y_grid': y_grid,
            'sources': sources,
        }
    
    def rubinize_images(self, images, psf_sigma=0.7, noise_std=None, 
                       pixel_scale=0.2):
        """
        Apply Rubin-like observational effects using SLSim.
        
        Parameters
        ----------
        images : dict
            Perfect images dict from generate_perfect_images
        psf_sigma : float
            PSF standard deviation (arcsec)
        noise_std : float, optional
            Noise standard deviation
        pixel_scale : float
            Pixel scale (arcsec/pixel)
        
        Returns
        -------
        rubinized : dict
            - 'image': Rubinized image
            - 'psf': PSF kernel
            - 'metadata': processing metadata
        """
        # Placeholder: would call SLSim Rubinization here
        
        return {
            'image': images['image'].copy(),
            'psf': np.array([[1.0]]),  # Placeholder
            'metadata': {
                'psf_sigma': psf_sigma,
                'pixel_scale': pixel_scale,
                'noise_std': noise_std,
            }
        }
    
    def detect_ggsl_events(self, sources, x_range=None, y_range=None, n_pixels=None,
                          source_z=1.0, lens_z=0.3):
        """
        Detect GGSL events for a source catalog.
        
        Parameters
        ----------
        sources : ndarray of shape (n_sources, 2)
            Source positions
        x_range, y_range, n_pixels : optional
            Grid parameters
        source_z : float
            Source redshift
        lens_z : float
            Lens redshift
        
        Returns
        -------
        events : list of GGSLObservable
            Detected events
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
        
        summary = self.ggsl_counter.summary()
        
        return events, summary
    
    def generate_mock_images(self, sources_list, x_range=None, y_range=None, 
                            image_size=256, pixel_scale=0.2):
        """
        Generate mock lensed images from sources using SLSim.
        
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
        lensed_images : list of SlsimLensedImage
            Generated images
        combined_image : ndarray
            Combined image array
        """
        from agent_sim.slsim_branch.imaging import (
            SlsimSourcePlacer, SlsimImageGenerator
        )
        
        if x_range is None:
            x_range = (-150, 150)
        if y_range is None:
            y_range = (-150, 150)
        
        # Initialize generators
        source_placer = SlsimSourcePlacer(self.deflection, self.magnification)
        image_gen = SlsimImageGenerator(self.deflection, self.magnification, source_placer)
        
        # Generate images
        lensed_images, combined = image_gen.generate_perfect_images(
            sources_list,
            use_slsim=False,  # Fallback for now
            x_range=x_range,
            y_range=y_range,
            image_size=image_size,
            pixel_scale=pixel_scale
        )
        
        return lensed_images, combined


import numpy as np
from agent_sim.core import models, lensing, observables, utils
