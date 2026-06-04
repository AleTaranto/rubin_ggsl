"""
Source placement utilities for SLSim branch.

Places background sources on the lensing map using SLSim interfaces.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SourcePlacement:
    """Data class for a placed source."""
    source_id: int
    source_position: Tuple[float, float]  # (x, y) in source plane
    redshift: float
    magnitude: float
    radius_arcsec: float
    

class SlsimSourcePlacer:
    """
    Place background sources using SLSim interface.
    """
    
    def __init__(self, deflection_field, magnification_calc):
        """
        Initialize SLSim source placer.
        
        Parameters
        ----------
        deflection_field : DeflectionField
            Deflection angle calculator
        magnification_calc : Magnification
            Magnification calculator
        """
        self.deflection = deflection_field
        self.magnification = magnification_calc
    
    def place_sources_from_config(self, config: dict, redshift=1.0):
        """
        Place sources from configuration dictionary.
        
        Parameters
        ----------
        config : dict
            Configuration with 'n_sources', 'region', etc.
        redshift : float
            Source redshift
        
        Returns
        -------
        sources : list of SourcePlacement
            Placed sources
        """
        n_sources = config.get('n_sources', 50)
        region = config.get('region', [(-100, 100), (-100, 100)])
        
        return self.place_uniform_sources(
            n_sources,
            region_x=region[0],
            region_y=region[1],
            redshift=redshift
        )
    
    def place_uniform_sources(self, n_sources, region_x=(-100, 100), 
                             region_y=(-100, 100), redshift=1.0):
        """
        Place sources uniformly in a region.
        
        Parameters
        ----------
        n_sources : int
            Number of sources
        region_x, region_y : tuple
            Coordinate ranges
        redshift : float
            Source redshift
        
        Returns
        -------
        sources : list of SourcePlacement
            Placed sources
        """
        sources = []
        
        # Sample positions uniformly
        x_positions = np.random.uniform(region_x[0], region_x[1], n_sources)
        y_positions = np.random.uniform(region_y[0], region_y[1], n_sources)
        
        # Sample magnitudes
        magnitudes = np.random.uniform(18, 26, n_sources)
        
        # Sample sizes
        radii = np.random.exponential(0.15, n_sources)
        radii = np.clip(radii, 0.03, 0.5)
        
        for i in range(n_sources):
            source = SourcePlacement(
                source_id=i,
                source_position=(x_positions[i], y_positions[i]),
                redshift=redshift,
                magnitude=magnitudes[i],
                radius_arcsec=radii[i]
            )
            sources.append(source)
        
        return sources
    
    def compute_image_positions_slsim(self, source, use_slsim=False):
        """
        Compute image positions using SLSim if available.
        
        Parameters
        ----------
        source : SourcePlacement
            Source to trace
        use_slsim : bool
            If True, attempt to use actual SLSim; otherwise use fallback
        
        Returns
        -------
        images : list of tuple
            (x, y) image positions
        magnifications : list of float
            Magnifications
        """
        if use_slsim:
            # TODO: Interface with actual SLSim
            pass
        
        # Fallback: use deflection field directly
        return self._trace_images_fallback(source)
    
    def _trace_images_fallback(self, source):
        """
        Fallback image tracing when SLSim not available.
        
        Parameters
        ----------
        source : SourcePlacement
            Source to trace
        
        Returns
        -------
        images : list of tuple
            Image positions
        magnifications : list of float
            Magnifications
        """
        beta_x, beta_y = source.source_position
        images = []
        mags = []
        
        # Try multiple starting points
        for offset in [0, 30, 50, 80, 120]:
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                theta_x = beta_x + offset * np.cos(angle)
                theta_y = beta_y + offset * np.sin(angle)
                
                # Newton-Raphson iteration
                for iteration in range(20):
                    alpha_x, alpha_y = self.deflection.deflection_angle(theta_x, theta_y)
                    
                    residual_x = theta_x - beta_x - alpha_x
                    residual_y = theta_y - beta_y - alpha_y
                    residual = np.sqrt(residual_x**2 + residual_y**2)
                    
                    if residual < 1e-4:
                        # Check if new
                        is_new = True
                        for existing in images:
                            if np.sqrt((theta_x - existing[0])**2 + 
                                      (theta_y - existing[1])**2) < 0.05:
                                is_new = False
                                break
                        
                        if is_new:
                            mag = self.magnification.magnification(theta_x, theta_y)
                            images.append((theta_x, theta_y))
                            mags.append(mag)
                        break
                    
                    # Approximate Jacobian
                    eps = 0.05
                    alpha_x_p, _ = self.deflection.deflection_angle(theta_x + eps, theta_y)
                    alpha_x_m, _ = self.deflection.deflection_angle(theta_x - eps, theta_y)
                    _, alpha_y_p = self.deflection.deflection_angle(theta_x, theta_y + eps)
                    _, alpha_y_m = self.deflection.deflection_angle(theta_x, theta_y - eps)
                    
                    J = np.array([
                        [1 - (alpha_x_p - alpha_x_m) / (2*eps), -(alpha_y_p - alpha_y_m) / (2*eps)],
                        [-(alpha_x_p - alpha_x_m) / (2*eps), 1 - (alpha_y_p - alpha_y_m) / (2*eps)]
                    ])
                    
                    try:
                        delta = np.linalg.solve(J, np.array([residual_x, residual_y]))
                        theta_x -= delta[0]
                        theta_y -= delta[1]
                    except:
                        break
        
        return images, mags
