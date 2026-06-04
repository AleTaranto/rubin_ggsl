"""
Source placement utilities for PyLensLib branch.

Places background sources on the lensing map and computes their positions
in the source plane and image plane.
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
    

class SourcePlacer:
    """
    Place background sources on lensing map and compute image positions.
    """
    
    def __init__(self, deflection_field, magnification_calc):
        """
        Initialize source placer.
        
        Parameters
        ----------
        deflection_field : DeflectionField
            Deflection angle calculator
        magnification_calc : Magnification
            Magnification calculator
        """
        self.deflection = deflection_field
        self.magnification = magnification_calc
    
    def place_uniform_sources(self, n_sources, region_x=(-100, 100), 
                              region_y=(-100, 100), redshift=1.0):
        """
        Place sources uniformly in a region of the source plane.
        
        Parameters
        ----------
        n_sources : int
            Number of sources to place
        region_x, region_y : tuple
            Coordinate ranges (arcsec or kpc)
        redshift : float
            Source redshift
        
        Returns
        -------
        sources : list of SourcePlacement
            Placed sources with sampled properties
        """
        sources = []
        
        # Sample positions uniformly
        x_positions = np.random.uniform(region_x[0], region_x[1], n_sources)
        y_positions = np.random.uniform(region_y[0], region_y[1], n_sources)
        
        # Sample magnitudes (Rubin survey range)
        magnitudes = np.random.uniform(18, 26, n_sources)
        
        # Sample sizes (effective radii in arcsec)
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
    
    def place_clustered_sources(self, n_sources, n_clusters=5, 
                               cluster_scale=20.0, redshift=1.0):
        """
        Place sources in clusters (e.g., galaxy clusters as sources).
        
        Parameters
        ----------
        n_sources : int
            Total number of sources
        n_clusters : int
            Number of clusters
        cluster_scale : float
            Cluster scale (arcsec)
        redshift : float
            Source redshift
        
        Returns
        -------
        sources : list of SourcePlacement
            Clustered sources
        """
        sources = []
        sources_per_cluster = n_sources // n_clusters
        
        # Place cluster centers randomly
        cluster_centers_x = np.random.uniform(-100, 100, n_clusters)
        cluster_centers_y = np.random.uniform(-100, 100, n_clusters)
        
        source_id = 0
        magnitudes = np.random.uniform(18, 26, n_sources)
        radii = np.random.exponential(0.15, n_sources)
        radii = np.clip(radii, 0.03, 0.5)
        
        for cluster_idx in range(n_clusters):
            cx = cluster_centers_x[cluster_idx]
            cy = cluster_centers_y[cluster_idx]
            
            # Place sources around cluster center
            n_in_cluster = sources_per_cluster
            if cluster_idx == n_clusters - 1:
                n_in_cluster = n_sources - source_id  # Last cluster gets remainder
            
            for i in range(n_in_cluster):
                # Gaussian distribution around cluster
                r = np.random.normal(0, cluster_scale, 1)[0]
                theta = np.random.uniform(0, 2*np.pi, 1)[0]
                
                x = cx + r * np.cos(theta)
                y = cy + r * np.sin(theta)
                
                # Clip to reasonable bounds
                x = np.clip(x, -150, 150)
                y = np.clip(y, -150, 150)
                
                source = SourcePlacement(
                    source_id=source_id,
                    source_position=(x, y),
                    redshift=redshift,
                    magnitude=magnitudes[source_id],
                    radius_arcsec=radii[source_id]
                )
                sources.append(source)
                source_id += 1
        
        return sources
    
    def compute_image_positions(self, source, grid_size=0.05, max_iterations=20):
        """
        Compute image positions for a source via ray-tracing.
        
        Solves: theta = beta + alpha(theta)
        
        Parameters
        ----------
        source : SourcePlacement
            Source to trace
        grid_size : float
            Grid size for numerical derivatives
        max_iterations : int
            Maximum Newton-Raphson iterations
        
        Returns
        -------
        images : list of tuple
            (x, y) positions of images
        magnifications : list of float
            Magnification of each image
        """
        beta_x, beta_y = source.source_position
        
        # Initial guess: image at source position
        theta_x = beta_x
        theta_y = beta_y
        
        images = []
        mags_found = []
        
        # Try multiple starting points to find all images
        search_radius = 150
        n_search = 12
        
        for search_idx in range(n_search):
            # Different starting positions
            angle = 2 * np.pi * search_idx / n_search
            offset = 50 * (search_idx / n_search)
            
            theta_x = beta_x + offset * np.cos(angle)
            theta_y = beta_y + offset * np.sin(angle)
            
            # Newton-Raphson iteration
            for iteration in range(max_iterations):
                # Get deflection at current position
                alpha_x, alpha_y = self.deflection.deflection_angle(theta_x, theta_y)
                
                # Lens equation: theta = beta + alpha(theta)
                residual_x = theta_x - beta_x - alpha_x
                residual_y = theta_y - beta_y - alpha_y
                residual = np.sqrt(residual_x**2 + residual_y**2)
                
                if residual < 1e-5:
                    # Converged!
                    # Check if this is a new image
                    is_new = True
                    for existing_image in images:
                        dist = np.sqrt((theta_x - existing_image[0])**2 + 
                                      (theta_y - existing_image[1])**2)
                        if dist < 0.1:  # Tolerance
                            is_new = False
                            break
                    
                    if is_new:
                        mag = self.magnification.magnification(theta_x, theta_y)
                        images.append((theta_x, theta_y))
                        mags_found.append(mag)
                    break
                
                # Compute Jacobian (approximate)
                eps = grid_size
                alpha_x_plus, alpha_y_plus = self.deflection.deflection_angle(
                    theta_x + eps, theta_y
                )
                alpha_x_minus, alpha_y_minus = self.deflection.deflection_angle(
                    theta_x - eps, theta_y
                )
                
                dalpha_x_dtheta_x = (alpha_x_plus - alpha_x_minus) / (2 * eps)
                dalpha_y_dtheta_x = (alpha_y_plus - alpha_y_minus) / (2 * eps)
                
                alpha_x_plus, alpha_y_plus = self.deflection.deflection_angle(
                    theta_x, theta_y + eps
                )
                alpha_x_minus, alpha_y_minus = self.deflection.deflection_angle(
                    theta_x, theta_y - eps
                )
                
                dalpha_x_dtheta_y = (alpha_x_plus - alpha_x_minus) / (2 * eps)
                dalpha_y_dtheta_y = (alpha_y_plus - alpha_y_minus) / (2 * eps)
                
                # Jacobian: A_ij = delta_ij - d(alpha_i)/d(theta_j)
                A11 = 1 - dalpha_x_dtheta_x
                A12 = -dalpha_x_dtheta_y
                A21 = -dalpha_y_dtheta_x
                A22 = 1 - dalpha_y_dtheta_y
                
                det = A11 * A22 - A12 * A21
                if abs(det) < 1e-10:
                    break  # Singular
                
                # Update: delta_theta = A^(-1) * residual
                inv_A11 = A22 / det
                inv_A12 = -A12 / det
                inv_A21 = -A21 / det
                inv_A22 = A11 / det
                
                delta_x = inv_A11 * residual_x + inv_A12 * residual_y
                delta_y = inv_A21 * residual_x + inv_A22 * residual_y
                
                theta_x -= delta_x
                theta_y -= delta_y
        
        return images, mags_found
    
    def __repr__(self):
        return f"SourcePlacer()"
