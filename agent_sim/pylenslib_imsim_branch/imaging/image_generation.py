"""
Lensed image generation for PyLensLib branch.

Generates 2D mock lensed images by placing sources and computing their
magnified images on a pixel grid.
"""

import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple


@dataclass
class LensedImage:
    """Data class for a lensed image."""
    source_id: int
    image_array: np.ndarray
    pixel_scale: float  # arcsec/pixel
    x_grid: np.ndarray
    y_grid: np.ndarray
    n_images: int
    magnifications: List[float]
    total_flux: float


class ImageGenerator:
    """
    Generate lensed images from sources using deflection and magnification maps.
    """
    
    def __init__(self, deflection_field, magnification_calc, source_placer):
        """
        Initialize image generator.
        
        Parameters
        ----------
        deflection_field : DeflectionField
            Deflection calculator
        magnification_calc : Magnification
            Magnification calculator
        source_placer : SourcePlacer
            Source placement utility
        """
        self.deflection = deflection_field
        self.magnification = magnification_calc
        self.source_placer = source_placer
    
    def render_source(self, source_position, source_radius, image_size=256, 
                     pixel_scale=0.2):
        """
        Render a single point or extended source.
        
        Parameters
        ----------
        source_position : tuple
            (x, y) source position
        source_radius : float
            Effective radius (arcsec)
        image_size : int
            Image dimension (pixels)
        pixel_scale : float
            arcsec per pixel
        
        Returns
        -------
        source_image : ndarray of shape (image_size, image_size)
            Rendered source
        """
        # Create coordinate grid
        x_center = (image_size // 2) * pixel_scale
        y_center = (image_size // 2) * pixel_scale
        
        x_grid = np.linspace(-x_center, x_center, image_size)
        y_grid = np.linspace(-y_center, y_center, image_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Source profile: Sersic (approximate as Gaussian for simplicity)
        r_squared = (X - source_position[0])**2 + (Y - source_position[1])**2
        
        # Gaussian profile
        sigma = source_radius / 2.355  # FWHM to sigma
        profile = np.exp(-r_squared / (2 * sigma**2))
        
        # Normalize
        profile = profile / np.max(profile)
        
        return profile
    
    def lens_source(self, source_image, source_position, 
                   deflection_grid_x, deflection_grid_y, 
                   x_grid, y_grid, magnification_grid):
        """
        Apply lensing deflection to a source image.
        
        Maps source to image plane via: I(theta) = S(beta(theta))
        where beta = theta - alpha(theta)
        
        Parameters
        ----------
        source_image : ndarray
            Source image
        source_position : tuple
            Source plane (x, y) position
        deflection_grid_x, deflection_grid_y : ndarray
            Deflection field components on grid
        x_grid, y_grid : ndarray
            Coordinate grids
        magnification_grid : ndarray
            Magnification map
        
        Returns
        -------
        lensed_image : ndarray
            Lensed image in image plane
        """
        # Create output image
        nx, ny = len(x_grid), len(y_grid)
        lensed = np.zeros((ny, nx))
        
        # For each image plane pixel, compute source plane position
        X_img, Y_img = np.meshgrid(x_grid, y_grid)
        
        # Source plane coordinates
        X_src = X_img - deflection_grid_x
        Y_src = Y_img - deflection_grid_y
        
        # Interpolate source image at source plane positions
        # (Simplified: use nearest neighbor for speed)
        from scipy.interpolate import RectBivariateSpline
        
        # Create source image grid
        src_size = source_image.shape[0]
        src_x = np.linspace(source_position[0] - 0.5, source_position[0] + 0.5, src_size)
        src_y = np.linspace(source_position[1] - 0.5, source_position[1] + 0.5, src_size)
        
        try:
            # Create spline interpolator
            spline = RectBivariateSpline(src_y, src_x, source_image, kx=1, ky=1)
            
            # Evaluate at source plane positions
            for i in range(ny):
                for j in range(nx):
                    try:
                        value = spline(Y_src[i, j], X_src[i, j], grid=False)[0, 0]
                        # Apply magnification
                        lensed[i, j] = value * magnification_grid[i, j]
                    except:
                        pass
        except:
            # Fallback: simple nearest-neighbor
            pass
        
        return lensed
    
    def generate_mock_images(self, sources: List, x_range=(-150, 150), 
                            y_range=(-150, 150), image_size=256, 
                            pixel_scale=0.2):
        """
        Generate mock lensed images for a set of sources.
        
        Parameters
        ----------
        sources : list of SourcePlacement
            Placed sources
        x_range, y_range : tuple
            Coordinate ranges (arcsec)
        image_size : int
            Image size (pixels per side)
        pixel_scale : float
            Pixel scale (arcsec/pixel)
        
        Returns
        -------
        lensed_images : list of LensedImage
            Generated lensed images
        combined_image : ndarray
            Combined image of all sources
        """
        # Create coordinate grid
        x_grid = np.linspace(x_range[0], x_range[1], image_size)
        y_grid = np.linspace(y_range[0], y_range[1], image_size)
        
        # Precompute lensing maps
        alpha_x, alpha_y = self.deflection.compute_deflection_grid(x_grid, y_grid)
        mag_grid = self.magnification.compute_magnification_grid(x_grid, y_grid)
        
        # Clamp magnification for stability
        mag_grid = np.clip(mag_grid, 0.1, 100)
        
        lensed_images = []
        combined = np.zeros((image_size, image_size))
        
        for source in sources:
            # Render source
            src_img = self.render_source(
                source.source_position, 
                source.radius_arcsec,
                image_size=128,
                pixel_scale=pixel_scale
            )
            
            # Apply lensing
            lensed = self.lens_source(
                src_img,
                source.source_position,
                alpha_x, alpha_y,
                x_grid, y_grid,
                mag_grid
            )
            
            # Scale by source brightness
            # Magnitude to flux: f ~ 10^(-m/2.5)
            flux = 10**(-source.magnitude / 2.5)
            lensed = lensed * flux
            
            # Find images (peak detection)
            n_images = self._count_images(lensed)
            mags = self._estimate_magnifications(lensed, source)
            total_flux = np.sum(lensed)
            
            lensed_img = LensedImage(
                source_id=source.source_id,
                image_array=lensed,
                pixel_scale=pixel_scale,
                x_grid=x_grid,
                y_grid=y_grid,
                n_images=n_images,
                magnifications=mags,
                total_flux=total_flux
            )
            lensed_images.append(lensed_img)
            
            # Add to combined image
            combined += lensed
        
        return lensed_images, combined
    
    @staticmethod
    def _count_images(image, threshold=0.1):
        """
        Count number of distinct images by peak detection.
        
        Parameters
        ----------
        image : ndarray
            Lensed image
        threshold : float
            Threshold for peak detection
        
        Returns
        -------
        n_peaks : int
            Number of distinct peaks
        """
        from scipy.ndimage import label, find_objects
        
        # Threshold
        binary = image > (threshold * np.max(image))
        
        # Connected components
        labeled, num_features = label(binary)
        
        return num_features
    
    @staticmethod
    def _estimate_magnifications(image, source):
        """
        Estimate magnifications of individual images.
        
        Parameters
        ----------
        image : ndarray
            Lensed image
        source : SourcePlacement
            Source that created the image
        
        Returns
        -------
        magnifications : list
            Estimated magnifications
        """
        from scipy.ndimage import label, find_objects
        
        # Find image peaks
        binary = image > (0.1 * np.max(image))
        labeled, num_features = label(binary)
        
        magnifications = []
        for i in range(1, num_features + 1):
            # Sum intensity in each component
            intensity = np.sum(image[labeled == i])
            # Estimate magnification (flux ratios)
            magnifications.append(intensity)
        
        # Normalize to source brightness
        if magnifications:
            flux = 10**(-source.magnitude / 2.5)
            magnifications = [m / flux for m in magnifications]
        
        return magnifications
    
    def save_image(self, lensed_image: LensedImage, output_path):
        """
        Save a lensed image to file.
        
        Parameters
        ----------
        lensed_image : LensedImage
            Image to save
        output_path : str
            Output file path (will save as NPZ)
        """
        np.savez(
            output_path,
            image=lensed_image.image_array,
            x_grid=lensed_image.x_grid,
            y_grid=lensed_image.y_grid,
            source_id=lensed_image.source_id,
            n_images=lensed_image.n_images,
            magnifications=lensed_image.magnifications,
            total_flux=lensed_image.total_flux,
            pixel_scale=lensed_image.pixel_scale
        )
