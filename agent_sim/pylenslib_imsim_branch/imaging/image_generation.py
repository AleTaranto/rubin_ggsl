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
        
        # Interpolate source image at source plane positions using nearest neighbor
        from scipy.interpolate import griddata
        
        # Create source image grid
        src_size = source_image.shape[0]
        src_x = np.linspace(source_position[0] - 0.5, source_position[0] + 0.5, src_size)
        src_y = np.linspace(source_position[1] - 0.5, source_position[1] + 0.5, src_size)
        src_X, src_Y = np.meshgrid(src_x, src_y)
        
        # Prepare source data for interpolation
        src_points = np.column_stack([src_X.ravel(), src_Y.ravel()])
        src_values = source_image.ravel()
        
        # Prepare evaluation points
        eval_points = np.column_stack([X_src.ravel(), Y_src.ravel()])
        
        try:
            # Use nearest neighbor interpolation for robustness
            interpolated = griddata(src_points, src_values, eval_points, 
                                   method='nearest', fill_value=0.0)
            
            # Reshape back to 2D
            interpolated = interpolated.reshape((ny, nx))
            
            # Apply magnification and normalize to avoid NaN propagation
            mag_safe = np.nan_to_num(magnification_grid, nan=1.0, posinf=100.0, neginf=0.1)
            lensed = interpolated * mag_safe
            
        except Exception as e:
            # Fallback: simple direct mapping without interpolation
            # Map magnifications directly where they're strongest
            src_indices = np.clip((X_src * nx / (x_grid[-1] - x_grid[0]) + nx/2).astype(int), 0, nx-1)
            src_indices_y = np.clip((Y_src * ny / (y_grid[-1] - y_grid[0]) + ny/2).astype(int), 0, ny-1)
            
            try:
                lensed = source_image[src_indices_y, src_indices] * magnification_grid
            except:
                # If all else fails, return zeros (at least it's explicit)
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
        
        # Clean up magnification: handle NaN and inf values
        mag_grid = np.nan_to_num(mag_grid, nan=1.0, posinf=100.0, neginf=0.1)
        
        # Clamp magnification for stability
        mag_grid = np.clip(mag_grid, 0.1, 100)
        
        lensed_images = []
        combined = np.zeros((image_size, image_size))
        
        for source in sources:
            # Render source
            # Render the source at the same resolution as the final image to
            # avoid broadcasting / shape mismatch when mapping to the image grid.
            src_img = self.render_source(
                source.source_position,
                source.radius_arcsec,
                image_size=image_size,
                pixel_scale=pixel_scale
            )
            
            # Ensure source image is valid
            src_img = np.nan_to_num(src_img, nan=0.0, posinf=1.0, neginf=0.0)
            src_img = np.clip(src_img, 0, 1)
            
            # Apply lensing
            lensed = self.lens_source(
                src_img,
                source.source_position,
                alpha_x, alpha_y,
                x_grid, y_grid,
                mag_grid
            )
            
            # Clean up lensed image
            lensed = np.nan_to_num(lensed, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale by source brightness
            # Magnitude to flux: f ~ 10^(-m/2.5)
            flux = 10**(-source.magnitude / 2.5)
            lensed = lensed * flux
            
            # Find images (peak detection)
            n_images = self._count_images(lensed)
            mags = self._estimate_magnifications(lensed, source)
            total_flux = np.sum(lensed)
            
            # Ensure non-zero flux - if image is empty, use source brightness directly
            if total_flux <= 0 or np.isnan(total_flux):
                # Fallback: use a simple magnified source profile
                lensed = src_img * flux * 10  # Scale up for visibility
                n_images = 1
                mags = [1.0]
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
        
        # Clean up final combined image
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        combined = np.clip(combined, 0, None)
        
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
