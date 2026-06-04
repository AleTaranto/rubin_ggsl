"""
Lensed image generation for SLSim branch.

Generates 2D mock lensed images using SLSim interface or fallback methods.
"""

import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter, label
from typing import Dict, List, Tuple


@dataclass
class SlsimLensedImage:
    """Data class for SLSim-generated lensed image."""
    source_id: int
    image_array: np.ndarray
    pixel_scale: float
    x_grid: np.ndarray
    y_grid: np.ndarray
    n_images: int
    magnifications: List[float]
    total_flux: float
    metadata: Dict = None


class SlsimImageGenerator:
    """
    Generate lensed images using SLSim or fallback methods.
    """
    
    def __init__(self, deflection_field, magnification_calc, source_placer):
        """
        Initialize SLSim image generator.
        
        Parameters
        ----------
        deflection_field : DeflectionField
            Deflection calculator
        magnification_calc : Magnification
            Magnification calculator
        source_placer : SlsimSourcePlacer
            Source placement utility
        """
        self.deflection = deflection_field
        self.magnification = magnification_calc
        self.source_placer = source_placer
    
    def generate_perfect_images(self, sources, use_slsim=False, 
                               x_range=(-150, 150), y_range=(-150, 150),
                               image_size=256, pixel_scale=0.2):
        """
        Generate perfect (noise-free) lensed images.
        
        Parameters
        ----------
        sources : list of SourcePlacement
            Placed sources
        use_slsim : bool
            Whether to use actual SLSim (if available)
        x_range, y_range : tuple
            Coordinate ranges
        image_size : int
            Image size
        pixel_scale : float
            Pixel scale
        
        Returns
        -------
        lensed_images : list of SlsimLensedImage
            Generated images
        combined_image : ndarray
            Combined image
        """
        if use_slsim:
            # TODO: Actual SLSim call would go here
            return self._generate_with_slsim_fallback(
                sources, x_range, y_range, image_size, pixel_scale
            )
        else:
            return self._generate_with_fallback(
                sources, x_range, y_range, image_size, pixel_scale
            )
    
    def _generate_with_fallback(self, sources, x_range, y_range, 
                               image_size, pixel_scale):
        """
        Generate images using fallback method (no SLSim).
        
        Parameters
        ----------
        sources : list
            Sources to image
        x_range, y_range : tuple
            Ranges
        image_size : int
            Image size
        pixel_scale : float
            Pixel scale
        
        Returns
        -------
        lensed_images : list
            Generated images
        combined : ndarray
            Combined image
        """
        # Create grids
        x_grid = np.linspace(x_range[0], x_range[1], image_size)
        y_grid = np.linspace(y_range[0], y_range[1], image_size)
        
        # Precompute lensing
        alpha_x, alpha_y = self.deflection.compute_deflection_grid(x_grid, y_grid)
        mag_grid = self.magnification.compute_magnification_grid(x_grid, y_grid)
        mag_grid = np.clip(mag_grid, 0.1, 100)
        
        lensed_images = []
        combined = np.zeros((image_size, image_size))
        
        X_img, Y_img = np.meshgrid(x_grid, y_grid)
        X_src = X_img - alpha_x
        Y_src = Y_img - alpha_y
        
        for source in sources:
            # Simple source profile
            src_profile = self._create_source_profile(
                source.source_position,
                source.radius_arcsec,
                image_size
            )
            
            # Magnify via interpolation
            lensed = self._apply_magnification(
                src_profile, X_src, Y_src, 
                source.source_position, mag_grid
            )
            
            # Scale by brightness
            flux = 10**(-source.magnitude / 2.5)
            lensed = lensed * flux
            
            # Count images and estimate magnifications
            n_images, mags = self._analyze_image(lensed, source)
            
            img = SlsimLensedImage(
                source_id=source.source_id,
                image_array=lensed,
                pixel_scale=pixel_scale,
                x_grid=x_grid,
                y_grid=y_grid,
                n_images=n_images,
                magnifications=mags,
                total_flux=np.sum(lensed),
                metadata={'method': 'fallback'}
            )
            lensed_images.append(img)
            combined += lensed
        
        return lensed_images, combined
    
    def _generate_with_slsim_fallback(self, sources, x_range, y_range, 
                                      image_size, pixel_scale):
        """
        Generate images with SLSim framework (placeholder).
        
        Parameters
        ----------
        sources : list
            Sources
        x_range, y_range : tuple
            Ranges
        image_size : int
            Image size
        pixel_scale : float
            Pixel scale
        
        Returns
        -------
        images : list
            Generated images
        combined : ndarray
            Combined image
        """
        # For now, same as fallback; will integrate with real SLSim
        return self._generate_with_fallback(
            sources, x_range, y_range, image_size, pixel_scale
        )
    
    @staticmethod
    def _create_source_profile(position, radius, size):
        """
        Create a source profile (Gaussian approximation).
        
        Parameters
        ----------
        position : tuple
            Source position
        radius : float
            Effective radius
        size : int
            Profile size
        
        Returns
        -------
        profile : ndarray
            Source profile
        """
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        r_squared = X**2 + Y**2
        sigma = radius / 2.355
        profile = np.exp(-r_squared / (2 * sigma**2))
        
        return profile / np.max(profile)
    
    @staticmethod
    def _apply_magnification(src_profile, X_src, Y_src, src_pos, mag_grid):
        """
        Apply magnification to source profile via interpolation.
        
        Parameters
        ----------
        src_profile : ndarray
            Source profile
        X_src, Y_src : ndarray
            Source plane coordinates
        src_pos : tuple
            Source position
        mag_grid : ndarray
            Magnification grid
        
        Returns
        -------
        lensed : ndarray
            Lensed image
        """
        from scipy.interpolate import RectBivariateSpline
        
        lensed = np.zeros_like(mag_grid)
        ny, nx = mag_grid.shape
        
        # Create source interpolator
        src_x = np.linspace(src_pos[0] - 1, src_pos[0] + 1, src_profile.shape[0])
        src_y = np.linspace(src_pos[1] - 1, src_pos[1] + 1, src_profile.shape[1])
        
        try:
            spline = RectBivariateSpline(src_y, src_x, src_profile, kx=1, ky=1)
            
            for i in range(ny):
                for j in range(nx):
                    try:
                        val = spline(Y_src[i, j], X_src[i, j], grid=False)[0, 0]
                        lensed[i, j] = val * mag_grid[i, j]
                    except:
                        pass
        except:
            pass
        
        return lensed
    
    @staticmethod
    def _analyze_image(image, source):
        """
        Analyze lensed image for number of images and magnifications.
        
        Parameters
        ----------
        image : ndarray
            Lensed image
        source : SourcePlacement
            Original source
        
        Returns
        -------
        n_images : int
            Number of images
        magnifications : list
            Magnifications
        """
        # Find peaks
        threshold = 0.1 * np.max(image)
        binary = image > threshold
        
        labeled, num = label(binary)
        
        # Estimate magnifications
        magnifications = []
        flux = 10**(-source.magnitude / 2.5)
        
        for i in range(1, num + 1):
            intensity = np.sum(image[labeled == i])
            mag = intensity / flux if flux > 0 else 0
            magnifications.append(mag)
        
        return num, magnifications
    
    def save_image(self, image: SlsimLensedImage, output_path):
        """
        Save SLSim image to file.
        
        Parameters
        ----------
        image : SlsimLensedImage
            Image to save
        output_path : str
            Output path
        """
        np.savez(
            output_path,
            image=image.image_array,
            x_grid=image.x_grid,
            y_grid=image.y_grid,
            source_id=image.source_id,
            n_images=image.n_images,
            magnifications=image.magnifications,
            total_flux=image.total_flux,
            pixel_scale=image.pixel_scale
        )
