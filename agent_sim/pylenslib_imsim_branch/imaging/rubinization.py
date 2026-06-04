"""
Rubinization wrapper for PyLensLib + ImSim branch.

Applies Rubin-like observational effects to perfect mock images:
- PSF convolution
- Detector noise
- Sky background
- Cosmic rays
- Pixel discretization
"""

import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter, convolve
from typing import Dict, Optional


@dataclass
class RubinizedImage:
    """Rubinized (realistic) image with observational effects."""
    source_id: int
    image_array: np.ndarray
    pixel_scale: float
    psf: np.ndarray
    noise_std: float
    sky_background: float
    metadata: Dict


class ImSimRubinizer:
    """
    Apply Rubin-like observational effects to images using ImSim integration.
    
    Effects applied:
    1. PSF convolution (Gaussian or Moffat)
    2. Photon noise (Poisson)
    3. Read noise
    4. Sky background
    5. Cosmic rays
    6. Pixel saturation
    """
    
    def __init__(self, psf_sigma=0.7, read_noise_e=10, 
                 sky_brightness_mag=22, gain=3.0):
        """
        Initialize Rubinizer.
        
        Parameters
        ----------
        psf_sigma : float
            PSF standard deviation (arcsec)
        read_noise_e : float
            Read noise (electrons)
        sky_brightness_mag : float
            Sky brightness (mag/arcsec^2)
        gain : float
            Detector gain (e-/ADU)
        """
        self.psf_sigma = psf_sigma
        self.read_noise = read_noise_e
        self.sky_mag = sky_brightness_mag
        self.gain = gain
    
    def create_psf_kernel(self, pixel_scale=0.2, size=31):
        """
        Create PSF kernel for convolution.
        
        Parameters
        ----------
        pixel_scale : float
            Pixel scale (arcsec/pixel)
        size : int
            Kernel size (pixels, odd)
        
        Returns
        -------
        psf : ndarray
            PSF kernel
        """
        # PSF FWHM
        fwhm = 2 * self.psf_sigma
        
        # Convert to pixels
        sigma_pix = fwhm / (2.355 * pixel_scale)
        
        # Create kernel
        center = size // 2
        x = np.arange(size) - center
        y = np.arange(size) - center
        X, Y = np.meshgrid(x, y)
        r_sq = X**2 + Y**2
        
        # Gaussian PSF
        psf = np.exp(-r_sq / (2 * sigma_pix**2))
        
        # Normalize
        psf = psf / np.sum(psf)
        
        return psf
    
    def apply_psf(self, image, psf_kernel):
        """
        Apply PSF convolution to image.
        
        Parameters
        ----------
        image : ndarray
            Input image
        psf_kernel : ndarray
            PSF kernel
        
        Returns
        -------
        convolved : ndarray
            PSF-convolved image
        """
        # Use scipy convolve for accurate convolution
        convolved = convolve(image, psf_kernel, mode='constant', cval=0.0)
        return convolved
    
    def add_noise(self, image, sky_flux):
        """
        Add realistic noise to image.
        
        Parameters
        ----------
        image : ndarray
            Input image (in electrons)
        sky_flux : float
            Sky background flux (e-/pixel)
        
        Returns
        -------
        noisy_image : ndarray
            Image with noise
        """
        # Make a copy
        noisy = image.copy()
        
        # Add sky background (Poisson)
        sky_noise = np.random.poisson(sky_flux, image.shape)
        noisy += sky_noise
        
        # Add photon noise (Poisson)
        # Already in the image via the source convolution
        photon_noise = np.random.poisson(np.maximum(image, 0)) - image
        noisy += photon_noise
        
        # Add read noise (Gaussian)
        read_noise = np.random.normal(0, self.read_noise, image.shape)
        noisy += read_noise
        
        return noisy
    
    def add_cosmic_rays(self, image, cr_rate=0.01, cr_strength=100):
        """
        Add simulated cosmic rays.
        
        Parameters
        ----------
        image : ndarray
            Input image
        cr_rate : float
            Cosmic ray rate (fraction of pixels)
        cr_strength : float
            Cosmic ray signal (electrons)
        
        Returns
        -------
        image_with_cr : ndarray
            Image with cosmic rays
        """
        image_cr = image.copy()
        
        # Random cosmic ray hits
        ny, nx = image.shape
        n_pixels = ny * nx
        n_hits = int(n_pixels * cr_rate)
        
        if n_hits > 0:
            hit_indices = np.random.choice(n_pixels, n_hits, replace=False)
            hit_y = hit_indices // nx
            hit_x = hit_indices % nx
            
            # Add cosmic ray streaks (simple 3x3 kernel)
            for y, x in zip(hit_y, hit_x):
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        py, px = y + dy, x + dx
                        if 0 <= py < ny and 0 <= px < nx:
                            image_cr[py, px] += cr_strength * (1 - abs(dy) * 0.3) * (1 - abs(dx) * 0.3)
        
        return image_cr
    
    def apply_saturation(self, image, saturation_level=65000):
        """
        Apply detector saturation.
        
        Parameters
        ----------
        image : ndarray
            Input image
        saturation_level : float
            Saturation level (ADU)
        
        Returns
        -------
        saturated : ndarray
            Image with saturation
        """
        saturated = np.minimum(image, saturation_level)
        return saturated
    
    def rubinize(self, perfect_image, pixel_scale=0.2, add_psf=True, 
                add_noise_flag=True, add_cr=True):
        """
        Apply full Rubinization to perfect image.
        
        Parameters
        ----------
        perfect_image : ndarray
            Perfect (noise-free) image in flux units
        pixel_scale : float
            Pixel scale (arcsec/pixel)
        add_psf : bool
            Whether to apply PSF
        add_noise_flag : bool
            Whether to add noise
        add_cr : bool
            Whether to add cosmic rays
        
        Returns
        -------
        rubinized : RubinizedImage data (in ADU)
        psf_kernel : ndarray
            PSF kernel used
        """
        # Create PSF kernel
        psf_kernel = self.create_psf_kernel(pixel_scale=pixel_scale, size=31)
        
        # Start with perfect image (convert to electrons)
        # Assume 1 flux unit = 100 electrons
        image_e = perfect_image * 100
        
        # Apply PSF
        if add_psf:
            image_e = self.apply_psf(image_e, psf_kernel)
        
        # Convert sky brightness mag to flux
        # mag/arcsec^2 -> flux per pixel
        sky_flux = 10**(-self.sky_mag / 2.5) * 100  # electrons/pixel
        
        # Add noise
        if add_noise_flag:
            image_e = self.add_noise(image_e, sky_flux)
        
        # Add cosmic rays
        if add_cr:
            image_e = self.add_cosmic_rays(image_e, cr_rate=0.001)
        
        # Apply saturation
        image_e = self.apply_saturation(image_e, saturation_level=100000)
        
        # Convert to ADU
        image_adu = image_e / self.gain
        
        return image_adu, psf_kernel
    
    def save_rubinized(self, rubinized_array, output_path, metadata=None):
        """
        Save Rubinized image to file.
        
        Parameters
        ----------
        rubinized_array : ndarray
            Rubinized image (in ADU)
        output_path : str
            Output file path
        metadata : dict, optional
            Additional metadata
        """
        save_data = {
            'image': rubinized_array,
            'psf_sigma': self.psf_sigma,
            'read_noise': self.read_noise,
            'sky_mag': self.sky_mag,
            'gain': self.gain,
        }
        
        if metadata:
            save_data.update(metadata)
        
        np.savez(output_path, **save_data)
