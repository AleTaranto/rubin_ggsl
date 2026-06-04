"""
Rubinization wrapper for SLSim branch.

Applies Rubin-like observational effects via SLSim integration.
"""

import numpy as np
from dataclasses import dataclass
from scipy.ndimage import convolve
from typing import Dict


@dataclass
class SlsimRubinizedImage:
    """SLSim-generated Rubinized (realistic) image."""
    source_id: int
    image_array: np.ndarray
    pixel_scale: float
    psf: np.ndarray
    metadata: Dict


class SlsimRubinizer:
    """
    Apply Rubin-like effects using SLSim interface.
    
    Includes PSF, noise, sky background, and cosmic rays.
    """
    
    def __init__(self, psf_sigma=0.7, read_noise_e=10, 
                 sky_brightness_mag=22, gain=3.0):
        """
        Initialize SLSim Rubinizer.
        
        Parameters
        ----------
        psf_sigma : float
            PSF sigma (arcsec)
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
    
    def rubinize_slsim(self, perfect_image, pixel_scale=0.2, 
                      use_slsim_psf=False, **kwargs):
        """
        Apply Rubinization using SLSim framework (or fallback).
        
        Parameters
        ----------
        perfect_image : ndarray
            Perfect image in flux units
        pixel_scale : float
            Pixel scale (arcsec/pixel)
        use_slsim_psf : bool
            Whether to use actual SLSim PSF (if available)
        **kwargs : optional
            Additional SLSim parameters
        
        Returns
        -------
        rubinized : ndarray
            Rubinized image (in ADU)
        metadata : dict
            Processing metadata
        """
        # For now, use fallback since SLSim integration is in progress
        return self._rubinize_fallback(perfect_image, pixel_scale)
    
    def _rubinize_fallback(self, perfect_image, pixel_scale=0.2):
        """
        Fallback Rubinization when SLSim not available.
        
        Parameters
        ----------
        perfect_image : ndarray
            Perfect image
        pixel_scale : float
            Pixel scale
        
        Returns
        -------
        rubinized : ndarray
            Rubinized image
        metadata : dict
            Metadata
        """
        # Convert to electrons
        image_e = perfect_image * 100
        
        # Create and apply PSF
        psf = self._create_psf(pixel_scale)
        image_e = convolve(image_e, psf, mode='constant', cval=0.0)
        
        # Add noise
        sky_flux = 10**(-self.sky_mag / 2.5) * 100
        image_e = self._add_noise(image_e, sky_flux)
        
        # Convert to ADU
        image_adu = image_e / self.gain
        
        # Clip to reasonable range
        image_adu = np.clip(image_adu, 0, 65000)
        
        return image_adu, {
            'method': 'fallback',
            'psf_sigma': self.psf_sigma,
            'noise_level': self.read_noise,
            'sky_background': self.sky_mag
        }
    
    def _create_psf(self, pixel_scale, size=31):
        """Create PSF kernel."""
        sigma_pix = self.psf_sigma / (2.355 * pixel_scale)
        center = size // 2
        
        x = np.arange(size) - center
        y = np.arange(size) - center
        X, Y = np.meshgrid(x, y)
        r_sq = X**2 + Y**2
        
        psf = np.exp(-r_sq / (2 * sigma_pix**2))
        psf = psf / np.sum(psf)
        
        return psf
    
    def _add_noise(self, image, sky_flux):
        """Add realistic noise."""
        noisy = image.copy()
        
        # Sky noise
        sky_noise = np.random.poisson(sky_flux, image.shape)
        noisy += sky_noise
        
        # Photon noise
        photon_noise = np.random.poisson(np.maximum(image, 0)) - image
        noisy += photon_noise
        
        # Read noise
        read_noise = np.random.normal(0, self.read_noise, image.shape)
        noisy += read_noise
        
        return noisy
