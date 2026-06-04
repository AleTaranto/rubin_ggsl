"""
Dark matter density profile models.

This module provides base classes and implementations of common dark matter
density profiles used in strong gravitational lensing studies.
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.integrate import quad


class DarkMatterModel(ABC):
    """
    Abstract base class for dark matter models.
    
    All dark matter models should inherit from this class and implement
    the required methods for computing density, potential, and lensing quantities.
    """
    
    def __init__(self, redshift=None, cosmology=None):
        """
        Initialize dark matter model.
        
        Parameters
        ----------
        redshift : float, optional
            Redshift of the halo
        cosmology : astropy.cosmology.Cosmology, optional
            Cosmological model (default: Planck18)
        """
        self.redshift = redshift
        self.cosmology = cosmology
    
    @abstractmethod
    def density(self, r):
        """
        Density profile rho(r).
        
        Parameters
        ----------
        r : float or array_like
            Radius (in physical or comoving units as specified)
        
        Returns
        -------
        rho : float or ndarray
            Density at radius r
        """
        pass
    
    @abstractmethod
    def surface_density(self, R):
        """
        Surface density (column density) Sigma(R).
        
        Parameters
        ----------
        R : float or array_like
            Projected radius
        
        Returns
        -------
        Sigma : float or ndarray
            Surface density at radius R
        """
        pass
    
    @abstractmethod
    def mass_enclosed(self, r):
        """
        Mass enclosed within radius r.
        
        Parameters
        ----------
        r : float or array_like
            Radius
        
        Returns
        -------
        M : float or ndarray
            Total mass within radius r
        """
        pass
    
    def convergence(self, R, critical_density):
        """
        Convergence (dimensionless surface density).
        
        Parameters
        ----------
        R : float or array_like
            Projected radius
        critical_density : float
            Critical density for lensing
        
        Returns
        -------
        kappa : float or ndarray
            Convergence at radius R
        """
        return self.surface_density(R) / critical_density
    
    def __repr__(self):
        return f"{self.__class__.__name__}(redshift={self.redshift})"


class NFW(DarkMatterModel):
    """
    Navarro-Frenk-White (NFW) dark matter profile.
    
    The NFW profile is one of the most widely used dark matter density models
    derived from N-body simulations:
    rho(r) = rho_s / [(r/r_s)(1 + r/r_s)^2]
    """
    
    def __init__(self, M200=1e14, c=4.0, redshift=None, cosmology=None):
        """
        Initialize NFW profile.
        
        Parameters
        ----------
        M200 : float
            Mass within r200 (virial mass, in solar masses)
        c : float
            Concentration parameter (default: 4.0)
        redshift : float, optional
            Redshift of the halo
        cosmology : astropy.cosmology.Cosmology, optional
            Cosmological model
        """
        super().__init__(redshift=redshift, cosmology=cosmology)
        self.M200 = M200
        self.c = c
        self._set_scale_params()
    
    def _set_scale_params(self):
        """Compute scale radius and scale density from M200 and c."""
        # Using approximate relation: r200 ~ 1/H(z) for a given M200
        # For simplicity, use characteristic scale
        # r_s = r_200 / c
        # Then compute rho_s from enclosed mass
        
        # Compute r200 approximately (in kpc)
        self.r200 = 220 * (self.M200 / 1e14) ** (1/3)  # rough approximation
        self.rs = self.r200 / self.c
        
        # Compute scale density from M200 enclosed
        # M200 = 4*pi*rho_s*r_s^3 * f(c) where f(c) = ln(1+c) - c/(1+c)
        f_c = np.log(1 + self.c) - self.c / (1 + self.c)
        self.rho_s = self.M200 / (4 * np.pi * self.rs**3 * f_c)
    
    def density(self, r):
        """NFW density profile."""
        x = r / self.rs
        return self.rho_s / (x * (1 + x)**2)
    
    def surface_density(self, R):
        """NFW surface density (integrated from 0 to infinity)."""
        x = R / self.rs
        
        def integrand(z):
            return self.rho_s / np.sqrt(z**2 - x**2) / (z * (1 + z)**2)
        
        if np.isscalar(R):
            result, _ = quad(integrand, x + 1e-6, 1e3 * x + 10, limit=100)
            return 2 * result
        else:
            return np.array([self.surface_density(R_i) for R_i in np.atleast_1d(R)])
    
    def mass_enclosed(self, r):
        """NFW enclosed mass."""
        x = r / self.rs
        M_s = 4 * np.pi * self.rho_s * self.rs**3
        return M_s * (np.log(1 + x)**2 / x - np.log(1 + x) + np.arctan(x))
    
    def __repr__(self):
        return f"NFW(M200={self.M200:.2e}, c={self.c:.2f}, z={self.redshift})"


class CoreNFW(DarkMatterModel):
    """
    NFW profile with an added central core.
    
    Models: rho(r) = rho_s / [(r/r_s + r_core/r_s)(1 + r/r_s)^2]
    """
    
    def __init__(self, M200=1e14, c=4.0, r_core=0.1, redshift=None, cosmology=None):
        """
        Initialize CoreNFW profile.
        
        Parameters
        ----------
        M200 : float
            Mass within r200 (virial mass, in solar masses)
        c : float
            Concentration parameter
        r_core : float
            Core radius relative to scale radius (r_core/r_s)
        redshift : float, optional
            Redshift of the halo
        cosmology : astropy.cosmology.Cosmology, optional
            Cosmological model
        """
        super().__init__(redshift=redshift, cosmology=cosmology)
        self.M200 = M200
        self.c = c
        self.r_core_rel = r_core
        
        # Initialize like NFW to get r_s and rho_s
        nfw = NFW(M200=M200, c=c, redshift=redshift, cosmology=cosmology)
        self.rs = nfw.rs
        self.r200 = nfw.r200
        self.r_core = r_core * self.rs
        
        # Renormalize density to maintain M200
        self.rho_s = nfw.rho_s  # This is approximate; full renormalization would be needed
    
    def density(self, r):
        """CoreNFW density profile."""
        x = r / self.rs
        x_core = self.r_core / self.rs
        return self.rho_s / ((x + x_core) * (1 + x)**2)
    
    def surface_density(self, R):
        """CoreNFW surface density (simplified)."""
        # Approximate: similar to NFW but with smoother core
        x = R / self.rs
        x_core = self.r_core / self.rs
        
        def integrand(z):
            z_c = z + x_core
            return self.rho_s / np.sqrt(z**2 - x**2) / (z_c * (1 + z)**2)
        
        if np.isscalar(R):
            result, _ = quad(integrand, x + 1e-6, 1e3 * x + 10, limit=100)
            return 2 * result
        else:
            return np.array([self.surface_density(R_i) for R_i in np.atleast_1d(R)])
    
    def mass_enclosed(self, r):
        """CoreNFW enclosed mass (approximate)."""
        x = r / self.rs
        x_core = self.r_core / self.rs
        # Approximate formula
        M_s = 4 * np.pi * self.rho_s * self.rs**3
        numerator = (np.log(1 + x) - x_core * np.log(1 + x / x_core))**2
        return M_s * numerator / x
    
    def __repr__(self):
        return f"CoreNFW(M200={self.M200:.2e}, c={self.c:.2f}, r_core/r_s={self.r_core_rel:.2f}, z={self.redshift})"


class SIS(DarkMatterModel):
    """
    Singular Isothermal Sphere (SIS) dark matter profile.
    
    A simple model with constant velocity dispersion:
    rho(r) = sigma_v^2 / (2 * pi * G * r^2)
    """
    
    def __init__(self, sigma_v=200.0, redshift=None, cosmology=None):
        """
        Initialize SIS profile.
        
        Parameters
        ----------
        sigma_v : float
            Velocity dispersion (km/s)
        redshift : float, optional
            Redshift of the halo
        cosmology : astropy.cosmology.Cosmology, optional
            Cosmological model
        """
        super().__init__(redshift=redshift, cosmology=cosmology)
        self.sigma_v = sigma_v
    
    def density(self, r):
        """SIS density profile: rho(r) ~ 1/r^2."""
        G = 4.3e-3  # in units where distances are kpc and velocities are km/s
        return self.sigma_v**2 / (2 * np.pi * G * r**2)
    
    def surface_density(self, R):
        """SIS surface density (convergence)."""
        G = 4.3e-3
        return self.sigma_v / (2 * G * R)
    
    def mass_enclosed(self, r):
        """SIS enclosed mass: M(r) ~ r."""
        G = 4.3e-3
        return self.sigma_v**2 * r / (2 * G)
    
    def __repr__(self):
        return f"SIS(sigma_v={self.sigma_v:.1f} km/s, z={self.redshift})"


class CustomDensity(DarkMatterModel):
    """
    User-defined dark matter model using arbitrary density function.
    """
    
    def __init__(self, density_func, surface_density_func=None, 
                 mass_enclosed_func=None, redshift=None, cosmology=None):
        """
        Initialize custom dark matter model.
        
        Parameters
        ----------
        density_func : callable
            Function rho(r) defining density profile
        surface_density_func : callable, optional
            Function Sigma(R) for surface density (computed if not provided)
        mass_enclosed_func : callable, optional
            Function M(r) for enclosed mass (computed if not provided)
        redshift : float, optional
            Redshift of the halo
        cosmology : astropy.cosmology.Cosmology, optional
            Cosmological model
        """
        super().__init__(redshift=redshift, cosmology=cosmology)
        self._density_func = density_func
        self._surface_density_func = surface_density_func
        self._mass_enclosed_func = mass_enclosed_func
    
    def density(self, r):
        """Evaluate density using provided function."""
        return self._density_func(r)
    
    def surface_density(self, R):
        """Surface density (user-provided or computed)."""
        if self._surface_density_func is not None:
            return self._surface_density_func(R)
        else:
            # Numerical integration (slow, for backup)
            def integrand(z):
                return 2 * self._density_func(np.sqrt(R**2 + z**2))
            
            if np.isscalar(R):
                result, _ = quad(integrand, 0, 1e3 * R, limit=100)
                return result
            else:
                return np.array([self.surface_density(R_i) for R_i in np.atleast_1d(R)])
    
    def mass_enclosed(self, r):
        """Enclosed mass (user-provided or computed)."""
        if self._mass_enclosed_func is not None:
            return self._mass_enclosed_func(r)
        else:
            # Numerical integration (slow, for backup)
            def integrand(r_prime):
                return 4 * np.pi * r_prime**2 * self._density_func(r_prime)
            
            if np.isscalar(r):
                result, _ = quad(integrand, 0, r, limit=100)
                return result
            else:
                return np.array([self.mass_enclosed(r_i) for r_i in np.atleast_1d(r)])
