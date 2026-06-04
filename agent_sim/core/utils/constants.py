"""Physical constants in various unit systems."""

import numpy as np

# SI units
G_SI = 6.674e-11  # m^3 kg^-1 s^-2
c_SI = 2.998e8    # m/s
M_sun_SI = 1.989e30  # kg

# Astronomical units (commonly used in lensing)
# Distance units: kpc (kiloparsec)
# Velocity units: km/s
# Mass units: solar masses (M_sun)

G_AST = 4.3e-3  # kpc km^2 s^-2 M_sun^-1 (gravitational constant in astronomical units)
c_AST = 299792.458  # km/s

# Conversion factors
kpc_to_m = 3.086e22  # meters per kpc
Mpc_to_m = 3.086e25  # meters per Mpc
Gyr_to_s = 3.154e16  # seconds per Gyr

# Cosmology
H0 = 67.4  # Hubble constant (km/s/Mpc, Planck 2018)
Omega_m = 0.315  # Matter density
Omega_L = 0.685  # Dark energy density
Omega_k = 0.0    # Spatial curvature

# Critical density
rho_crit_0 = 2.775e11 * H0**2  # M_sun/kpc^3 (at z=0)

def critical_surface_density(z_s, z_l, z_ls=None, H0=67.4):
    """
    Compute critical surface density for lensing.
    
    Sigma_crit = (c^2 / (4*pi*G)) * (D_A(z_s)) / (D_A(z_l) * D_A(z_ls))
    
    Parameters
    ----------
    z_s : float
        Source redshift
    z_l : float
        Lens redshift
    z_ls : float, optional
        Lens-source redshift (z_s - z_l). If None, computed from z_s, z_l
    H0 : float
        Hubble constant (km/s/Mpc)
    
    Returns
    -------
    Sigma_crit : float
        Critical surface density (M_sun/kpc^2)
    """
    from scipy.integrate import quad
    
    def integrand(z, H0):
        # Simplified: assume flat Lambda-CDM
        E_z = np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
        return 1.0 / E_z
    
    # Comoving distance: D_c(z) = c/H0 * integral_0^z dz'/E(z')
    D_c_l, _ = quad(integrand, 0, z_l, args=(H0,))
    D_c_s, _ = quad(integrand, 0, z_s, args=(H0,))
    D_c_ls, _ = quad(integrand, z_l, z_s, args=(H0,))
    
    # Angular diameter distances
    D_A_l = D_c_l / (1 + z_l)
    D_A_s = D_c_s / (1 + z_s)
    D_A_ls = D_c_ls / (1 + z_s)
    
    # Critical surface density (c^2/(4*pi*G) = 1.66e15 M_sun/kpc^2 in our units)
    factor = c_AST**2 / (4 * np.pi * G_AST)  # M_sun/kpc^2
    
    if D_A_l > 0 and D_A_ls > 0:
        Sigma_crit = factor * D_A_s / (D_A_l * D_A_ls)
    else:
        Sigma_crit = np.inf
    
    return Sigma_crit
