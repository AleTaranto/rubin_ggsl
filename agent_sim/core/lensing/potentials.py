"""
Gravitational potential and lensing quantities.

Computes lensing quantities from density profiles:
- Gravitational potential: phi(x, y)
- Deflection angles: alpha(x, y)
- Magnification tensor and matrices
- Critical lines and caustics
"""

import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.ndimage import gaussian_filter


class LensingPotential:
    """
    Compute gravitational potential from density distributions.
    
    The lensing potential Psi is related to the surface density by:
    Laplacian(Psi) = 2 * Sigma
    """
    
    def __init__(self, cluster):
        """
        Initialize lensing potential calculator.
        
        Parameters
        ----------
        cluster : Cluster
            Cluster object with density profile
        """
        self.cluster = cluster
    
    def potential(self, X, Y):
        """
        Compute lensing potential psi(X, Y).
        
        Uses Green's function approach:
        psi(X, Y) = (1/2pi) * integral[ Sigma(x', y') * ln(|(X,Y)-(x',y')|) dx' dy' ]
        
        Parameters
        ----------
        X, Y : float or ndarray
            Projected coordinates
        
        Returns
        -------
        psi : float or ndarray
            Lensing potential at (X, Y)
        """
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        R = np.sqrt(X**2 + Y**2)
        
        # Simplified: Use cluster_halo contribution mainly
        # Full integration would be slow; use fast multipole or direct grid method
        
        # For cluster halo: approximate potential from enclosed mass
        M = self.cluster.cluster_halo.mass_enclosed(R)
        # psi ~ M * ln(R) for isothermal-like
        psi = M / 4 * np.log(R**2 + 0.01)
        
        if psi.size == 1:
            return psi.item()
        return psi
    
    def compute_potential_grid(self, x_grid, y_grid):
        """
        Compute potential on a grid (faster for batch processing).
        
        Parameters
        ----------
        x_grid, y_grid : ndarray
            1D arrays defining grid
        
        Returns
        -------
        psi_grid : ndarray of shape (len(y_grid), len(x_grid))
            Potential on grid
        """
        X, Y = np.meshgrid(x_grid, y_grid)
        return self.potential(X, Y)


class DeflectionField:
    """
    Compute deflection angles from lensing potential.
    
    The deflection angle is: alpha = nabla(Psi) = (d Psi/dx, d Psi/dy)
    """
    
    def __init__(self, cluster, grid_size=None):
        """
        Initialize deflection field calculator.
        
        Parameters
        ----------
        cluster : Cluster
            Cluster object
        grid_size : float, optional
            Grid spacing for numerical derivatives
        """
        self.cluster = cluster
        self.grid_size = grid_size or 0.05  # arcsec or kpc
        self.potential_calc = LensingPotential(cluster)
    
    def deflection_angle(self, X, Y):
        """
        Compute deflection angle alpha(X, Y) = [alpha_x, alpha_y].
        
        Computed numerically from potential derivatives.
        
        Parameters
        ----------
        X, Y : float or ndarray
            Coordinates
        
        Returns
        -------
        alpha_x, alpha_y : float or ndarray
            Deflection components
        """
        eps = self.grid_size
        
        # Finite difference: alpha_x = d(psi)/dx
        psi_x_plus = self.potential_calc.potential(X + eps, Y)
        psi_x_minus = self.potential_calc.potential(X - eps, Y)
        alpha_x = (psi_x_plus - psi_x_minus) / (2 * eps)
        
        # alpha_y = d(psi)/dy
        psi_y_plus = self.potential_calc.potential(X, Y + eps)
        psi_y_minus = self.potential_calc.potential(X, Y - eps)
        alpha_y = (psi_y_plus - psi_y_minus) / (2 * eps)
        
        return alpha_x, alpha_y
    
    def compute_deflection_grid(self, x_grid, y_grid):
        """
        Compute deflection field on a grid.
        
        Parameters
        ----------
        x_grid, y_grid : ndarray
            1D arrays defining grid
        
        Returns
        -------
        alpha_x_grid, alpha_y_grid : ndarray of shape (len(y_grid), len(x_grid))
            Deflection components on grid
        """
        X, Y = np.meshgrid(x_grid, y_grid)
        alpha_x, alpha_y = self.deflection_angle(X, Y)
        return alpha_x, alpha_y
    
    def image_position(self, source_x, source_y):
        """
        Map source position to image position via deflection.
        
        Image position: theta = beta + alpha(theta)
        where theta is image, beta is source, alpha is deflection
        
        Parameters
        ----------
        source_x, source_y : float
            Source position
        
        Returns
        -------
        image_x, image_y : float
            Image position (or None if no solution found)
        """
        # Solve for image position iteratively
        # theta = beta + alpha(theta)
        # This is a fixed-point equation
        
        image_x = source_x
        image_y = source_y
        
        for iteration in range(20):  # Newton-Raphson iterations
            alpha_x, alpha_y = self.deflection_angle(image_x, image_y)
            
            # Update: theta_new = beta + alpha(theta)
            image_x_new = source_x + alpha_x
            image_y_new = source_y + alpha_y
            
            # Check convergence
            delta = np.sqrt((image_x_new - image_x)**2 + (image_y_new - image_y)**2)
            if delta < 1e-6:
                return image_x_new, image_y_new
            
            image_x = image_x_new
            image_y = image_y_new
        
        return image_x, image_y


class Magnification:
    """
    Compute magnification matrix and related quantities.
    
    The magnification matrix A = I - kappa - gamma * [cos(2*phi), sin(2*phi); sin(2*phi), -cos(2*phi)]
    where kappa is convergence and gamma is shear.
    """
    
    def __init__(self, cluster, grid_size=None):
        """
        Initialize magnification calculator.
        
        Parameters
        ----------
        cluster : Cluster
            Cluster object
        grid_size : float, optional
            Grid spacing for numerical derivatives
        """
        self.cluster = cluster
        self.grid_size = grid_size or 0.05
        self.deflection = DeflectionField(cluster, grid_size=grid_size)
    
    def convergence(self, X, Y, critical_density=1.0):
        """
        Compute convergence kappa(X, Y).
        
        Convergence: kappa = Sigma(X, Y) / Sigma_crit
        
        Parameters
        ----------
        X, Y : float or ndarray
            Coordinates
        critical_density : float
            Critical surface density
        
        Returns
        -------
        kappa : float or ndarray
            Convergence
        """
        return self.cluster.surface_density(X, Y) / critical_density
    
    def shear(self, X, Y):
        """
        Compute shear components gamma_1, gamma_2.
        
        Shear is computed from second derivatives of potential:
        gamma_1 = (d^2 psi/dx^2 - d^2 psi/dy^2) / 2
        gamma_2 = d^2 psi/dxdy
        
        Parameters
        ----------
        X, Y : float or ndarray
            Coordinates
        
        Returns
        -------
        gamma_1, gamma_2 : float or ndarray
            Shear components
        """
        eps = self.grid_size
        
        # Compute second derivatives numerically
        # d^2 psi/dx^2
        psi_xx_plus = self.deflection.potential_calc.potential(X + eps, Y)
        psi_center = self.deflection.potential_calc.potential(X, Y)
        psi_xx_minus = self.deflection.potential_calc.potential(X - eps, Y)
        psi_xx = (psi_xx_plus - 2*psi_center + psi_xx_minus) / eps**2
        
        # d^2 psi/dy^2
        psi_yy_plus = self.deflection.potential_calc.potential(X, Y + eps)
        psi_yy_minus = self.deflection.potential_calc.potential(X, Y - eps)
        psi_yy = (psi_yy_plus - 2*psi_center + psi_yy_minus) / eps**2
        
        # d^2 psi/dxdy
        psi_xy_pp = self.deflection.potential_calc.potential(X + eps, Y + eps)
        psi_xy_pm = self.deflection.potential_calc.potential(X + eps, Y - eps)
        psi_xy_mp = self.deflection.potential_calc.potential(X - eps, Y + eps)
        psi_xy_mm = self.deflection.potential_calc.potential(X - eps, Y - eps)
        psi_xy = (psi_xy_pp - psi_xy_pm - psi_xy_mp + psi_xy_mm) / (4 * eps**2)
        
        gamma_1 = (psi_xx - psi_yy) / 2
        gamma_2 = psi_xy
        
        return gamma_1, gamma_2
    
    def magnification_tensor(self, X, Y, critical_density=1.0):
        """
        Compute magnification matrix A.
        
        A = [[1 - kappa - gamma_1, -gamma_2],
             [-gamma_2, 1 - kappa + gamma_1]]
        
        The magnification is: mu = 1 / |det(A)|
        The amplification is: |mu|
        
        Parameters
        ----------
        X, Y : float or ndarray
            Coordinates
        critical_density : float
            Critical density
        
        Returns
        -------
        A11, A12, A21, A22 : float or ndarray
            Components of magnification matrix
        """
        kappa = self.convergence(X, Y, critical_density)
        gamma_1, gamma_2 = self.shear(X, Y)
        
        A11 = 1 - kappa - gamma_1
        A12 = -gamma_2
        A21 = -gamma_2
        A22 = 1 - kappa + gamma_1
        
        return A11, A12, A21, A22
    
    def magnification(self, X, Y, critical_density=1.0):
        """
        Compute magnification mu = 1 / |det(A)|.
        
        Parameters
        ----------
        X, Y : float or ndarray
            Coordinates
        critical_density : float
            Critical density
        
        Returns
        -------
        mu : float or ndarray
            Magnification (dimensionless factor)
        """
        A11, A12, A21, A22 = self.magnification_tensor(X, Y, critical_density)
        det_A = A11 * A22 - A12 * A21
        
        # Avoid division by zero
        det_A = np.where(np.abs(det_A) < 1e-10, 1e-10, det_A)
        
        return 1.0 / np.abs(det_A)
    
    def compute_magnification_grid(self, x_grid, y_grid, critical_density=1.0):
        """
        Compute magnification on a grid.
        
        Parameters
        ----------
        x_grid, y_grid : ndarray
            1D arrays defining grid
        critical_density : float
            Critical density
        
        Returns
        -------
        mu_grid : ndarray of shape (len(y_grid), len(x_grid))
            Magnification on grid
        """
        X, Y = np.meshgrid(x_grid, y_grid)
        return self.magnification(X, Y, critical_density)
