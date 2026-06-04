"""
Cluster mass distribution models.

Clusters are composed of a cluster-scale halo (typically an NFW or similar)
plus a collection of subhalos and smaller galaxies.
"""

import numpy as np
from .dark_matter import DarkMatterModel, NFW


class Cluster:
    """
    Composite cluster model: cluster halo + subhalos + galaxies.
    
    A cluster consists of:
    1. Main cluster-scale dark matter halo
    2. Collection of subhalos (dwarf galaxies)
    3. Optional stellar components
    """
    
    def __init__(self, cluster_halo, subhalos=None, galaxies=None, name="Cluster"):
        """
        Initialize cluster model.
        
        Parameters
        ----------
        cluster_halo : DarkMatterModel
            Main cluster-scale dark matter halo
        subhalos : list of DarkMatterModel, optional
            List of subhalo models with positions (to be stored separately)
        galaxies : list of dict, optional
            List of galaxy models with positions
        name : str
            Name identifier for the cluster
        """
        self.cluster_halo = cluster_halo
        self.subhalos = subhalos or []
        self.galaxies = galaxies or []
        self.name = name
        
        # Positions will be stored in separate arrays
        self.subhalo_positions = []  # List of (x, y) in arcsec or kpc
        self.galaxy_positions = []
        self.galaxy_masses = []
    
    def add_subhalo(self, halo_model, position):
        """
        Add a subhalo to the cluster.
        
        Parameters
        ----------
        halo_model : DarkMatterModel
            The subhalo density model
        position : tuple
            (x, y) position of subhalo center
        """
        self.subhalos.append(halo_model)
        self.subhalo_positions.append(position)
    
    def add_galaxy(self, mass, position, size=None):
        """
        Add a point-mass galaxy to the cluster.
        
        Parameters
        ----------
        mass : float
            Galaxy mass (in solar masses)
        position : tuple
            (x, y) position
        size : float, optional
            Effective radius (if extended model needed)
        """
        self.galaxies.append({"mass": mass, "size": size})
        self.galaxy_positions.append(position)
        self.galaxy_masses.append(mass)
    
    def surface_density(self, X, Y):
        """
        Total surface density at projected positions (X, Y).
        
        Parameters
        ----------
        X, Y : float or array_like
            Projected coordinates
        
        Returns
        -------
        Sigma : float or ndarray
            Total surface density at (X, Y)
        """
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        R = np.sqrt(X**2 + Y**2)
        
        # Cluster halo contribution
        sigma = self.cluster_halo.surface_density(R)
        
        # Add subhalos (point masses or extended)
        for i, subhalo in enumerate(self.subhalos):
            x_i, y_i = self.subhalo_positions[i]
            R_i = np.sqrt((X - x_i)**2 + (Y - y_i)**2)
            sigma = sigma + subhalo.surface_density(R_i)
        
        # Add point-mass galaxies
        for i, gal in enumerate(self.galaxies):
            x_g, y_g = self.galaxy_positions[i]
            R_g = np.sqrt((X - x_g)**2 + (Y - y_g)**2)
            # Point mass: Sigma = M / (2*pi*R) (not quite physical but effective)
            # Better: use small softening
            softening = 0.01  # arcsec or kpc
            R_soft = np.sqrt(R_g**2 + softening**2)
            sigma = sigma + gal["mass"] / (2 * np.pi * R_soft**2) * softening
        
        if sigma.size == 1:
            return sigma.item()
        return sigma
    
    def mass_enclosed_2d(self, R):
        """
        Total projected mass enclosed within radius R.
        
        Parameters
        ----------
        R : float
            Projected radius
        
        Returns
        -------
        M : float
            Total projected mass within R
        """
        M = self.cluster_halo.mass_enclosed(R)
        
        for i, subhalo in enumerate(self.subhalos):
            x_i, y_i = self.subhalo_positions[i]
            # Project distance
            r_proj = np.sqrt(x_i**2 + y_i**2)
            if r_proj < R:
                # Approximate: add fraction of subhalo mass
                M = M + subhalo.mass_enclosed(np.inf)  # Total subhalo mass
        
        for gal_mass in self.galaxy_masses:
            M = M + gal_mass
        
        return M
    
    def __repr__(self):
        n_sub = len(self.subhalos)
        n_gal = len(self.galaxies)
        return f"{self.name}(cluster={self.cluster_halo}, subhalos={n_sub}, galaxies={n_gal})"


class ClusterGenerator:
    """
    Generate cluster models with various configurations.
    """
    
    @staticmethod
    def simple_cluster(M200_cluster=1e15, c_cluster=4.0, 
                       n_subhalos=10, M200_sub_range=(1e11, 1e12),
                       redshift=0.3, seed=None):
        """
        Generate a simple cluster with subhalos.
        
        Parameters
        ----------
        M200_cluster : float
            Cluster virial mass
        c_cluster : float
            Cluster concentration
        n_subhalos : int
            Number of subhalos
        M200_sub_range : tuple
            Range of subhalo masses (min, max)
        redshift : float
            Redshift
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        cluster : Cluster
            Generated cluster model
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Create main halo
        main_halo = NFW(M200=M200_cluster, c=c_cluster, redshift=redshift)
        cluster = Cluster(main_halo, name="GeneratedCluster")
        
        # Add subhalos
        for i in range(n_subhalos):
            # Random mass in range
            mass = 10**np.random.uniform(np.log10(M200_sub_range[0]), 
                                         np.log10(M200_sub_range[1]))
            # Random position (within some radius)
            r = np.random.uniform(0, 500)  # kpc or arcsec
            theta = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            subhalo = NFW(M200=mass, c=4.0, redshift=redshift)
            cluster.add_subhalo(subhalo, (x, y))
        
        return cluster
