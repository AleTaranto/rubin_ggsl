"""
Galaxy source distributions for strong lensing simulations.

Defines how background source galaxies are distributed (spatially and 
in terms of properties like luminosity, size, etc.)
"""

import numpy as np
from scipy.stats import powerlaw, lognorm


class SourceDistribution:
    """
    Distribution of background source galaxies.
    """
    
    def __init__(self, redshift=1.0, density_per_arcmin2=100):
        """
        Initialize source distribution.
        
        Parameters
        ----------
        redshift : float
            Redshift of source population
        density_per_arcmin2 : float
            Source number density (sources per arcmin^2)
        """
        self.redshift = redshift
        self.density = density_per_arcmin2
    
    def sample_positions(self, n_sources, region_size=10.0):
        """
        Sample source positions in a region.
        
        Parameters
        ----------
        n_sources : int
            Number of sources to sample
        region_size : float
            Size of region (arcmin) where sources are sampled
        
        Returns
        -------
        positions : ndarray of shape (n_sources, 2)
            (x, y) positions of sources (in arcsec)
        """
        # Uniform distribution in region
        x = np.random.uniform(-region_size/2, region_size/2, n_sources) * 60  # to arcsec
        y = np.random.uniform(-region_size/2, region_size/2, n_sources) * 60
        return np.column_stack([x, y])
    
    def sample_luminosities(self, n_sources, magnitude_range=(18, 26)):
        """
        Sample source luminosities/magnitudes.
        
        Parameters
        ----------
        n_sources : int
            Number of sources
        magnitude_range : tuple
            Min and max apparent magnitudes
        
        Returns
        -------
        magnitudes : ndarray
            Apparent magnitudes of sources
        """
        # Uniform in magnitude (simplified; could use Schechter function)
        return np.random.uniform(magnitude_range[0], magnitude_range[1], n_sources)
    
    def sample_sizes(self, n_sources, size_range=(0.05, 0.3)):
        """
        Sample source effective radii.
        
        Parameters
        ----------
        n_sources : int
            Number of sources
        size_range : tuple
            Min and max effective radius (arcsec)
        
        Returns
        -------
        sizes : ndarray
            Effective radii (arcsec)
        """
        # Log-normal distribution in size
        log_sizes = np.random.normal(np.log(np.mean(size_range)), 0.3, n_sources)
        sizes = np.exp(log_sizes)
        sizes = np.clip(sizes, size_range[0], size_range[1])
        return sizes
    
    def __repr__(self):
        return f"SourceDistribution(z={self.redshift}, density={self.density:.1f}/arcmin^2)"


class CatalogSourceDistribution(SourceDistribution):
    """
    Source distribution based on external catalog.
    """
    
    def __init__(self, catalog_path, redshift=1.0):
        """
        Initialize from catalog.
        
        Parameters
        ----------
        catalog_path : str
            Path to source catalog (FITS or HDF5)
        redshift : float
            Redshift of source population
        """
        super().__init__(redshift=redshift)
        self.catalog_path = catalog_path
        self.catalog = self._load_catalog()
    
    def _load_catalog(self):
        """Load source catalog from file."""
        # Placeholder: would load FITS/HDF5 file here
        raise NotImplementedError("Catalog loading not implemented yet")
    
    def sample_positions(self, n_sources=None, **kwargs):
        """Sample positions from catalog."""
        if n_sources is None:
            return self.catalog['x'], self.catalog['y']
        else:
            indices = np.random.choice(len(self.catalog), n_sources, replace=True)
            return self.catalog['x'][indices], self.catalog['y'][indices]


class ClusteredSourceDistribution(SourceDistribution):
    """
    Sources with spatial clustering (e.g., galaxy clusters).
    """
    
    def __init__(self, redshift=1.0, density_per_arcmin2=100, 
                 cluster_positions=None, cluster_scale=1.0):
        """
        Initialize clustered distribution.
        
        Parameters
        ----------
        redshift : float
            Redshift of source population
        density_per_arcmin2 : float
            Source number density
        cluster_positions : ndarray, optional
            Positions of clusters (if None, random Poisson)
        cluster_scale : float
            Clustering scale (arcmin)
        """
        super().__init__(redshift=redshift, density_per_arcmin2=density_per_arcmin2)
        self.cluster_positions = cluster_positions
        self.cluster_scale = cluster_scale
    
    def sample_positions(self, n_sources, region_size=10.0):
        """Sample positions with clustering."""
        # Simplified: Poisson cluster process
        # First place cluster centers, then sources around them
        if self.cluster_positions is None:
            n_clusters = max(1, int(region_size**2 / (self.cluster_scale**2 * 4)))
            cluster_x = np.random.uniform(-region_size/2, region_size/2, n_clusters) * 60
            cluster_y = np.random.uniform(-region_size/2, region_size/2, n_clusters) * 60
        else:
            cluster_x = self.cluster_positions[:, 0]
            cluster_y = self.cluster_positions[:, 1]
        
        # Place sources around clusters
        positions = []
        sources_per_cluster = n_sources // len(cluster_x)
        for cx, cy in zip(cluster_x, cluster_y):
            # Gaussian around cluster center
            n = sources_per_cluster
            r = np.random.normal(0, self.cluster_scale * 30, n)  # to arcsec
            theta = np.random.uniform(0, 2*np.pi, n)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            positions.append(np.column_stack([x, y]))
        
        positions = np.vstack(positions)
        # Trim or pad to exact n_sources
        if len(positions) > n_sources:
            positions = positions[:n_sources]
        elif len(positions) < n_sources:
            x = np.random.uniform(-region_size/2, region_size/2, n_sources - len(positions)) * 60
            y = np.random.uniform(-region_size/2, region_size/2, n_sources - len(positions)) * 60
            positions = np.vstack([positions, np.column_stack([x, y])])
        
        return positions
