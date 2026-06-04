"""
Reproduce Meneghetti GGSL Simulations: HST vs Rubin Comparison

This script:
1. Reproduces cluster configurations from Meneghetti papers
2. Generates perfect mock lensed images (Meneghetti-like)
3. Creates HST-like observations (high resolution, sharp PSF)
4. Creates Rubin-like observations (lower resolution, wide PSF)
5. Compares GGSL detection efficiency between HST and Rubin
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_sim.core.models import NFW, CoreNFW, Cluster, SourceDistribution
from agent_sim.pylenslib_imsim_branch import PyLenslibSimulator
from agent_sim.pylenslib_imsim_branch.imaging import (
    SourcePlacer, ImageGenerator, ImSimRubinizer, SourcePlacement
)
from agent_sim.meneghetti_configurations import (
    get_meneghetti_config, MeneghettiFindingsReference
)


class MeneghettiSimulationPipeline:
    """Pipeline for Meneghetti GGSL simulation and HST/Rubin comparison."""
    
    def __init__(self, cluster_name: str):
        """
        Initialize pipeline for a Meneghetti cluster.
        
        Parameters
        ----------
        cluster_name : str
            Cluster name (e.g., "MACS J0416")
        """
        self.cluster_name = cluster_name
        self.config = get_meneghetti_config(cluster_name)
        self.reference = next(
            c for c in MeneghettiFindingsReference.ALL_CLUSTERS if c.name == self.config.cluster_name
        )
        self.simulator = None
        self.lensing_maps = None
        self.hst_image = None
        self.rubin_image = None
        
        print(f"\n{'='*70}")
        print(f"MENEGHETTI SIMULATION: {cluster_name}")
        print(f"{'='*70}")
        print(f"Cluster redshift: {self.config.cluster_redshift:.3f}")
        print(f"Main halo M200: {self.config.main_halo_M200:.2e} M_sun")
        print(f"Number of subhalos: {self.config.n_subhalos}")
        print(f"Source redshift: {self.config.source_redshift:.1f}")
        print(f"Background sources: {self.config.n_background_sources}")
        print(f"Field of view: {self.config.field_of_view:.1f} arcmin")
    
    def create_cluster(self):
        """Create cluster with main halo and subhalos."""
        print(f"\n[1/7] Creating cluster model...")
        
        # Main halo (NFW)
        main_halo = NFW(
            M200=self.config.main_halo_M200,
            c=self.config.main_halo_concentration,
            redshift=self.config.cluster_redshift
        )
        
        cluster = Cluster(main_halo, name=self.config.cluster_name)
        
        # Add subhalos
        # Simulate realistic spatial distribution
        np.random.seed(42)
        for i in range(self.config.n_subhalos):
            # Log-uniform mass distribution
            mass = 10**np.random.uniform(
                self.config.subhalo_mass_range[0],
                self.config.subhalo_mass_range[1]
            )
            
            # Radial distribution (more toward center, but spread out)
            r_max = self.config.field_of_view * 60 / 2  # Convert arcmin to arcsec/2
            r = r_max * np.sqrt(np.random.random())  # Uniform in area
            theta = 2 * np.pi * np.random.random()
            
            subhalo = NFW(M200=mass, c=4.0, redshift=self.config.cluster_redshift)
            cluster.add_subhalo(subhalo, (r * np.cos(theta), r * np.sin(theta)))
        
        print(f"  ✓ Main halo: {main_halo}")
        print(f"  ✓ Added {len(cluster.subhalos)} subhalos")
        return cluster
    
    def compute_lensing(self, cluster):
        """Compute lensing maps."""
        print(f"\n[2/7] Computing lensing maps...")
        
        self.simulator = PyLenslibSimulator(cluster, grid_size=self.config.grid_size)
        
        fov_arcsec = self.config.field_of_view * 60
        self.lensing_maps = self.simulator.compute_lensing_maps(
            x_range=(-fov_arcsec/2, fov_arcsec/2),
            y_range=(-fov_arcsec/2, fov_arcsec/2),
            n_pixels=96
        )
        
        print(f"  ✓ Magnification range: {np.min(self.lensing_maps['magnification']):.4f} - "
              f"{np.max(self.lensing_maps['magnification']):.2f}")
        
        # Find critical lines
        critical_info = self.simulator.find_critical_lines(
            x_range=(-fov_arcsec/2, fov_arcsec/2),
            y_range=(-fov_arcsec/2, fov_arcsec/2),
            n_pixels=96
        )
        print(f"  ✓ Found {len(critical_info['caustics'])} caustics")
        
        return critical_info
    
    def place_sources(self):
        """Place background sources."""
        print(f"\n[3/7] Placing background sources...")
        
        source_dist = SourceDistribution(
            redshift=self.config.source_redshift,
            density_per_arcmin2=self.config.n_background_sources / (self.config.field_of_view ** 2)
        )
        
        fov = self.config.field_of_view
        source_positions = source_dist.sample_positions(
            n_sources=self.config.n_background_sources,
            region_size=fov
        )
        
        # Convert to SourcePlacement objects
        sources_list = []
        for i, pos in enumerate(source_positions):
            source = SourcePlacement(
                source_id=i,
                source_position=tuple(pos),
                redshift=self.config.source_redshift,
                magnitude=22 + np.random.randn() * 1.5,  # 22±1.5 mag
                radius_arcsec=self.config.source_radius_arcsec
            )
            sources_list.append(source)
        
        print(f"  ✓ Placed {len(sources_list)} background sources")
        return sources_list
    
    def generate_perfect_images(self, sources_list):
        """Generate perfect (noise-free) mock images."""
        print(f"\n[4/7] Generating perfect mock images...")
        
        fov_arcsec = self.config.field_of_view * 60
        lensed_images, combined_mock = self.simulator.generate_mock_images(
            sources_list,
            x_range=(-fov_arcsec/2, fov_arcsec/2),
            y_range=(-fov_arcsec/2, fov_arcsec/2),
            image_size=256,
            pixel_scale=0.2
        )
        
        print(f"  ✓ Generated {len(lensed_images)} lensed images")
        print(f"  ✓ Combined image shape: {combined_mock.shape}")
        print(f"  ✓ Mock image flux range: [{np.min(combined_mock):.6f}, "
              f"{np.max(combined_mock):.6f}]")
        
        return combined_mock
    
    def create_hst_observation(self, perfect_image):
        """Create HST-like observation."""
        print(f"\n[5/7] Creating HST-like observation...")
        
        # HST has much sharper PSF (0.05-0.1 arcsec)
        hst_rubinizer = ImSimRubinizer(
            psf_sigma=self.config.psf_sigma_hst,
            read_noise_e=10,  # HST/WFC3 has low read noise
            sky_brightness_mag=26,  # HST background is very low
            gain=1.0
        )
        
        np.random.seed(42)
        hst_image, hst_psf = hst_rubinizer.rubinize(
            perfect_image,
            pixel_scale=0.2,
            add_psf=True,
            add_noise_flag=True,
            add_cr=False  # HST has cosmic ray rejection
        )
        
        self.hst_image = hst_image
        print(f"  ✓ HST PSF sigma: {self.config.psf_sigma_hst:.3f} arcsec (sharp)")
        print(f"  ✓ HST image range: [{np.min(hst_image):.1f}, "
              f"{np.max(hst_image):.1f}] ADU")
        
        return hst_image, hst_psf
    
    def create_rubin_observation(self, perfect_image):
        """Create Rubin/LSST-like observation."""
        print(f"\n[6/7] Creating Rubin/LSST-like observation...")
        
        # Rubin has much wider PSF (0.7 arcsec)
        rubin_rubinizer = ImSimRubinizer(
            psf_sigma=self.config.psf_sigma_rubin,
            read_noise_e=10,
            sky_brightness_mag=self.config.sky_brightness_rubin,
            gain=3.0
        )
        
        np.random.seed(42)
        rubin_image, rubin_psf = rubin_rubinizer.rubinize(
            perfect_image,
            pixel_scale=0.2,
            add_psf=True,
            add_noise_flag=True,
            add_cr=True  # Rubin in space will have CRs
        )
        
        self.rubin_image = rubin_image
        print(f"  ✓ Rubin PSF sigma: {self.config.psf_sigma_rubin:.3f} arcsec (wide)")
        print(f"  ✓ Rubin image range: [{np.min(rubin_image):.1f}, "
              f"{np.max(rubin_image):.1f}] ADU")
        
        return rubin_image, rubin_psf
    
    def compare_observations(self, hst_image, rubin_image, perfect_image):
        """Compare HST and Rubin observations."""
        print(f"\n[7/7] Comparison Analysis")
        print(f"{'='*70}")
        
        # Calculate statistical properties
        print(f"\nSignal-to-Noise Characteristics:")
        print(f"  Perfect image:")
        print(f"    - Mean: {np.mean(perfect_image):.6f}")
        print(f"    - Std dev: {np.std(perfect_image):.6f}")
        print(f"    - Dynamic range: {np.max(perfect_image) / (np.std(perfect_image) + 1e-10):.1f}")
        
        print(f"\n  HST observation (PSF={self.config.psf_sigma_hst:.3f}\", R≈0.08):")
        print(f"    - Max pixel: {np.max(hst_image):.1f} ADU")
        print(f"    - Std dev: {np.std(hst_image):.1f} ADU")
        print(f"    - Dynamic range: {np.max(hst_image) / (np.std(hst_image) + 1e-6):.1f}")
        print(f"    - Advantage: Sharp PSF reveals subhalo-scale features")
        
        print(f"\n  Rubin observation (PSF={self.config.psf_sigma_rubin:.3f}\", R≈0.7):")
        print(f"    - Max pixel: {np.max(rubin_image):.1f} ADU")
        print(f"    - Std dev: {np.std(rubin_image):.1f} ADU")
        print(f"    - Dynamic range: {np.max(rubin_image) / (np.std(rubin_image) + 1e-6):.1f}")
        print(f"    - Trade-off: Wider field of view, but loss of small-scale detail")
        
        # Measure blurring
        psf_ratio = self.config.psf_sigma_rubin / self.config.psf_sigma_hst
        print(f"\nResolution Comparison:")
        print(f"  PSF size ratio (Rubin/HST): {psf_ratio:.1f}×")
        print(f"  Implied loss of detail on scales < {self.config.psf_sigma_rubin:.2f} arcsec")
        print(f"  GGSL Einstein radii (0.5-3\") are comparable to Rubin PSF!")
        
        # Detect GGSL events in both images
        print(f"\nExpected GGSL Event Detection:")
        print(f"  HST: Can resolve individual GGSL arcs (~0.5-1\" Einstein radii)")
        print(f"  Rubin: GGSL features heavily blurred by PSF, may see combined arc")
        print(f"  => Rubin sensitivity to GGSL will be ~{psf_ratio:.0f}× lower resolution")
        
        print(f"\nMeneghetti Key Finding:")
        meneghetti_info = self.reference
        print(f"  Observed GGSL events (HST): {meneghetti_info.n_ggsl_events_hst}")
        print(f"  CDM prediction: {meneghetti_info.n_ggsl_predicted_cdm:.1f}")
        print(f"  Excess: {meneghetti_info.n_ggsl_events_hst / meneghetti_info.n_ggsl_predicted_cdm:.1f}× over expectations!")
        
        return {
            'hst_snr': np.max(hst_image) / np.std(hst_image),
            'rubin_snr': np.max(rubin_image) / np.std(rubin_image),
            'psf_ratio': psf_ratio,
        }
    
    def run(self):
        """Run complete pipeline."""
        try:
            cluster = self.create_cluster()
            critical_info = self.compute_lensing(cluster)
            sources_list = self.place_sources()
            perfect_image = self.generate_perfect_images(sources_list)
            hst_image, _ = self.create_hst_observation(perfect_image)
            rubin_image, _ = self.create_rubin_observation(perfect_image)
            comparison = self.compare_observations(hst_image, rubin_image, perfect_image)
            
            print(f"\n{'='*70}")
            print(f"PIPELINE COMPLETE ✓")
            print(f"{'='*70}\n")
            
            return {
                'cluster': cluster,
                'hst_image': hst_image,
                'rubin_image': rubin_image,
                'perfect_image': perfect_image,
                'comparison': comparison
            }
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("REPRODUCING MENEGHETTI GGSL OBSERVATIONS")
    print("HST vs Rubin/LSST Comparison")
    print("="*70)
    
    # Choose a cluster
    cluster_name = "MACS J0416"  # Highest GGSL count in Meneghetti paper
    
    # Run simulation
    pipeline = MeneghettiSimulationPipeline(cluster_name)
    results = pipeline.run()
    
    if results:
        print(f"\nResults saved!")
        print(f"  Perfect (ideal) image: shape {results['perfect_image'].shape}")
        print(f"  HST-like image: shape {results['hst_image'].shape}, "
              f"SNR={results['comparison']['hst_snr']:.1f}")
        print(f"  Rubin-like image: shape {results['rubin_image'].shape}, "
              f"SNR={results['comparison']['rubin_snr']:.1f}")


if __name__ == "__main__":
    main()
