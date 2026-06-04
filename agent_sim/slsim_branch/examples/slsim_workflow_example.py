"""
Example: SLSim branch workflow with Rubinization

This example demonstrates:
1. Creating cluster models
2. Generating mock lensed images
3. Applying Rubinization (realistic observational effects)
4. Comparing mock vs. Rubinized images
"""

import numpy as np
from agent_sim.core.models import NFW, Cluster, SourceDistribution
from agent_sim.slsim_branch import SlsimSimulator
from agent_sim.slsim_branch.imaging import SlsimSourcePlacer, SlsimImageGenerator, SlsimRubinizer


def main():
    print("=" * 60)
    print("AGENT_sim: SLSim Workflow with Rubinization")
    print("=" * 60)
    
    # Step 1: Create cluster
    print("\n[1] Creating cluster model...")
    main_halo = NFW(M200=1e15, c=4.0, redshift=0.3)
    cluster = Cluster(main_halo, name="SLSimCluster")
    
    # Add a few subhalos
    np.random.seed(42)
    for i in range(3):
        mass = 10**np.random.uniform(12, 12.5)
        r = 50 * np.random.random()
        theta = 2 * np.pi * np.random.random()
        
        subhalo = NFW(M200=mass, c=4.0, redshift=0.3)
        cluster.add_subhalo(subhalo, (r * np.cos(theta), r * np.sin(theta)))
    
    print(f"  Cluster: {cluster}")
    print(f"  Subhalos: {len(cluster.subhalos)}")
    
    # Step 2: Initialize SLSim simulator
    print("\n[2] Initializing SLSim simulator...")
    simulator = SlsimSimulator(cluster, grid_size=0.05)
    print("  Simulator ready")
    
    # Step 3: Compute lensing maps
    print("\n[3] Computing lensing maps...")
    lensing_maps = simulator.compute_lensing_maps(
        x_range=(-150, 150),
        y_range=(-150, 150),
        n_pixels=100
    )
    print(f"  Magnification range: [{np.min(lensing_maps['magnification']):.4f}, "
          f"{np.max(lensing_maps['magnification']):.2f}]")
    
    # Step 4: Place sources
    print("\n[4] Placing background sources...")
    source_dist = SourceDistribution(redshift=1.0, density_per_arcmin2=30)
    source_positions = source_dist.sample_positions(n_sources=10, region_size=5)
    print(f"  Placed {len(source_positions)} sources")
    
    # Step 5: Generate mock images
    print("\n[5] Generating perfect mock images...")
    from agent_sim.slsim_branch.imaging import SourcePlacement
    sources_list = []
    for i, pos in enumerate(source_positions):
        source = SourcePlacement(
            source_id=i,
            source_position=tuple(pos),
            redshift=1.0,
            magnitude=20 + np.random.randn() * 2,
            radius_arcsec=0.1
        )
        sources_list.append(source)
    
    # Generate mock images
    try:
        lensed_images, combined_mock = simulator.generate_mock_images(
            sources_list,
            x_range=(-150, 150),
            y_range=(-150, 150),
            image_size=128,
            pixel_scale=0.2
        )
        print(f"  Generated {len(lensed_images)} mock images")
        print(f"  Combined mock image shape: {combined_mock.shape}")
        print(f"  Mock image flux range: [{np.min(combined_mock):.4f}, "
              f"{np.max(combined_mock):.4f}]")
        
        # Step 6: Apply Rubinization
        print("\n[6] Applying Rubinization (realistic effects)...")
        rubinizer = SlsimRubinizer(
            psf_sigma=0.7,
            read_noise_e=10,
            sky_brightness_mag=22,
            gain=3.0
        )
        
        rubinized, metadata = rubinizer.rubinize_slsim(
            combined_mock,
            pixel_scale=0.2,
            use_slsim_psf=False
        )
        
        print(f"  Rubinization method: {metadata.get('method', 'unknown')}")
        print(f"  PSF sigma: {metadata.get('psf_sigma', 'N/A')} arcsec")
        print(f"  Rubinized image shape: {rubinized.shape}")
        print(f"  Rubinized image range [ADU]: [{np.min(rubinized):.1f}, "
              f"{np.max(rubinized):.1f}]")
        
        # Step 7: Compare before/after
        print("\n[7] Comparison: Perfect vs. Rubinized")
        print(f"  Mock max flux: {np.max(combined_mock):.6f}")
        print(f"  Rubinized max (ADU): {np.max(rubinized):.1f}")
        print(f"  Mock std dev: {np.std(combined_mock):.6f}")
        print(f"  Rubinized std dev: {np.std(rubinized):.1f}")
        print(f"  Noise level added: ~{rubinizer.read_noise} e-")
        print(f"  Sky background: {rubinizer.sky_mag} mag/arcsec^2")
        
        # Step 8: GGSL event detection
        print("\n[8] Detecting GGSL events...")
        events, summary = simulator.detect_ggsl_events(
            source_positions,
            x_range=(-150, 150),
            y_range=(-150, 150),
            n_pixels=100,
            source_z=1.0,
            lens_z=0.3
        )
        
        print(f"  Events detected: {summary['n_events']}")
        if summary['n_events'] > 0:
            print(f"  Mean magnification: {summary['mean_magnification']:.2f}x")
            print(f"  Mean arc length: {summary['mean_arc_length']:.2f} arcsec")
        
    except Exception as e:
        print(f"  Error during image generation: {e}")
        print("  (This is expected if SLSim integration is not yet complete)")
    
    print("\n" + "=" * 60)
    print("SLSim workflow example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
