"""
Example: Generate mock lensed images with AGENT_sim

This example demonstrates:
1. Creating a cluster with subhalos
2. Placing background sources
3. Generating lensed mock images
4. Detecting GGSL events
"""

import numpy as np
import matplotlib.pyplot as plt
from agent_sim.core.models import NFW, Cluster, SourceDistribution
from agent_sim.pylenslib_imsim_branch import PyLenslibSimulator
from agent_sim.pylenslib_imsim_branch.imaging import SourcePlacer, ImageGenerator


def main():
    print("=" * 60)
    print("AGENT_sim: Mock Lensed Image Generation Example")
    print("=" * 60)
    
    # Step 1: Create cluster with subhalos
    print("\n[1] Creating cluster model...")
    main_halo = NFW(M200=1e15, c=4.0, redshift=0.3)
    cluster = Cluster(main_halo, name="ExampleCluster")
    
    # Add subhalos
    np.random.seed(42)
    for i in range(5):
        mass = 10**np.random.uniform(11.5, 12.5)
        r = 50 * np.sqrt(np.random.random())
        theta = 2 * np.pi * np.random.random()
        
        subhalo = NFW(M200=mass, c=4.0, redshift=0.3)
        cluster.add_subhalo(subhalo, (r * np.cos(theta), r * np.sin(theta)))
    
    print(f"  Created: {cluster}")
    print(f"  Main halo: {cluster.cluster_halo}")
    print(f"  Subhalos: {len(cluster.subhalos)}")
    
    # Step 2: Initialize simulator
    print("\n[2] Initializing PyLensLib simulator...")
    simulator = PyLenslibSimulator(cluster, grid_size=0.05)
    print("  Simulator ready")
    
    # Step 3: Compute lensing maps
    print("\n[3] Computing lensing maps...")
    lensing_maps = simulator.compute_lensing_maps(
        x_range=(-150, 150),
        y_range=(-150, 150),
        n_pixels=100
    )
    print(f"  Deflection field shape: {lensing_maps['deflection_x'].shape}")
    print(f"  Magnification field shape: {lensing_maps['magnification'].shape}")
    print(f"  Max magnification: {np.max(lensing_maps['magnification']):.2f}")
    
    # Step 4: Find critical lines
    print("\n[4] Finding critical lines and caustics...")
    critical_info = simulator.find_critical_lines(
        x_range=(-150, 150),
        y_range=(-150, 150),
        n_pixels=100
    )
    print(f"  Found {len(critical_info['caustics'])} caustics")
    for i, props in enumerate(critical_info['properties']):
        if props:
            print(f"    Caustic {i}: length={props.get('length', 0):.2f}, "
                  f"curvature={props.get('curvature', 0):.4f}")
    
    # Step 5: Place background sources
    print("\n[5] Placing background sources...")
    source_dist = SourceDistribution(redshift=1.0, density_per_arcmin2=50)
    source_positions = source_dist.sample_positions(n_sources=20, region_size=5)
    print(f"  Placed {len(source_positions)} sources")
    
    # Step 6: Generate mock lensed images
    print("\n[6] Generating mock lensed images...")
    
    # Create source placement objects
    from agent_sim.pylenslib_imsim_branch.imaging import SourcePlacement
    sources_list = []
    for i, pos in enumerate(source_positions):
        source = SourcePlacement(
            source_id=i,
            source_position=tuple(pos),
            redshift=1.0,
            magnitude=np.random.uniform(18, 24),
            radius_arcsec=np.random.uniform(0.05, 0.3)
        )
        sources_list.append(source)
    
    # Generate images
    lensed_images, combined_image = simulator.generate_mock_images(
        sources_list,
        x_range=(-150, 150),
        y_range=(-150, 150),
        image_size=256,
        pixel_scale=0.2
    )
    
    print(f"  Generated {len(lensed_images)} lensed images")
    print(f"  Combined image shape: {combined_image.shape}")
    print(f"  Combined image flux: {np.sum(combined_image):.2e}")
    
    # Step 7: Detect GGSL events
    print("\n[7] Detecting GGSL events...")
    events, summary = simulator.detect_ggsl_events(
        source_positions,
        x_range=(-150, 150),
        y_range=(-150, 150),
        n_pixels=100,
        source_z=1.0,
        lens_z=0.3
    )
    
    print(f"  Detected {summary['n_events']} GGSL events")
    if summary['n_events'] > 0:
        print(f"  Mean magnification: {summary['mean_magnification']:.2f}")
        print(f"  Median magnification: {summary['median_magnification']:.2f}")
        print(f"  Mean arc length: {summary['mean_arc_length']:.2f} arcsec")
        print(f"  Median arc length: {summary['median_arc_length']:.2f} arcsec")
    
    # Step 8: Print event details
    if len(events) > 0:
        print("\n[8] GGSL Event Details:")
        for event in events[:5]:  # Show first 5
            print(f"  Event {event.event_id}:")
            print(f"    Source ID: {event.source_id}")
            print(f"    N images: {event.n_images}")
            print(f"    Total mag: {event.total_magnification:.2f}")
            print(f"    Arc length: {event.arc_length_arcsec:.2f} arcsec")
    
    # Step 9: Summary statistics
    print("\n[9] Simulation Summary:")
    print(f"  Total sources placed: {len(sources_list)}")
    print(f"  GGSL events detected: {summary['n_events']}")
    print(f"  Detection efficiency: {100*summary['n_events']/len(sources_list):.1f}%")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
