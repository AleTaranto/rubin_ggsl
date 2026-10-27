"""
Quick verification test for AGENT_sim package structure and imports.
"""

import sys
import numpy as np


def test_imports():
    """Test that all main modules import correctly."""
    print("Testing imports...")
    
    import agent_sim
    print("✓ agent_sim")

    from agent_sim.core import models
    print("✓ agent_sim.core.models")

    from agent_sim.core import lensing
    print("✓ agent_sim.core.lensing")

    from agent_sim.core import observables
    print("✓ agent_sim.core.observables")

    from agent_sim.core import utils
    print("✓ agent_sim.core.utils")

    from agent_sim.pylenslib_imsim_branch import PyLenslibSimulator
    print("✓ agent_sim.pylenslib_imsim_branch.PyLenslibSimulator")

    from agent_sim.slsim_branch import SlsimSimulator
    print("✓ agent_sim.slsim_branch.SlsimSimulator")


def test_basic_models():
    """Test basic model instantiation."""
    print("\nTesting basic models...")
    
    from agent_sim.core.models import NFW, CoreNFW, SIS, Cluster, ClusterGenerator

    # Create NFW model
    nfw = NFW(M200=1e14, c=4.0, redshift=0.3)
    print(f"✓ Created NFW model: {nfw}")

    # Test density evaluation
    rho = nfw.density(100)  # 100 kpc
    print(f"  - Density at 100 kpc: {rho:.2e}")
    assert rho > 0

    # Create CoreNFW
    core_nfw = CoreNFW(M200=1e14, c=4.0, r_core=0.1, redshift=0.3)
    print(f"✓ Created CoreNFW model: {core_nfw}")

    # Create SIS
    sis = SIS(sigma_v=200.0, redshift=0.3)
    print(f"✓ Created SIS model: {sis}")

    # Create cluster
    cluster = Cluster(nfw, name="TestCluster")
    print(f"✓ Created cluster: {cluster}")

    # Test surface density
    sigma = cluster.surface_density(50, 50)
    print(f"  - Surface density at (50, 50): {sigma:.2e}")
    assert sigma > 0


def test_lensing():
    """Test lensing calculations."""
    print("\nTesting lensing calculations...")
    
    from agent_sim.core.models import NFW, Cluster
    from agent_sim.core.lensing import DeflectionField, Magnification

    # Create cluster
    nfw = NFW(M200=1e14, c=4.0, redshift=0.3)
    cluster = Cluster(nfw)

    # Create deflection field
    defl = DeflectionField(cluster)
    print("✓ Created DeflectionField")

    # Test deflection
    alpha_x, alpha_y = defl.deflection_angle(50, 50)
    print(f"  - Deflection at (50, 50): ({alpha_x:.4f}, {alpha_y:.4f})")
    assert not np.isnan(alpha_x) and not np.isnan(alpha_y)

    # Create magnification
    mag = Magnification(cluster)
    print("✓ Created Magnification")

    # Test magnification
    mu = mag.magnification(50, 50)
    print(f"  - Magnification at (50, 50): {mu:.4f}")
    assert not np.isnan(mu)


def test_simulators():
    """Test simulator initialization."""
    print("\nTesting simulators...")
    
    from agent_sim.core.models import NFW, Cluster
    from agent_sim.pylenslib_imsim_branch import PyLenslibSimulator
    from agent_sim.slsim_branch import SlsimSimulator

    # Create cluster
    nfw = NFW(M200=1e14, c=4.0, redshift=0.3)
    cluster = Cluster(nfw)

    # Create PyLensLib simulator
    sim_pyl = PyLenslibSimulator(cluster)
    print("✓ Created PyLenslibSimulator")

    # Create SLSim simulator
    sim_sl = SlsimSimulator(cluster)
    print("✓ Created SlsimSimulator")

    # Test lensing map computation
    maps = sim_pyl.compute_lensing_maps(x_range=(-100, 100),
                                        y_range=(-100, 100), n_pixels=50)
    print(f"  - Magnification map shape: {maps['magnification'].shape}")
    assert maps['magnification'].shape == (50, 50)


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("AGENT_sim Package Verification")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Basic Models", test_basic_models),
        ("Lensing", test_lensing),
        ("Simulators", test_simulators),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, True))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    print("=" * 60)
    print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
