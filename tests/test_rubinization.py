"""
Integration tests for Rubinization modules.

Tests the complete pipeline from mock images to Rubinized images for both branches.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_sim.pylenslib_imsim_branch.imaging import ImSimRubinizer
from agent_sim.slsim_branch.imaging import SlsimRubinizer


def test_imsim_rubinizer_basic():
    """Test basic ImSimRubinizer functionality."""
    print("\n[TEST] ImSimRubinizer basic functionality...")
    
    rubinizer = ImSimRubinizer(
        psf_sigma=0.7,
        read_noise_e=10,
        sky_brightness_mag=22,
        gain=3.0
    )
    
    # Create simple test image
    perfect_image = np.ones((100, 100), dtype=float)
    perfect_image[40:60, 40:60] = 10.0  # Central source
    
    # Test PSF creation
    psf = rubinizer.create_psf_kernel(pixel_scale=0.2, size=31)
    assert psf.shape == (31, 31), f"PSF shape mismatch: {psf.shape}"
    assert abs(np.sum(psf) - 1.0) < 0.01, "PSF not normalized"
    print("  ✓ PSF kernel creation works")
    
    # Test full Rubinization
    np.random.seed(42)
    rubinized, psf_used = rubinizer.rubinize(
        perfect_image,
        pixel_scale=0.2,
        add_psf=True,
        add_noise_flag=True,
        add_cr=False
    )
    
    assert rubinized.shape == perfect_image.shape, "Shape mismatch after Rubinization"
    assert rubinized.dtype == np.ndarray, "Wrong output type"
    assert np.all(rubinized >= 0), "Negative values in Rubinized image"
    print("  ✓ Full Rubinization pipeline works")
    
    # Check that noise was actually added
    assert np.std(rubinized) > 0, "No noise added"
    print("  ✓ Noise simulation works")
    
    # Check output range (should be in ADU)
    max_adu = np.max(rubinized)
    assert 0 < max_adu < 100000, f"Unrealistic ADU range: {max_adu}"
    print(f"  ✓ Output in realistic ADU range: 0-{max_adu:.0f}")
    
    print("  [PASS] ImSimRubinizer basic tests")


def test_imsim_rubinizer_noise_levels():
    """Test that noise levels are reasonable."""
    print("\n[TEST] ImSimRubinizer noise levels...")
    
    rubinizer = ImSimRubinizer(
        psf_sigma=0.7,
        read_noise_e=10,
        sky_brightness_mag=22,
        gain=3.0
    )
    
    # Uniform image (should just have noise)
    perfect_image = np.ones((200, 200), dtype=float) * 5.0
    
    np.random.seed(42)
    rubinized, _ = rubinizer.rubinize(perfect_image, add_psf=False, add_cr=False)
    
    # Calculate noise statistics
    noise = rubinized - 5.0 * 100 / 3.0  # Subtract expected value in ADU
    noise_std = np.std(noise)
    
    # Expected noise should be ~10 e- read noise + sky noise
    # In ADU, read noise should be ~10/3 = 3.3 ADU
    assert noise_std > 1, f"Noise too low: {noise_std:.2f} ADU"
    assert noise_std < 50, f"Noise too high: {noise_std:.2f} ADU"
    
    print(f"  ✓ Noise std dev reasonable: {noise_std:.2f} ADU")
    print("  [PASS] Noise level tests")


def test_slsim_rubinizer_basic():
    """Test basic SlsimRubinizer functionality."""
    print("\n[TEST] SlsimRubinizer basic functionality...")
    
    rubinizer = SlsimRubinizer(
        psf_sigma=0.7,
        read_noise_e=10,
        sky_brightness_mag=22,
        gain=3.0
    )
    
    # Create test image
    perfect_image = np.ones((100, 100), dtype=float)
    perfect_image[40:60, 40:60] = 10.0
    
    # Test Rubinization
    np.random.seed(42)
    rubinized, metadata = rubinizer.rubinize_slsim(
        perfect_image,
        pixel_scale=0.2,
        use_slsim_psf=False
    )
    
    assert rubinized.shape == perfect_image.shape, "Shape mismatch"
    assert rubinized.dtype == np.ndarray, "Wrong type"
    assert np.all(rubinized >= 0), "Negative values"
    
    print("  ✓ SlsimRubinizer pipeline works")
    
    # Check metadata
    assert 'method' in metadata, "Missing method in metadata"
    assert metadata['method'] in ['fallback', 'slsim'], f"Unknown method: {metadata['method']}"
    print(f"  ✓ Metadata recorded: method={metadata['method']}")
    
    print("  [PASS] SlsimRubinizer basic tests")


def test_rubinization_comparison():
    """Compare outputs from both Rubinizers."""
    print("\n[TEST] Comparing PyLensLib and SLSim Rubinization...")
    
    # Same test image
    perfect_image = np.random.randn(50, 50) + 10
    perfect_image = np.maximum(perfect_image, 0)
    
    np.random.seed(42)
    imsim_rubinizer = ImSimRubinizer(psf_sigma=0.7, read_noise_e=10, 
                                     sky_brightness_mag=22, gain=3.0)
    pyl_output, _ = imsim_rubinizer.rubinize(perfect_image, add_cr=False)
    
    np.random.seed(42)
    slsim_rubinizer = SlsimRubinizer(psf_sigma=0.7, read_noise_e=10,
                                     sky_brightness_mag=22, gain=3.0)
    slsim_output, _ = slsim_rubinizer.rubinize_slsim(perfect_image)
    
    # Both should produce reasonable images
    assert pyl_output.shape == slsim_output.shape, "Shape mismatch between branches"
    
    # Outputs may differ due to random seeds, but should be in same ballpark
    ratio = np.max(pyl_output) / (np.max(slsim_output) + 1e-6)
    assert 0.5 < ratio < 2.0, f"Outputs differ too much: ratio={ratio:.2f}"
    
    print(f"  ✓ PyLensLib output: {np.max(pyl_output):.1f} ADU max")
    print(f"  ✓ SLSim output: {np.max(slsim_output):.1f} ADU max")
    print(f"  ✓ Ratio reasonable: {ratio:.2f}x")
    
    print("  [PASS] Comparison tests")


def test_rubinization_with_real_lensing():
    """Test Rubinization with actual lensing output (mock)."""
    print("\n[TEST] Rubinization with simulated lensing map...")
    
    # Simulate lensed source (Einstein ring-like)
    from scipy.ndimage import gaussian_filter
    
    x = np.linspace(-1, 1, 128)
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2)
    
    # Simple ring pattern
    perfect_image = np.exp(-100*(r - 0.3)**2) + 0.01 * np.random.randn(128, 128)
    perfect_image = np.maximum(perfect_image, 0)
    
    rubinizer = ImSimRubinizer(psf_sigma=0.7, read_noise_e=10,
                               sky_brightness_mag=22, gain=3.0)
    
    np.random.seed(42)
    rubinized, _ = rubinizer.rubinize(perfect_image)
    
    # Ring structure should still be visible after Rubinization
    center = rubinized[64, 64]
    edge = np.mean(rubinized[0:10, 0:10])
    
    # Center should be brighter than edge
    assert center > edge, "Ring structure lost after Rubinization"
    
    print(f"  ✓ Ring structure preserved: center={center:.1f}, edge={edge:.1f}")
    print("  [PASS] Realistic lensing Rubinization tests")


def run_all_tests():
    """Run all Rubinization tests."""
    print("=" * 70)
    print("AGENT_sim Rubinization Integration Tests")
    print("=" * 70)
    
    tests = [
        test_imsim_rubinizer_basic,
        test_imsim_rubinizer_noise_levels,
        test_slsim_rubinizer_basic,
        test_rubinization_comparison,
        test_rubinization_with_real_lensing,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
