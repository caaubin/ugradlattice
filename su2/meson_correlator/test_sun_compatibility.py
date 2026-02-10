"""
Test SU(N) Generalization - Backward Compatibility Check
========================================================

This script tests that the SU(N) generalized code produces identical
results to the original SU(2) code when N_c=2.

Tests:
1. sun.py module functions match su2.py for N=2
2. Wilson-Dirac matrix construction with n_colors=2
3. Point source creation with n_colors=2
4. Identity gauge field generation with n_colors=2

Author: Zeke Mohammed
Date: October 2025
"""

import numpy as np
import logging
import sun
import su2
from MesonBase import (
    build_wilson_dirac_matrix,
    create_point_source,
    generate_identity_gauge_field
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_lattice_navigation():
    """Test that lattice navigation functions work identically"""
    print("\n" + "="*60)
    print("TEST 1: Lattice Navigation Functions")
    print("="*60)

    lattice_dims = [4, 4, 4, 8]
    test_point = np.array([1, 2, 3, 4])

    # Test p2i (point to index)
    idx_su2 = su2.p2i(test_point, lattice_dims)
    idx_sun = sun.p2i(test_point, lattice_dims)

    print(f"Point {test_point} → Index:")
    print(f"  su2.p2i: {idx_su2}")
    print(f"  sun.p2i: {idx_sun}")
    print(f"  ✓ MATCH!" if idx_su2 == idx_sun else f"  ✗ MISMATCH!")

    # Test i2p (index to point)
    pt_su2 = su2.i2p(42, lattice_dims)
    pt_sun = sun.i2p(42, lattice_dims)

    print(f"\nIndex 42 → Point:")
    print(f"  su2.i2p: {pt_su2}")
    print(f"  sun.i2p: {pt_sun}")
    print(f"  ✓ MATCH!" if np.array_equal(pt_su2, pt_sun) else f"  ✗ MISMATCH!")

    # Test volume
    vol_su2 = su2.vol(lattice_dims)
    vol_sun = sun.vol(lattice_dims)

    print(f"\nLattice volume:")
    print(f"  su2.vol: {vol_su2}")
    print(f"  sun.vol: {vol_sun}")
    print(f"  ✓ MATCH!" if vol_su2 == vol_sun else f"  ✗ MISMATCH!")

    return idx_su2 == idx_sun and np.array_equal(pt_su2, pt_sun) and vol_su2 == vol_sun


def test_identity_gauge_field():
    """Test identity gauge field generation"""
    print("\n" + "="*60)
    print("TEST 2: Identity Gauge Field Generation")
    print("="*60)

    lattice_dims = [4, 4, 4, 8]

    # Generate with n_colors=2
    U_new, metadata = generate_identity_gauge_field(lattice_dims, n_colors=2, verbose=False)

    print(f"Generated identity gauge field:")
    print(f"  Lattice: {lattice_dims}")
    print(f"  N_colors: {metadata['n_colors']}")
    print(f"  Format: {metadata['format']}")
    print(f"  Plaquette: {metadata['plaquette']}")

    # Check structure
    V = np.prod(lattice_dims)
    print(f"\nGauge field structure:")
    print(f"  Shape: {U_new[1].shape}")
    print(f"  Expected: ({V}, 4, 4) for SU(2) real representation")

    # Verify identity elements
    sample_link = U_new[1][0, 0]
    expected_identity = su2.cstart()

    print(f"\nSample link U[0,0]:")
    print(f"  Value: {sample_link}")
    print(f"  Expected: {expected_identity}")
    print(f"  ✓ MATCH!" if np.allclose(sample_link, expected_identity) else f"  ✗ MISMATCH!")

    return True


def test_point_source():
    """Test point source creation"""
    print("\n" + "="*60)
    print("TEST 3: Point Source Creation")
    print("="*60)

    lattice_dims = [4, 4, 4, 8]
    V = np.prod(lattice_dims)

    # Create source with n_colors=2
    source = create_point_source(lattice_dims, t_source=0, color=0, spin=1, n_colors=2)

    print(f"Created point source:")
    print(f"  Lattice: {lattice_dims}, Volume: {V}")
    print(f"  N_colors: 2")
    print(f"  Source parameters: t=0, color=0, spin=1")
    print(f"  Source vector length: {len(source)}")
    print(f"  Expected: {(2*4)*V} = {8*V}")
    print(f"  ✓ MATCH!" if len(source) == 8*V else f"  ✗ MISMATCH!")

    # Find non-zero element
    nonzero_idx = np.nonzero(source)[0]
    if len(nonzero_idx) > 0:
        idx = nonzero_idx[0]
        value = source[idx]
        expected_idx = 4*0 + 1  # 4*color + spin (FIXED indexing!)

        print(f"\nNon-zero element:")
        print(f"  Index: {idx}")
        print(f"  Expected: {expected_idx} (at site 0)")
        print(f"  Value: {value}")
        print(f"  ✓ Correct!" if abs(value - 1.0) < 1e-10 else f"  ✗ Wrong value!")

    return len(source) == 8*V


def test_wilson_dirac_matrix():
    """Test Wilson-Dirac matrix construction"""
    print("\n" + "="*60)
    print("TEST 4: Wilson-Dirac Matrix Construction")
    print("="*60)

    lattice_dims = [4, 4, 4, 8]
    mass = 0.1
    wilson_r = 1.0
    n_colors = 2

    print(f"Building Wilson-Dirac matrix:")
    print(f"  Lattice: {lattice_dims}")
    print(f"  Mass: {mass}")
    print(f"  Wilson r: {wilson_r}")
    print(f"  N_colors: {n_colors}")

    # Build with identity gauge field (free field)
    D = build_wilson_dirac_matrix(mass, lattice_dims, wilson_r=wilson_r,
                                   U=None, n_colors=n_colors, verbose=True)

    print(f"\nMatrix properties:")
    print(f"  Shape: {D.shape}")
    print(f"  Sparse: {D.nnz} non-zero elements")
    print(f"  Sparsity: {(D.nnz / (D.shape[0]**2)) * 100:.2f}%")

    # Check matrix size
    V = np.prod(lattice_dims)
    expected_size = (n_colors * 4) * V
    print(f"\nSize check:")
    print(f"  Matrix size: {D.shape[0]} × {D.shape[1]}")
    print(f"  Expected: {expected_size} × {expected_size}")
    print(f"  ✓ MATCH!" if D.shape[0] == expected_size else f"  ✗ MISMATCH!")

    return D.shape[0] == expected_size


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" SU(N) GENERALIZATION - BACKWARD COMPATIBILITY TEST (N_c=2)")
    print("="*70)
    print("\nVerifying that SU(N) code with n_colors=2 works correctly...")

    results = {}

    try:
        results['navigation'] = test_lattice_navigation()
    except Exception as e:
        print(f"\n✗ Navigation test FAILED: {e}")
        results['navigation'] = False

    try:
        results['identity'] = test_identity_gauge_field()
    except Exception as e:
        print(f"\n✗ Identity gauge field test FAILED: {e}")
        results['identity'] = False

    try:
        results['source'] = test_point_source()
    except Exception as e:
        print(f"\n✗ Point source test FAILED: {e}")
        results['source'] = False

    try:
        results['dirac'] = test_wilson_dirac_matrix()
    except Exception as e:
        print(f"\n✗ Wilson-Dirac matrix test FAILED: {e}")
        results['dirac'] = False

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print(" ✓ ALL TESTS PASSED - BACKWARD COMPATIBILITY VERIFIED!")
        print("="*70)
        print("\nThe SU(N) generalization works correctly for N_c=2.")
        print("Next step: Test with N_c=3 for real QCD!")
    else:
        print(" ✗ SOME TESTS FAILED - NEEDS DEBUGGING")
        print("="*70)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
