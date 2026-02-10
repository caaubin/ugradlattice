#!/usr/bin/env python3
"""
================================================================================
Production Ensemble Generator for Lattice QCD
================================================================================

This program generates an ensemble of thermalized SU(3) gauge configurations
for use in statistical analysis of QCD observables. Ensemble averaging is
essential for extracting physical quantities with controlled statistical errors.

WHY ENSEMBLE AVERAGING?
-----------------------
In lattice QCD, observables are computed as path integral expectation values:

    <O> = ∫ DU O[U] exp(-S[U]) / ∫ DU exp(-S[U])

Monte Carlo methods estimate this by generating configurations {U_i} from the
Boltzmann distribution exp(-S[U]) and averaging:

    <O> ≈ (1/N) Σᵢ O[Uᵢ]

The statistical error decreases as 1/√N, so more configurations give more
precise results. Typical production runs use 100-1000 configurations.

DECORRELATION:
--------------
Successive configurations from a Markov chain are correlated. To obtain
statistically independent samples, we run "separation sweeps" between
saved configurations. The autocorrelation time τ determines how many
sweeps are needed:

    - Near phase transitions: τ can be large (critical slowing down)
    - Deep in confined phase: τ ~ 5-10 sweeps typically
    - Rule of thumb: separation = 2-3 × τ

THERMALIZATION:
---------------
Before generating the ensemble, we must thermalize from the initial
configuration (cold or hot start) to reach equilibrium. This is monitored
by watching the plaquette evolution:

    - Start: <P> = 1.0 (cold) or 0.0 (hot)
    - Equilibrium: <P> ≈ 0.60 at β=6.0
    - Thermalization time: 20-100 sweeps typically

OUTPUT FORMAT:
--------------
Each configuration is saved as a pickle file containing:
    - U: The gauge field array, shape (V, 4, 3, 3)
    - plaquette: Average plaquette value
    - beta: Inverse coupling
    - Lx, Ly, Lz, Lt: Lattice dimensions
    - n_colors: Number of colors (3 for SU(3))
    - timestamp: When configuration was generated

An ensemble metadata file summarizes the run parameters and plaquette statistics.

USAGE:
------
    # Generate 10 configurations on 6³×12 lattice at β=6.0
    python generate_ensemble.py --beta 6.0 --ls 6 --lt 12 --configs 10

    # Larger ensemble with more thermalization
    python generate_ensemble.py --beta 6.0 --ls 8 --lt 16 --configs 50 --therm 100

Author: Zeke Mohammed
Advisor: Dr. Aubin
Institution: Fordham University
Date: January 2026
================================================================================
"""

import numpy as np
import pickle
import sys
import os
import argparse
from datetime import datetime

# Add parent directory for imports
# Adjust this path as needed for your installation
sys.path.insert(0, '..')

from Thermal_Generator_SU3 import generate_su3_config, average_plaquette_su3, thermalize_su3


def generate_ensemble(beta, lattice_dims, n_configs, n_therm=50, n_separation=10, output_dir='configs'):
    """
    Generate an ensemble of thermalized SU(3) gauge configurations.

    This is the main production function that:
    1. Creates the output directory
    2. Generates and thermalizes the first configuration
    3. Generates subsequent configurations with separation sweeps
    4. Saves all configurations and metadata

    Parameters:
    -----------
    beta : float
        Inverse coupling β = 6/g². Controls the lattice spacing.
        - β = 5.7: Near deconfinement (a ≈ 0.17 fm)
        - β = 6.0: Standard choice (a ≈ 0.1 fm)
        - β = 6.5: Fine lattice (a ≈ 0.065 fm)

    lattice_dims : list of int
        [Lx, Ly, Lz, Lt] lattice dimensions.
        Physical volume: V = (Lx × a)³ × (Lt × a)
        For finite-temperature studies, Lt sets the temperature: T = 1/(Lt × a)

    n_configs : int
        Number of configurations to generate.
        Statistical error scales as 1/√n_configs.
        Typical values: 10-100 for exploratory, 100-1000 for production.

    n_therm : int
        Number of thermalization sweeps for first configuration.
        Should be large enough to reach equilibrium from cold start.
        Monitor plaquette evolution to verify.
        Default: 50 (sufficient for small lattices)

    n_separation : int
        Number of sweeps between saved configurations.
        Should be larger than the autocorrelation time τ.
        Default: 10 (conservative choice)

    output_dir : str
        Directory to save configuration files.
        Created if it doesn't exist.

    Returns:
    --------
    config_list : list of dict
        List of all configuration data dictionaries
    metadata : dict
        Ensemble metadata including statistics

    FILE OUTPUT:
    ------------
    config_b{beta}_L{Ls}x{Lt}_{num:04d}.pkl : Individual configurations
    ensemble_b{beta}_L{Ls}x{Lt}_meta.pkl   : Ensemble metadata

    COMPUTATIONAL COST:
    -------------------
    Time per sweep ≈ O(V) = O(L³ × Lt)
    Total time ≈ (n_therm + n_configs × n_separation) × time_per_sweep

    For L=4, Lt=8: ~2 sec/sweep → 10 configs ~ 3 minutes
    For L=6, Lt=12: ~10 sec/sweep → 10 configs ~ 20 minutes
    For L=8, Lt=16: ~30 sec/sweep → 10 configs ~ 1 hour
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Extract spatial and temporal extents
    Ls = lattice_dims[0]
    Lt = lattice_dims[3]

    # =========================================================================
    # Print run parameters
    # =========================================================================
    print("=" * 70)
    print("PRODUCTION ENSEMBLE GENERATOR")
    print("=" * 70)
    print(f"Beta: {beta}")
    print(f"Lattice: {Ls}³ × {Lt}")
    print(f"Configurations: {n_configs}")
    print(f"Thermalization sweeps: {n_therm}")
    print(f"Separation sweeps: {n_separation}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # =========================================================================
    # Generate first configuration with full thermalization
    # =========================================================================
    # Start from cold (identity links) and thermalize to equilibrium
    print(f"\n[1/{n_configs}] Generating initial thermalized configuration...")
    U, plaq, history = generate_su3_config(lattice_dims, beta, n_therm, 'cold', verbose=True)

    config_list = []

    # Save first configuration
    config_data = {
        'U': U.copy(),          # Copy to avoid aliasing issues
        'plaquette': plaq,
        'beta': beta,
        'Lx': Ls, 'Ly': Ls, 'Lz': Ls, 'Lt': Lt,
        'n_colors': 3,
        'config_number': 0,
        'timestamp': datetime.now().isoformat()
    }
    config_list.append(config_data)

    # Save to pickle file
    fname = f"{output_dir}/config_b{beta:.2f}_L{Ls}x{Lt}_{0:04d}.pkl"
    with open(fname, 'wb') as f:
        pickle.dump(config_data, f)
    print(f"  Saved: {fname}, plaquette = {plaq:.6f}")

    # =========================================================================
    # Generate remaining configurations with separation sweeps
    # =========================================================================
    # Continue from the thermalized configuration, running separation sweeps
    # between each saved configuration to reduce autocorrelations

    for cfg_num in range(1, n_configs):
        print(f"\n[{cfg_num+1}/{n_configs}] Generating configuration {cfg_num}...")

        # Run separation sweeps (continues to evolve U in place)
        history = thermalize_su3(U, beta, n_separation, lattice_dims, verbose=False)
        plaq = history[-1]

        # Package configuration data
        config_data = {
            'U': U.copy(),
            'plaquette': plaq,
            'beta': beta,
            'Lx': Ls, 'Ly': Ls, 'Lz': Ls, 'Lt': Lt,
            'n_colors': 3,
            'config_number': cfg_num,
            'timestamp': datetime.now().isoformat()
        }
        config_list.append(config_data)

        # Save to pickle file
        fname = f"{output_dir}/config_b{beta:.2f}_L{Ls}x{Lt}_{cfg_num:04d}.pkl"
        with open(fname, 'wb') as f:
            pickle.dump(config_data, f)
        print(f"  Saved: {fname}, plaquette = {plaq:.6f}")

    # =========================================================================
    # Print ensemble summary
    # =========================================================================
    plaquettes = [c['plaquette'] for c in config_list]

    print("\n" + "=" * 70)
    print("ENSEMBLE SUMMARY")
    print("=" * 70)
    print(f"Generated {n_configs} configurations")
    print(f"Average plaquette: {np.mean(plaquettes):.6f} ± {np.std(plaquettes):.6f}")
    print(f"Min plaquette: {np.min(plaquettes):.6f}")
    print(f"Max plaquette: {np.max(plaquettes):.6f}")

    # Check that plaquette is reasonable for this beta
    expected_plaq = {5.7: 0.58, 6.0: 0.60, 6.5: 0.70}
    if beta in expected_plaq:
        print(f"Expected at β={beta}: ~{expected_plaq[beta]:.2f}")

    # =========================================================================
    # Save ensemble metadata
    # =========================================================================
    metadata = {
        'beta': beta,
        'lattice_dims': lattice_dims,
        'n_configs': n_configs,
        'n_therm': n_therm,
        'n_separation': n_separation,
        'plaquettes': plaquettes,
        'mean_plaquette': np.mean(plaquettes),
        'std_plaquette': np.std(plaquettes),
        'timestamp': datetime.now().isoformat()
    }

    meta_fname = f"{output_dir}/ensemble_b{beta:.2f}_L{Ls}x{Lt}_meta.pkl"
    with open(meta_fname, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"\nMetadata saved: {meta_fname}")

    return config_list, metadata


# =============================================================================
#  COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """
    Command-line interface for ensemble generation.

    Examples:
    ---------
    # Standard run
    python generate_ensemble.py --beta 6.0 --ls 4 --lt 8 --configs 10

    # Larger lattice, more configs
    python generate_ensemble.py --beta 6.0 --ls 6 --lt 12 --configs 20 --outdir large_ensemble

    # Near critical temperature
    python generate_ensemble.py --beta 5.7 --ls 8 --lt 8 --configs 50 --therm 100
    """
    parser = argparse.ArgumentParser(
        description='Generate SU(3) gauge configuration ensemble',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    %(prog)s --beta 6.0 --ls 4 --lt 8 --configs 10
    %(prog)s --beta 6.0 --ls 6 --lt 12 --configs 20 --outdir my_ensemble
    %(prog)s --beta 5.7 --ls 8 --lt 16 --configs 50 --therm 100

PHYSICS NOTES:
    β = 6/g² controls the lattice spacing: larger β = finer lattice
    Lt controls temperature in finite-T studies: T = 1/(Lt × a)
    More configs = smaller statistical errors (∝ 1/√N)
        """
    )

    parser.add_argument('--beta', type=float, default=6.0,
                       help='Inverse coupling β=6/g² (default: 6.0)')
    parser.add_argument('--ls', type=int, default=6,
                       help='Spatial lattice size (default: 6)')
    parser.add_argument('--lt', type=int, default=12,
                       help='Temporal lattice size (default: 12)')
    parser.add_argument('--configs', type=int, default=10,
                       help='Number of configurations (default: 10)')
    parser.add_argument('--therm', type=int, default=50,
                       help='Thermalization sweeps (default: 50)')
    parser.add_argument('--sep', type=int, default=10,
                       help='Separation sweeps between configs (default: 10)')
    parser.add_argument('--outdir', type=str, default='configs',
                       help='Output directory (default: configs)')

    args = parser.parse_args()

    # Build lattice dimensions (cubic spatial)
    lattice_dims = [args.ls, args.ls, args.ls, args.lt]

    # Generate the ensemble
    generate_ensemble(
        beta=args.beta,
        lattice_dims=lattice_dims,
        n_configs=args.configs,
        n_therm=args.therm,
        n_separation=args.sep,
        output_dir=args.outdir
    )


if __name__ == "__main__":
    main()
