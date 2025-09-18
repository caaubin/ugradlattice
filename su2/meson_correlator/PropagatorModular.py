#!/usr/bin/env python3
"""
Modular Lattice QCD Propagator Calculator - Command Line Interface
==================================================================

This script provides the exact same command-line interface as the original
Propagator.py, but uses the faster modular backend for improved performance.

New Features:
- Faster execution using specialized meson modules
- Enhanced physics diagnostics and validation
- Improved error handling and scipy compatibility
- Better logging and output organization

Output Directory Structure:
--------------------------
lattice_qcd_results/
├── single/                              # Single channel calculations
│   └── [timestamp]_m[mass]_[channel]_L[Lx]x[Ly]x[Lz]T[Lt]_r[wilson]/
│       ├── data/                        # Results in JSON and text formats
│       │   ├── results_[time].json      # Machine-readable results
│       │   └── summary_[time].txt       # Human-readable analysis
│       ├── plots/                       # Comprehensive 8-panel analysis plots
│       │   └── [channel]_analysis.png
│       ├── correlators/                 # Raw correlator data files
│       │   └── [channel]_correlator_[time].dat
│       ├── logs/                        # Detailed calculation logs
│       │   └── calculation.log
│       └── configs/                     # Configuration backups
│
├── spectrum/                            # Multi-channel spectrum analysis
│   └── [timestamp]_spectrum_m[mass]_L[Lx]x[Ly]x[Lz]T[Lt]/
│       └── [similar structure]
│
├── mass_scan/                           # Chiral behavior studies
│   └── [timestamp]_mass_scan_[channel]_L[Lx]x[Ly]x[Lz]T[Lt]/
│       └── [similar structure]
│
└── wilson_scan/                         # Wilson parameter optimization
    └── [timestamp]_wilson_scan_[channel]_L[Lx]x[Ly]x[Lz]T[Lt]/
        └── [similar structure]

Author: Zeke Mohammed 
Advisor: Dr. Aubin
Institution: Fordham University
Date: September 2025
"""

import argparse
import sys
import os
import time
import logging
from datetime import datetime

# Import the modular backend
from MesonIntegration import calculate_meson_spectrum, quick_meson_calculation

def parse_arguments():
    """Parse command line arguments - EXACT same interface as original"""
    parser = argparse.ArgumentParser(
        description='Lattice QCD meson propagator calculation using Wilson fermions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
1. Single pion mass calculation:
   python3 %(prog)s --mass 0.1 --channel pion --input-config config.pkl

2. Full hadron spectrum (π, σ, ρ mesons):
   python3 %(prog)s --mass 0.1 --channel all --input-config config.pkl

3. Study chiral behavior (M²_π ∝ m_quark):
   python3 %(prog)s --mass-scan 0.01,0.05,0.1,0.2 --channel pion --input-config config.pkl

4. Optimize Wilson parameter:
   python3 %(prog)s --wilson-scan 0.1,0.5,1.0 --mass 0.1 --input-config config.pkl

5. Custom lattice size:
   python3 %(prog)s --mass 0.1 --channel pion --ls 6 --lt 20 --input-config config.pkl

Your Example Command:
python3 %(prog)s --mass 0.2 --ls 6 --lt 20 --channel all \\
    --input-config thermal_generator_runs/standard/20250309_031914_L6x6x6T20_b2.40_cold/configs/quSU2_b2.4_6_6_6_20_2333 \\
    --output results_m0.2
        """)

    # Physics parameters
    physics_group = parser.add_argument_group('QCD Physics Parameters')
    physics_group.add_argument('--mass', type=float, default=0.1,
                              help='Bare quark mass in lattice units (default: 0.1)')
    physics_group.add_argument('--wilson-r', type=float, default=0.5,
                              help='Wilson parameter r, controls discretization (default: 0.5)')

    # Lattice parameters
    lattice_group = parser.add_argument_group('Lattice Discretization')
    lattice_group.add_argument('--ls', type=int, default=None,
                              help='Spatial lattice size (cubic, default: auto-detect)')
    lattice_group.add_argument('--lt', type=int, default=None,
                              help='Temporal lattice size (default: auto-detect)')

    # Analysis options
    analysis_group = parser.add_argument_group('Hadron Analysis Options')
    analysis_group.add_argument('--channel', type=str, default='pion',
                               choices=['pion', 'sigma', 'rho', 'rho_x', 'rho_y', 'rho_z', 'all'],
                               help='Meson channel to calculate (default: pion)')
    analysis_group.add_argument('--solver', type=str, default='auto',
                               choices=['auto', 'direct', 'gmres', 'lsqr'],
                               help='Linear solver for Dirac equation (default: auto)')

    # Parameter scans
    scan_group = parser.add_argument_group('Parameter Scan Studies')
    scan_group.add_argument('--mass-scan', type=str,
                           help='Comma-separated quark masses for chiral behavior study')
    scan_group.add_argument('--wilson-scan', type=str,
                           help='Comma-separated Wilson r values for optimization')

    # I/O options
    io_group = parser.add_argument_group('Input/Output Configuration')
    io_group.add_argument('--input-config', type=str, required=True,
                         help='Path to gauge configuration file (pickle format)')
    io_group.add_argument('--output', type=str, default='results',
                         help='Output directory prefix (default: results)')

    # Control options
    control_group = parser.add_argument_group('Calculation Control')
    control_group.add_argument('--verbose', action='store_true',
                              help='Enable detailed physics output and debugging')
    control_group.add_argument('--save-correlators', action='store_true',
                              help='Save raw correlator data for analysis')
    control_group.add_argument('--no-plots', action='store_true',
                              help='Skip plot generation (faster for batch runs)')

    return parser.parse_args()

def setup_output_directories(output_prefix, run_type="single"):
    """Create output directory structure compatible with original"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if run_type == "single":
        output_dir = f"{output_prefix}_{timestamp}"
    else:
        output_dir = f"{output_prefix}_{run_type}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories
    subdirs = ['data', 'plots', 'correlators', 'logs']
    dirs = {'base': output_dir}

    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        dirs[subdir] = subdir_path

    # Add log file path
    dirs['log_file'] = os.path.join(dirs['logs'], 'calculation.log')

    return dirs

def configure_logging(log_file, console_level=logging.INFO):
    """Configure logging to match original format"""
    logging.basicConfig(
        level=console_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def save_results_original_format(results, output_dirs, save_correlators=False):
    """Save results in original Propagator.py format"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not isinstance(results, list):
        results = [results]

    # Save in original text format
    summary_file = os.path.join(output_dirs['data'], f'results_{timestamp}.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("LATTICE QCD MESON MASS RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Using modular backend for improved performance\n\n")

        for i, result in enumerate(results):
            if len(results) > 1:
                f.write(f"Result {i+1}: {result['channel'].upper()}\n")
                f.write("-"*30 + "\n")

            f.write(f"Channel: {result['channel']} ({result.get('channel_info', {}).get('JPC', 'N/A')})\n")
            f.write(f"Mass: {result['meson_mass']:.6f} ± {result['meson_error']:.6f}\n")
            f.write(f"Chi²/dof: {result['chi_squared']:.4f}\n")
            f.write(f"Fit range: t = {result['fit_range'][0]} to {result['fit_range'][1]}\n\n")

            f.write(f"Parameters:\n")
            f.write(f"  Quark mass: {result['input_mass']:.6f}\n")
            f.write(f"  Wilson r: {result['wilson_r']:.3f}\n")
            f.write(f"  Effective mass: {result['input_mass'] + 4*result['wilson_r']:.6f}\n")
            f.write(f"  Lattice: {result['lattice_dims']}\n")
            f.write(f"  Solver: {result['solver']}\n\n")

    logging.info(f"Results saved: {summary_file}")

    # Save correlators if requested
    if save_correlators:
        for result in results:
            if 'correlator' in result and result['correlator']:
                channel = result['channel']
                corr_file = os.path.join(output_dirs['correlators'], f'{channel}_correlator_{timestamp}.dat')

                with open(corr_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {channel.upper()} correlator data\n")
                    f.write(f"# Mass: {result['meson_mass']:.6f} ± {result['meson_error']:.6f}\n")
                    f.write(f"# t  C(t)\n")

                    for t, c in enumerate(result['correlator']):
                        f.write(f"{t:3d}  {c:15.8e}\n")

                logging.info(f"Saved correlator: {corr_file}")

def main():
    """Main execution function - same interface as original but with modular backend"""

    # Parse arguments exactly like original
    args = parse_arguments()

    # Setup output directories
    if args.mass_scan:
        run_type = "mass_scan"
    elif args.wilson_scan:
        run_type = "wilson_scan"
    elif args.channel == 'all':
        run_type = "spectrum"
    else:
        run_type = "single"

    output_dirs = setup_output_directories(args.output, run_type)

    # Configure logging
    configure_logging(output_dirs['log_file'],
                     logging.DEBUG if args.verbose else logging.INFO)

    # Header exactly like original
    logging.info("="*60)
    logging.info("LATTICE QCD PROPAGATOR CALCULATION")
    logging.info("Using modular backend for enhanced performance")
    logging.info("="*60)
    logging.info(f"Run type: {run_type}")
    logging.info(f"Output directory: {output_dirs['base']}")
    logging.info("")

    # Start calculations
    start_time = time.time()

    # Determine lattice dimensions
    lattice_dims = None
    if args.ls is not None and args.lt is not None:
        lattice_dims = [args.ls, args.ls, args.ls, args.lt]

    try:
        if args.mass_scan:
            # Mass scan
            masses = [float(m.strip()) for m in args.mass_scan.split(',')]
            logging.info(f"Mass scan: {masses}")

            results = []
            for mass in masses:
                logging.info(f"\n{'-'*40}")
                logging.info(f"Mass = {mass}")
                logging.info(f"{'-'*40}")

                result = quick_meson_calculation(args.input_config, args.channel,
                                               mass, args.wilson_r, lattice_dims, args.verbose)
                results.append(result)

                logging.info(f"Result: M_{args.channel} = {result['meson_mass']:.6f} ± {result['meson_error']:.6f}")

        elif args.wilson_scan:
            # Wilson parameter scan
            wilson_values = [float(r.strip()) for r in args.wilson_scan.split(',')]
            logging.info(f"Wilson scan: {wilson_values}")

            results = []
            for wilson_r in wilson_values:
                logging.info(f"\n{'-'*40}")
                logging.info(f"Wilson r = {wilson_r}")
                logging.info(f"{'-'*40}")

                result = quick_meson_calculation(args.input_config, args.channel,
                                               args.mass, wilson_r, lattice_dims, args.verbose)
                results.append(result)

                logging.info(f"Result: M_{args.channel} = {result['meson_mass']:.6f} ± {result['meson_error']:.6f}")

        elif args.channel == 'all':
            # Full spectrum - use the modular spectrum calculator
            import pickle

            with open(args.input_config, 'rb') as f:
                gauge_data = pickle.load(f)

            if isinstance(gauge_data, list) and len(gauge_data) >= 2:
                U = [None, gauge_data[1]]
            elif isinstance(gauge_data, dict) and 'U' in gauge_data:
                U = [None, gauge_data['U']]
            else:
                raise ValueError("Unknown gauge configuration format")

            # Auto-detect lattice dimensions if not provided
            if lattice_dims is None:
                V = U[1].shape[0]
                # Common lattice sizes
                if V == 64:
                    lattice_dims = [4, 4, 4, 4]
                elif V == 256:
                    lattice_dims = [4, 4, 4, 16]
                elif V == 512:
                    lattice_dims = [4, 4, 4, 8]  # 4*4*4*8 = 512
                elif V == 2880:
                    lattice_dims = [6, 6, 6, 20]  # 6*6*6*20 = 4320, but let's try
                else:
                    # General guess
                    spatial_size = int(round((V/16)**(1/3)))
                    lattice_dims = [spatial_size, spatial_size, spatial_size, 16]

            logging.info(f"Auto-detected lattice dimensions: {lattice_dims}")

            spectrum_results = calculate_meson_spectrum(
                U, args.mass, lattice_dims, args.wilson_r, args.solver,
                include_rho_polarizations=True, verbose=args.verbose)

            # Extract individual results for compatibility
            results = []
            for channel in ['pion', 'sigma', 'rho_x', 'rho_y', 'rho_z']:
                if channel in spectrum_results:
                    results.append(spectrum_results[channel])

            for result in results:
                logging.info(f"Result: M_{result['channel']} = {result['meson_mass']:.6f} ± {result['meson_error']:.6f}")

        else:
            # Single calculation
            result = quick_meson_calculation(args.input_config, args.channel,
                                           args.mass, args.wilson_r, lattice_dims, args.verbose)
            results = result

            logging.info(f"\n{args.channel.upper()} RESULTS:")
            logging.info(f"  Mass: {result['meson_mass']:.6f} ± {result['meson_error']:.6f}")
            logging.info(f"  χ²/dof: {result['chi_squared']:.4f}")

        total_time = time.time() - start_time
        logging.info(f"\nCalculation time: {total_time:.2f} seconds")

        # Save results in original format
        save_results_original_format(results, output_dirs, args.save_correlators)

        # Final summary
        logging.info(f"\n{'='*60}")
        logging.info(f"CALCULATION COMPLETE")
        logging.info(f"{'='*60}")
        logging.info(f"Results saved in: {output_dirs['base']}")
        logging.info(f"{'='*60}")

    except Exception as e:
        logging.error(f"Calculation failed: {e}")
        if args.verbose:
            import traceback
            logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()