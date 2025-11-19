#!/usr/bin/env python3
"""
Overnight Correlator
==============================================

Runs Propagator.py on gauge configurations and outputs YAML correlator files.
Uses the FIXED code with correct propagator indexing (pion=0.018 Goldstone boson).

Output: pi_corr_l6t20f0b240m020.{config_num}
One file per configuration, YAML format matching Dr. Aubin's reference.

Author: Zeke Mohammed
Date: October 2025
"""

import subprocess
import json
import os
import glob
from datetime import datetime
import argparse
import re
import sys
import time

def log(message, logfile=None):
    """Log message to stdout and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg, flush=True)
    if logfile:
        with open(logfile, 'a') as f:
            f.write(msg + '\n')

def get_config_number(config_path):
    """Extract configuration number from filename."""
    basename = os.path.basename(config_path)
    match = re.search(r'_(\d{4})\.pkl$', basename)
    if match:
        return match.group(1)
    return None

def format_mass(mass_float):
    """Format mass for filename: 0.2 -> 020."""
    return str(int(mass_float * 1000)).zfill(3)

def format_beta(beta_float):
    """Format beta for filename: 2.40 -> 240."""
    return str(int(beta_float * 100))

def create_yaml_from_json(json_file, config_num, output_dir, lattice_dims, beta, mass, logfile=None):
    """
    Read JSON output from Propagator.py and create YAML in Dr. Aubin's format.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        log(f"  ERROR reading JSON: {e}", logfile)
        return None

    Lx, Ly, Lz, Lt = lattice_dims

    # File naming: pi_corr_l{L}t{T}f0b{beta}m{mass}.{config}
    # NO 'a' before the config number!
    filename = f"pi_corr_l{Lx}t{Lt}f0b{format_beta(beta)}m{format_mass(mass)}.{config_num}"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, 'w') as f:
            # Write YAML for each channel
            for channel_data in data:
                channel = channel_data['channel']
                correlator = channel_data.get('correlator', [])

                # Map channel names to Dr. Aubin's format
                if channel == 'pion':
                    correlator_name = "PION_5"
                    spin_taste = "pion5"
                elif channel == 'sigma':
                    correlator_name = "SIGMA"
                    spin_taste = "sigma"
                elif 'rho' in channel:
                    direction = channel.split('_')[-1]  # x, y, or z
                    correlator_name = f"RHO_{direction.upper()}"
                    spin_taste = f"rho_{direction}"
                else:
                    continue

                # Write YAML header
                f.write("---\n")
                f.write(f"JobID:                        su2_lqcd_fall2025\n")
                date_str = datetime.now().strftime("%a %b %d %H:%M:%S %Y UTC")
                f.write(f'date:                         "{date_str}"\n')
                f.write(f"lattice_size:                 {Lx},{Ly},{Lz},{Lt}\n")
                f.write(f"antiquark_type:               wilson\n")
                f.write(f"antiquark_source_type:        point\n")
                f.write(f"antiquark_source_subset:      full\n")
                f.write(f"antiquark_source_t0:          0\n")
                f.write(f"antiquark_source_label:       q\n")
                f.write(f"antiquark_sink_label:         d\n")
                f.write(f'antiquark_mass:               "{mass}"\n')
                f.write(f"antiquark_epsilon:            0\n")
                f.write(f"quark_type:                   wilson\n")
                f.write(f"quark_source_type:            point\n")
                f.write(f"quark_source_subset:          full\n")
                f.write(f"quark_source_t0:              0\n")
                f.write(f"quark_source_label:           q\n")
                f.write(f"quark_sink_label:             d\n")
                f.write(f'quark_mass:                   "{mass}"\n')
                f.write(f"quark_epsilon:                0\n")
                f.write(f"...\n")

                # Write correlator metadata
                f.write("---\n")
                f.write(f"correlator:                   {correlator_name}\n")
                f.write(f"momentum:                     p000\n")
                f.write(f"spin_taste_sink:              {spin_taste}\n")
                f.write(f"correlator_key:               {correlator_name}_q_q_d_d_m{mass}_m{mass}_p000\n")
                f.write(f"...\n")

                # Write correlator values
                for t, val in enumerate(correlator):
                    f.write(f"{t:2d}  {val:+.16e}\n")

                f.write("\n")

        log(f"  Created: {filename}", logfile)
        return filepath

    except Exception as e:
        log(f"  ERROR writing YAML: {e}", logfile)
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate Aubin-format correlator files (OVERNIGHT RUN)')
    parser.add_argument('--config-dir', default='gauge_configs', help='Directory containing gauge configurations')
    parser.add_argument('--output-dir', default='correlator_outputs/m020_b240', help='Directory for output YAML files')
    parser.add_argument('--mass', type=float, default=0.2, help='Quark mass')
    parser.add_argument('--beta', type=float, default=2.4, help='Beta value')
    parser.add_argument('--ls', type=int, default=6, help='Spatial lattice size')
    parser.add_argument('--lt', type=int, default=20, help='Temporal lattice size')
    parser.add_argument('--every-n', type=int, default=5, help='Process every Nth configuration')
    parser.add_argument('--logfile', default='logs/overnight_run.log', help='Log file path')

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.logfile), exist_ok=True)

    # Initialize log
    log("="*70, args.logfile)
    log("OVERNIGHT CORRELATOR GENERATION - STARTING", args.logfile)
    log("="*70, args.logfile)
    log(f"Config directory: {args.config_dir}", args.logfile)
    log(f"Output directory: {args.output_dir}", args.logfile)
    log(f"Parameters: mass={args.mass}, beta={args.beta}, lattice={args.ls}^3×{args.lt}", args.logfile)
    log(f"Processing: every {args.every_n}th configuration", args.logfile)
    log("", args.logfile)

    # Get all configuration files
    config_files = sorted(glob.glob(os.path.join(args.config_dir, "*.pkl")))

    if not config_files:
        log(f"ERROR: No configuration files found in {args.config_dir}", args.logfile)
        return

    log(f"Found {len(config_files)} total configuration files", args.logfile)

    # Process every Nth configuration
    configs_to_process = config_files[::args.every_n]
    log(f"Will process {len(configs_to_process)} configurations", args.logfile)
    log("", args.logfile)

    lattice_dims = [args.ls, args.ls, args.ls, args.lt]

    start_time = time.time()
    successful = 0
    failed = 0

    for i, config_path in enumerate(configs_to_process, 1):
        config_num = get_config_number(config_path)
        if not config_num:
            log(f"[{i}/{len(configs_to_process)}] SKIP: Couldn't extract config number from {config_path}", args.logfile)
            failed += 1
            continue

        log(f"[{i}/{len(configs_to_process)}] Processing config {config_num}...", args.logfile)

        # Run Propagator.py (the monolithic version with JSON output)
        cmd = [
            "python3", "Propagator.py",
            "--mass", str(args.mass),
            "--ls", str(args.ls),
            "--lt", str(args.lt),
            "--channel", "all",
            "--input-config", config_path,
            "--output", f"temp_cfg{config_num}"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                log(f"  ERROR: Propagator.py failed", args.logfile)
                log(f"  {result.stderr[:300]}", args.logfile)
                failed += 1
                continue

            # Find the JSON output
            json_pattern = f"lattice_qcd_results/spectrum/*/data/results_*.json"
            json_files = sorted(glob.glob(json_pattern), key=os.path.getmtime)

            if not json_files:
                log(f"  ERROR: No JSON output found", args.logfile)
                failed += 1
                continue

            json_file = json_files[-1]  # Most recent
            log(f"  Found JSON: {os.path.basename(json_file)}", args.logfile)

            # Create YAML
            yaml_file = create_yaml_from_json(
                json_file,
                config_num,
                args.output_dir,
                lattice_dims,
                args.beta,
                args.mass,
                args.logfile
            )

            if yaml_file:
                successful += 1
                # Clean up the temp output directory
                import shutil
                temp_dirs = glob.glob(f"lattice_qcd_results/spectrum/*{config_num}*")
                for d in temp_dirs:
                    try:
                        shutil.rmtree(d)
                    except:
                        pass
                log(f"  ✓ Success ({successful}/{len(configs_to_process)})", args.logfile)
            else:
                failed += 1
                log(f"  ✗ Failed to create YAML", args.logfile)

        except subprocess.TimeoutExpired:
            log(f"  ERROR: Timeout (>10 min)", args.logfile)
            failed += 1
        except Exception as e:
            log(f"  ERROR: {e}", args.logfile)
            failed += 1

        log("", args.logfile)

    # Final summary
    elapsed = time.time() - start_time
    log("="*70, args.logfile)
    log("OVERNIGHT RUN COMPLETE", args.logfile)
    log("="*70, args.logfile)
    log(f"Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)", args.logfile)
    log(f"Successful: {successful}/{len(configs_to_process)}", args.logfile)
    log(f"Failed: {failed}/{len(configs_to_process)}", args.logfile)
    log(f"Output files in: {args.output_dir}", args.logfile)
    log("="*70, args.logfile)

if __name__ == "__main__":
    main()
