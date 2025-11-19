"""
Correlator Analysis Script
===========================

Analyzes YAML correlator files to extract meson masses.
Reads YAML files and performs:
1. Effective mass calculation
2. Plateau fitting
3. Ensemble averaging over configurations
4. Physics validation (mass hierarchy, chiral symmetry)

Author: Zeke Mohammed
Date: October 2025
"""

import numpy as np
import yaml
import os
import logging
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')

def parse_correlator_file(filepath):
    """
    Parse a YAML correlator file and extract all correlators

    Args:
        filepath (str): Path to YAML correlator file

    Returns:
        dict: Dictionary mapping channel names to correlator arrays
    """
    correlators = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Split by YAML document separator
    docs = content.split('---\n')

    for doc in docs:
        if not doc.strip():
            continue

        # Look for correlator key and data
        lines = doc.strip().split('\n')

        channel = None
        correlator_data = []
        in_data_section = False

        for line in lines:
            # Find correlator name
            if line.startswith('correlator:'):
                channel = line.split(':')[1].strip()
                in_data_section = True
                continue

            # Skip metadata lines after correlator name
            if in_data_section and ':' in line and not line[0].isdigit():
                continue

            # Parse correlator data (lines like " 0  -1.539e-03")
            if in_data_section and line.strip():
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    t = int(parts[0])
                    value = float(parts[1])
                    correlator_data.append((t, value))

        if channel and correlator_data:
            # Convert to array
            correlator_data.sort(key=lambda x: x[0])  # Sort by time
            times = np.array([t for t, v in correlator_data])
            values = np.array([v for t, v in correlator_data])

            correlators[channel] = {
                'times': times,
                'correlator': values
            }

    return correlators


def calculate_effective_mass(correlator, method='log'):
    """
    Calculate effective mass from correlator

    For exponential decay C(t) ~ exp(-m*t):
    m_eff(t) = log(C(t)/C(t+1))

    Args:
        correlator (array): Correlator values C(t)
        method (str): Method to use ('log' or 'cosh')

    Returns:
        array: Effective mass at each time slice
    """
    Lt = len(correlator)
    m_eff = np.zeros(Lt)

    for t in range(Lt-1):
        C_t = abs(correlator[t])
        C_tp1 = abs(correlator[t+1])

        if C_t > 1e-15 and C_tp1 > 1e-15:
            m_eff[t] = np.log(C_t / C_tp1)
        else:
            m_eff[t] = np.nan

    m_eff[-1] = np.nan  # Last point has no t+1

    return m_eff


def fit_correlator_plateau(correlator, t_min=3, t_max=15):
    """
    Fit correlator to extract mass from plateau region

    Fits C(t) ~ A * exp(-m*t) in plateau region

    Args:
        correlator (array): Correlator values
        t_min (int): Start of plateau region
        t_max (int): End of plateau region

    Returns:
        tuple: (mass, mass_error, chi_squared, A, A_error)
    """
    Lt = len(correlator)
    t_max = min(t_max, Lt-1)

    # Select plateau region
    times = np.arange(t_min, t_max+1)
    C_fit = np.abs(correlator[t_min:t_max+1])

    # Skip if too few points or zeros
    if len(C_fit) < 3 or np.any(C_fit < 1e-15):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Exponential fit function
    def exp_decay(t, A, m):
        return A * np.exp(-m * t)

    try:
        # Initial guesses
        A_guess = C_fit[0]
        m_guess = np.log(C_fit[0] / C_fit[-1]) / (t_max - t_min)

        # Fit
        popt, pcov = curve_fit(exp_decay, times, C_fit,
                               p0=[A_guess, max(m_guess, 0.01)],
                               maxfev=5000)

        A_fit, m_fit = popt
        A_err, m_err = np.sqrt(np.diag(pcov))

        # Calculate chi-squared
        C_model = exp_decay(times, A_fit, m_fit)
        residuals = C_fit - C_model
        chi_sq = np.sum(residuals**2 / (C_fit + 1e-10)) / (len(C_fit) - 2)

        return m_fit, m_err, chi_sq, A_fit, A_err

    except Exception as e:
        logging.debug(f"Fit failed: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan


def analyze_single_config(filepath, verbose=False):
    """
    Analyze all correlators from a single configuration

    Args:
        filepath (str): Path to YAML file
        verbose (bool): Print detailed output

    Returns:
        dict: Analysis results for each channel
    """
    if verbose:
        logging.info(f"\nAnalyzing: {os.path.basename(filepath)}")

    correlators = parse_correlator_file(filepath)

    results = {}

    for channel, data in correlators.items():
        C = data['correlator']

        # Calculate effective mass
        m_eff = calculate_effective_mass(C)

        # Fit plateau
        mass, mass_err, chi_sq, A, A_err = fit_correlator_plateau(C, t_min=3, t_max=15)

        results[channel] = {
            'correlator': C,
            'effective_mass': m_eff,
            'mass': mass,
            'mass_error': mass_err,
            'chi_squared': chi_sq,
            'amplitude': A,
            'amplitude_error': A_err
        }

        if verbose and not np.isnan(mass):
            logging.info(f"  {channel:12s}: m = {mass:.6f} ± {mass_err:.6f}, χ²/dof = {chi_sq:.2f}")

    return results


def ensemble_average(all_results):
    """
    Average results over all configurations

    Args:
        all_results (list): List of result dicts from each config

    Returns:
        dict: Ensemble averages and errors
    """
    # Collect masses for each channel
    channel_masses = {}

    for results in all_results:
        for channel, data in results.items():
            if channel not in channel_masses:
                channel_masses[channel] = []

            if not np.isnan(data['mass']):
                channel_masses[channel].append(data['mass'])

    # Calculate ensemble averages
    ensemble_results = {}

    for channel, masses in channel_masses.items():
        if len(masses) > 0:
            mean_mass = np.mean(masses)
            std_mass = np.std(masses, ddof=1) if len(masses) > 1 else 0
            sem_mass = std_mass / np.sqrt(len(masses)) if len(masses) > 1 else std_mass

            ensemble_results[channel] = {
                'mass': mean_mass,
                'std': std_mass,
                'sem': sem_mass,
                'n_configs': len(masses),
                'all_masses': masses
            }

    return ensemble_results


def plot_correlators(all_results, output_dir='plots'):
    """
    Create diagnostic plots

    Args:
        all_results (list): Analysis results from all configs
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all unique channels
    channels = set()
    for results in all_results:
        channels.update(results.keys())

    channels = sorted(list(channels))

    for channel in channels:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Correlators from all configs
        ax = axes[0]
        for results in all_results:
            if channel in results:
                C = results[channel]['correlator']
                times = np.arange(len(C))
                ax.semilogy(times, np.abs(C), 'o-', alpha=0.3, markersize=3)

        ax.set_xlabel('Time slice t')
        ax.set_ylabel('|C(t)|')
        ax.set_title(f'{channel} Correlator (all configs)')
        ax.grid(True, alpha=0.3)

        # Plot 2: Effective masses
        ax = axes[1]
        for results in all_results:
            if channel in results:
                m_eff = results[channel]['effective_mass']
                times = np.arange(len(m_eff))
                ax.plot(times, m_eff, 'o-', alpha=0.3, markersize=3)

        ax.set_xlabel('Time slice t')
        ax.set_ylabel('m_eff(t)')
        ax.set_title(f'{channel} Effective Mass')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 2)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{channel}_analysis.png'), dpi=150)
        plt.close()


def main():
    """Main analysis workflow"""

    print("="*70)
    print(" CORRELATOR ANALYSIS - Extracting Meson Masses")
    print("="*70)

    # Find correlator files
    corr_dir = "correlator_outputs/m020_b240"

    if not os.path.exists(corr_dir):
        logging.error(f"Correlator directory not found: {corr_dir}")
        return

    yaml_files = sorted([f for f in os.listdir(corr_dir) if f.startswith('pi_corr_')])

    if not yaml_files:
        logging.error(f"No correlator files found in {corr_dir}")
        return

    logging.info(f"\nFound {len(yaml_files)} correlator files")
    logging.info(f"Directory: {corr_dir}\n")

    # Analyze each configuration
    all_results = []

    for yaml_file in yaml_files:
        filepath = os.path.join(corr_dir, yaml_file)

        try:
            results = analyze_single_config(filepath, verbose=True)
            all_results.append(results)
        except Exception as e:
            logging.error(f"Failed to analyze {yaml_file}: {e}")
            continue

    if not all_results:
        logging.error("No successful analyses")
        return

    # Ensemble average
    print("\n" + "="*70)
    print(" ENSEMBLE AVERAGES (over all configurations)")
    print("="*70)

    ensemble = ensemble_average(all_results)

    # Sort channels by mass for display
    channels_sorted = sorted(ensemble.keys(), key=lambda c: ensemble[c]['mass'])

    print(f"\n{'Channel':<15} {'Mass':<20} {'Std Dev':<15} {'N_configs':<10}")
    print("-"*70)

    for channel in channels_sorted:
        data = ensemble[channel]
        print(f"{channel:<15} {data['mass']:>8.6f} ± {data['sem']:<8.6f} "
              f"{data['std']:>8.6f}      {data['n_configs']:<10}")

    # Physics validation
    print("\n" + "="*70)
    print(" PHYSICS VALIDATION")
    print("="*70)

    if 'PION_5' in ensemble and 'SIGMA' in ensemble:
        m_pi = ensemble['PION_5']['mass']
        m_sigma = ensemble['SIGMA']['mass']

        print(f"\nPion mass (Goldstone boson): {m_pi:.6f}")
        print(f"Sigma mass (scalar):         {m_sigma:.6f}")
        print(f"Mass ratio M_σ/M_π:          {m_sigma/m_pi:.2f}")

        if m_pi < m_sigma:
            print("✓ Correct hierarchy: Pion is lighter than sigma")
        else:
            print("✗ WARNING: Unexpected hierarchy!")

        if m_pi < 0.1:
            print(f"✓ Pion shows Goldstone behavior (light, m_π={m_pi:.4f})")
        else:
            print(f"⚠ Pion mass relatively heavy for Goldstone boson")

    # Create plots
    try:
        logging.info("\nCreating diagnostic plots...")
        plot_correlators(all_results, output_dir='correlator_plots')
        logging.info("Plots saved to: correlator_plots/")
    except Exception as e:
        logging.warning(f"Plotting failed: {e}")

    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nProcessed {len(all_results)} configurations successfully")
    print(f"Extracted masses for {len(ensemble)} channels")

    return ensemble


if __name__ == "__main__":
    results = main()
