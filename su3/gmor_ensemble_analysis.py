#!/usr/bin/env python3
"""
================================================================================
GMOR Relation Analysis with Ensemble Averaging and Jackknife Errors
================================================================================

This program computes pion masses from an ensemble of SU(3) gauge configurations
and extracts the critical quark mass using the Gell-Mann-Oakes-Renner (GMOR)
relation with proper statistical error analysis.

THE GMOR RELATION:
------------------
The GMOR relation is a fundamental result connecting pion mass to quark mass:

    m²_π = 2B(m_q + m_crit)

or equivalently:
    m²_π = 2B × m_q + const

where:
    - m_π = pion mass
    - m_q = bare quark mass
    - m_crit = critical mass (where pion becomes massless)
    - B = parameter related to chiral condensate ⟨q̄q⟩

PHYSICAL SIGNIFICANCE:
---------------------
The pion is special in QCD:
    - It is the Goldstone boson of spontaneous chiral symmetry breaking
    - In the chiral limit (m_q → m_crit), the pion becomes MASSLESS
    - This is why the physical pion (m_π ≈ 140 MeV) is much lighter than
      other hadrons (m_ρ ≈ 770 MeV, m_p ≈ 938 MeV)

The GMOR relation shows that m²_π (not m_π!) is linear in quark mass.
This is a direct consequence of the Goldstone nature of the pion.

WILSON FERMION CRITICAL MASS:
----------------------------
For Wilson fermions, the critical mass is NEGATIVE due to "additive
mass renormalization":

    m_eff = m_bare + 4r  (where r ≈ 0.5 is the Wilson parameter)

The critical mass compensates for this shift:
    - β = 5.7: m_crit ≈ -0.90
    - β = 6.0: m_crit ≈ -0.80
    - β = 6.2: m_crit ≈ -0.70

(More negative at stronger coupling)

EXTRACTION METHOD:
-----------------
1. Generate ensemble of N gauge configurations at fixed β
2. For each configuration, compute m_π at several quark masses m_q
3. Average over ensemble: <m_π>(m_q), with jackknife errors
4. Fit: m²_π = slope × m_q + intercept
5. Extract: m_crit = -intercept / slope
6. Compare with literature values

JACKKNIFE ERROR ANALYSIS:
------------------------
Standard error propagation assumes independent measurements. For correlated
data from Monte Carlo, we use the jackknife method:

1. Full sample estimate: θ̂ = f(all data)
2. Leave-one-out estimates: θ̂_i = f(data excluding sample i)
3. Jackknife error: σ_θ = √[(N-1) × var(θ̂_i)]

The factor (N-1) accounts for the fact that jackknife samples are not independent.

PION CORRELATOR:
---------------
The pion correlator measures ⟨π(t) π†(0)⟩ where π = q̄γ₅q:

    C_π(t) = Σ_x Tr[Γ S(0;x,t) Γ S†(0;x,t)]

where:
    - Γ = γ₅ ⊗ I_color (combines Dirac and color structure)
    - S = quark propagator from origin to (x,t)
    - The sum over x projects onto zero spatial momentum

For large t: C(t) ~ exp(-m_π t), so:
    m_eff(t) = ln[C(t)/C(t+1)] → m_π

REFERENCE:
----------
M. Gell-Mann, R.J. Oakes, B. Renner, "Behavior of Current Divergences under
SU(3) x SU(3)", Phys. Rev. 175, 2195 (1968).

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
import glob
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.sparse.linalg import gmres

# Add parent directory for imports
sys.path.insert(0, '..')

from MesonBase import build_wilson_dirac_matrix, create_point_source, get_gamma_matrices
import su2


# ============================================================================
#  PION CORRELATOR COMPUTATION
# ============================================================================

def compute_pion_correlator(U, quark_mass, lattice_dims, n_colors=3, wilson_r=0.5):
    """
    Compute pion two-point correlation function from gauge configuration.

    The pion correlator C(t) gives access to the pion mass through:
        C(t) ~ exp(-m_π t)  for large t

    CORRELATOR FORMULA:
    ------------------
    The correct formula for a meson correlator is:

        C(t) = Σ_x Tr[Γ S(0;x,t) Γ S†(0;x,t)]

    where:
        - Γ = γ₅ ⊗ I_color for the pion
        - S(0;x,t) = quark propagator from origin to (x,t)
        - Sum over x gives zero momentum projection

    IMPORTANT: The formula involves S S†, NOT just Tr[Γ S]!
    The latter gives wrong results (can even be negative).

    Parameters:
    -----------
    U : ndarray, shape (V, 4, n_c, n_c)
        Gauge configuration
    quark_mass : float
        Bare quark mass (can be negative for Wilson fermions)
    lattice_dims : list of int
        [Lx, Ly, Lz, Lt] lattice dimensions
    n_colors : int
        Number of colors (3 for SU(3))
    wilson_r : float
        Wilson parameter (default: 0.5)

    Returns:
    --------
    pion_correlator : ndarray, shape (Lt,)
        Pion correlator C(t) at each time slice

    COMPUTATIONAL COST:
    -------------------
    Requires n_colors × 4 = 12 propagator inversions (for point source).
    Each inversion is O(V²) for direct methods, O(V log V) for iterative.
    Total: dominant cost of the calculation.
    """
    Lx, Ly, Lz, Lt = lattice_dims
    V = np.prod(lattice_dims)
    dof_per_site = n_colors * 4  # Color × Dirac indices

    # =========================================================================
    # Build Wilson-Dirac matrix
    # =========================================================================
    # D_W = m + 4r - (1/2) Σ_μ [(1-γ_μ)U_μ T_+ + (1+γ_μ)U†_μ T_-]
    U_fmt = [None, U]  # Format expected by build_wilson_dirac_matrix
    D = build_wilson_dirac_matrix(quark_mass, lattice_dims, wilson_r, U_fmt,
                                   n_colors=n_colors, verbose=False)

    # =========================================================================
    # Compute all propagators from origin
    # =========================================================================
    # Need S(0,y) for all y, all color/spin combinations
    # This requires solving D × S = b for point sources at origin

    propagators = []  # Will store 12 propagator vectors

    for color in range(n_colors):
        for spin in range(4):
            # Create point source at origin with specific color/spin
            source = create_point_source(lattice_dims, 0, color, spin,
                                        n_colors=n_colors, verbose=False)

            # Solve D × propagator = source using GMRES iterative method
            # rtol=1e-10 gives sufficient precision for mass extraction
            prop, info = gmres(D, source, rtol=1e-10, maxiter=1000)
            propagators.append(prop)

    # =========================================================================
    # Get gamma5 matrix for pion operator
    # =========================================================================
    gamma_matrices = get_gamma_matrices()
    gamma5 = gamma_matrices['gamma5']  # 4×4 Dirac matrix

    # =========================================================================
    # Compute pion correlator at each time slice
    # =========================================================================
    # C(t) = Σ_x Tr[Γ S(0;x,t) Γ S†(0;x,t)]

    pion_correlator = np.zeros(Lt, dtype=float)

    for t in range(Lt):
        corr_t = 0.0

        # Sum over all spatial sites at fixed time t
        for x in range(Lx):
            for y in range(Ly):
                for z in range(Lz):
                    # Convert spatial coordinates to linear site index
                    sink_point = np.array([x, y, z, t])
                    sink_site_idx = su2.p2i(sink_point, lattice_dims)
                    sink_base_idx = dof_per_site * sink_site_idx

                    # Build full propagator matrix S(0 → sink) with all indices
                    # S_full[sink_color,sink_spin ; src_color,src_spin]
                    S_full = np.zeros((n_colors*4, n_colors*4), dtype=complex)

                    for src_color in range(n_colors):
                        for src_spin in range(4):
                            src_idx = 4*src_color + src_spin  # Index into propagators list
                            prop = propagators[src_idx]

                            for sink_color in range(n_colors):
                                for sink_spin in range(4):
                                    # Global index in propagator vector
                                    sink_global_idx = sink_base_idx + 4*sink_color + sink_spin

                                    # Matrix indices
                                    row = 4*sink_color + sink_spin
                                    col = 4*src_color + src_spin

                                    S_full[row, col] = prop[sink_global_idx]

                    # Apply Γ = γ₅ ⊗ I_color
                    # This is a (n_colors*4) × (n_colors*4) matrix
                    Gamma = np.kron(np.eye(n_colors), gamma5)

                    # Compute Tr[Γ S Γ S†]
                    # This is the CORRECT correlator formula for mesons
                    GS = Gamma @ S_full
                    contrib = np.real(np.trace(GS @ GS.conj().T))
                    corr_t += contrib

        pion_correlator[t] = corr_t

    return pion_correlator


# ============================================================================
#  MASS EXTRACTION
# ============================================================================

def extract_mass_from_correlator(correlator):
    """
    Extract effective pion mass from correlator using log ratio method.

    The effective mass at time t is defined as:

        m_eff(t) = ln[C(t) / C(t+1)]

    For a single exponential decay C(t) = A exp(-m t), this gives m_eff = m exactly.
    For multiple states: C(t) = Σ_n A_n exp(-m_n t), we get:

        m_eff(t) → m_0 as t → ∞  (ground state dominates)

    We average over the "plateau region" where m_eff(t) is approximately constant.

    Parameters:
    -----------
    correlator : ndarray
        Pion correlator C(t)

    Returns:
    --------
    mass : float or None
        Extracted pion mass (average over plateau)
    error : float or None
        Uncertainty from variation in plateau region

    SYSTEMATIC EFFECTS:
    -------------------
    - Small t: Excited state contamination (m_eff > m_π)
    - Large t: Statistical noise (C(t) → 0, fluctuations dominate)
    - Periodic boundary: Use sinh cosh form for full accuracy
    """
    Lt = len(correlator)
    effective_masses = []

    # Compute effective mass at each time slice
    for t in range(1, Lt//2):  # Use first half to avoid periodic effects
        if correlator[t] > 0 and correlator[t+1] > 0:
            m_eff = np.log(correlator[t] / correlator[t+1])
            effective_masses.append(m_eff)

    if effective_masses:
        # Simple average and std dev
        # More sophisticated: fit plateau region only
        return np.mean(effective_masses), np.std(effective_masses)

    return None, None


# ============================================================================
#  JACKKNIFE ERROR ANALYSIS
# ============================================================================

def jackknife_analysis(data_list, func):
    """
    Perform jackknife error analysis for correlated data.

    The jackknife method provides reliable error estimates when:
    - Data points are correlated (as in Monte Carlo chains)
    - The estimator is not simply the mean

    ALGORITHM:
    ----------
    1. Compute full-sample estimate: θ̂ = f(x_1, x_2, ..., x_N)

    2. For each i = 1, ..., N:
       Compute leave-one-out estimate: θ̂_i = f(all x except x_i)

    3. Jackknife error:
       σ_θ = √[(N-1)/N × Σᵢ (θ̂_i - θ̂)²]

       or equivalently:
       σ_θ = √[(N-1) × var(θ̂_i)]

    The factor (N-1) corrects for the correlation between jackknife samples.

    Parameters:
    -----------
    data_list : list
        List of N measurements
    func : callable
        Function to apply to data (e.g., np.mean for average)

    Returns:
    --------
    mean : float
        Full-sample estimate
    error : float
        Jackknife error estimate

    EXAMPLE:
    --------
    >>> data = [1.2, 1.5, 1.3, 1.4, 1.6]
    >>> mean, err = jackknife_analysis(data, np.mean)
    >>> print(f"{mean:.3f} ± {err:.3f}")
    """
    n = len(data_list)

    if n < 2:
        return np.mean(data_list), 0.0

    # Full sample estimate
    full_estimate = func(data_list)

    # Leave-one-out (jackknife) estimates
    jackknife_estimates = []
    for i in range(n):
        # Create sample with element i removed
        sample = [data_list[j] for j in range(n) if j != i]
        jackknife_estimates.append(func(sample))

    jackknife_estimates = np.array(jackknife_estimates)

    # Jackknife error formula
    error = np.sqrt((n-1) * np.mean((jackknife_estimates - full_estimate)**2))

    return full_estimate, error


# ============================================================================
#  GMOR FIT FUNCTION
# ============================================================================

def fit_gmor_relation(quark_masses, pion_masses_sq, pion_errors_sq):
    """
    Fit GMOR relation: m²_π = slope × m_q + intercept

    Uses weighted least squares with errors from jackknife analysis.
    Extracts critical mass: m_crit = -intercept / slope

    Parameters:
    -----------
    quark_masses : array-like
        Bare quark mass values
    pion_masses_sq : array-like
        Squared pion masses <m²_π>
    pion_errors_sq : array-like
        Errors on squared pion masses

    Returns:
    --------
    slope : float
        Fit slope (= 2B in GMOR)
    intercept : float
        Fit intercept
    m_crit : float
        Critical mass = -intercept/slope
    errors : dict
        Dictionary containing slope_err, intercept_err, m_crit_err

    PHYSICS:
    --------
    The GMOR relation predicts:
        m²_π = 2B(m_q - m_crit)
             = (2B) × m_q + (-2B × m_crit)
             = slope × m_q + intercept

    So:
        slope = 2B (related to chiral condensate)
        intercept = -2B × m_crit
        m_crit = -intercept / slope
    """
    # Convert to numpy arrays
    quark_masses = np.array(quark_masses)
    pion_masses_sq = np.array(pion_masses_sq)
    pion_errors_sq = np.array(pion_errors_sq)

    # Weights for least squares (inverse variance)
    weights = 1.0 / (pion_errors_sq**2 + 1e-10)  # Small epsilon prevents division by zero

    # Weighted polynomial fit (degree 1 = linear)
    coeffs, cov = np.polyfit(quark_masses, pion_masses_sq, 1,
                             w=np.sqrt(weights), cov=True)

    slope = coeffs[0]
    intercept = coeffs[1]
    slope_err = np.sqrt(cov[0, 0])
    intercept_err = np.sqrt(cov[1, 1])

    # Extract critical mass
    if abs(slope) > 1e-10:
        m_crit = -intercept / slope
        # Error propagation for ratio
        m_crit_err = abs(m_crit) * np.sqrt(
            (slope_err/slope)**2 + (intercept_err/intercept)**2
        )
    else:
        m_crit = 0.0
        m_crit_err = 0.0

    errors = {
        'slope_err': slope_err,
        'intercept_err': intercept_err,
        'm_crit_err': m_crit_err
    }

    return slope, intercept, m_crit, errors


# ============================================================================
#  MAIN ANALYSIS FUNCTION
# ============================================================================

def run_ensemble_analysis(config_dir, quark_masses, output_prefix='gmor'):
    """
    Run complete GMOR analysis on an ensemble of gauge configurations.

    This is the main analysis function that:
    1. Loads all configurations from a directory
    2. Computes pion masses at each quark mass on each configuration
    3. Performs ensemble averaging with jackknife errors
    4. Fits GMOR relation to extract critical mass
    5. Creates diagnostic plots
    6. Saves results to pickle file

    Parameters:
    -----------
    config_dir : str
        Directory containing gauge configuration pickle files
        Files should be named: config_*.pkl
    quark_masses : list of float
        Bare quark masses to compute pion mass at
        Example: [0.4, 0.3, 0.2, 0.15, 0.1, 0.05]
    output_prefix : str
        Prefix for output files (plots, results pickle)

    Returns:
    --------
    results_data : dict or None
        Dictionary containing all results, or None if analysis failed

    OUTPUT FILES:
    -------------
    {output_prefix}_b{beta}_N{n_configs}.png  : GMOR plot
    {output_prefix}_results_b{beta}_N{n_configs}.pkl : Results data

    TYPICAL WORKFLOW:
    -----------------
    1. Generate ensemble: python generate_ensemble.py --configs 10
    2. Run analysis: python gmor_ensemble_analysis.py --config_dir configs
    3. Check plot and results
    """
    print("=" * 70)
    print("GMOR ENSEMBLE ANALYSIS")
    print("=" * 70)
    print(f"Config directory: {config_dir}")
    print(f"Quark masses: {quark_masses}")
    print("=" * 70)

    # =========================================================================
    # Find and load configurations
    # =========================================================================
    config_files = sorted(glob.glob(f"{config_dir}/config_*.pkl"))
    n_configs = len(config_files)

    if n_configs == 0:
        print("ERROR: No configuration files found!")
        return None

    print(f"\nFound {n_configs} configurations")

    # Load first config to get lattice parameters
    with open(config_files[0], 'rb') as f:
        data = pickle.load(f)
    lattice_dims = [data['Lx'], data['Ly'], data['Lz'], data['Lt']]
    beta = data['beta']
    print(f"Lattice: {lattice_dims[0]}³ × {lattice_dims[3]}")
    print(f"Beta: {beta}")

    # =========================================================================
    # Compute pion masses on each configuration
    # =========================================================================
    all_pion_masses = {m: [] for m in quark_masses}

    for cfg_idx, config_file in enumerate(config_files):
        print(f"\n--- Config {cfg_idx + 1}/{n_configs}: {os.path.basename(config_file)} ---")

        with open(config_file, 'rb') as f:
            data = pickle.load(f)
        U = data['U']
        print(f"  Plaquette: {data['plaquette']:.6f}")

        # Compute pion mass at each quark mass
        for m_q in quark_masses:
            print(f"  Computing m_q = {m_q}...", end=" ", flush=True)

            correlator = compute_pion_correlator(U, m_q, lattice_dims)
            m_pi, m_pi_err = extract_mass_from_correlator(correlator)

            if m_pi is not None:
                all_pion_masses[m_q].append(m_pi)
                print(f"m_pi = {m_pi:.4f}")
            else:
                print("FAILED (correlator not positive)")

    # =========================================================================
    # Ensemble averaging with jackknife errors
    # =========================================================================
    print("\n" + "=" * 70)
    print("ENSEMBLE AVERAGED RESULTS")
    print("=" * 70)

    results = []
    print(f"\n{'m_q':>8} {'<m_pi>':>10} {'error':>10} {'<m_pi^2>':>12} {'error':>10}")
    print("-" * 55)

    for m_q in quark_masses:
        masses = all_pion_masses[m_q]

        if len(masses) >= 2:
            # Jackknife average for m_pi
            mean_mpi, err_mpi = jackknife_analysis(masses, np.mean)

            # Jackknife average for m_pi^2
            masses_sq = np.array(masses)**2
            mean_mpi_sq, err_mpi_sq = jackknife_analysis(list(masses_sq), np.mean)

            results.append({
                'quark_mass': m_q,
                'pion_mass': mean_mpi,
                'pion_error': err_mpi,
                'pion_mass_sq': mean_mpi_sq,
                'pion_mass_sq_err': err_mpi_sq,
                'n_samples': len(masses)
            })

            print(f"{m_q:>8.3f} {mean_mpi:>10.4f} {err_mpi:>10.4f} "
                  f"{mean_mpi_sq:>12.4f} {err_mpi_sq:>10.4f}")

    # =========================================================================
    # Fit GMOR relation
    # =========================================================================
    if len(results) >= 3:
        m_q_arr = np.array([r['quark_mass'] for r in results])
        m_pi_sq_arr = np.array([r['pion_mass_sq'] for r in results])
        m_pi_sq_err = np.array([r['pion_mass_sq_err'] for r in results])

        slope, intercept, m_crit, fit_errors = fit_gmor_relation(
            m_q_arr, m_pi_sq_arr, m_pi_sq_err
        )

        slope_err = fit_errors['slope_err']
        intercept_err = fit_errors['intercept_err']
        m_crit_err = fit_errors['m_crit_err']

        print("\n" + "=" * 70)
        print("GMOR FIT RESULTS")
        print("=" * 70)
        print(f"Fit: m_pi^2 = {slope:.4f}(±{slope_err:.4f}) × m_q + "
              f"{intercept:.4f}(±{intercept_err:.4f})")
        print(f"\nSlope (2B): {slope:.4f} ± {slope_err:.4f}")
        print(f"Critical mass m_crit: {m_crit:.4f} ± {m_crit_err:.4f}")

        # ---------------------------------------------------------------------
        # Compare with literature
        # ---------------------------------------------------------------------
        print("\n" + "-" * 50)
        print("LITERATURE COMPARISON:")
        print("-" * 50)

        lit_val = -0.80  # Literature value for quenched SU(3) at β=6.0
        lit_err = 0.05
        print(f"Literature (quenched SU(3), β=6.0): m_crit = {lit_val:.2f} ± {lit_err:.2f}")
        print(f"This work: m_crit = {m_crit:.4f} ± {m_crit_err:.4f}")

        diff = m_crit - lit_val
        sigma = abs(diff) / np.sqrt(m_crit_err**2 + lit_err**2)
        print(f"Difference: {diff:+.4f} ({sigma:.1f}σ)")

        if sigma < 2:
            print("✓ Result is consistent with literature within 2σ")
        elif sigma < 3:
            print("⚠ Result differs by 2-3σ from literature")
        else:
            print("✗ Result differs significantly from literature")
            print("  (Expected for small lattices due to finite-size effects)")

        # =====================================================================
        # Create diagnostic plots
        # =====================================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # -----------------------------------------------------------------
        # Plot 1: m²_π vs m_q (GMOR relation)
        # -----------------------------------------------------------------
        ax1.errorbar(m_q_arr, m_pi_sq_arr, yerr=m_pi_sq_err, fmt='o',
                    markersize=10, capsize=5, label='Data', color='blue')

        # Plot fit line extending to m_crit
        x_fit = np.linspace(min(m_crit, 0), max(m_q_arr) * 1.1, 100)
        y_fit = slope * x_fit + intercept
        ax1.plot(x_fit, y_fit, 'r--', linewidth=2,
                label=f'Fit: $m_{{crit}}$ = {m_crit:.3f}±{m_crit_err:.3f}')

        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.axvline(x=m_crit, color='green', linestyle=':', linewidth=2)

        ax1.set_xlabel(r'Quark Mass $m_q$', fontsize=12)
        ax1.set_ylabel(r'$m_\pi^2$', fontsize=12)
        ax1.set_title(f'GMOR Relation (β={beta}, {lattice_dims[0]}³×{lattice_dims[3]}, N={n_configs})',
                     fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # -----------------------------------------------------------------
        # Plot 2: m_π vs m_q
        # -----------------------------------------------------------------
        m_pi_arr = np.array([r['pion_mass'] for r in results])
        m_pi_err_arr = np.array([r['pion_error'] for r in results])

        ax2.errorbar(m_q_arr, m_pi_arr, yerr=m_pi_err_arr, fmt='s',
                    markersize=10, capsize=5, label='Data', color='purple')

        # GMOR prediction: m_π = √(slope × m_q + intercept)
        x_theory = np.linspace(m_crit + 0.01, max(m_q_arr) * 1.1, 100)
        y_theory = np.sqrt(np.maximum(0, slope * x_theory + intercept))
        ax2.plot(x_theory, y_theory, 'r--', linewidth=2, label='GMOR prediction')

        ax2.axvline(x=m_crit, color='green', linestyle=':', linewidth=2,
                   label=f'$m_{{crit}}$ = {m_crit:.3f}')
        ax2.set_xlabel(r'Quark Mass $m_q$', fontsize=12)
        ax2.set_ylabel(r'$m_\pi$', fontsize=12)
        ax2.set_title('Pion Mass vs Quark Mass', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_fname = f'{output_prefix}_b{beta:.2f}_N{n_configs}.png'
        plt.savefig(plot_fname, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved: {plot_fname}")
        plt.close()

        # =====================================================================
        # Save results to pickle file
        # =====================================================================
        results_data = {
            'beta': beta,
            'lattice_dims': lattice_dims,
            'n_configs': n_configs,
            'quark_masses': quark_masses,
            'results': results,
            'slope': slope,
            'slope_err': slope_err,
            'intercept': intercept,
            'intercept_err': intercept_err,
            'm_crit': m_crit,
            'm_crit_err': m_crit_err,
            'timestamp': datetime.now().isoformat()
        }

        results_fname = f'{output_prefix}_results_b{beta:.2f}_N{n_configs}.pkl'
        with open(results_fname, 'wb') as f:
            pickle.dump(results_data, f)
        print(f"Results saved: {results_fname}")

        return results_data

    return None


# ============================================================================
#  COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """
    Command-line interface for GMOR ensemble analysis.

    Examples:
        python gmor_ensemble_analysis.py --config_dir production_configs
        python gmor_ensemble_analysis.py --config_dir large_configs --output gmor_large
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='GMOR Relation Analysis with Ensemble Averaging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PHYSICS BACKGROUND:
    The GMOR relation m²_π = 2B(m_q + m_crit) connects pion mass to quark mass.
    The critical mass m_crit is where the pion becomes massless (chiral limit).
    For Wilson fermions at β=6.0, m_crit ≈ -0.80 (literature value).

EXAMPLES:
    %(prog)s --config_dir production_configs
    %(prog)s --config_dir large_configs --output gmor_large

OUTPUT:
    - PNG plot showing GMOR fit
    - Pickle file with all results
        """
    )

    parser.add_argument('--config_dir', type=str, default='production_configs',
                       help='Directory containing config files')
    parser.add_argument('--output', type=str, default='gmor_ensemble',
                       help='Output file prefix')

    args = parser.parse_args()

    # Default quark masses to probe
    # These should span from heavy (easier to compute) to light (near chiral limit)
    quark_masses = [0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

    run_ensemble_analysis(
        config_dir=args.config_dir,
        quark_masses=quark_masses,
        output_prefix=args.output
    )


if __name__ == "__main__":
    main()
