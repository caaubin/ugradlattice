"""
Sigma Meson Calculator for Lattice QCD
======================================

This module specializes in calculating sigma (σ) scalar meson masses from
lattice QCD gauge configurations using Wilson fermions. The sigma meson
has quantum numbers J^PC = 0^(++) and serves as the chiral partner of the pion.

Physics Background:
The sigma meson operator σ(x) = ψ̄(x)ψ(x) is the scalar density operator.
Unlike the pion, the sigma does not vanish in the chiral limit due to 
explicit chiral symmetry breaking by quark masses.

Key Properties:
- Chiral partner of pion: (π, σ) form chiral multiplet
- Remains massive in chiral limit (M_σ ≠ 0 as m_quark → 0)  
- Typically heavier than vector mesons: M_σ > M_ρ > M_π
- Large width in nature: Γ_σ ≈ 400-700 MeV (very broad resonance)
- Controversial experimental status (f₀(500) identification)

Lattice Considerations:
- Scalar correlators often have poor signal-to-noise ratio
- May require specialized techniques (e.g., smearing, variational methods)
- Sensitive to chiral symmetry restoration on the lattice
- Wilson fermions explicitly break chiral symmetry

Author: Zeke Mohammed
Advisor: Dr. Aubin  
Institution: Fordham University
Date: September 2025
"""

import numpy as np
import logging
import time
from MesonBase import (
    build_wilson_dirac_matrix,
    create_point_source,
    solve_dirac_system, 
    calculate_effective_mass,
    fit_plateau,
    get_gamma_matrices
)
import su2

def get_sigma_operator(verbose=False):
    """
    Return the sigma meson operator (scalar density)
    
    The sigma meson is created by the scalar operator:
    σ(x) = ψ̄(x)ψ(x) = ψ̄(x)Iψ(x)
    
    where I is the 4×4 identity matrix in Dirac space.
    This creates a meson with quantum numbers J^PC = 0^(++).
    
    Physics Notes:
    - Scalar density operator, couples to quark condensate ⟨ψ̄ψ⟩
    - Chiral partner of pion in SU(2)_L × SU(2)_R multiplet  
    - Does not vanish in chiral limit (explicit symmetry breaking)
    - Experimental identification controversial (f₀(500) or σ pole?)
    - Very broad resonance: Γ_σ ≈ 400-700 MeV
    
    Args:
        verbose (bool): Enable detailed physics output
        
    Returns:
        dict: Sigma channel information with identity operator
    """
    if verbose:
        logging.info("Setting up sigma meson channel:")
        logging.info("  Operator: σ(x) = ψ̄(x)ψ(x)")
        logging.info("  Quantum numbers: J^PC = 0^(++)")
        logging.info("  Physics: Scalar meson, chiral partner of pion")
        logging.info("  Chiral behavior: M_σ remains finite as m_quark → 0")
        logging.info("  Experimental: Very broad resonance, controversial identification")
    
    gamma_matrices = get_gamma_matrices()
    
    sigma_info = {
        'gamma': gamma_matrices['identity'],
        'JPC': '0++',
        'name': 'Sigma', 
        'description': 'Scalar meson, chiral partner of pion',
        'operator': 'I (identity)',
        'physics': 'Couples to chiral symmetry breaking, remains massive in chiral limit'
    }
    
    return sigma_info

def calculate_sigma_correlator(propagators, lattice_dims, n_colors=2, verbose=False):
    """
    Calculate sigma correlator C_σ(t) = Tr[I S(0,t)]
    
    Constructs the sigma scalar meson correlation function from quark propagators.
    The correlator measures ⟨σ(t)σ†(0)⟩ and exhibits exponential decay:
    
    C_σ(t) ~ A exp(-M_σ t) + excited states
    
    Scalar correlators are notoriously difficult due to:
    - Poor overlap with ground state (small A coefficient)
    - Strong mixing with higher Fock states
    - Sensitivity to quark-antiquark vs. tetraquark components
    - Large statistical fluctuations
    
    Args:
        propagators (list): Quark propagators for all color-spin combinations
        lattice_dims (list): [Lx, Ly, Lz, Lt] lattice dimensions
        n_colors (int): Number of colors for SU(N) (default: 2)
        verbose (bool): Enable detailed correlator diagnostics
        
    Returns:
        numpy.ndarray: Sigma correlator C_σ(t) for all time slices
        
    Numerical Challenges:
    - Scalar correlators often have poor signal-to-noise
    - May require enhanced statistics or specialized techniques
    - Disconnected diagrams important but computationally expensive
    - Wilson fermions break chiral symmetry explicitly
    """
    Lx, Ly, Lz, Lt = lattice_dims
    sigma_correlator = np.zeros(Lt, dtype=complex)
    
    # Get sigma operator (identity matrix)
    sigma_info = get_sigma_operator(verbose=False)
    identity_op = sigma_info['gamma']
    
    if verbose:
        logging.info(f"  Computing sigma correlator C_σ(t) = Tr[I S(0,t)]:")
        logging.info(f"    Operator: scalar density ψ̄ψ")
        logging.info(f"    Time extent: {Lt} slices")
        logging.info(f"    Propagators: {len(propagators)} (should be 8)")
        logging.info(f"    Note: Scalar correlators typically have poor signal-to-noise")
        
        if len(propagators) != 8:
            logging.warning(f"    Expected 8 propagators, got {len(propagators)}")
    
    # Loop over all time slices
    for t in range(Lt):
        # Sink at spatial origin, time t
        sink_point = np.array([0, 0, 0, t])
        sink_site_idx = su2.p2i(sink_point, lattice_dims)
        sink_base_idx = (n_colors * 4) * sink_site_idx
        
        correlator_sum = 0.0
        
        # Sum over colors 
        for color in range(n_colors):
            # Build 4×4 propagator matrix for this color
            S_matrix = np.zeros((4, 4), dtype=complex)
            
            # Fill matrix from propagator solutions
            for source_spin in range(4):
                source_prop_idx = 4 * color + source_spin
                
                if source_prop_idx < len(propagators):
                    source_propagator = propagators[source_prop_idx]
                    
                    for sink_spin in range(4):
                        sink_global_idx = sink_base_idx + 4 * color + sink_spin
                        
                        if sink_global_idx < len(source_propagator):
                            S_matrix[sink_spin, source_spin] = source_propagator[sink_global_idx]
            
            # Compute Tr[I S] = Tr[S] for this color
            trace_S = np.trace(S_matrix)
            correlator_sum += trace_S
            
            if verbose and t < 3 and color == 0:
                # Debug output for first few time slices
                S_norm = np.linalg.norm(S_matrix)
                logging.info(f"      t={t}, color={color}: |S|={S_norm:.2e}, Tr[S]={trace_S:.6e}")
        
        sigma_correlator[t] = correlator_sum
    
    # Convert to real values (sigma correlator should be real)
    sigma_correlator_real = np.real(sigma_correlator)
    
    if verbose:
        # Correlator quality diagnostics
        max_imaginary = np.max(np.abs(np.imag(sigma_correlator)))
        correlator_range = np.max(sigma_correlator_real) - np.min(sigma_correlator_real)
        
        logging.info(f"  Sigma correlator computed:")
        logging.info(f"    Real part range: [{np.min(sigma_correlator_real):.2e}, {np.max(sigma_correlator_real):.2e}]")
        logging.info(f"    Maximum imaginary component: {max_imaginary:.2e}")
        
        if max_imaginary > 1e-10:
            logging.warning(f"    Large imaginary part may indicate numerical issues")
        
        # Signal-to-noise assessment
        if len(sigma_correlator_real) > 1 and np.abs(sigma_correlator_real[0]) > 0:
            signal_ratio = np.abs(sigma_correlator_real[-1]) / np.abs(sigma_correlator_real[0])
            logging.info(f"    Signal decay: C(T-1)/C(0) = {signal_ratio:.2e}")
            
            if signal_ratio > 0.1:
                logging.warning(f"    Poor signal-to-noise: correlator not well-decayed")
                logging.info(f"      Scalar correlators often require larger time extent or statistics")
            elif signal_ratio < 1e-12:
                logging.warning(f"    Correlator may have decayed to numerical noise")
        
        # Check for problematic values
        zero_count = np.sum(np.abs(sigma_correlator_real) < 1e-15)
        negative_count = np.sum(sigma_correlator_real < 0)
        
        if zero_count > 0:
            logging.warning(f"    {zero_count} time slices have near-zero values")
        if negative_count > 0:
            logging.info(f"    {negative_count} time slices have negative values")
            logging.info(f"      Negative scalar correlator values can occur due to quantum fluctuations")
        
        # Sigma-specific warnings
        if np.std(sigma_correlator_real) / np.mean(np.abs(sigma_correlator_real)) > 2.0:
            logging.warning(f"    Very noisy correlator - typical for scalar channel")
            logging.info(f"      Consider: longer time evolution, smeared sources, variational methods")
    
    return sigma_correlator_real

def calculate_sigma_mass(U, mass, lattice_dims, wilson_r=0.5, n_colors=2, solver='auto', verbose=False):
    """
    Complete sigma meson mass calculation from gauge configuration
    
    Performs the full lattice QCD workflow to extract sigma meson mass:
    1. Build Wilson-Dirac matrix with gauge field coupling
    2. Solve for quark propagators (8 total: 2 colors × 4 spins) 
    3. Construct sigma correlator C_σ(t) = Tr[I S(0,t)]
    4. Extract effective mass M_eff(t) = ln[C(t)/C(t+1)]
    5. Fit plateau to obtain ground state mass M_σ
    
    Args:
        U (array): Gauge field configuration from thermal generation
        mass (float): Bare quark mass in lattice units
        lattice_dims (list): [Lx, Ly, Lz, Lt] lattice dimensions
        wilson_r (float): Wilson parameter (default: 0.5)
        n_colors (int): Number of colors for SU(N) (default: 2)
        solver (str): Linear solver method
        verbose (bool): Enable detailed diagnostics
        
    Returns:
        dict: Complete sigma analysis results with mass, errors, and diagnostics
        
    Physics Expectations:
    - M_σ > M_ρ > M_π (scalar heaviest in light meson spectrum)
    - M_σ remains finite in chiral limit (≠ Goldstone boson)
    - Large statistical errors typical for scalar channel
    - May require specialized techniques for clean extraction
    """
    if verbose:
        logging.info(f"SIGMA MESON CALCULATION")
        logging.info(f"=" * 40)
        logging.info(f"Input parameters:")
        logging.info(f"  Bare quark mass: {mass:.6f}")
        logging.info(f"  Wilson parameter: {wilson_r:.3f}")
        logging.info(f"  Effective mass: m_eff = m + 4r = {mass + 4*wilson_r:.6f}")
        logging.info(f"  Lattice dimensions: {lattice_dims}")
        logging.info(f"  Linear solver: {solver}")
        
        logging.info(f"\nSigma meson physics context:")
        logging.info(f"  • Scalar meson, J^PC = 0^(++)")
        logging.info(f"  • Operator: σ(x) = ψ̄(x)ψ(x) (scalar density)")
        logging.info(f"  • Chiral partner of pion, but remains massive in chiral limit")
        logging.info(f"  • Expected: M_σ > M_ρ > M_π (heaviest light meson)")
        logging.info(f"  • Experimental: Very broad resonance, controversial identification")
        logging.info(f"  • Computational challenge: Poor signal-to-noise in scalar channel")
    
    # Step 1: Get sigma operator information
    sigma_info = get_sigma_operator(verbose=verbose)
    
    # Step 2: Build Wilson-Dirac matrix
    if verbose:
        logging.info(f"\nStep 1: Building Wilson-Dirac matrix...")
    
    D = build_wilson_dirac_matrix(mass, lattice_dims, wilson_r, U, n_colors, verbose)
    
    if D.nnz == 0:
        logging.error("Wilson-Dirac matrix construction failed!")
        return create_failed_result(mass, lattice_dims, wilson_r, solver)
    
    # Step 3: Solve for quark propagators
    if verbose:
        logging.info(f"\nStep 2: Solving for quark propagators...")
        logging.info(f"  Scalar mesons use same propagators as other channels")
        logging.info(f"  Computing {n_colors*4} propagators ({n_colors} colors × 4 spins)")
    
    propagators = []
    solve_start_time = time.time()
    
    for color in range(n_colors):
        for spin in range(4):
            prop_idx = len(propagators) + 1
            
            if verbose:
                logging.info(f"    Propagator {prop_idx}/8: color={color}, spin={spin}")
            
            # Create point source at t=0
            source = create_point_source(lattice_dims, t_source=0, color=color, spin=spin, n_colors=n_colors, verbose=False)
            
            # Solve Dirac equation
            propagator = solve_dirac_system(D, source, method=solver, verbose=False)
            propagators.append(propagator)
            
            # Check convergence
            prop_norm = np.linalg.norm(propagator)
            if prop_norm == 0:
                logging.warning(f"      Zero propagator for color={color}, spin={spin}")
    
    solve_time = time.time() - solve_start_time
    
    if verbose:
        successful_props = sum(1 for p in propagators if np.linalg.norm(p) > 0)
        logging.info(f"  Propagator calculation completed in {solve_time:.2f}s")
        logging.info(f"  Successful propagators: {successful_props}/{n_colors*4}")
        
        if successful_props < n_colors*4:
            logging.warning(f"  Some propagators failed - results may be inaccurate")
    
    # Step 4: Calculate sigma correlator
    if verbose:
        logging.info(f"\nStep 3: Computing sigma correlator...")
        logging.info(f"  Warning: Scalar correlators typically have poor signal-to-noise")
    
    correlator = calculate_sigma_correlator(propagators, lattice_dims, n_colors, verbose)
    
    # Step 5: Extract effective mass
    if verbose:
        logging.info(f"\nStep 4: Extracting effective mass...")
        logging.info(f"  Note: Scalar effective masses often very noisy")
    
    mass_eff, mass_err = calculate_effective_mass(correlator, verbose)
    
    # Step 6: Fit plateau
    if verbose:
        logging.info(f"\nStep 5: Plateau fitting...")
        logging.info(f"  Scalar mesons often require larger t_min due to noise")
    
    # Use larger t_min for scalar mesons (poor signal-to-noise)
    t_min = 4 if len(mass_eff) > 6 else 3
    plateau_mass, plateau_err, chi2, fit_range = fit_plateau(mass_eff, mass_err, t_min=t_min, verbose=verbose)
    
    # Compile complete results
    results = {
        'channel': 'sigma',
        'input_mass': mass,
        'meson_mass': plateau_mass,
        'meson_error': plateau_err,
        'chi_squared': chi2,
        'fit_range': fit_range,
        'correlator': correlator.tolist(),
        'effective_mass': mass_eff.tolist() if len(mass_eff) > 0 else [],
        'mass_errors': mass_err.tolist() if len(mass_err) > 0 else [],
        'lattice_dims': lattice_dims,
        'wilson_r': wilson_r,
        'solver': solver,
        'channel_info': sigma_info,
        'physics_notes': {
            'effective_mass': mass + 4*wilson_r,
            'mass_ratio': plateau_mass / (mass + 4*wilson_r) if (mass + 4*wilson_r) > 0 else 0,
            'meson_type': 'scalar',
            'chiral_partner': 'pion'
        }
    }
    
    if verbose:
        logging.info(f"\nSIGMA CALCULATION COMPLETE")
        logging.info(f"=" * 40)
        logging.info(f"Final Results:")
        logging.info(f"  Sigma mass: M_σ = {plateau_mass:.6f} ± {plateau_err:.6f}")
        logging.info(f"  Mass ratio: M_σ/m_eff = {results['physics_notes']['mass_ratio']:.3f}")
        logging.info(f"  Fit quality: χ²/dof = {chi2:.3f}")
        
        # Physics interpretation
        mass_ratio = results['physics_notes']['mass_ratio']
        if mass_ratio < 3.0:
            logging.warning(f"  → Unusually light sigma meson")
            logging.info(f"    Physical sigma should be heavier than rho and pion")
        elif mass_ratio < 6.0:
            logging.info(f"  → Reasonable sigma mass")
        else:
            logging.info(f"  → Heavy sigma meson")
            
        if chi2 > 10.0:
            logging.warning(f"  → Very large χ² indicates poor plateau extraction")
            logging.info(f"    This is common for scalar correlators")
            logging.info(f"    Consider: smeared sources, longer evolution, ensemble averaging")
        elif chi2 < 0.05:
            logging.warning(f"  → Very small χ² may indicate constant or over-fitted data")
            
        # Scalar-specific diagnostics
        statistical_error = plateau_err / plateau_mass if plateau_mass > 0 else 1.0
        if statistical_error > 0.5:
            logging.warning(f"  → Very large statistical error ({statistical_error*100:.0f}%)")
            logging.info(f"    This is typical for scalar channel on single configurations")
            logging.info(f"    Recommendations:")
            logging.info(f"      • Use ensemble of gauge configurations")
            logging.info(f"      • Try smeared interpolating operators")
            logging.info(f"      • Consider variational analysis with multiple operators")
            logging.info(f"      • Increase temporal lattice extent")
    
    return results

def create_failed_result(mass, lattice_dims, wilson_r, solver):
    """Create error result structure when calculation fails"""
    return {
        'channel': 'sigma',
        'input_mass': mass,
        'meson_mass': 0.0,
        'meson_error': 1.0,
        'chi_squared': 0.0,
        'fit_range': (0, 0),
        'correlator': [],
        'effective_mass': [],
        'mass_errors': [],
        'lattice_dims': lattice_dims,
        'wilson_r': wilson_r,
        'solver': solver,
        'channel_info': get_sigma_operator(),
        'physics_notes': {
            'status': 'failed',
            'effective_mass': mass + 4*wilson_r,
            'mass_ratio': 0.0,
            'meson_type': 'scalar',
            'chiral_partner': 'pion'
        }
    }

def analyze_chiral_multiplet(pion_result, sigma_result, verbose=False):
    """
    Analyze (π, σ) chiral multiplet structure
    
    Studies the mass splitting between chiral partners and approach to
    chiral symmetry restoration as quark mass decreases.
    
    Args:
        pion_result (dict): Pion calculation results
        sigma_result (dict): Sigma calculation results  
        verbose (bool): Enable detailed analysis output
        
    Returns:
        dict: Chiral multiplet analysis
        
    Physics:
    - Chiral symmetry: (π, σ) form SU(2)_L × SU(2)_R multiplet
    - Spontaneous breaking: M_π → 0, M_σ finite in chiral limit
    - Splitting: Δ = M_σ - M_π measures explicit symmetry breaking
    """
    if verbose:
        logging.info(f"CHIRAL MULTIPLET ANALYSIS")
        logging.info(f"=" * 40)
    
    M_pi = pion_result['meson_mass']
    M_sigma = sigma_result['meson_mass']
    
    splitting = M_sigma - M_pi
    relative_splitting = splitting / M_pi if M_pi > 0 else float('inf')
    
    analysis = {
        'pion_mass': M_pi,
        'sigma_mass': M_sigma,
        'mass_splitting': splitting,
        'relative_splitting': relative_splitting,
        'multiplet_quality': 'good' if splitting > 0.5 * M_pi else 'anomalous'
    }
    
    if verbose:
        logging.info(f"Chiral partner masses:")
        logging.info(f"  M_π = {M_pi:.6f} ± {pion_result['meson_error']:.6f}")
        logging.info(f"  M_σ = {M_sigma:.6f} ± {sigma_result['meson_error']:.6f}")
        logging.info(f"\nChiral symmetry measures:")
        logging.info(f"  Mass splitting: Δ = M_σ - M_π = {splitting:.6f}")
        logging.info(f"  Relative splitting: Δ/M_π = {relative_splitting:.3f}")
        
        if analysis['multiplet_quality'] == 'good':
            logging.info(f"  → Normal chiral symmetry breaking pattern")
        else:
            logging.warning(f"  → Anomalous splitting - M_σ too light or M_π too heavy")
            
        logging.info(f"\nPhysics interpretation:")
        if splitting > 0:
            logging.info(f"  • Correct hierarchy: M_σ > M_π")
            logging.info(f"  • Explicit chiral symmetry breaking by quark masses")
        else:
            logging.warning(f"  • Inverted hierarchy: M_σ < M_π")
            logging.info(f"    This can occur on single configurations due to noise")
            
        logging.info(f"  • Pion: Goldstone boson, vanishes in chiral limit")
        logging.info(f"  • Sigma: Non-Goldstone, remains massive in chiral limit")
    
    return analysis