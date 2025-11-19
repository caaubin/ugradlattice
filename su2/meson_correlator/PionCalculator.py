"""
Pion Meson Calculator for Lattice QCD
=====================================

This module specializes in calculating pion (π) masses from lattice QCD 
gauge configurations using Wilson fermions. The pion is the lightest 
meson and serves as the Goldstone boson of spontaneous chiral symmetry breaking.

Physics Background:
The pion operator π(x) = ψ̄(x)γ₅ψ(x) has quantum numbers J^PC = 0^(-+).
As the Goldstone boson, the pion mass satisfies the Gell-Mann-Oakes-Renner relation:
M²_π = 2Bm_quark, where B measures chiral symmetry breaking.

In the chiral limit (m_quark → 0), the pion mass vanishes: M_π → 0.
This makes the pion calculation crucial for understanding chiral dynamics
and extrapolating to physical quark masses.

Key Features:
- Optimized for pseudoscalar (γ₅) meson operator
- Specialized correlator construction for pion channel  
- Enhanced error analysis for light meson extraction
- Chiral behavior monitoring and diagnostics

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

def get_pion_operator(verbose=False):
    """
    Return the pion meson operator γ₅
    
    The pion is created by the pseudoscalar operator:
    π(x) = ψ̄(x)γ₅ψ(x)
    
    where γ₅ = diag(-1, -1, 1, 1) in our Dirac representation.
    This creates a meson with quantum numbers J^PC = 0^(-+).
    
    Physics Notes:
    - Pion is the lightest hadron (Goldstone boson)  
    - Couples to chiral symmetry breaking: M²_π ∝ m_quark
    - In chiral limit: M_π → 0 (massless Goldstone mode)
    - Experimental: M_π ≈ 140 MeV (physical pion mass)
    
    Args:
        verbose (bool): Enable detailed physics output
        
    Returns:
        dict: Pion channel information with gamma matrix
    """
    if verbose:
        logging.info("Setting up pion channel:")
        logging.info("  Operator: π(x) = ψ̄(x)γ₅ψ(x)")
        logging.info("  Quantum numbers: J^PC = 0^(-+)")
        logging.info("  Physics: Goldstone boson of chiral symmetry")
        logging.info("  Mass behavior: M²_π ∝ m_quark (chiral limit)")
    
    gamma_matrices = get_gamma_matrices()
    
    pion_info = {
        'gamma': gamma_matrices['gamma5'],
        'JPC': '0-+',
        'name': 'Pion',
        'description': 'Pseudoscalar Goldstone boson',
        'operator': 'γ₅',
        'physics': 'Lightest meson, chiral symmetry breaking probe'
    }
    
    return pion_info

def calculate_pion_correlator(propagators, lattice_dims, n_colors=2, verbose=False):
    """
    Calculate pion correlator C_π(t) = Tr[γ₅ S(0,t)]

    Constructs the pion two-point correlation function from quark propagators.
    The correlator measures ⟨π(t)π†(0)⟩ and exhibits exponential decay:

    C_π(t) ~ A exp(-M_π t) + excited states

    The ground state mass M_π is extracted from the asymptotic behavior.

    Correlator Construction:
    1. Sum over all color indices (SU(2): colors 0,1; SU(3): colors 0,1,2)
    2. Contract Dirac indices with γ₅ matrix
    3. Trace over resulting 4×4 matrix at each time slice
    4. Take real part (correlator should be real for γ₅)

    Args:
        propagators (list): Quark propagators for all color-spin combinations
        lattice_dims (list): [Lx, Ly, Lz, Lt] lattice dimensions
        n_colors (int): Number of colors for SU(N) (default: 2)
        verbose (bool): Enable detailed correlator diagnostics

    Returns:
        numpy.ndarray: Pion correlator C_π(t) for all time slices

    Physics Expectations:
    - Monotonic decay for large t (ground state dominance)
    - Positive values (π† = π for γ₅ operator)
    - Exponential falloff with rate M_π
    """
    Lx, Ly, Lz, Lt = lattice_dims
    pion_correlator = np.zeros(Lt, dtype=complex)
    
    # Get pion gamma matrix (γ₅)
    pion_info = get_pion_operator(verbose=False)
    gamma5 = pion_info['gamma']
    
    if verbose:
        logging.info("  Computing pion correlator C_π(t) = Tr[γ₅ S(0,t)]:")
        logging.info(f"    Time extent: {Lt} slices")
        logging.info(f"    Propagators: {len(propagators)} (should be {n_colors*4} for full calculation)")

        # Validate propagator set
        expected_propagators = n_colors * 4  # n_colors × 4 spins
        if len(propagators) != expected_propagators:
            logging.warning(f"    Expected {expected_propagators} propagators, got {len(propagators)}")
    
    # Loop over all time slices
    for t in range(Lt):
        # Source at spatial origin, sink at time t
        sink_point = np.array([0, 0, 0, t])
        sink_site_idx = su2.p2i(sink_point, lattice_dims)
        sink_base_idx = (n_colors * 4) * sink_site_idx

        correlator_sum = 0.0

        # Sum over all color indices
        for color in range(n_colors):
            # Construct 4×4 propagator matrix for this color
            S_matrix = np.zeros((4, 4), dtype=complex)
            
            # Fill matrix elements from propagator solutions
            for source_spin in range(4):
                source_prop_idx = 4 * color + source_spin
                
                if source_prop_idx < len(propagators):
                    source_propagator = propagators[source_prop_idx]
                    
                    for sink_spin in range(4):
                        sink_global_idx = sink_base_idx + 4 * color + sink_spin
                        
                        if sink_global_idx < len(source_propagator):
                            S_matrix[sink_spin, source_spin] = source_propagator[sink_global_idx]
            
            # Compute Tr[γ₅ S] for this color
            trace_gamma5_S = np.trace(gamma5 @ S_matrix)
            correlator_sum += trace_gamma5_S
            
            if verbose and t < 3 and color == 0:
                # Debug output for first few time slices
                S_norm = np.linalg.norm(S_matrix)
                logging.info(f"      t={t}, color={color}: |S|={S_norm:.2e}, Tr[γ₅S]={trace_gamma5_S:.6e}")
        
        pion_correlator[t] = correlator_sum
    
    # Convert to real values (pion correlator should be real)
    pion_correlator_real = np.real(pion_correlator)
    
    if verbose:
        # Correlator quality diagnostics
        max_imaginary = np.max(np.abs(np.imag(pion_correlator)))
        correlator_range = np.max(pion_correlator_real) - np.min(pion_correlator_real)
        
        logging.info(f"  Pion correlator computed:")
        logging.info(f"    Real part range: [{np.min(pion_correlator_real):.2e}, {np.max(pion_correlator_real):.2e}]")
        logging.info(f"    Maximum imaginary component: {max_imaginary:.2e}")
        
        if max_imaginary > 1e-10:
            logging.warning(f"    Large imaginary part may indicate numerical issues")
        
        # Check for expected exponential behavior
        if len(pion_correlator_real) > 1:
            if pion_correlator_real[0] > 0 and pion_correlator_real[1] > 0:
                ratio = pion_correlator_real[0] / pion_correlator_real[1]
                if ratio > 1.0:
                    logging.info(f"    Good: C(0)/C(1) = {ratio:.3f} > 1 (expected exponential decay)")
                else:
                    logging.warning(f"    Unusual: C(0)/C(1) = {ratio:.3f} ≤ 1 (non-monotonic)")
            
        # Check for zeros (problematic for mass extraction)
        zero_count = np.sum(np.abs(pion_correlator_real) < 1e-15)
        if zero_count > 0:
            logging.warning(f"    {zero_count} time slices have near-zero correlator values")
    
    return pion_correlator_real

def calculate_pion_mass(U, mass, lattice_dims, wilson_r=0.5, n_colors=2, solver='auto', verbose=False):
    """
    Complete pion mass calculation from gauge configuration

    Performs the full lattice QCD workflow to extract the pion mass:
    1. Build Wilson-Dirac matrix D with gauge field U
    2. Solve D·S = δ for all required quark propagators
    3. Construct pion correlator C_π(t) = Tr[γ₅ S(0,t)]
    4. Extract effective mass M_eff(t) = ln[C(t)/C(t+1)]
    5. Fit plateau to obtain ground state mass M_π

    This is the primary interface for pion mass calculations.

    Args:
        U (array): Gauge field configuration from thermal generation
        mass (float): Bare quark mass in lattice units
        lattice_dims (list): [Lx, Ly, Lz, Lt] lattice dimensions
        wilson_r (float): Wilson parameter (default: 0.5)
        n_colors (int): Number of colors for SU(N) (default: 2)
        solver (str): Linear solver method ('auto', 'direct', 'gmres', 'lsqr')
        verbose (bool): Enable detailed physics diagnostics
        
    Returns:
        dict: Complete pion analysis results including:
            - 'meson_mass': Extracted pion mass M_π  
            - 'meson_error': Statistical uncertainty
            - 'chi_squared': Plateau fit quality (χ²/dof)
            - 'correlator': Raw correlator data C_π(t)
            - 'effective_mass': Effective mass M_eff(t)  
            - 'fit_range': Time range used for plateau fit
            - Plus additional metadata and physics parameters
            
    Physics Interpretation:
        - Light pion (M_π < 2×m_eff): Good chiral behavior
        - Heavy pion (M_π > 5×m_eff): Non-chiral regime
        - χ²/dof ≈ 1: Clean plateau extraction
        - χ²/dof >> 1: Excited state contamination or poor statistics
    """
    if verbose:
        logging.info(f"PION MASS CALCULATION")
        logging.info(f"=" * 40)
        logging.info(f"Input parameters:")
        logging.info(f"  Bare quark mass: {mass:.6f}")
        logging.info(f"  Wilson parameter: {wilson_r:.3f}")
        logging.info(f"  Effective mass: m_eff = m + 4r = {mass + 4*wilson_r:.6f}")
        logging.info(f"  Lattice dimensions: {lattice_dims}")
        logging.info(f"  Linear solver: {solver}")
        
        # Physics context for pion
        logging.info(f"\nPion physics context:")
        logging.info(f"  • Lightest meson, Goldstone boson of chiral symmetry")
        logging.info(f"  • Operator: π(x) = ψ̄(x)γ₅ψ(x), J^PC = 0^(-+)")
        logging.info(f"  • Chiral limit: M_π → 0 as m_quark → 0")
        logging.info(f"  • Physical mass: M_π ≈ 140 MeV")
    
    # Step 1: Get pion operator information
    pion_info = get_pion_operator(verbose=verbose)
    
    # Step 2: Build Wilson-Dirac matrix
    if verbose:
        logging.info(f"\nStep 1: Building Wilson-Dirac matrix...")

    D = build_wilson_dirac_matrix(mass, lattice_dims, wilson_r, U, n_colors, verbose)

    if D.nnz == 0:
        logging.error("Wilson-Dirac matrix construction failed!")
        return create_failed_result(mass, lattice_dims, wilson_r, solver)

    # Step 3: Solve for all required propagators (n_colors × 4 spins)
    if verbose:
        logging.info(f"\nStep 2: Solving for quark propagators...")
        logging.info(f"  Computing {n_colors*4} propagators ({n_colors} colors × 4 spins)")

    propagators = []
    solve_start_time = time.time()

    for color in range(n_colors):
        for spin in range(4):
            prop_idx = len(propagators) + 1

            if verbose:
                logging.info(f"    Propagator {prop_idx}/{n_colors*4}: color={color}, spin={spin}")

            # Create point source at t=0
            source = create_point_source(lattice_dims, t_source=0, color=color, spin=spin, n_colors=n_colors, verbose=False)
            
            # Solve Dirac equation
            propagator = solve_dirac_system(D, source, method=solver, verbose=False)
            propagators.append(propagator)
            
            # Quick convergence check
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

    # Step 4: Calculate pion correlator
    if verbose:
        logging.info(f"\nStep 3: Computing pion correlator...")

    correlator = calculate_pion_correlator(propagators, lattice_dims, n_colors, verbose)
    
    # Step 5: Extract effective mass
    if verbose:
        logging.info(f"\nStep 4: Extracting effective mass...")
    
    mass_eff, mass_err = calculate_effective_mass(correlator, verbose)
    
    # Step 6: Fit plateau for ground state mass
    if verbose:
        logging.info(f"\nStep 5: Plateau fitting for ground state mass...")
    
    plateau_mass, plateau_err, chi2, fit_range = fit_plateau(mass_eff, mass_err, verbose=verbose)
    
    # Compile complete results
    results = {
        'channel': 'pion',
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
        'channel_info': pion_info,
        'physics_notes': {
            'effective_mass': mass + 4*wilson_r,
            'mass_ratio': plateau_mass / (mass + 4*wilson_r) if (mass + 4*wilson_r) > 0 else 0,
            'chiral_behavior': 'light' if plateau_mass < 2*(mass + 4*wilson_r) else 'heavy'
        }
    }
    
    if verbose:
        logging.info(f"\nPION CALCULATION COMPLETE")
        logging.info(f"=" * 40)
        logging.info(f"Final Results:")
        logging.info(f"  Pion mass: M_π = {plateau_mass:.6f} ± {plateau_err:.6f}")
        logging.info(f"  Mass ratio: M_π/m_eff = {results['physics_notes']['mass_ratio']:.3f}")
        logging.info(f"  Fit quality: χ²/dof = {chi2:.3f}")
        logging.info(f"  Chiral regime: {results['physics_notes']['chiral_behavior']}")
        
        # Physics interpretation
        mass_ratio = results['physics_notes']['mass_ratio']
        if mass_ratio < 1.5:
            logging.info(f"  → Light pion, approaching chiral limit")
        elif mass_ratio < 3.0:
            logging.info(f"  → Intermediate pion mass")  
        else:
            logging.info(f"  → Heavy pion, far from chiral limit")
            
        if chi2 > 5.0:
            logging.warning(f"  → Large χ² suggests excited state contamination")
        elif chi2 < 0.1:
            logging.warning(f"  → Very small χ² may indicate fitting issues")
    
    return results

def create_failed_result(mass, lattice_dims, wilson_r, solver):
    """Create error result structure when calculation fails"""
    return {
        'channel': 'pion',
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
        'channel_info': get_pion_operator(),
        'physics_notes': {
            'status': 'failed',
            'effective_mass': mass + 4*wilson_r,
            'mass_ratio': 0.0,
            'chiral_behavior': 'unknown'
        }
    }

# Convenience function for quick pion calculations
def quick_pion_mass(gauge_config_file, mass=0.1, wilson_r=0.5, lattice_dims=None, verbose=True):
    """
    Simplified interface for pion mass calculation from gauge configuration file
    
    Args:
        gauge_config_file (str): Path to pickled gauge configuration
        mass (float): Bare quark mass (default: 0.1)
        wilson_r (float): Wilson parameter (default: 0.5)
        lattice_dims (list): Lattice dimensions (auto-detected if None)
        verbose (bool): Enable detailed output
        
    Returns:
        dict: Pion calculation results
    """
    import pickle
    
    # Load gauge configuration
    try:
        with open(gauge_config_file, 'rb') as f:
            gauge_data = pickle.load(f)
            
        if isinstance(gauge_data, list) and len(gauge_data) >= 2:
            U = [None, gauge_data[1]]  # Standard format
        else:
            raise ValueError("Unsupported gauge configuration format")
            
    except Exception as e:
        logging.error(f"Failed to load gauge configuration: {e}")
        return None
    
    # Auto-detect lattice dimensions if not provided
    if lattice_dims is None:
        # Try to infer from gauge field shape
        if U[1] is not None:
            V = U[1].shape[0]
            # Assume cubic spatial volume with Lt = 4*Ls
            Ls = int(round((V/16)**(1/3)))
            lattice_dims = [Ls, Ls, Ls, 4*Ls]
            
            if verbose:
                logging.info(f"Auto-detected lattice dimensions: {lattice_dims}")
        else:
            lattice_dims = [4, 4, 4, 16]  # Default fallback
    
    # Run pion calculation
    return calculate_pion_mass(U, mass, lattice_dims, wilson_r, 'auto', verbose)