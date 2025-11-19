"""
Rho Meson Calculator for Lattice QCD
====================================

This module specializes in calculating rho (ρ) vector meson masses from 
lattice QCD gauge configurations using Wilson fermions. The rho meson is 
the lightest vector meson with quantum numbers J^PC = 1^(--).

Physics Background:
The rho meson operators ρᵢ(x) = ψ̄(x)γᵢψ(x) (i = 1,2,3) correspond to the 
three spatial polarizations of the vector meson. In continuum QCD, these 
are degenerate due to rotational symmetry.

On the lattice:
- Finite lattice spacing breaks rotational symmetry  
- Single configurations may show ρₓ ≠ ρᵧ ≠ ρᵧ
- Ensemble averages restore degeneracy
- Physical mass: M_ρ ≈ 775 MeV

Vector mesons probe the non-Abelian gauge structure and provide important
benchmarks for lattice QCD calculations.

Key Features:
- Separate calculations for ρₓ, ρᵧ, ρᵧ polarizations
- Rotational symmetry analysis and averaging
- Enhanced signal-to-noise techniques for vector correlators
- Systematic uncertainty estimation

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

def get_rho_operator(polarization='average', verbose=False):
    """
    Return rho meson operator for specified polarization
    
    The rho meson operators are:
    - ρₓ(x) = ψ̄(x)γ₁ψ(x)  (x-polarization)
    - ρᵧ(x) = ψ̄(x)γ₂ψ(x)  (y-polarization)  
    - ρᵧ(x) = ψ̄(x)γ₃ψ(x)  (z-polarization)
    - ρ_avg = (ρₓ + ρᵧ + ρᵧ)/3  (rotationally averaged)
    
    All have quantum numbers J^PC = 1^(--) (vector meson).
    
    Args:
        polarization (str): 'x', 'y', 'z', or 'average'
        verbose (bool): Enable detailed physics output
        
    Returns:
        dict: Rho channel information with gamma matrix
        
    Physics Notes:
    - Vector mesons couple to conserved vector currents
    - Physical rho decays: ρ → π + π (dominant mode)
    - Mass hierarchy: M_π < M_ρ < M_a₁ (vector dominance)
    - Experimental: M_ρ ≈ 775 MeV, Γ_ρ ≈ 149 MeV (broad resonance)
    """
    gamma_matrices = get_gamma_matrices()
    
    if polarization.lower() == 'x':
        gamma_op = gamma_matrices['gamma1']
        name = 'Rho (x-polarization)'
        operator = 'γ₁'
    elif polarization.lower() == 'y':
        gamma_op = gamma_matrices['gamma2'] 
        name = 'Rho (y-polarization)'
        operator = 'γ₂'
    elif polarization.lower() == 'z':
        gamma_op = gamma_matrices['gamma3']
        name = 'Rho (z-polarization)' 
        operator = 'γ₃'
    elif polarization.lower() == 'average':
        # Rotationally averaged rho operator
        gamma_op = (gamma_matrices['gamma1'] + gamma_matrices['gamma2'] + gamma_matrices['gamma3']) / 3.0
        name = 'Rho (averaged)'
        operator = '(γ₁ + γ₂ + γ₃)/3'
    else:
        raise ValueError(f"Unknown rho polarization: {polarization}")
    
    if verbose:
        logging.info(f"Setting up rho meson channel:")
        logging.info(f"  Polarization: {polarization}")
        logging.info(f"  Operator: ρ(x) = ψ̄(x){operator}ψ(x)")
        logging.info(f"  Quantum numbers: J^PC = 1^(--)") 
        logging.info(f"  Physics: Vector meson, couples to vector currents")
        logging.info(f"  Experimental mass: M_ρ ≈ 775 MeV")
    
    rho_info = {
        'gamma': gamma_op,
        'JPC': '1--',
        'name': name,
        'description': 'Vector meson',
        'operator': operator, 
        'polarization': polarization,
        'physics': 'Lightest vector meson, couples to vector currents'
    }
    
    return rho_info

def calculate_rho_correlator(propagators, lattice_dims, polarization='average', n_colors=2, verbose=False):
    """
    Calculate rho correlator C_ρ(t) = Tr[γᵢ S(0,t)]
    
    Constructs the rho vector meson correlation function from quark propagators.
    The correlator measures ⟨ρᵢ(t)ρ†ᵢ(0)⟩ and exhibits exponential decay:
    
    C_ρ(t) ~ A exp(-M_ρ t) + excited states
    
    Vector correlators typically have poorer signal-to-noise than pseudoscalar
    due to the more complex Dirac structure and smaller overlap with ground state.
    
    Args:
        propagators (list): Quark propagators for all color-spin combinations
        lattice_dims (list): [Lx, Ly, Lz, Lt] lattice dimensions
        n_colors (int): Number of colors for SU(N) (default: 2)
        polarization (str): Rho polarization ('x', 'y', 'z', 'average')
        verbose (bool): Enable detailed correlator diagnostics
        
    Returns:
        numpy.ndarray: Rho correlator C_ρ(t) for all time slices
        
    Physics Notes:
    - Vector correlators often noisier than pseudoscalar
    - May require larger statistics for clean plateau
    - Single configs can show large ρₓ ≠ ρᵧ ≠ ρᵧ splittings
    - Ensemble averaging restores rotational symmetry
    """
    Lx, Ly, Lz, Lt = lattice_dims
    rho_correlator = np.zeros(Lt, dtype=complex)
    
    # Get rho gamma matrix for specified polarization
    rho_info = get_rho_operator(polarization, verbose=False)
    gamma_rho = rho_info['gamma']
    
    if verbose:
        logging.info(f"  Computing rho correlator C_ρ(t) = Tr[{rho_info['operator']} S(0,t)]:")
        logging.info(f"    Polarization: {polarization}")
        logging.info(f"    Time extent: {Lt} slices")
        logging.info(f"    Propagators: {len(propagators)} (should be 8)")
        
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
            
            # Compute Tr[γᵢ S] for this color
            trace_gamma_S = np.trace(gamma_rho @ S_matrix)
            correlator_sum += trace_gamma_S
            
            if verbose and t < 3 and color == 0:
                # Debug output for first few time slices
                S_norm = np.linalg.norm(S_matrix)
                logging.info(f"      t={t}, color={color}: |S|={S_norm:.2e}, Tr[γᵢS]={trace_gamma_S:.6e}")
        
        rho_correlator[t] = correlator_sum
    
    # Convert to real values (vector correlators should be real)
    rho_correlator_real = np.real(rho_correlator)
    
    if verbose:
        # Correlator quality diagnostics
        max_imaginary = np.max(np.abs(np.imag(rho_correlator)))
        correlator_range = np.max(rho_correlator_real) - np.min(rho_correlator_real)
        
        logging.info(f"  Rho correlator ({polarization}) computed:")
        logging.info(f"    Real part range: [{np.min(rho_correlator_real):.2e}, {np.max(rho_correlator_real):.2e}]")
        logging.info(f"    Maximum imaginary component: {max_imaginary:.2e}")
        
        if max_imaginary > 1e-10:
            logging.warning(f"    Large imaginary part may indicate numerical issues")
        
        # Signal quality assessment
        if len(rho_correlator_real) > 1 and np.abs(rho_correlator_real[0]) > 0:
            signal_ratio = np.abs(rho_correlator_real[-1]) / np.abs(rho_correlator_real[0])
            logging.info(f"    Signal decay: C(T-1)/C(0) = {signal_ratio:.2e}")
            
            if signal_ratio > 0.1:
                logging.warning(f"    Poor signal-to-noise: correlator not well-decayed")
            elif signal_ratio < 1e-10:
                logging.warning(f"    Correlator may have decayed to numerical noise")
        
        # Check for problematic values
        zero_count = np.sum(np.abs(rho_correlator_real) < 1e-15)
        negative_count = np.sum(rho_correlator_real < 0)
        
        if zero_count > 0:
            logging.warning(f"    {zero_count} time slices have near-zero values")
        if negative_count > 0:
            logging.info(f"    {negative_count} time slices have negative values (can be physical)")
    
    return rho_correlator_real

def calculate_rho_mass(U, mass, lattice_dims, polarization='average', wilson_r=0.5, n_colors=2, solver='auto', verbose=False):
    """
    Complete rho meson mass calculation from gauge configuration
    
    Performs the full lattice QCD workflow to extract rho meson mass:
    1. Build Wilson-Dirac matrix with gauge field coupling
    2. Solve for quark propagators (8 total: 2 colors × 4 spins)
    3. Construct rho correlator C_ρ(t) = Tr[γᵢ S(0,t)]
    4. Extract effective mass M_eff(t) = ln[C(t)/C(t+1)]
    5. Fit plateau to obtain ground state mass M_ρ
    
    Args:
        U (array): Gauge field configuration from thermal generation
        mass (float): Bare quark mass in lattice units
        lattice_dims (list): [Lx, Ly, Lz, Lt] lattice dimensions
        polarization (str): Vector polarization ('x', 'y', 'z', 'average')
        wilson_r (float): Wilson parameter (default: 0.5)
        solver (str): Linear solver method
        verbose (bool): Enable detailed diagnostics
        
    Returns:
        dict: Complete rho analysis results with mass, errors, and diagnostics
        
    Physics Expectations:
    - M_ρ > M_π (vector meson heavier than pseudoscalar)
    - Vector correlators typically noisier (larger statistical errors)
    - Single config may show polarization dependence
    - Ensemble average gives degenerate ρₓ = ρᵧ = ρᵧ
    """
    if verbose:
        logging.info(f"RHO MESON CALCULATION ({polarization.upper()})")
        logging.info(f"=" * 50)
        logging.info(f"Input parameters:")
        logging.info(f"  Bare quark mass: {mass:.6f}")
        logging.info(f"  Wilson parameter: {wilson_r:.3f}")
        logging.info(f"  Effective mass: m_eff = m + 4r = {mass + 4*wilson_r:.6f}")
        logging.info(f"  Lattice dimensions: {lattice_dims}")
        logging.info(f"  Vector polarization: {polarization}")
        logging.info(f"  Linear solver: {solver}")
        
        logging.info(f"\nRho meson physics context:")
        logging.info(f"  • Lightest vector meson, J^PC = 1^(--)")
        logging.info(f"  • Operator: ρᵢ(x) = ψ̄(x)γᵢψ(x)")
        logging.info(f"  • Physical mass: M_ρ ≈ 775 MeV")
        logging.info(f"  • Broad resonance: Γ_ρ ≈ 149 MeV")
        logging.info(f"  • Expected: M_ρ > M_π (vector > pseudoscalar)")
    
    # Step 1: Get rho operator information
    rho_info = get_rho_operator(polarization, verbose=verbose)
    
    # Step 2: Build Wilson-Dirac matrix  
    if verbose:
        logging.info(f"\nStep 1: Building Wilson-Dirac matrix...")
    
    D = build_wilson_dirac_matrix(mass, lattice_dims, wilson_r, U, n_colors, verbose)
    
    if D.nnz == 0:
        logging.error("Wilson-Dirac matrix construction failed!")
        return create_failed_result(mass, lattice_dims, polarization, wilson_r, solver)
    
    # Step 3: Solve for quark propagators
    if verbose:
        logging.info(f"\nStep 2: Solving for quark propagators...")
        logging.info(f"  Vector mesons require same propagators as pseudoscalar")
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
    
    # Step 4: Calculate rho correlator
    if verbose:
        logging.info(f"\nStep 3: Computing rho correlator...")
    
    correlator = calculate_rho_correlator(propagators, lattice_dims, polarization, n_colors, verbose)
    
    # Step 5: Extract effective mass
    if verbose:
        logging.info(f"\nStep 4: Extracting effective mass...")
    
    mass_eff, mass_err = calculate_effective_mass(correlator, verbose)
    
    # Step 6: Fit plateau
    if verbose:
        logging.info(f"\nStep 5: Plateau fitting...")
        logging.info(f"  Note: Vector correlators often require larger t_min than pseudoscalar")
    
    # Use slightly larger t_min for vector mesons (worse signal-to-noise)
    t_min = 3 if len(mass_eff) > 5 else 2
    plateau_mass, plateau_err, chi2, fit_range = fit_plateau(mass_eff, mass_err, t_min=t_min, verbose=verbose)
    
    # Compile complete results
    channel_name = f'rho_{polarization}' if polarization != 'average' else 'rho'
    
    results = {
        'channel': channel_name,
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
        'channel_info': rho_info,
        'physics_notes': {
            'effective_mass': mass + 4*wilson_r,
            'mass_ratio': plateau_mass / (mass + 4*wilson_r) if (mass + 4*wilson_r) > 0 else 0,
            'polarization': polarization,
            'meson_type': 'vector'
        }
    }
    
    if verbose:
        logging.info(f"\nRHO CALCULATION COMPLETE ({polarization.upper()})")
        logging.info(f"=" * 50) 
        logging.info(f"Final Results:")
        logging.info(f"  Rho mass: M_ρ = {plateau_mass:.6f} ± {plateau_err:.6f}")
        logging.info(f"  Mass ratio: M_ρ/m_eff = {results['physics_notes']['mass_ratio']:.3f}")
        logging.info(f"  Fit quality: χ²/dof = {chi2:.3f}")
        logging.info(f"  Polarization: {polarization}")
        
        # Physics interpretation
        mass_ratio = results['physics_notes']['mass_ratio']
        if mass_ratio < 2.0:
            logging.info(f"  → Light vector meson")
        elif mass_ratio < 4.0:
            logging.info(f"  → Intermediate vector meson mass")
        else:
            logging.info(f"  → Heavy vector meson")
            
        if chi2 > 5.0:
            logging.warning(f"  → Large χ² suggests poor signal or excited states")
        elif chi2 < 0.1:
            logging.warning(f"  → Very small χ² may indicate fitting issues")
            
        # Vector-specific diagnostics
        statistical_error = plateau_err / plateau_mass if plateau_mass > 0 else 1.0
        if statistical_error > 0.3:
            logging.warning(f"  → Large statistical error ({statistical_error*100:.0f}%) typical for vector mesons")
            logging.info(f"    Recommendation: Use ensemble averaging for better statistics")
    
    return results

def calculate_all_rho_polarizations(U, mass, lattice_dims, wilson_r=0.5, solver='auto', verbose=False):
    """
    Calculate rho masses for all three spatial polarizations
    
    Computes ρₓ, ρᵧ, ρᵧ masses separately to study rotational symmetry breaking
    on single gauge configurations. Returns analysis of splitting patterns.
    
    Args:
        U (array): Gauge field configuration
        mass (float): Bare quark mass 
        lattice_dims (list): Lattice dimensions
        wilson_r (float): Wilson parameter
        solver (str): Linear solver method
        verbose (bool): Enable detailed output
        
    Returns:
        dict: Results for all polarizations plus symmetry analysis
    """
    if verbose:
        logging.info(f"RHO POLARIZATION ANALYSIS")
        logging.info(f"=" * 40)
        logging.info(f"Computing ρₓ, ρᵧ, ρᵧ masses separately...")
        logging.info(f"Note: Single configs may show large splittings due to")
        logging.info(f"broken rotational symmetry. Ensemble averaging restores degeneracy.")
    
    polarizations = ['x', 'y', 'z']
    results = {}
    
    # Calculate each polarization
    for pol in polarizations:
        if verbose:
            logging.info(f"\n{'-'*30}")
            logging.info(f"POLARIZATION: {pol.upper()}")
            logging.info(f"{'-'*30}")
        
        result = calculate_rho_mass(U, mass, lattice_dims, pol, wilson_r, solver, verbose)
        results[f'rho_{pol}'] = result
    
    # Symmetry analysis
    masses = [results[f'rho_{pol}']['meson_mass'] for pol in polarizations]
    errors = [results[f'rho_{pol}']['meson_error'] for pol in polarizations]
    
    mass_avg = np.mean(masses)
    mass_std = np.std(masses) 
    mass_spread = max(masses) - min(masses)
    
    symmetry_analysis = {
        'average_mass': mass_avg,
        'mass_spread': mass_spread,
        'relative_spread': mass_spread / mass_avg if mass_avg > 0 else 0,
        'individual_masses': {f'rho_{pol}': masses[i] for i, pol in enumerate(polarizations)},
        'symmetry_quality': 'good' if mass_spread / mass_avg < 0.1 else 'broken'
    }
    
    results['symmetry_analysis'] = symmetry_analysis
    
    if verbose:
        logging.info(f"\n{'='*40}")
        logging.info(f"ROTATIONAL SYMMETRY ANALYSIS")
        logging.info(f"{'='*40}")
        logging.info(f"Individual masses:")
        for i, pol in enumerate(polarizations):
            logging.info(f"  M_ρ{pol}: {masses[i]:.6f} ± {errors[i]:.6f}")
        
        logging.info(f"\nSymmetry measures:")
        logging.info(f"  Average mass: {mass_avg:.6f}")
        logging.info(f"  Mass spread: {mass_spread:.6f}")
        logging.info(f"  Relative spread: {mass_spread/mass_avg*100:.1f}%")
        
        if symmetry_analysis['symmetry_quality'] == 'good':
            logging.info(f"  → Good rotational symmetry on this configuration")
        else:
            logging.warning(f"  → Large symmetry breaking (expected on single config)")
            logging.info(f"    Ensemble averaging will restore ρₓ = ρᵧ = ρᵧ degeneracy")
    
    return results

def create_failed_result(mass, lattice_dims, polarization, wilson_r, solver):
    """Create error result structure when calculation fails"""
    return {
        'channel': f'rho_{polarization}' if polarization != 'average' else 'rho',
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
        'channel_info': get_rho_operator(polarization),
        'physics_notes': {
            'status': 'failed',
            'effective_mass': mass + 4*wilson_r,
            'mass_ratio': 0.0,
            'polarization': polarization,
            'meson_type': 'vector'
        }
    }