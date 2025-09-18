"""
Meson Integration Module for Lattice QCD  
=========================================

This module provides a unified interface for all meson calculations,
integrating PionCalculator, RhoCalculator, and SigmaCalculator into
a cohesive system. It offers:

- Unified function calls for any meson type
- Spectrum calculations (π, ρ, σ together)
- Comparison and physics validation tools
- Integration with the original Propagator.py workflow

Usage Examples:
--------------
# Single meson calculation
result = calculate_meson_mass(U, mass=0.1, channel='pion', lattice_dims=[6,6,6,20])

# Full spectrum  
results = calculate_meson_spectrum(U, mass=0.1, lattice_dims=[6,6,6,20])

# Quick calculation from file
result = quick_meson_calculation('config.pkl', channel='pion')

Author: Zeke Mohammed
Advisor: Dr. Aubin
Institution: Fordham University  
Date: September 2025
"""

import numpy as np
import logging
import time
import pickle
from typing import Union, List, Dict, Optional

# Import specialized meson calculators
from PionCalculator import calculate_pion_mass, get_pion_operator
from RhoCalculator import calculate_rho_mass, calculate_all_rho_polarizations, get_rho_operator  
from SigmaCalculator import calculate_sigma_mass, get_sigma_operator, analyze_chiral_multiplet
from MesonBase import build_wilson_dirac_matrix, generate_identity_gauge_field

def calculate_meson_mass(U, mass: float, channel: str, lattice_dims: List[int], 
                        wilson_r: float = 0.5, solver: str = 'auto', 
                        verbose: bool = False) -> Dict:
    """
    Unified interface for calculating any meson mass
    
    Routes calculation to the appropriate specialized module based on channel.
    Provides consistent interface across all meson types.
    
    Args:
        U: Gauge field configuration [None, U_links]
        mass: Bare quark mass in lattice units
        channel: Meson channel ('pion', 'rho', 'sigma', 'rho_x', 'rho_y', 'rho_z')  
        lattice_dims: [Lx, Ly, Lz, Lt] lattice dimensions
        wilson_r: Wilson parameter (default: 0.5)
        solver: Linear solver method ('auto', 'direct', 'gmres', 'lsqr')
        verbose: Enable detailed physics output
        
    Returns:
        dict: Meson calculation results with mass, errors, diagnostics
        
    Supported Channels:
        - 'pion': π meson, J^PC = 0^(-+), Goldstone boson
        - 'sigma': σ meson, J^PC = 0^(++), scalar, chiral partner of π
        - 'rho': ρ meson averaged over polarizations, J^PC = 1^(--)
        - 'rho_x', 'rho_y', 'rho_z': Individual ρ polarizations
    """
    if verbose:
        logging.info(f"UNIFIED MESON CALCULATION: {channel.upper()}")
        logging.info(f"=" * 50)
    
    # Route to appropriate calculator
    if channel.lower() == 'pion':
        return calculate_pion_mass(U, mass, lattice_dims, wilson_r, solver, verbose)
    
    elif channel.lower() == 'sigma':
        return calculate_sigma_mass(U, mass, lattice_dims, wilson_r, solver, verbose)
    
    elif channel.lower() == 'rho':
        return calculate_rho_mass(U, mass, lattice_dims, 'average', wilson_r, solver, verbose)
    
    elif channel.lower() in ['rho_x', 'rho_y', 'rho_z']:
        polarization = channel.split('_')[1]  # Extract x, y, z
        return calculate_rho_mass(U, mass, lattice_dims, polarization, wilson_r, solver, verbose)
    
    else:
        raise ValueError(f"Unknown meson channel: {channel}")

def calculate_meson_spectrum(U, mass: float, lattice_dims: List[int], 
                           wilson_r: float = 0.5, solver: str = 'auto',
                           include_rho_polarizations: bool = True,
                           verbose: bool = False) -> Dict:
    """
    Calculate complete light meson spectrum (π, ρ, σ)
    
    Performs systematic calculation of all light meson masses to study
    the hadron spectrum structure and mass hierarchy.
    
    Args:
        U: Gauge field configuration
        mass: Bare quark mass
        lattice_dims: Lattice dimensions
        wilson_r: Wilson parameter
        solver: Linear solver method
        include_rho_polarizations: Calculate ρₓ, ρᵧ, ρᵧ separately
        verbose: Enable detailed output
        
    Returns:
        dict: Complete spectrum results with physics analysis
        
    Physics Analysis:
        - Mass hierarchy: Should observe M_π < M_ρ < M_σ on average
        - Chiral behavior: M²_π ∝ m_quark (Goldstone nature) 
        - Symmetry breaking: Single configs may show ρₓ ≠ ρᵧ ≠ ρᵧ
        - Quality assessment: Statistical errors, fit quality, physics consistency
    """
    if verbose:
        logging.info(f"FULL MESON SPECTRUM CALCULATION")
        logging.info(f"=" * 60)
        logging.info(f"Computing complete light meson spectrum:")
        logging.info(f"  • Pion (π): J^PC = 0^(-+), pseudoscalar Goldstone boson")
        logging.info(f"  • Rho (ρ): J^PC = 1^(--), vector meson") 
        logging.info(f"  • Sigma (σ): J^PC = 0^(++), scalar, chiral partner of π")
        logging.info(f"")
        logging.info(f"Expected mass hierarchy: M_π < M_ρ < M_σ")
        logging.info(f"Note: Single configurations may show statistical fluctuations")
    
    results = {}
    start_time = time.time()
    
    # Calculate pion mass
    if verbose:
        logging.info(f"\n{'-'*40}")
        logging.info(f"1. PION CALCULATION") 
        logging.info(f"{'-'*40}")
    
    try:
        pion_result = calculate_pion_mass(U, mass, lattice_dims, wilson_r, solver, verbose)
        results['pion'] = pion_result
        if verbose:
            logging.info(f"✓ Pion: M_π = {pion_result['meson_mass']:.6f} ± {pion_result['meson_error']:.6f}")
    except Exception as e:
        logging.error(f"✗ Pion calculation failed: {e}")
        results['pion'] = create_failed_result('pion', mass, lattice_dims, wilson_r, solver)
    
    # Calculate rho masses  
    if verbose:
        logging.info(f"\n{'-'*40}")
        logging.info(f"2. RHO CALCULATION")
        logging.info(f"{'-'*40}")
    
    if include_rho_polarizations:
        try:
            rho_results = calculate_all_rho_polarizations(U, mass, lattice_dims, wilson_r, solver, verbose)
            results.update(rho_results)  # Includes rho_x, rho_y, rho_z, symmetry_analysis
            
            if verbose and 'symmetry_analysis' in rho_results:
                sym_analysis = rho_results['symmetry_analysis'] 
                logging.info(f"✓ Rho polarizations computed:")
                logging.info(f"  Average: {sym_analysis['average_mass']:.6f}")
                logging.info(f"  Spread: {sym_analysis['mass_spread']:.6f} ({sym_analysis['relative_spread']*100:.1f}%)")
        except Exception as e:
            logging.error(f"✗ Rho polarization calculation failed: {e}")
            for pol in ['x', 'y', 'z']:
                results[f'rho_{pol}'] = create_failed_result(f'rho_{pol}', mass, lattice_dims, wilson_r, solver)
    else:
        try:
            rho_result = calculate_rho_mass(U, mass, lattice_dims, 'average', wilson_r, solver, verbose)
            results['rho'] = rho_result
            if verbose:
                logging.info(f"✓ Rho: M_ρ = {rho_result['meson_mass']:.6f} ± {rho_result['meson_error']:.6f}")
        except Exception as e:
            logging.error(f"✗ Rho calculation failed: {e}")
            results['rho'] = create_failed_result('rho', mass, lattice_dims, wilson_r, solver)
    
    # Calculate sigma mass
    if verbose:
        logging.info(f"\n{'-'*40}")
        logging.info(f"3. SIGMA CALCULATION")
        logging.info(f"{'-'*40}")
    
    try:
        sigma_result = calculate_sigma_mass(U, mass, lattice_dims, wilson_r, solver, verbose)
        results['sigma'] = sigma_result
        if verbose:
            logging.info(f"✓ Sigma: M_σ = {sigma_result['meson_mass']:.6f} ± {sigma_result['meson_error']:.6f}")
    except Exception as e:
        logging.error(f"✗ Sigma calculation failed: {e}")
        results['sigma'] = create_failed_result('sigma', mass, lattice_dims, wilson_r, solver)
    
    # Physics analysis of spectrum
    spectrum_time = time.time() - start_time
    
    if verbose:
        logging.info(f"\n{'='*60}")
        logging.info(f"SPECTRUM ANALYSIS")
        logging.info(f"{'='*60}")
        
        # Extract masses for analysis
        masses = {}
        successful_channels = []
        
        for channel in ['pion', 'sigma']:
            if channel in results and results[channel]['meson_mass'] > 0:
                masses[channel] = results[channel]['meson_mass']
                successful_channels.append(channel)
        
        # Handle rho masses (may be polarization-split or averaged)
        if 'rho' in results and results['rho']['meson_mass'] > 0:
            masses['rho'] = results['rho']['meson_mass'] 
            successful_channels.append('rho')
        elif 'symmetry_analysis' in results:
            masses['rho'] = results['symmetry_analysis']['average_mass']
            successful_channels.append('rho')
        
        # Mass hierarchy analysis
        if len(successful_channels) >= 2:
            logging.info(f"Computed masses:")
            for channel in ['pion', 'rho', 'sigma']:
                if channel in masses:
                    error = results[channel]['meson_error'] if channel in results else 0.0
                    logging.info(f"  M_{channel}: {masses[channel]:.6f} ± {error:.6f}")
            
            # Check expected hierarchy: M_π < M_ρ < M_σ
            hierarchy_check = []
            if 'pion' in masses and 'rho' in masses:
                if masses['pion'] < masses['rho']:
                    hierarchy_check.append("M_π < M_ρ ✓")
                else:
                    hierarchy_check.append("M_π > M_ρ ⚠")
            
            if 'rho' in masses and 'sigma' in masses:
                if masses['rho'] < masses['sigma']:
                    hierarchy_check.append("M_ρ < M_σ ✓")
                else:
                    hierarchy_check.append("M_ρ > M_σ ⚠")
            
            if 'pion' in masses and 'sigma' in masses:
                if masses['pion'] < masses['sigma']:
                    hierarchy_check.append("M_π < M_σ ✓")
                else:
                    hierarchy_check.append("M_π > M_σ ⚠")
            
            if hierarchy_check:
                logging.info(f"\nMass hierarchy:")
                for check in hierarchy_check:
                    logging.info(f"  {check}")
                    
                # Overall assessment
                correct_count = sum(1 for check in hierarchy_check if '✓' in check)
                total_count = len(hierarchy_check)
                
                if correct_count == total_count:
                    logging.info(f"  → Excellent: All {total_count}/{total_count} hierarchy relations correct")
                elif correct_count >= total_count // 2:
                    logging.info(f"  → Good: {correct_count}/{total_count} hierarchy relations correct")
                else:
                    logging.warning(f"  → Poor: Only {correct_count}/{total_count} hierarchy relations correct")
                    logging.info(f"    This can occur on single configurations due to statistical fluctuations")
        
        # Chiral multiplet analysis
        if 'pion' in results and 'sigma' in results:
            try:
                chiral_analysis = analyze_chiral_multiplet(results['pion'], results['sigma'], verbose=True)
                results['chiral_analysis'] = chiral_analysis
            except Exception as e:
                logging.error(f"Chiral multiplet analysis failed: {e}")
        
        logging.info(f"\nSpectrum calculation completed in {spectrum_time:.2f} seconds")
        logging.info(f"Successful channels: {len(successful_channels)}")
    
    # Add metadata to results
    results['spectrum_metadata'] = {
        'input_mass': mass,
        'wilson_r': wilson_r,
        'lattice_dims': lattice_dims,
        'solver': solver,
        'calculation_time': spectrum_time,
        'include_polarizations': include_rho_polarizations,
        'successful_channels': successful_channels if verbose else None
    }
    
    return results

def quick_meson_calculation(gauge_config_file: str, channel: str = 'pion',
                          mass: float = 0.1, wilson_r: float = 0.5,
                          lattice_dims: Optional[List[int]] = None,
                          verbose: bool = True) -> Optional[Dict]:
    """
    Simplified interface for single meson calculation from file
    
    Automatically loads gauge configuration and performs meson mass calculation
    with sensible defaults. Ideal for quick analysis and testing.
    
    Args:
        gauge_config_file: Path to pickled gauge configuration file
        channel: Meson channel to calculate ('pion', 'rho', 'sigma')
        mass: Bare quark mass (default: 0.1)
        wilson_r: Wilson parameter (default: 0.5)  
        lattice_dims: Lattice dimensions (auto-detected if None)
        verbose: Enable detailed output
        
    Returns:
        dict: Meson calculation results, or None if loading fails
    """
    if verbose:
        logging.info(f"QUICK MESON CALCULATION: {channel.upper()}")
        logging.info(f"=" * 50)
        logging.info(f"Loading gauge configuration: {gauge_config_file}")
    
    # Load gauge configuration
    try:
        with open(gauge_config_file, 'rb') as f:
            gauge_data = pickle.load(f)
            
        # Handle different gauge configuration formats
        if isinstance(gauge_data, list) and len(gauge_data) >= 2:
            U = [None, gauge_data[1]]  # Standard [plaquette, U] format
            if verbose:
                plaquette = gauge_data[0] if len(gauge_data) > 0 else "unknown"
                logging.info(f"  Format: List [plaquette, U]")
                logging.info(f"  Plaquette: {plaquette}")
        elif isinstance(gauge_data, dict) and 'U' in gauge_data:
            U = [None, gauge_data['U']]  # Dictionary format
            if verbose:
                logging.info(f"  Format: Dictionary with 'U' field")
                if 'plaquette' in gauge_data:
                    logging.info(f"  Plaquette: {gauge_data['plaquette']}")
        else:
            raise ValueError(f"Unknown gauge configuration format: {type(gauge_data)}")
            
    except Exception as e:
        logging.error(f"Failed to load gauge configuration: {e}")
        if verbose:
            logging.info(f"Falling back to identity gauge field for testing...")
        
        # Use identity gauge field as fallback  
        default_dims = lattice_dims if lattice_dims else [4, 4, 4, 16]
        U, metadata = generate_identity_gauge_field(default_dims, verbose=verbose)
    
    # Auto-detect lattice dimensions if not provided
    if lattice_dims is None:
        if U[1] is not None and hasattr(U[1], 'shape'):
            V = U[1].shape[0]
            # Common assumption: cubic spatial volume
            if V == 64:  # 4³ × 4
                lattice_dims = [4, 4, 4, 4]
            elif V == 256:  # 4³ × 16  
                lattice_dims = [4, 4, 4, 16]
            elif V == 1536:  # 6³ × 20
                lattice_dims = [6, 6, 6, 20]
            else:
                # General guess: assume Lt = 4*Ls  
                Ls = int(round((V/16)**(1/3)))
                if Ls**3 * 16 == V:
                    lattice_dims = [Ls, Ls, Ls, 16]
                else:
                    lattice_dims = [4, 4, 4, V//64]
                    
            if verbose:
                logging.info(f"  Auto-detected lattice dimensions: {lattice_dims}")
        else:
            lattice_dims = [4, 4, 4, 16]  # Fallback default
            if verbose:
                logging.warning(f"  Using default lattice dimensions: {lattice_dims}")
    
    # Run meson calculation
    try:
        result = calculate_meson_mass(U, mass, channel, lattice_dims, wilson_r, 'auto', verbose)
        
        if verbose:
            logging.info(f"\nQUICK CALCULATION COMPLETE")
            logging.info(f"=" * 50)
            logging.info(f"Result: M_{channel} = {result['meson_mass']:.6f} ± {result['meson_error']:.6f}")
            
        return result
        
    except Exception as e:
        logging.error(f"Meson calculation failed: {e}")
        return None

def validate_meson_modules(test_lattice_dims: List[int] = [4, 4, 4, 8],
                          test_mass: float = 0.2, wilson_r: float = 0.5,
                          verbose: bool = True) -> Dict[str, bool]:
    """
    Validate all meson calculation modules with identity gauge field
    
    Performs systematic testing of PionCalculator, RhoCalculator, and 
    SigmaCalculator modules using a controlled identity gauge field environment.
    
    Args:
        test_lattice_dims: Small lattice for fast testing
        test_mass: Test quark mass (not too small to avoid chiral issues)
        wilson_r: Wilson parameter
        verbose: Enable detailed test output
        
    Returns:
        dict: Test results for each module {module_name: success_boolean}
    """
    if verbose:
        logging.info(f"MESON MODULE VALIDATION")
        logging.info(f"=" * 50)
        logging.info(f"Testing all meson calculators with identity gauge field")
        logging.info(f"Test parameters:")
        logging.info(f"  Lattice: {test_lattice_dims}")
        logging.info(f"  Mass: {test_mass}")
        logging.info(f"  Wilson r: {wilson_r}")
    
    # Generate identity gauge field for testing
    U, metadata = generate_identity_gauge_field(test_lattice_dims, verbose=False)
    
    test_results = {}
    
    # Test each meson calculator
    test_channels = ['pion', 'rho', 'sigma']
    
    for channel in test_channels:
        if verbose:
            logging.info(f"\n{'-'*30}")
            logging.info(f"Testing {channel.upper()} Calculator")
            logging.info(f"{'-'*30}")
        
        try:
            # Run calculation with minimal verbosity for testing
            result = calculate_meson_mass(U, test_mass, channel, test_lattice_dims, 
                                        wilson_r, 'auto', verbose=False)
            
            # Validate result structure
            required_keys = ['meson_mass', 'meson_error', 'channel', 'correlator']
            has_required = all(key in result for key in required_keys)
            
            # Check for reasonable physics results  
            reasonable_mass = 0.1 < result['meson_mass'] < 10.0
            reasonable_error = 0.0 < result['meson_error'] < result['meson_mass']
            has_correlator = len(result['correlator']) > 0
            
            success = has_required and reasonable_mass and reasonable_error and has_correlator
            test_results[channel] = success
            
            if verbose:
                if success:
                    logging.info(f"  ✓ {channel.capitalize()} calculator: PASSED")
                    logging.info(f"    Mass: {result['meson_mass']:.6f} ± {result['meson_error']:.6f}")
                    logging.info(f"    Correlator points: {len(result['correlator'])}")
                else:
                    logging.error(f"  ✗ {channel.capitalize()} calculator: FAILED")
                    if not has_required:
                        logging.error(f"    Missing required keys")
                    if not reasonable_mass:
                        logging.error(f"    Unreasonable mass: {result.get('meson_mass', 'N/A')}")
                    if not reasonable_error:
                        logging.error(f"    Unreasonable error: {result.get('meson_error', 'N/A')}")
                    if not has_correlator:
                        logging.error(f"    Empty correlator data")
                        
        except Exception as e:
            test_results[channel] = False
            if verbose:
                logging.error(f"  ✗ {channel.capitalize()} calculator: EXCEPTION")
                logging.error(f"    Error: {e}")
    
    # Overall validation summary
    passed_count = sum(test_results.values())
    total_count = len(test_results)
    
    if verbose:
        logging.info(f"\n{'='*50}")
        logging.info(f"VALIDATION SUMMARY")
        logging.info(f"{'='*50}")
        logging.info(f"Tests passed: {passed_count}/{total_count}")
        
        if passed_count == total_count:
            logging.info(f"✓ All meson calculators validated successfully")
            logging.info(f"  Modules are ready for production calculations")
        else:
            logging.warning(f"⚠ {total_count - passed_count} module(s) failed validation")
            logging.info(f"  Check individual calculator implementations")
    
    return test_results

def create_failed_result(channel: str, mass: float, lattice_dims: List[int], 
                        wilson_r: float, solver: str) -> Dict:
    """Create standardized failed result structure"""
    return {
        'channel': channel,
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
        'channel_info': {'name': f'{channel.capitalize()} (failed)', 'JPC': 'N/A'},
        'physics_notes': {
            'status': 'failed',
            'effective_mass': mass + 4*wilson_r,
            'mass_ratio': 0.0
        }
    }

# Module testing when run directly
if __name__ == "__main__":
    """Test the meson integration module"""
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("Meson Integration Module Test")
    print("=" * 40)
    
    # Run validation tests
    validation_results = validate_meson_modules(verbose=True)
    
    # Quick spectrum test  
    print(f"\n" + "=" * 40)
    print("Quick Spectrum Test")
    print("=" * 40)
    
    try:
        U, metadata = generate_identity_gauge_field([4, 4, 4, 8], verbose=False)
        spectrum = calculate_meson_spectrum(U, mass=0.15, lattice_dims=[4, 4, 4, 8], 
                                          include_rho_polarizations=False, verbose=True)
        print("✓ Spectrum calculation completed successfully")
        
    except Exception as e:
        print(f"✗ Spectrum test failed: {e}")
    
    print(f"\n" + "=" * 40)
    print("Integration module testing complete!")
    print("=" * 40)