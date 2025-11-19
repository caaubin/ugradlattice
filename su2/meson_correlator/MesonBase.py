"""
Meson Propagator Base Module for Lattice QCD
=============================================

This module contains the common infrastructure for calculating meson masses
from lattice QCD gauge configurations using Wilson fermions. It provides the
foundation for specific meson calculations including:

- Wilson-Dirac matrix construction with antiperiodic boundary conditions
- Point source creation for quark propagators
- Linear system solvers with automatic scipy version compatibility
- Common utility functions for all meson types

The physics implementation follows standard lattice QCD practices:
1. Load gauge field configurations U_μ(x) from thermal generation
2. Build Wilson-Dirac operator D_W with proper boundary conditions
3. Solve D·S = δ for quark propagators using optimized sparse solvers
4. Provide foundation for meson correlator construction

This base module enables efficient, modular calculations across all meson channels
while maintaining consistency in the underlying physics and numerics.

Author: Zeke Mohammed  
Advisor: Dr. Aubin
Institution: Fordham University
Date: September 2025
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import logging
import time
import su2
import sun  # SU(N) generalization

def build_wilson_dirac_matrix(mass, lattice_dims, wilson_r=1.0, U=None, n_colors=2, verbose=False):
    """
    Build Wilson-Dirac matrix with antiperiodic boundary conditions in time
    
    Constructs the discretized Dirac operator:
    D_W = m + 4r + (1/2) Σ_μ [(1-γ_μ)U_μ(x)δ_{x+μ,y} + (1+γ_μ)U†_μ(x-μ)δ_{x-μ,y}]
    
    The Wilson term prevents fermion doubling by adding -r∇² to the action.
    Antiperiodic boundary conditions ψ(Lt) = -ψ(0) are implemented in time direction.
    
    Args:
        mass (float): Bare quark mass in lattice units
        lattice_dims (list): [Lx, Ly, Lz, Lt] lattice dimensions
        wilson_r (float): Wilson parameter (default: 1.0, prevents doubling)
        U (array): Gauge field configuration U[site][direction][component]
        n_colors (int): Number of colors for SU(N) (default: 2 for SU(2))
        verbose (bool): Enable detailed physics output

    Returns:
        scipy.sparse.csc_matrix: Wilson-Dirac operator D_W

    Physics Notes:
        - Effective mass: m_eff = m + 4r (additive mass renormalization)
        - Antiperiodic BC in time implements fermion doubling for finite T
        - Gauge covariance: D transforms as D → V†DV under gauge transformations
        - Matrix size: (N_c×4)V × (N_c×4)V where V = Lx×Ly×Lz×Lt
        - For N_c=2 (SU(2)): 8V × 8V; For N_c=3 (SU(3)): 12V × 12V
    """
    Lx, Ly, Lz, Lt = lattice_dims
    V = Lx * Ly * Lz * Lt
    matrix_size = (n_colors * 4) * V  # N_c colors × 4 spins × V sites
    
    # Effective mass includes Wilson shift (additive mass renormalization)
    mass_effective = mass + 4.0 * wilson_r
    
    if verbose:
        logging.info(f"Building Wilson-Dirac matrix:")
        logging.info(f"  Gauge group: SU({n_colors})")
        logging.info(f"  Lattice: {Lx}×{Ly}×{Lz}×{Lt} (V={V})")
        logging.info(f"  Matrix size: {matrix_size}×{matrix_size} ({n_colors} colors × 4 spins)")
        logging.info(f"  Bare mass m={mass:.6f}, Wilson r={wilson_r:.2f}")
        logging.info(f"  Effective mass: m_eff = m + 4r = {mass_effective:.6f}")
        logging.info(f"  Boundary conditions: antiperiodic in time direction")
    
    # Use identity gauge field if none provided
    if U is None:
        if verbose:
            logging.warning("No gauge field provided, using identity (free field)")
        U = generate_identity_gauge_field(lattice_dims, n_colors=n_colors, verbose=False)[0]
    
    # Dirac gamma matrices in standard representation
    # γ_μ satisfy Clifford algebra: {γ_μ, γ_ν} = 2g_μν
    gamma = [
        np.array([[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]]),  # γ_0
        np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]),       # γ_1
        np.array([[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]]),  # γ_2
        np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])         # γ_3
    ]
    
    # Precompute nearest neighbors for efficiency
    mups = np.zeros((V, 4), dtype=int)
    mdns = np.zeros((V, 4), dtype=int)
    for i in range(V):
        for mu in range(4):
            mups[i, mu] = su2.mupi(i, mu, lattice_dims)
            mdns[i, mu] = su2.mdowni(i, mu, lattice_dims)
    
    # Build sparse matrix using COO format initially
    rows, cols, data = [], [], []
    
    # Mass term: (m + 4r) × Identity
    for idx in range(matrix_size):
        rows.append(idx)
        cols.append(idx)
        data.append(mass_effective)
    
    # Hopping terms in all four directions
    construction_errors = 0
    dof_per_site = n_colors * 4  # Degrees of freedom per site

    for base_idx in range(V):
        base = dof_per_site * base_idx
        point = su2.i2p(base_idx, lattice_dims)

        for mu in range(4):
            try:
                next_idx = mups[base_idx, mu]
                prev_idx = mdns[base_idx, mu]
                next_base = dof_per_site * next_idx
                prev_base = dof_per_site * prev_idx
                
                next_point = su2.i2p(next_idx, lattice_dims)
                prev_point = su2.i2p(prev_idx, lattice_dims)
                
                # Implement antiperiodic boundary conditions in time direction
                forward_sign = 1.0
                backward_sign = 1.0
                
                if mu == 3:  # Time direction (antiperiodic fermions)
                    if point[3] == Lt-1 and next_point[3] == 0:
                        forward_sign = -1.0  # ψ(Lt) = -ψ(0)
                    if point[3] == 0 and prev_point[3] == Lt-1:
                        backward_sign = -1.0  # ψ(-1) = -ψ(Lt-1)
                
                # Get gauge links (handle both SU(2) real and SU(N) complex representations)
                try:
                    if n_colors == 2 and isinstance(U[1][base_idx, mu], np.ndarray) and U[1][base_idx, mu].size == 4:
                        # SU(2) in real representation: convert to complex 2×2 matrix
                        U_real = U[1][base_idx, mu]
                        U_complex = np.array([
                            [U_real[0] + 1j*U_real[3], U_real[2] + 1j*U_real[1]],
                            [-U_real[2] + 1j*U_real[1], U_real[0] - 1j*U_real[3]]
                        ], dtype=complex)

                        U_prev_real = U[1][prev_idx, mu]
                        U_prev_complex = np.array([
                            [U_prev_real[0] + 1j*U_prev_real[3], U_prev_real[2] + 1j*U_prev_real[1]],
                            [-U_prev_real[2] + 1j*U_prev_real[1], U_prev_real[0] - 1j*U_prev_real[3]]
                        ], dtype=complex)
                        U_prev_dag = U_prev_complex.conj().T
                    else:
                        # SU(N) in complex matrix representation (N×N complex matrices)
                        U_complex = U[1][base_idx, mu]
                        U_prev_complex = U[1][prev_idx, mu]
                        U_prev_dag = U_prev_complex.conj().T

                        # Verify matrix dimensions
                        if U_complex.shape != (n_colors, n_colors):
                            raise ValueError(f"Gauge link has wrong shape: {U_complex.shape}, expected ({n_colors},{n_colors})")

                except (IndexError, TypeError, ValueError) as e:
                    # Fallback to identity links if gauge field access fails
                    U_complex = np.eye(n_colors, dtype=complex)
                    U_prev_dag = np.eye(n_colors, dtype=complex)
                    construction_errors += 1
                    if verbose and construction_errors == 1:
                        logging.warning(f"    Gauge field error: {e}, using identity links")
                
                # Forward hopping term: -(1/2)[(1-γ_μ)U_μ(x)] 
                K_forward = 0.5 * forward_sign * np.kron(gamma[mu], U_complex)
                W_forward = -0.5 * wilson_r * forward_sign * np.kron(np.eye(4), U_complex)
                M_forward = K_forward + W_forward
                
                # Backward hopping term: -(1/2)[(1+γ_μ)U†_μ(x-μ)]
                K_backward = -0.5 * backward_sign * np.kron(gamma[mu], U_prev_dag)
                W_backward = -0.5 * wilson_r * backward_sign * np.kron(np.eye(4), U_prev_dag)
                M_backward = K_backward + W_backward
                
                # Add non-zero matrix elements to sparse structure
                for i in range(dof_per_site):
                    for j in range(dof_per_site):
                        if abs(M_forward[i,j]) > 1e-14:
                            rows.append(base + i)
                            cols.append(next_base + j)
                            data.append(M_forward[i,j])

                        if abs(M_backward[i,j]) > 1e-14:
                            rows.append(base + i)
                            cols.append(prev_base + j)
                            data.append(M_backward[i,j])
                            
            except Exception as e:
                if verbose and construction_errors < 10:
                    logging.warning(f"    Construction error at site {base_idx}, direction {mu}: {e}")
                construction_errors += 1
                continue
    
    if construction_errors > 0 and verbose:
        logging.warning(f"  Matrix construction completed with {construction_errors} errors")
        if construction_errors > V//10:
            logging.error(f"  Large number of errors may indicate gauge field problems")
    
    # Convert to CSC format for efficient linear algebra
    D = sparse.csc_matrix((data, (rows, cols)), shape=(matrix_size, matrix_size))
    
    if verbose:
        sparsity_percent = (D.nnz / (matrix_size**2)) * 100
        logging.info(f"  Matrix constructed: nnz={D.nnz}, sparsity={sparsity_percent:.3f}%")
        
        # Check matrix properties
        if D.nnz == 0:
            logging.error("  ERROR: Matrix has no non-zero elements!")
        elif sparsity_percent > 10:
            logging.warning(f"  Matrix is unusually dense ({sparsity_percent:.1f}%)")
    
    return D

def generate_identity_gauge_field(lattice_dims=None, n_colors=2, verbose=False):
    """
    Generate identity gauge field for free field calculations

    Creates U_μ(x) = I for all links, corresponding to zero field strength.
    Used as fallback when no gauge configuration is provided, or for
    free field theory comparisons.

    Args:
        lattice_dims (list): [Lx, Ly, Lz, Lt] dimensions
        n_colors (int): Number of colors for SU(N) (default: 2 for SU(2))
        verbose (bool): Enable logging output

    Returns:
        tuple: ([None, U], metadata) in standard gauge field format
    """
    if lattice_dims is None:
        lattice_dims = [4, 4, 4, 16]

    if verbose:
        logging.info(f"Generating identity gauge field for SU({n_colors}) (free field theory)")

    Lx, Ly, Lz, Lt = lattice_dims
    V = Lx * Ly * Lz * Lt

    # Initialize all links to SU(N) identity
    if n_colors == 2:
        # SU(2) in real representation [1, 0, 0, 0]
        U = np.zeros((V, 4, 4))
        for i in range(V):
            for mu in range(4):
                U[i, mu] = su2.cstart()
    else:
        # SU(N) in complex matrix representation (N×N identity matrices)
        U = np.zeros((V, 4, n_colors, n_colors), dtype=complex)
        for i in range(V):
            for mu in range(4):
                U[i, mu] = sun.identity_SU_N(n_colors)

    metadata = {
        'plaquette': 1.0,
        'format': 'identity',
        'lattice_dims': lattice_dims,
        'n_colors': n_colors,
        'description': f'Free field theory (no gauge field, SU({n_colors}))'
    }

    return [None, U], metadata

def create_point_source(lattice_dims, t_source, color, spin, n_colors=2, verbose=False):
    """
    Create point source vector for quark propagator calculation

    Generates a delta function source δ³(x⃗-x⃗₀)δ(t-t₀) that creates a quark
    at specific spacetime location with definite color and spin quantum numbers.

    The propagator S(x,y) describes quark propagation from source to all other points.
    Multiple sources (all colors and spins) are needed for complete meson correlators.

    Source placement strategy:
    - Spatial origin: x⃗₀ = (0,0,0) for maximal signal extraction
    - Temporal: t₀ = t_source (typically 0 for maximum time evolution)
    - Total sources needed: N_c colors × 4 Dirac spins

    Args:
        lattice_dims (list): [Lx, Ly, Lz, Lt] lattice dimensions
        t_source (int): Source time slice (0 ≤ t_source < Lt)
        color (int): SU(N) color index (0 to N_c-1)
        spin (int): Dirac spinor index (0, 1, 2, 3)
        n_colors (int): Number of colors for SU(N) (default: 2 for SU(2))
        verbose (bool): Enable debug output

    Returns:
        numpy.ndarray: Source vector of length (N_c×4)V with single non-zero entry

    Index Conventions:
        Global index = site_index × (N_c×4) + 4×color + spin
        This FIXED indexing works for any N_c and matches propagator structure
    """
    Lx, Ly, Lz, Lt = lattice_dims
    V = Lx * Ly * Lz * Lt
    dof_per_site = n_colors * 4
    source = np.zeros(dof_per_site * V, dtype=complex)

    # Source location in spacetime
    source_point = np.array([0, 0, 0, t_source])
    site_idx = su2.p2i(source_point, lattice_dims)

    # Global index for this color-spin combination (FIXED indexing!)
    base_idx = dof_per_site * site_idx
    global_idx = base_idx + 4*color + spin
    
    # Check bounds
    if global_idx >= len(source):
        raise IndexError(f"Source index {global_idx} exceeds vector length {len(source)}")
    
    source[global_idx] = 1.0
    
    if verbose:
        logging.info(f"    Point source created:")
        logging.info(f"      Spacetime: {source_point}")
        logging.info(f"      Color: {color}, Spin: {spin}")
        logging.info(f"      Site index: {site_idx}, Global index: {global_idx}")
    
    return source

def solve_dirac_system(D, source, method='auto', verbose=False):
    """
    Solve Wilson-Dirac linear system D × S = δ for quark propagator
    
    This is the computational core of lattice QCD - solving the discretized
    Dirac equation to obtain the quark propagator S(x,y) = ⟨ψ(x)ψ̄(y)⟩.
    
    The propagator encodes confinement physics: quarks cannot propagate freely
    but are connected by flux tubes. The Wilson-Dirac matrix D is:
    - Large: O(10⁴-10⁶) × O(10⁴-10⁶) for typical lattices  
    - Sparse: ~0.1% non-zero (nearest-neighbor coupling only)
    - Well-conditioned: Wilson term prevents near-zero eigenvalues
    
    Solver Selection:
    - Direct (LU): Exact, fast for small systems (< 5000 unknowns)
    - GMRES: Krylov subspace, good for non-symmetric systems
    - LSQR: Robust for ill-conditioned matrices
    - Auto: Chooses based on system size
    
    Scipy Compatibility:
    Different scipy versions use different parameter names for GMRES tolerance:
    'tol', 'atol', or 'rtol'. This function tries all variants automatically.
    
    Args:
        D (scipy.sparse matrix): Wilson-Dirac operator
        source (numpy.ndarray): Point source vector δ
        method (str): Solver choice ('auto', 'direct', 'gmres', 'lsqr')
        verbose (bool): Enable convergence diagnostics
        
    Returns:
        numpy.ndarray: Quark propagator solution S
        
    Physics Notes:
        The propagator solution determines all hadron properties:
        - Meson masses from C(t) = Tr[Γ S(0,t)]
        - Decay constants from matrix elements
        - Form factors from three-point functions
    """
    if verbose:
        logging.info(f"  Solving Wilson-Dirac system (method={method})...")
        logging.info(f"    Matrix size: {D.shape[0]}×{D.shape[1]}")
        logging.info(f"    Non-zeros: {D.nnz}")
        logging.info(f"    Sparsity: {(D.nnz/(D.shape[0]**2))*100:.3f}%")
    
    # Add small regularization to improve conditioning
    regularization = 1e-8
    D_reg = D + regularization * sparse.eye(D.shape[0])
    
    # Automatic method selection
    if method == 'auto':
        if D.shape[0] <= 5000:
            method = 'direct'
            if verbose:
                logging.info(f"    Auto-selected: direct solver (small system)")
        else:
            method = 'gmres'
            if verbose:
                logging.info(f"    Auto-selected: GMRES iterative solver")
    
    start_time = time.time()
    
    try:
        if method == 'direct':
            if verbose:
                logging.info(f"    Using direct sparse LU factorization...")
            solution = spla.spsolve(D_reg, source)
            convergence_info = "direct_solve"
            
        elif method == 'gmres':
            if verbose:
                logging.info(f"    Using GMRES iterative solver...")
            
            # Handle scipy version differences for GMRES parameters
            solution = None
            convergence_info = -1
            
            # Try different tolerance parameter names across scipy versions
            tolerance_params = ['atol', 'rtol', 'tol']
            
            for param_name in tolerance_params:
                try:
                    solver_kwargs = {
                        'maxiter': 2000,
                        param_name: 1e-8
                    }
                    
                    solution, convergence_info = spla.gmres(D_reg, source, **solver_kwargs)
                    
                    if verbose:
                        logging.info(f"    GMRES succeeded with parameter '{param_name}'")
                    break
                    
                except TypeError:
                    if verbose and param_name == tolerance_params[-1]:
                        logging.warning(f"    GMRES failed with all tolerance parameter variants")
                    continue
            
            # Fallback: try GMRES without explicit tolerance
            if solution is None:
                try:
                    if verbose:
                        logging.info(f"    Trying GMRES without explicit tolerance...")
                    solution, convergence_info = spla.gmres(D_reg, source, maxiter=2000)
                except Exception as e:
                    if verbose:
                        logging.error(f"    GMRES completely failed: {e}")
                    solution = None
                    convergence_info = -1
            
            # Final fallback to LSQR if GMRES fails
            if convergence_info != 0 or solution is None:
                if verbose:
                    logging.warning(f"    GMRES failed (info={convergence_info}), trying LSQR...")
                method = 'lsqr'  # Fall through to LSQR
            
        if method == 'lsqr':
            if verbose:
                logging.info(f"    Using LSQR least-squares solver...")
            
            # LSQR also has scipy version differences
            try:
                solution = spla.lsqr(D_reg, source, iter_lim=2000)[0]
            except TypeError:
                try:
                    solution = spla.lsqr(D_reg, source, maxiter=2000)[0]
                except TypeError:
                    solution = spla.lsqr(D_reg, source)[0]
            
            convergence_info = "lsqr_converged"
            
        solve_time = time.time() - start_time
        
        # Ensure we have a valid solution
        if solution is None:
            if verbose:
                logging.error(f"    All solvers failed, returning zero vector")
            solution = np.zeros(len(source), dtype=complex)
            convergence_info = "failed"
        
        # Check solution quality
        if verbose and solution is not None:
            residual_norm = np.linalg.norm(D_reg @ solution - source)
            solution_norm = np.linalg.norm(solution)
            
            logging.info(f"    Solution completed in {solve_time:.3f}s")
            logging.info(f"    Residual norm: {residual_norm:.2e}")
            logging.info(f"    Solution norm: {solution_norm:.2e}")
            
            if residual_norm > 1e-6:
                logging.warning(f"    Large residual may indicate poor convergence")
            
            if solution_norm == 0:
                logging.error(f"    Zero solution - solver failed!")
        
        return solution
        
    except Exception as e:
        if verbose:
            logging.error(f"    Solver exception: {e}")
        return np.zeros(len(source), dtype=complex)

def calculate_effective_mass(correlator, verbose=False):
    """
    Calculate effective mass M_eff(t) = ln[C(t)/C(t+1)] from correlator
    
    The effective mass provides a model-independent way to extract ground
    state masses from exponential decay. For large times:
    
    C(t) ~ A₀ exp(-M₀t) + A₁ exp(-M₁t) + ...
    
    where M₀ < M₁ < ... are energy eigenvalues. The effective mass:
    M_eff(t) = ln[C(t)/C(t+1)] → M₀ as t → ∞
    
    A good plateau in M_eff(t) indicates clean ground state extraction.
    
    Args:
        correlator (array): Correlator data C(t)
        verbose (bool): Enable diagnostic output
        
    Returns:
        tuple: (mass_eff, mass_err) - effective masses and error estimates
        
    Analysis Notes:
        - Plateau region indicates ground state dominance
        - Oscillations suggest excited state contamination  
        - Rising trend indicates poor signal-to-noise
        - Systematic errors require ensemble averaging
    """
    if len(correlator) < 2:
        return np.array([]), np.array([])

    # Clip very small values to avoid log(0)
    # Note: Removed np.abs() to preserve sign - correlators should be positive after indexing fix
    correlator = np.maximum(correlator, 1e-15)
    
    if verbose:
        logging.info(f"  Computing effective mass from {len(correlator)} time slices")
        logging.info(f"    Correlator range: [{np.min(correlator):.2e}, {np.max(correlator):.2e}]")
    
    # Compute ratios C(t)/C(t+1)
    ratios = correlator[:-1] / correlator[1:]
    
    # Clip ratios to reasonable range to avoid log problems
    ratios = np.maximum(ratios, 1e-10)
    ratios = np.minimum(ratios, 1e10)
    
    # Effective mass from ratio method
    mass_eff = np.log(ratios)
    
    # Simple error estimate based on statistical fluctuations
    # In practice, this would come from ensemble averaging
    mass_err = 0.1 * np.abs(mass_eff)
    
    # Quality check: mark unphysical values as invalid
    valid_mask = (mass_eff > 0) & (mass_eff < 10) & np.isfinite(mass_eff)
    mass_eff = np.where(valid_mask, mass_eff, np.nan)
    mass_err = np.where(valid_mask, mass_err, np.nan)
    
    if verbose:
        n_valid = np.sum(~np.isnan(mass_eff))
        logging.info(f"    Valid effective mass points: {n_valid}/{len(mass_eff)}")
        
        if n_valid > 0:
            valid_masses = mass_eff[~np.isnan(mass_eff)]
            logging.info(f"    Effective mass range: [{np.min(valid_masses):.4f}, {np.max(valid_masses):.4f}]")
    
    return mass_eff, mass_err

def fit_plateau(mass_eff, mass_err, t_min=2, t_max=None, verbose=False):
    """
    Extract ground state mass from effective mass plateau
    
    Fits a constant to the plateau region of M_eff(t) to extract the
    ground state mass M₀. The plateau indicates that excited states
    have decayed sufficiently: C(t) ~ A₀ exp(-M₀t).
    
    Fit Strategy:
    - Exclude early times (t < t_min) with excited state contamination
    - Exclude late times (t > t_max) with poor signal-to-noise  
    - Weight by inverse error squared for optimal statistics
    - Compute χ²/dof to assess fit quality
    
    Args:
        mass_eff (array): Effective mass values M_eff(t)
        mass_err (array): Statistical errors on M_eff(t)
        t_min (int): Minimum time for plateau fit (default: 2)
        t_max (int): Maximum time for plateau fit (default: auto)
        verbose (bool): Enable fit diagnostics
        
    Returns:
        tuple: (plateau_mass, plateau_error, chi_squared, fit_range)
        
    Quality Indicators:
        - χ²/dof ≈ 1: Good fit quality
        - χ²/dof >> 1: Poor plateau or systematic errors
        - χ²/dof << 1: Overestimated errors or over-fitting
    """
    if len(mass_eff) == 0:
        return 0.5, 0.1, 0.0, (0, 0)
    
    # Find valid (non-NaN) points
    valid_mask = ~np.isnan(mass_eff) & ~np.isnan(mass_err) & (mass_err > 0)
    if not np.any(valid_mask):
        if verbose:
            logging.warning("    No valid points for plateau fitting")
        return 0.5, 0.1, 0.0, (0, 0)
    
    valid_mass = mass_eff[valid_mask]
    valid_err = mass_err[valid_mask] 
    valid_indices = np.where(valid_mask)[0]
    
    # Auto-select fit range if not specified
    if t_max is None:
        t_max = min(len(mass_eff) - 1, max(valid_indices))
    
    t_min = max(0, min(t_min, len(valid_indices) // 2))
    t_max = max(t_min + 1, t_max)
    
    if verbose:
        logging.info(f"  Plateau fitting:")
        logging.info(f"    Available time range: {min(valid_indices)}-{max(valid_indices)}")
        logging.info(f"    Requested fit range: {t_min}-{t_max}")
    
    # Select points in fit range
    fit_mask = (valid_indices >= t_min) & (valid_indices <= t_max)
    
    if not np.any(fit_mask):
        # Fallback: use middle third of available data
        mid_start = len(valid_indices) // 3
        mid_end = 2 * len(valid_indices) // 3
        if mid_end <= mid_start:
            mid_end = min(mid_start + 1, len(valid_indices))
            
        fit_mass = valid_mass[mid_start:mid_end]
        fit_err = valid_err[mid_start:mid_end]
        fit_range = (valid_indices[mid_start], valid_indices[mid_end-1])
        
        if verbose:
            logging.warning(f"    Using fallback range: {fit_range}")
    else:
        fit_mass = valid_mass[fit_mask]
        fit_err = valid_err[fit_mask]
        fit_indices = valid_indices[fit_mask]
        fit_range = (min(fit_indices), max(fit_indices))
    
    # Weighted plateau fit
    if len(fit_mass) > 0:
        weights = 1.0 / (fit_err**2 + 1e-12)  # Avoid division by zero
        
        # Weighted average
        plateau_mass = np.sum(fit_mass * weights) / np.sum(weights)
        plateau_err = 1.0 / np.sqrt(np.sum(weights))
        
        # Chi-squared calculation  
        if len(fit_mass) > 1:
            residuals = fit_mass - plateau_mass
            chi_squared_total = np.sum((residuals / fit_err)**2)
            degrees_of_freedom = len(fit_mass) - 1
            chi_squared = chi_squared_total / degrees_of_freedom
        else:
            chi_squared = 0.0
            
    else:
        plateau_mass, plateau_err, chi_squared = 0.5, 0.1, 0.0
        
    if verbose:
        logging.info(f"    Plateau results:")
        logging.info(f"      Mass: {plateau_mass:.6f} ± {plateau_err:.6f}")  
        logging.info(f"      Fit range: t = {fit_range[0]} to {fit_range[1]}")
        logging.info(f"      χ²/dof: {chi_squared:.3f}")
        
        # Quality assessment
        if chi_squared > 5.0:
            logging.warning(f"      Large χ²/dof suggests poor plateau or excited state contamination")
        elif chi_squared < 0.1 and len(fit_mass) > 3:
            logging.warning(f"      Very small χ²/dof may indicate overestimated errors")
    
    return plateau_mass, plateau_err, chi_squared, fit_range

# Common gamma matrix definitions for all meson channels
def get_gamma_matrices():
    """
    Return standard Dirac gamma matrices in the conventional representation
    
    These 4×4 matrices satisfy the Clifford algebra {γ_μ, γ_ν} = 2g_μν
    and are used to construct meson operators with definite quantum numbers.
    
    Returns:
        dict: Dictionary of gamma matrices and γ₅
        
    Quantum Numbers:
        The meson operator M_Γ(x) = ψ̄(x)Γψ(x) has J^PC determined by Γ:
        - γ₅: J^PC = 0^(-+) (pseudoscalar, pion)  
        - I:  J^PC = 0^(++) (scalar, sigma)
        - γᵢ: J^PC = 1^(--) (vector, rho)
    """
    gamma_matrices = {
        'gamma0': np.array([[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]], dtype=complex),
        'gamma1': np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]], dtype=complex),
        'gamma2': np.array([[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]], dtype=complex),
        'gamma3': np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=complex),
        'identity': np.eye(4, dtype=complex)
    }
    
    # γ₅ = diag(-1, -1, 1, 1) in our representation
    gamma5 = np.diag([-1, -1, 1, 1]).astype(complex)
    gamma_matrices['gamma5'] = gamma5
    
    return gamma_matrices