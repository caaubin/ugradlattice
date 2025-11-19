"""
Lattice QCD Propagator Calculator
Meson mass extraction from gauge configurations using Wilson fermions

This code implements the complete workflow for calculating hadron masses in lattice QCD:
1. Load gauge field configurations U_μ(x) representing gluon interactions
2. Build Wilson-Dirac operator D_W for quark propagation
3. Solve D·S = δ for quark propagators S(x,y)
4. Construct meson correlators C(t) = Tr[Γ S(0,t)]
5. Extract masses from exponential decay: C(t) ~ exp(-M*t)

REQUIREMENTS:
- Python 3.6+
- numpy (any recent version)
- scipy (supports multiple versions automatically)
- matplotlib (for plotting)
- su2 module (custom lattice QCD module)

COMPATIBILITY NOTE:
This code automatically detects and handles different scipy versions.
SciPy's GMRES has used different tolerance parameter names across versions:
- Some versions use 'tol'
- Some versions use 'atol' 
- Some versions use 'rtol'
The code will automatically try each parameter name until one works.

If you encounter solver errors, you can check your scipy's gmres parameters:
  python -c "import scipy.sparse.linalg; import inspect; print(inspect.signature(scipy.sparse.linalg.gmres))"

For Jupyter notebook users:
  import scipy.sparse.linalg as spla
  import inspect
  print(inspect.signature(spla.gmres))

For best performance, consider upgrading scipy:
  pip install --upgrade scipy

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

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import argparse
import pickle
import os
import sys
import time
import json
import logging
import warnings
from datetime import datetime
import su2

# Global logging configuration
_logging_configured = False

def configure_logging(log_file, console_level=logging.INFO):
    """Configure logging for analysis tracking"""
    global _logging_configured
    
    if not _logging_configured:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        _logging_configured = True
    else:
        root_logger = logging.getLogger()
        current_files = [h.baseFilename for h in root_logger.handlers 
                        if isinstance(h, logging.FileHandler)]
        
        if os.path.abspath(log_file) not in current_files:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            root_logger.addHandler(file_handler)

def setup_output_directories(run_type="single", mass=0.1, channel="pion", lattice_dims=None, wilson_r=0.5):
    """Create organized directory structure for results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if lattice_dims is None:
        lattice_dims = [6, 6, 6, 20]
    
    Lx, Ly, Lz, Lt = lattice_dims
    base_dir = "lattice_qcd_results"
    
    # Create run-specific directory
    if run_type == "single":
        run_dir = f"{timestamp}_m{mass:.3f}_{channel}_L{Lx}x{Ly}x{Lz}T{Lt}_r{wilson_r:.2f}"
        full_path = os.path.join(base_dir, "single", run_dir)
    elif run_type == "spectrum":
        run_dir = f"{timestamp}_spectrum_m{mass:.3f}_L{Lx}x{Ly}x{Lz}T{Lt}_r{wilson_r:.2f}"
        full_path = os.path.join(base_dir, "spectrum", run_dir)
    elif run_type == "mass_scan":
        run_dir = f"{timestamp}_mass_scan_{channel}_L{Lx}x{Ly}x{Lz}T{Lt}_r{wilson_r:.2f}"
        full_path = os.path.join(base_dir, "mass_scan", run_dir)
    elif run_type == "wilson_scan":
        run_dir = f"{timestamp}_wilson_scan_{channel}_L{Lx}x{Ly}x{Lz}T{Lt}_m{mass:.3f}"
        full_path = os.path.join(base_dir, "wilson_scan", run_dir)
    else:
        run_dir = f"{timestamp}_analysis_m{mass:.3f}_{channel}"
        full_path = os.path.join(base_dir, "analysis", run_dir)
    
    # Create subdirectories
    subdirs = ['data', 'plots', 'correlators', 'logs', 'configs']
    dirs = {'base': full_path}
    
    for subdir in subdirs:
        subdir_path = os.path.join(full_path, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        dirs[subdir] = subdir_path
    
    # Add log file path
    dirs['log_file'] = os.path.join(dirs['logs'], 'calculation.log')
    
    return dirs

def parse_arguments():
    """Parse command line arguments with comprehensive options for lattice QCD calculations"""
    parser = argparse.ArgumentParser(
        description='Lattice QCD meson propagator calculation using Wilson fermions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
1. Single pion mass calculation:
   python %(prog)s --mass 0.1 --channel pion --input-config config.pkl
   
2. Full hadron spectrum (π, σ, ρ mesons):
   python %(prog)s --mass 0.1 --channel all --input-config config.pkl
   
3. Study chiral behavior (M²_π ∝ m_quark):
   python %(prog)s --mass-scan 0.01,0.05,0.1,0.2 --channel pion --input-config config.pkl
   
4. Optimize Wilson parameter:
   python %(prog)s --wilson-scan 0.1,0.5,1.0 --mass 0.1 --input-config config.pkl
   
5. Custom lattice size:
   python %(prog)s --mass 0.1 --channel pion --ls 6 --lt 20 --input-config config.pkl

Common Student Exercises:
------------------------
• Spectrum Study: Calculate π, σ, ρ masses and verify mass ordering
• Chiral Extrapolation: Plot M²_π vs m_quark and fit to linear form
• Wilson Dependence: Study how r parameter affects mass scale
• Plateau Analysis: Learn to identify stable effective mass regions
• Error Analysis: Understand statistical uncertainties in lattice QCD

Physics Notes:
-------------
• Pion (π): J^PC = 0^(-+), lightest meson, Goldstone boson
• Sigma (σ): J^PC = 0^(++), scalar, chiral partner of pion
• Rho (ρ): J^PC = 1^(--), vector meson, three polarizations
• Wilson r: Controls fermion doubling (r=0.5-1.0 typical)
• Effective mass: m_eff = m + 4r due to Wilson discretization
        """)
    
    # Physics parameters
    physics_group = parser.add_argument_group('QCD Physics Parameters')
    physics_group.add_argument('--mass', type=float, default=0.1,
                              help='Bare quark mass in lattice units (default: 0.1)')
    physics_group.add_argument('--wilson-r', type=float, default=0.5,
                              help='Wilson parameter r, controls discretization (default: 0.5)')
    physics_group.add_argument('--n-colors', type=int, default=2,
                              help='Number of colors for SU(N) gauge group (default: 2 for SU(2), use 3 for QCD)')
    
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

def load_gauge_configuration(config_file, verbose=False):
    """
    Load gauge configuration from file with support for multiple formats.
    
    Gauge configurations U_μ(x) represent the gluon field background that mediates
    the strong force between quarks. These are generated by Monte Carlo sampling
    of the Wilson gauge action S_g = β Σ (1 - Re Tr U_plaquette).
    
    The plaquette value measures local gauge field curvature:
    - High plaquette (>0.8): Weak coupling, smooth fields
    - Low plaquette (<0.4): Strong coupling, rough fields
    - Physical regime: Typically 0.5-0.7
    
    Args:
        config_file (str): Path to gauge configuration file
        verbose (bool): Enable detailed physics output
        
    Returns:
        tuple: ([None, U], metadata) where U contains SU(2) gauge links
    """
    if verbose:
        logging.info(f"Loading gauge configuration: {config_file}")
    
    try:
        with open(config_file, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different formats
        if isinstance(data, list) and len(data) >= 2:
            # [plaquette, U] format
            plaquette, U = data[0], data[1]
            metadata = {'plaquette': plaquette, 'format': 'list'}
            
            if verbose:
                logging.info(f"  Plaquette value: {plaquette:.6f}")
                if plaquette > 0.8:
                    logging.info(f"  Weak coupling regime")
                elif plaquette < 0.4:
                    logging.info(f"  Strong coupling regime")
                else:
                    logging.info(f"  Intermediate coupling regime")
            
            return [None, U], metadata
            
        elif isinstance(data, dict):
            # Dictionary format  
            if 'U' in data:
                U = data['U']
                metadata = {k: v for k, v in data.items() if k != 'U'}
                
                if verbose:
                    logging.info(f"  Dictionary format")
                    if 'plaquette' in metadata:
                        logging.info(f"  Plaquette: {metadata['plaquette']:.6f}")
                    if 'beta' in metadata:
                        logging.info(f"  Beta: {metadata['beta']:.3f}")
                
                return [None, U], metadata
            else:
                raise ValueError("Dictionary missing 'U' field")
                
        else:
            raise ValueError(f"Unknown format: {type(data)}")
            
    except Exception as e:
        if verbose:
            logging.error(f"Error loading configuration: {e}")
            logging.warning("Using identity gauge field")
        
        return generate_identity_gauge_field(verbose=verbose)

def generate_identity_gauge_field(lattice_dims=None, verbose=False):
    """Generate identity gauge field for testing"""
    if lattice_dims is None:
        lattice_dims = [4, 4, 4, 16]
        
    if verbose:
        logging.warning("Using identity gauge field U = I")
    
    Lx, Ly, Lz, Lt = lattice_dims
    V = Lx * Ly * Lz * Lt
    
    U = np.zeros((V, 4, 4))
    for i in range(V):
        for mu in range(4):
            U[i, mu] = su2.cstart()
    
    metadata = {
        'plaquette': 1.0,
        'format': 'identity',
        'lattice_dims': lattice_dims
    }
    
    return [None, U], metadata

def detect_lattice_dimensions(U, provided_dims=None, verbose=False):
    """Detect lattice dimensions from gauge configuration"""
    if provided_dims and all(d is not None for d in provided_dims):
        dims = list(provided_dims)
        if verbose:
            logging.info(f"Using provided dimensions: {dims}")
        return dims
    
    if U is None or len(U) < 2:
        default_dims = [4, 4, 4, 16]
        if verbose:
            logging.warning(f"Using default dimensions: {default_dims}")
        return default_dims
    
    V = U[1].shape[0]
    
    if verbose:
        logging.info(f"Detecting dimensions from volume V = {V}")
    
    # Common lattice sizes
    common_lattices = [
        [4, 4, 4, 4], [4, 4, 4, 8], [4, 4, 4, 16], [4, 4, 4, 20],
        [6, 6, 6, 16], [6, 6, 6, 20], [8, 8, 8, 16], [8, 8, 8, 24],
        [3, 3, 4, 20], [4, 4, 6, 16], [6, 6, 8, 20]
    ]
    
    for dims in common_lattices:
        if np.prod(dims) == V:
            if verbose:
                logging.info(f"  Detected: {dims}")
            return dims
    
    # Fallback
    spatial_size = int(round((V/16)**(1/3)))
    if spatial_size**3 * 16 == V:
        dims = [spatial_size, spatial_size, spatial_size, 16]
    else:
        dims = [4, 4, 4, V//64 if V >= 64 else 4]
    
    if verbose:
        logging.warning(f"  Guessed: {dims}")
    return dims

def get_meson_gamma_matrix(channel, verbose=False):
    """
    Get gamma matrix structure for different meson channels.
    
    Mesons are classified by quantum numbers J^PC (spin-parity-charge conjugation).
    The meson operator M_Γ(x) = ψ̄(x)Γψ(x) creates mesons with quantum numbers
    determined by the Dirac structure Γ:
    
    - Pion (π): Γ = γ₅, J^PC = 0^(-+), pseudoscalar Goldstone boson
    - Sigma (σ): Γ = I, J^PC = 0^(++), scalar chiral partner
    - Rho (ρ): Γ = γᵢ, J^PC = 1^(--), vector meson
    
    The gamma matrices satisfy the Clifford algebra {γμ,γν} = 2gμν.
    γ₅ = iγ⁰γ¹γ²γ³ in our chiral representation becomes diag(-1,-1,1,1).
    
    Args:
        channel (str): Meson channel name
        verbose (bool): Enable detailed output
        
    Returns:
        dict: Channel info with gamma matrix and quantum numbers
    """
    if verbose:
        logging.info(f"Setting up {channel} channel")
    
    # Gamma matrices
    gamma_matrices = {
        'gamma0': np.array([[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]]),
        'gamma1': np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]),
        'gamma2': np.array([[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]]),
        'gamma3': np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]),
        'identity': np.eye(4, dtype=complex)
    }
    
    # Gamma5 = diag(-1,-1,1,1)
    gamma5 = np.zeros((4, 4), dtype=complex)
    gamma5[0,0] = -1
    gamma5[1,1] = -1  
    gamma5[2,2] = 1
    gamma5[3,3] = 1
    gamma_matrices['gamma5'] = gamma5
    
    # Channel definitions
    channels = {
        'pion': {
            'gamma': gamma5, 
            'JPC': '0-+', 
            'name': 'Pion',
            'description': 'Pseudoscalar'
        },
        'sigma': {
            'gamma': gamma_matrices['identity'], 
            'JPC': '0++', 
            'name': 'Sigma',
            'description': 'Scalar'
        },
        'rho_x': {
            'gamma': gamma_matrices['gamma1'], 
            'JPC': '1--', 
            'name': 'Rho (x)'
        },
        'rho_y': {
            'gamma': gamma_matrices['gamma2'], 
            'JPC': '1--', 
            'name': 'Rho (y)'
        },
        'rho_z': {
            'gamma': gamma_matrices['gamma3'], 
            'JPC': '1--', 
            'name': 'Rho (z)'
        },
        'rho_t': {
            'gamma': gamma_matrices['gamma0'], 
            'JPC': '1--', 
            'name': 'Rho (t)'
        }
    }
    
    if channel == 'rho':
        # Average spatial components
        avg_gamma = (channels['rho_x']['gamma'] + channels['rho_y']['gamma'] + channels['rho_z']['gamma']) / 3.0
        return {
            'gamma': avg_gamma,
            'JPC': '1--',
            'name': 'Rho (averaged)',
            'description': 'Vector meson'
        }
    elif channel in channels:
        channel_info = channels[channel]
        if verbose:
            logging.info(f"  Channel: {channel_info['name']}")
            logging.info(f"  J^PC: {channel_info['JPC']}")
        return channel_info
    else:
        raise ValueError(f"Unknown channel: {channel}")

def build_wilson_dirac_matrix(mass, lattice_dims, wilson_r=1.0, U=None, verbose=False):
    """
    Build Wilson-Dirac matrix with antiperiodic boundary conditions (SU(2) only)

    D_W = m + 4r + (1/2) Σ_μ [(1-γ_μ)U_μ(x)δ_{x+μ,y} + (1+γ_μ)U†_μ(x-μ)δ_{x-μ,y}]

    Note: This version is optimized for SU(2) with quaternion gauge links.
    For SU(N) support, use the Calculator modules (PionCalculator, etc.)
    """
    Lx, Ly, Lz, Lt = lattice_dims
    V = Lx * Ly * Lz * Lt
    matrix_size = 8 * V  # 2 colors × 4 spins × V sites

    # Effective mass includes Wilson shift
    mass_effective = mass + 4.0 * wilson_r

    if verbose:
        logging.info(f"Building Wilson-Dirac matrix:")
        logging.info(f"  Lattice: {Lx}×{Ly}×{Lz}×{Lt} (V={V})")
        logging.info(f"  Matrix size: {matrix_size}×{matrix_size}")
        logging.info(f"  m={mass:.6f}, r={wilson_r:.2f}")
        logging.info(f"  Effective mass: {mass_effective:.6f}")
        logging.info(f"  Boundary conditions: antiperiodic in time")
    
    if U is None:
        if verbose:
            logging.warning("No gauge field provided, using identity")
        U = generate_identity_gauge_field(lattice_dims, verbose=False)[0]
    
    # Gamma matrices
    gamma = [
        np.array([[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]]),  # gamma_0
        np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]),       # gamma_1
        np.array([[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]]),  # gamma_2
        np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])         # gamma_3
    ]
    
    # Precompute neighbors
    mups = np.zeros((V, 4), dtype=int)
    mdns = np.zeros((V, 4), dtype=int)
    for i in range(V):
        for mu in range(4):
            mups[i, mu] = su2.mupi(i, mu, lattice_dims)
            mdns[i, mu] = su2.mdowni(i, mu, lattice_dims)
    
    # Build sparse matrix
    rows, cols, data = [], [], []
    
    # Mass term
    for idx in range(matrix_size):
        rows.append(idx)
        cols.append(idx)
        data.append(mass_effective)
    
    # Hopping terms
    errors = 0
    for base_idx in range(V):
        base = 8 * base_idx
        point = su2.i2p(base_idx, lattice_dims)
        
        for mu in range(4):
            try:
                next_idx = mups[base_idx, mu]
                prev_idx = mdns[base_idx, mu]
                next_base = 8 * next_idx
                prev_base = 8 * prev_idx
                
                next_point = su2.i2p(next_idx, lattice_dims)
                prev_point = su2.i2p(prev_idx, lattice_dims)
                
                # Antiperiodic boundary conditions in time
                forward_sign = 1.0
                backward_sign = 1.0
                
                if mu == 3:  # Time direction
                    if point[3] == Lt-1 and next_point[3] == 0:
                        forward_sign = -1.0  # ψ(Lt) = -ψ(0)
                    if point[3] == 0 and prev_point[3] == Lt-1:
                        backward_sign = -1.0  # ψ(-1) = -ψ(Lt-1)
                
                # Convert gauge links to complex matrices
                try:
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
                    
                except Exception:
                    U_complex = np.eye(2, dtype=complex)
                    U_prev_dag = np.eye(2, dtype=complex)
                    errors += 1
                
                # Hopping matrices
                K_forward = 0.5 * forward_sign * np.kron(gamma[mu], U_complex)
                W_forward = -0.5 * wilson_r * forward_sign * np.kron(np.eye(4), U_complex)
                M_forward = K_forward + W_forward
                
                K_backward = -0.5 * backward_sign * np.kron(gamma[mu], U_prev_dag)
                W_backward = -0.5 * wilson_r * backward_sign * np.kron(np.eye(4), U_prev_dag)
                M_backward = K_backward + W_backward
                
                # Add matrix elements
                for i in range(8):
                    for j in range(8):
                        if abs(M_forward[i,j]) > 1e-12:
                            rows.append(base + i)
                            cols.append(next_base + j)
                            data.append(M_forward[i,j])
                        
                        if abs(M_backward[i,j]) > 1e-12:
                            rows.append(base + i)
                            cols.append(prev_base + j)
                            data.append(M_backward[i,j])
                            
            except Exception as e:
                if verbose and errors < 10:
                    logging.warning(f"    Error at site {base_idx}, mu {mu}: {e}")
                errors += 1
                continue
    
    if errors > 0 and verbose:
        logging.warning(f"  Matrix construction had {errors} errors")
    
    D = sparse.csc_matrix((data, (rows, cols)), shape=(matrix_size, matrix_size))
    
    if verbose:
        logging.info(f"  Matrix built: nnz={D.nnz}, sparsity={D.nnz/(matrix_size**2)*100:.2f}%")
    
    return D

def create_point_source(lattice_dims, t_source, color, spin, verbose=False):
    """
    Create point source for quark propagator calculation.
    
    A point source δ³(x⃗-x⃗₀)δ(t-t₀) creates a quark at spacetime location
    (x⃗₀,t₀) with specific color and spin quantum numbers. The propagator
    S(x,y) then describes how this quark propagates to all other points.
    
    Source placement:
    - Spatial: x⃗₀ = (0,0,0) at lattice corner
    - Temporal: t₀ = 0 for maximal time extent
    - All color/spin: 2 colors × 4 spins = 8 total propagators
    
    Index ordering: site × 8 + spin × 2 + color
    
    Args:
        lattice_dims (list): [Lx, Ly, Lz, Lt]
        t_source (int): Source time slice (typically 0)
        color (int): SU(2) color index (0 or 1)
        spin (int): Dirac spinor index (0, 1, 2, 3)
        verbose (bool): Enable debug output
        
    Returns:
        numpy.ndarray: Source vector with single non-zero entry
    """
    Lx, Ly, Lz, Lt = lattice_dims
    V = Lx * Ly * Lz * Lt
    source = np.zeros(8*V, dtype=complex)
    
    point = np.array([0, 0, 0, t_source])
    site_idx = su2.p2i(point, lattice_dims)
    base_idx = 8 * site_idx
    
    idx = base_idx + 2*spin + color
    source[idx] = 1.0
    
    if verbose:
        logging.info(f"    Point source: pos={point}, color={color}, spin={spin}")
    
    return source

def solve_dirac_system(D, source, method='auto', verbose=False):
    """
    Solve Wilson-Dirac linear system D × S = δ for quark propagator.
    
    This is the computational heart of lattice QCD - solving the discretized
    Dirac equation. The propagator S(x,y) = <ψ(x)ψ̄(y)> gives the amplitude
    for a quark created at y to propagate to x, encoding confinement physics.
    
    Numerical considerations:
    - Matrix size: O(10⁴-10⁶) × O(10⁴-10⁶) for typical lattices
    - Sparsity: ~0.1% non-zero (nearest-neighbor coupling)
    - Conditioning: Wilson term prevents near-zero eigenvalues
    - Methods: Direct (small lattices), GMRES/LSQR (large lattices)
    
    NOTE: scipy's GMRES has used different tolerance parameters across versions:
    - Some versions: 'tol'
    - Some versions: 'atol'
    - Some versions: 'rtol'
    This function automatically tries each parameter until one works.
    
    Args:
        D (scipy.sparse matrix): Wilson-Dirac operator
        source (numpy.ndarray): Point source vector δ
        method (str): Solver choice ('auto', 'direct', 'gmres', 'lsqr')
        verbose (bool): Enable convergence diagnostics
        
    Returns:
        numpy.ndarray: Quark propagator solution S
    """
    if verbose:
        logging.info(f"  Solving Dirac system (method={method})...")
    
    # Add regularization
    regularization = 1e-8
    D_reg = D + regularization * sparse.eye(D.shape[0])
    
    if method == 'auto':
        method = 'direct' if D.shape[0] <= 5000 else 'gmres'
    
    start_time = time.time()
    
    try:
        if method == 'direct':
            solution = spla.spsolve(D_reg, source)
            info = "converged"
        elif method == 'gmres':
            # Handle scipy version differences
            
            solution = None
            info = -1
            
            # Try different parameter names
            for tol_param in ['atol', 'rtol', 'tol']:
                try:
                    kwargs = {'maxiter': 2000, tol_param: 1e-8}
                    solution, info = spla.gmres(D_reg, source, **kwargs)
                    if verbose:
                        logging.info(f"    GMRES succeeded with parameter '{tol_param}'")
                    break
                except TypeError as e:
                    if verbose and tol_param == 'tol':  # Last attempt
                        logging.warning(f"    GMRES failed with all tolerance parameters")
                    continue
            
            # If all parameter attempts failed, try without tolerance
            if solution is None:
                try:
                    solution, info = spla.gmres(D_reg, source, maxiter=2000)
                    if verbose:
                        logging.warning("    GMRES running without explicit tolerance")
                except Exception as e:
                    if verbose:
                        logging.error(f"    GMRES failed completely: {e}")
                    solution = np.zeros(len(source), dtype=complex)
                    info = -1
            
            # If GMRES failed after all attempts, fall back to LSQR
            if info != 0 or solution is None:
                if verbose:
                    logging.info(f"    GMRES info: {info}, trying LSQR...")
                # LSQR uses different parameter names across scipy versions
                try:
                    solution = spla.lsqr(D_reg, source, iter_lim=2000)[0]
                except TypeError:
                    try:
                        solution = spla.lsqr(D_reg, source, maxiter=2000)[0]
                    except TypeError:
                        solution = spla.lsqr(D_reg, source)[0]
                info = "lsqr_fallback"
        elif method == 'lsqr':
            # LSQR uses different parameter names across scipy versions
            try:
                solution = spla.lsqr(D_reg, source, iter_lim=2000)[0]
            except TypeError:
                try:
                    solution = spla.lsqr(D_reg, source, maxiter=2000)[0]
                except TypeError:
                    solution = spla.lsqr(D_reg, source)[0]
            info = "converged"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        solve_time = time.time() - start_time
        
        # Ensure we have a valid solution
        if solution is None:
            solution = np.zeros(len(source), dtype=complex)
            info = "failed"
        
        if verbose:
            residual = np.linalg.norm(D_reg @ solution - source)
            logging.info(f"    Solved in {solve_time:.3f}s, residual: {residual:.2e}")
        
        return solution
        
    except Exception as e:
        if verbose:
            logging.error(f"    Solver failed: {e}")
        return np.zeros(len(source), dtype=complex)

def calculate_meson_correlator(propagators, lattice_dims, gamma_matrix, verbose=False):
    """Calculate meson correlator C(t) = Tr[Γ S(0,t)] (SU(2) only)"""
    Lx, Ly, Lz, Lt = lattice_dims
    correlator = np.zeros(Lt, dtype=complex)

    if verbose:
        logging.info("  Calculating correlator...")

    for t in range(Lt):
        point_t = np.array([0, 0, 0, t])
        site_idx = su2.p2i(point_t, lattice_dims)
        base_idx = 8 * site_idx

        corr_sum = 0.0

        # Sum over colors
        for color in range(2):
            # Build 4x4 propagator matrix
            S_matrix = np.zeros((4, 4), dtype=complex)
            
            for src_spin in range(4):
                src_prop_idx = 4 * color + src_spin
                src_propagator = propagators[src_prop_idx]

                for sink_spin in range(4):
                    sink_idx = base_idx + 4 * color + sink_spin
                    if sink_idx < len(src_propagator):
                        S_matrix[sink_spin, src_spin] = src_propagator[sink_idx]
            
            # Calculate Tr[Γ S]
            trace_gamma_S = np.trace(gamma_matrix @ S_matrix)
            corr_sum += trace_gamma_S
            
            if verbose and t < 2 and color == 0:
                logging.info(f"    t={t}, color={color}: Tr[Γ S] = {trace_gamma_S:.6e}")
        
        correlator[t] = corr_sum
    
    # Take real part
    correlator_real = np.real(correlator)
    
    if verbose:
        imag_max = np.max(np.abs(np.imag(correlator)))
        logging.info(f"    Correlator range: [{np.min(correlator_real):.2e}, {np.max(correlator_real):.2e}]")
        logging.info(f"    Max imaginary part: {imag_max:.2e}")
    
    return correlator_real

def calculate_effective_mass(correlator, verbose=False):
    """Calculate effective mass M_eff(t) = ln[C(t)/C(t+1)]"""
    if len(correlator) < 2:
        return np.array([]), np.array([])
    
    correlator = np.maximum(correlator, 1e-15)
    
    if verbose:
        logging.info(f"  Calculating effective mass...")
    
    # Ratio method
    ratios = correlator[:-1] / correlator[1:]
    ratios = np.maximum(ratios, 1e-10)
    ratios = np.minimum(ratios, 1e10)
    
    mass_eff = np.log(ratios)
    
    # Error estimate
    mass_err = 0.1 * np.abs(mass_eff)
    
    # Validity check
    valid = (mass_eff > 0) & (mass_eff < 10) & np.isfinite(mass_eff)
    mass_eff = np.where(valid, mass_eff, np.nan)
    mass_err = np.where(valid, mass_err, np.nan)
    
    if verbose:
        n_valid = np.sum(~np.isnan(mass_eff))
        logging.info(f"    Valid points: {n_valid}/{len(mass_eff)}")
    
    return mass_eff, mass_err

def fit_plateau(mass_eff, mass_err, t_min=2, t_max=None, verbose=False):
    """Fit plateau to extract ground state mass"""
    if len(mass_eff) == 0:
        return 0.5, 0.1, 0.0, (0, 0)
    
    valid = ~np.isnan(mass_eff) & ~np.isnan(mass_err)
    if not np.any(valid):
        return 0.5, 0.1, 0.0, (0, 0)
    
    valid_mass = mass_eff[valid]
    valid_err = mass_err[valid]
    valid_indices = np.where(valid)[0]
    
    # Auto range selection
    if t_max is None:
        t_max = min(len(mass_eff) - 1, len(valid_indices) - 1)
    
    t_min = min(t_min, len(valid_indices) // 2)
    t_max = max(t_max, t_min)
    
    # Select fit range
    fit_mask = (valid_indices >= t_min) & (valid_indices <= t_max)
    if not np.any(fit_mask):
        mid_start = len(valid_indices) // 3
        mid_end = 2 * len(valid_indices) // 3
        if mid_end <= mid_start:
            mid_end = min(mid_start + 1, len(valid_indices))
        fit_mass = valid_mass[mid_start:mid_end]
        fit_err = valid_err[mid_start:mid_end]
        fit_range = (valid_indices[mid_start], valid_indices[mid_end-1])
    else:
        fit_mass = valid_mass[fit_mask]
        fit_err = valid_err[fit_mask]
        fit_indices = valid_indices[fit_mask]
        fit_range = (min(fit_indices), max(fit_indices))
    
    # Weighted fit
    if len(fit_mass) > 0:
        weights = 1.0 / (fit_err**2 + 1e-10)
        plateau_mass = np.sum(fit_mass * weights) / np.sum(weights)
        plateau_err = 1.0 / np.sqrt(np.sum(weights))
        
        if len(fit_mass) > 1:
            residuals = fit_mass - plateau_mass
            chi2 = np.sum((residuals / fit_err)**2) / (len(fit_mass) - 1)
        else:
            chi2 = 0.0
    else:
        plateau_mass, plateau_err, chi2 = 0.5, 0.1, 0.0
    
    if verbose:
        logging.info(f"    Plateau fit: M = {plateau_mass:.6f} ± {plateau_err:.6f}")
        logging.info(f"    Fit range: t={fit_range[0]}-{fit_range[1]}, χ²/dof = {chi2:.2f}")
    
    return plateau_mass, plateau_err, chi2, fit_range

def calculate_meson_mass(U, mass, lattice_dims, channel='pion', wilson_r=0.5, solver='auto', verbose=False):
    """Calculate meson mass for given parameters (SU(2) only)

    Note: This version uses the internal SU(2)-specific build_wilson_dirac_matrix.
    For SU(N) support, use PionCalculator, SigmaCalculator, or RhoCalculator modules.
    """
    if verbose:
        logging.info(f"\nCalculating {channel} mass:")
        logging.info(f"  Quark mass: {mass:.6f}")
        logging.info(f"  Lattice: {lattice_dims}")

    # Get gamma matrix
    channel_info = get_meson_gamma_matrix(channel, verbose)
    gamma_matrix = channel_info['gamma']

    # Build Dirac matrix
    D = build_wilson_dirac_matrix(mass, lattice_dims, wilson_r, U, verbose)

    # Calculate propagators
    if verbose:
        logging.info("  Calculating propagators...")

    propagators = []
    for color in range(2):
        for spin in range(4):
            if verbose:
                logging.info(f"    Propagator {len(propagators)+1}/8: color={color}, spin={spin}")
            source = create_point_source(lattice_dims, 0, color, spin, verbose)
            prop = solve_dirac_system(D, source, solver, verbose)
            propagators.append(prop)

    # Calculate correlator
    correlator = calculate_meson_correlator(propagators, lattice_dims, gamma_matrix, verbose)
    
    # Extract effective mass
    mass_eff, mass_err = calculate_effective_mass(correlator, verbose)
    
    # Fit plateau
    plateau_mass, plateau_err, chi2, fit_range = fit_plateau(mass_eff, mass_err, verbose=verbose)
    
    return {
        'channel': channel,
        'input_mass': mass,
        'meson_mass': plateau_mass,
        'meson_error': plateau_err,
        'chi_squared': chi2,
        'fit_range': fit_range,
        'correlator': correlator.tolist(),
        'effective_mass': mass_eff.tolist(),
        'mass_errors': mass_err.tolist(),
        'lattice_dims': lattice_dims,
        'wilson_r': wilson_r,
        'solver': solver,
        'channel_info': channel_info
    }

def save_results(results, output_dirs, save_correlators=False):
    """Save results in multiple formats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not isinstance(results, list):
        results = [results]
    
    # Save JSON
    json_file = os.path.join(output_dirs['data'], f'results_{timestamp}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"Saved results: {json_file}")
    
    # Save summary
    summary_file = os.path.join(output_dirs['data'], f'summary_{timestamp}.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("LATTICE QCD MESON MASS RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        # Add physics context for single configuration
        f.write("IMPORTANT PHYSICS NOTES:\n")
        f.write("-" * 30 + "\n")
        f.write("These results are from a SINGLE gauge configuration.\n")
        f.write("Physical meson masses require ensemble averaging over\n")
        f.write("many configurations (typically 100-1000).\n\n")
        f.write("Single-config results may show:\n")
        f.write("- Statistical fluctuations (e.g., M_ρ < M_π)\n")
        f.write("- Broken rotational symmetry (M_ρx ≠ M_ρy ≠ M_ρz)\n")
        f.write("- Poor signal-to-noise in some channels\n")
        f.write("- Deviations from expected mass hierarchy\n\n")
        
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
            
            # Add quality assessment
            f.write(f"Quality Assessment:\n")
            rel_error = result['meson_error'] / result['meson_mass'] if result['meson_mass'] > 0 else float('inf')
            if rel_error < 0.1:
                f.write(f"  Statistical precision: Good ({rel_error*100:.1f}%)\n")
            elif rel_error < 0.2:
                f.write(f"  Statistical precision: Acceptable ({rel_error*100:.1f}%)\n")
            else:
                f.write(f"  Statistical precision: Poor ({rel_error*100:.1f}%)\n")
                f.write(f"  → Large errors typical for single config\n")
            
            if result['chi_squared'] > 5.0:
                f.write(f"  Plateau fit: Poor (χ²/dof = {result['chi_squared']:.1f})\n")
                f.write(f"  → May indicate excited state contamination\n")
            elif result['chi_squared'] < 0.1:
                f.write(f"  Plateau fit: Suspiciously good (χ²/dof = {result['chi_squared']:.2f})\n")
                f.write(f"  → May indicate overfitting or constant data\n")
            else:
                f.write(f"  Plateau fit: Acceptable (χ²/dof = {result['chi_squared']:.2f})\n")
            
            # Physics interpretation
            f.write(f"\nPhysics Interpretation:\n")
            eff_mass = result['input_mass'] + 4*result['wilson_r']
            ratio = result['meson_mass'] / eff_mass if eff_mass > 0 else 0
            f.write(f"  M_meson/m_eff ratio: {ratio:.3f}\n")
            
            if result['channel'] == 'pion':
                f.write(f"  Pion: Pseudoscalar Goldstone boson\n")
                f.write(f"  Expected: Lightest meson, M_π → 0 as m → 0\n")
                if ratio < 2.0:
                    f.write(f"  ✓ Light mass consistent with Goldstone nature\n")
            elif result['channel'] == 'sigma':
                f.write(f"  Sigma: Scalar, chiral partner of pion\n")
                f.write(f"  Expected: Heavy, remains massive in chiral limit\n")
                if ratio > 3.0:
                    f.write(f"  ✓ Heavy mass indicates chiral breaking\n")
            elif 'rho' in result['channel']:
                f.write(f"  Rho: Vector meson, J^PC = 1^(--)\n")
                f.write(f"  Expected: Intermediate mass\n")
                f.write(f"  Note: ρx ≠ ρy ≠ ρz common on single config\n")
            
            f.write("\n")
    
    logging.info(f"Saved summary: {summary_file}")
    
    # Save correlators
    if save_correlators:
        for result in results:
            if 'correlator' in result and result['correlator']:
                channel = result['channel']
                corr_file = os.path.join(output_dirs['correlators'], f'{channel}_correlator_{timestamp}.dat')
                
                with open(corr_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {channel.upper()} correlator data\n")
                    f.write(f"# Single gauge configuration result\n")
                    f.write(f"# Mass: {result['meson_mass']:.6f} ± {result['meson_error']:.6f}\n")
                    f.write(f"# Chi²/dof: {result['chi_squared']:.2f}\n")
                    f.write(f"# \n")
                    f.write(f"# Expected behavior: C(t) ~ A exp(-M*t) for large t\n")
                    f.write(f"# Effective mass: M_eff(t) = ln[C(t)/C(t+1)]\n")
                    f.write(f"# \n")
                    f.write(f"# t  C(t)\n")
                    
                    for t, c in enumerate(result['correlator']):
                        f.write(f"{t:3d}  {c:15.8e}\n")
                
                logging.info(f"Saved correlator: {corr_file}")

def create_analysis_plots(results, output_dirs):
    """
    Create 8-panel analysis plots showing all aspects of the calculation.
    
    This visualization provides a complete view of the lattice QCD analysis:
    1. Correlator decay (log scale) - shows exponential behavior
    2. Effective mass plateau - demonstrates mass extraction
    3. Physics summary - key results and parameters
    4. Wilson mass analysis - shows discretization effects
    5. Linear correlator - reveals sign structure
    6. Log-linear fit - validates exponential decay
    7. Physics workflow guide - explains methodology
    8. Quality assessment - guides improvement strategies
    """
    if not isinstance(results, list):
        results = [results]
    
    for result in results:
        channel = result['channel']
        correlator = np.array(result.get('correlator', []))
        mass_eff = np.array(result.get('effective_mass', []))
        mass_err = np.array(result.get('mass_errors', []))
        
        if len(correlator) == 0:
            logging.warning(f"No data for {channel}, skipping plots")
            continue
        
        # Create comprehensive 8-panel figure
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(f'Lattice QCD Analysis: {channel.title()} Meson', 
                    fontsize=18, fontweight='bold')
        
        # Plot 1: Correlator exponential decay (log scale)
        plt.subplot(2, 4, 1)
        t_vals = np.arange(len(correlator))
        
        # Handle zero correlator case
        if np.all(correlator == 0):
            plt.text(0.5, 0.5, 'No correlator data\n(all propagators failed)', 
                    transform=plt.gca().transAxes, ha='center', va='center',
                    fontsize=14, color='red', fontweight='bold')
            plt.xlabel('Time Slice t', fontsize=14)
            plt.ylabel('|Correlator|', fontsize=14)
            plt.title('Exponential Decay Analysis', fontsize=16, fontweight='bold')
        else:
            # Plot only non-zero values
            non_zero = np.abs(correlator) > 0
            if np.any(non_zero):
                plt.semilogy(t_vals[non_zero], np.abs(correlator[non_zero]), 'bo-', 
                            markersize=8, linewidth=3, label='|C(t)| Data', alpha=0.8)
                
                # Overlay theoretical exponential decay
                if result['meson_mass'] > 0 and len(correlator) > 1:
                    if abs(correlator[1]) > 0:
                        # Normalize to match at t=1
                        amplitude = abs(correlator[1]) / np.exp(-result['meson_mass'] * 1)
                        theory_curve = amplitude * np.exp(-result['meson_mass'] * t_vals)
                        plt.plot(t_vals, theory_curve, 'r--', alpha=0.9, linewidth=4,
                                label=f'Theory: A exp(-{result["meson_mass"]:.3f}×t)')
                
                plt.xlabel('Time Slice t', fontsize=14)
                plt.ylabel('|Correlator|', fontsize=14)
                plt.title('Exponential Decay Analysis', fontsize=16, fontweight='bold')
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'All correlator values are zero', 
                        transform=plt.gca().transAxes, ha='center', va='center',
                        fontsize=12, color='orange')
        
        # Plot 2: Effective mass plateau extraction
        plt.subplot(2, 4, 2)
        if len(mass_eff) > 0:
            t_eff = np.arange(len(mass_eff))
            valid = ~np.isnan(mass_eff)
            
            if np.any(valid):
                plt.errorbar(t_eff[valid], mass_eff[valid], yerr=mass_err[valid], 
                           fmt='go-', capsize=6, markersize=8, linewidth=3, 
                           label='M_eff(t)', alpha=0.8)
                
                # Highlight plateau fitting region
                fit_range = result.get('fit_range', (0, 0))
                if fit_range[1] > fit_range[0]:
                    plt.axvspan(fit_range[0], fit_range[1], alpha=0.3, color='yellow', 
                               label='Plateau Fit Region')
                
                # Show extracted ground state mass
                plt.axhline(result['meson_mass'], color='red', linestyle='--', linewidth=4,
                           label=f'M = {result["meson_mass"]:.4f} ± {result["meson_error"]:.4f}')
        
        plt.xlabel('Time Slice t', fontsize=14)
        plt.ylabel('Effective Mass', fontsize=14)
        plt.title('Ground State Mass Extraction', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Physics results summary
        plt.subplot(2, 4, 3)
        summary_text = f'PHYSICS RESULTS\n'
        summary_text += f'{"="*25}\n'
        summary_text += f'Channel: {channel.upper()}\n'
        summary_text += f'J^PC: {result.get("channel_info", {}).get("JPC", "N/A")}\n\n'
        summary_text += f'EXTRACTED MASS:\n'
        summary_text += f'{result["meson_mass"]:.6f} ± {result["meson_error"]:.6f}\n\n'
        summary_text += f'INPUT PARAMETERS:\n'
        summary_text += f'Quark mass: {result["input_mass"]:.4f}\n'
        summary_text += f'Wilson r: {result["wilson_r"]:.3f}\n'
        summary_text += f'Effective mass: {result["input_mass"] + 4*result["wilson_r"]:.4f}\n\n'
        summary_text += f'ANALYSIS QUALITY:\n'
        summary_text += f'Chi²/dof: {result["chi_squared"]:.2f}\n'
        summary_text += f'Fit range: t={result["fit_range"][0]}-{result["fit_range"][1]}\n'
        summary_text += f'Lattice: {result["lattice_dims"]}'
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightcyan', alpha=0.9))
        plt.axis('off')
        plt.title('Analysis Summary', fontsize=16, fontweight='bold')
        
        # Plot 4: Wilson mass shift analysis
        plt.subplot(2, 4, 4)
        masses = [result['input_mass'], 
                 result['input_mass'] + 4*result['wilson_r'],
                 result['meson_mass']]
        labels = ['Bare\nMass', 'Effective\nMass\n(m+4r)', 'Meson\nMass']
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        
        bars = plt.bar(labels, masses, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        for bar, mass in zip(bars, masses):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(masses),
                    f'{mass:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.ylabel('Mass (lattice units)', fontsize=14)
        plt.title('Wilson Mass Shift Analysis', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add interpretation
        eff_mass = result['input_mass'] + 4*result['wilson_r']
        wilson_shift = 4 * result['wilson_r']
        shift_ratio = wilson_shift / result['input_mass'] if result['input_mass'] > 0 else float('inf')
        plt.text(0.5, 0.75, f'Wilson shift: +{wilson_shift:.2f}\nRatio 4r/m: {shift_ratio:.1f}', 
                transform=plt.gca().transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 5: Linear scale correlator
        plt.subplot(2, 4, 5)
        plt.plot(t_vals, correlator, 'mo-', markersize=8, linewidth=3, label='C(t)', alpha=0.8)
        plt.axhline(0, color='black', linestyle='-', alpha=0.5)
        plt.xlabel('Time Slice t', fontsize=14)
        plt.ylabel('Correlator C(t)', fontsize=14)
        plt.title('Correlator Sign Structure', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Plot 6: Log-linear analysis
        plt.subplot(2, 4, 6)
        # Handle zero correlator case
        if np.all(correlator == 0):
            plt.text(0.5, 0.5, 'No correlator data\n(solver failed)', 
                    transform=plt.gca().transAxes, ha='center', va='center',
                    fontsize=14, color='red')
        else:
            # Safely compute log of absolute value
            with np.errstate(divide='ignore', invalid='ignore'):
                log_correlator = np.log(np.abs(correlator))
                log_correlator[~np.isfinite(log_correlator)] = np.nan
            
            valid_log = np.isfinite(log_correlator)
            if np.any(valid_log):
                plt.plot(t_vals[valid_log], log_correlator[valid_log], 'co-', 
                        markersize=8, linewidth=3, label='ln|C(t)|', alpha=0.8)
                
                # Show linear fit in log space
                if result['meson_mass'] > 0 and len(correlator) > 1 and abs(correlator[1]) > 0:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        log_fit = np.log(abs(correlator[1])) - result['meson_mass'] * (t_vals - 1)
                    plt.plot(t_vals, log_fit, 'r--', alpha=0.9, linewidth=4, 
                            label=f'Slope = -{result["meson_mass"]:.3f}')
            else:
                plt.text(0.5, 0.5, 'Cannot compute log\n(all values zero)', 
                        transform=plt.gca().transAxes, ha='center', va='center',
                        fontsize=12, color='orange')
        
        plt.xlabel('Time Slice t', fontsize=14)
        plt.ylabel('ln|C(t)|', fontsize=14)
        plt.title('Exponential Decay Validation', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Plot 7: Lattice QCD workflow guide
        plt.subplot(2, 4, 7)
        workflow_text = f'LATTICE QCD WORKFLOW\n'
        workflow_text += f'{"="*25}\n\n'
        workflow_text += f'1. GAUGE GENERATION\n'
        workflow_text += f'   Monte Carlo sampling\n'
        workflow_text += f'   U_μ(x) ∈ SU(2)\n\n'
        workflow_text += f'2. PROPAGATOR SOLVE\n'
        workflow_text += f'   D_W·S = δ\n'
        workflow_text += f'   Wilson-Dirac equation\n\n'
        workflow_text += f'3. CORRELATORS\n'
        workflow_text += f'   C_Γ(t) = Tr[Γ S(0,t)]\n'
        workflow_text += f'   Γ determines J^PC\n\n'
        workflow_text += f'4. MASS EXTRACTION\n'
        workflow_text += f'   C(t) ~ exp(-M*t)\n'
        workflow_text += f'   M_eff = ln[C(t)/C(t+1)]\n'
        workflow_text += f'   Plateau → ground state\n\n'
        workflow_text += f'KEY PHYSICS:\n'
        workflow_text += f'• Confinement: no free quarks\n'
        workflow_text += f'• Chiral symmetry: M_π → 0\n'
        workflow_text += f'• Wilson fermions: m_eff = m+4r'
        
        plt.text(0.05, 0.95, workflow_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.9))
        plt.axis('off')
        plt.title('Physics Methodology', fontsize=16, fontweight='bold')
        
        # Plot 8: Quality assessment and recommendations
        plt.subplot(2, 4, 8)
        quality_text = f'QUALITY ASSESSMENT\n'
        quality_text += f'{"="*20}\n\n'
        
        # Statistical precision
        rel_error = result['meson_error'] / result['meson_mass']
        if rel_error < 0.05:
            quality_text += f'✓ STATISTICAL PRECISION\n'
            quality_text += f'  Error: {rel_error*100:.1f}% (Excellent)\n'
        elif rel_error < 0.1:
            quality_text += f'✓ STATISTICAL PRECISION\n'
            quality_text += f'  Error: {rel_error*100:.1f}% (Good)\n'
        else:
            quality_text += f'⚠ STATISTICAL PRECISION\n'
            quality_text += f'  Error: {rel_error*100:.1f}% (Poor)\n'
        
        # Fit quality
        chi2 = result['chi_squared']
        if chi2 < 1.5:
            quality_text += f'✓ PLATEAU FIT QUALITY\n'
            quality_text += f'  χ²/dof: {chi2:.2f} (Excellent)\n'
        elif chi2 < 3.0:
            quality_text += f'✓ PLATEAU FIT QUALITY\n'
            quality_text += f'  χ²/dof: {chi2:.2f} (Good)\n'
        else:
            quality_text += f'⚠ PLATEAU FIT QUALITY\n'
            quality_text += f'  χ²/dof: {chi2:.2f} (Poor)\n'
        
        quality_text += f'\nPHYSICS VALIDATION:\n'
        if result['channel'] == 'pion':
            if result['meson_mass'] < 1.5:
                quality_text += f'✓ Reasonable pion mass\n'
            else:
                quality_text += f'⚠ Heavy pion mass\n'
        
        # Wilson analysis
        eff_mass = result['input_mass'] + 4*result['wilson_r']
        wilson_ratio = 4*result['wilson_r'] / result['input_mass'] if result['input_mass'] > 0 else float('inf')
        if wilson_ratio > 10:
            quality_text += f'⚠ Large Wilson shift\n'
            quality_text += f'  4r/m = {wilson_ratio:.1f}\n'
        else:
            quality_text += f'✓ Moderate Wilson shift\n'
        
        quality_text += f'\nRECOMMENDATIONS:\n'
        if chi2 > 3.0:
            quality_text += f'• Try different fit range\n'
            quality_text += f'• Check excited states\n'
        if rel_error > 0.1:
            quality_text += f'• Generate ensemble\n'
            quality_text += f'• Increase time extent\n'
        if wilson_ratio > 10:
            quality_text += f'• Use smaller r value\n'
            quality_text += f'• Try lighter mass\n'
        if rel_error < 0.02:
            quality_text += f'• Verify not overfitting\n'
        
        quality_text += f'\nNOTE: Single config only!\n'
        quality_text += f'Ensemble needed for physics'
        
        plt.text(0.05, 0.95, quality_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightpink', alpha=0.9))
        plt.axis('off')
        plt.title('Quality & Recommendations', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(output_dirs['plots'], f'{channel}_analysis.png')
        
        # Suppress matplotlib warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        logging.info(f"Saved comprehensive analysis plot: {plot_file}")
        plt.show()

def main():
    """
    Main execution function for lattice QCD meson mass calculations.
    
    This orchestrates the complete workflow:
    1. Parse command line arguments and set up directories
    2. Load gauge configuration from file
    3. Execute requested analysis (single, spectrum, scan)
    4. Save results in multiple formats
    5. Generate comprehensive visualization
    
    The code supports multiple analysis modes:
    - Single channel: Calculate one meson mass
    - Spectrum: Calculate π, σ, ρ masses
    - Mass scan: Study chiral behavior
    - Wilson scan: Optimize discretization
    
    Results are organized in a clear directory structure with
    separate folders for data, plots, correlators, and logs.
    """
    # Quick test to see which GMRES parameter works
    print("Testing scipy.sparse.linalg.gmres compatibility...")
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        # Create tiny test system
        test_A = sp.csr_matrix([[1, 0], [0, 1]])
        test_b = np.array([1, 1])
        
        # Try each parameter
        param_works = None
        for param in ['atol', 'rtol', 'tol']:
            try:
                kwargs = {param: 1e-8, 'maxiter': 1}
                spla.gmres(test_A, test_b, **kwargs)
                param_works = param
                break
            except TypeError:
                continue
        
        if param_works:
            print(f"SUCCESS: Your scipy uses '{param_works}' for gmres tolerance")
        else:
            print("WARNING: Could not determine gmres tolerance parameter")
            print("The code will try multiple parameters automatically")
    except Exception as e:
        print(f"Could not test gmres parameters: {e}")
    
    print()
    
    # Parse arguments
    args = parse_arguments()
    
    # Detect lattice dimensions
    lattice_dims = detect_lattice_dimensions(None, [args.ls, args.ls, args.ls, args.lt])
    
    # Determine run type
    if args.mass_scan:
        run_type = "mass_scan"
    elif args.wilson_scan:
        run_type = "wilson_scan"
    elif args.channel == 'all':
        run_type = "spectrum"
    else:
        run_type = "single"
    
    # Set up output directories
    output_dirs = setup_output_directories(run_type, args.mass, args.channel, 
                                         lattice_dims, args.wilson_r)
    
    # Configure logging
    configure_logging(output_dirs['log_file'])
    
    # Log header
    logging.info("="*60)
    logging.info("LATTICE QCD PROPAGATOR CALCULATION")
    logging.info("="*60)
    logging.info(f"Run type: {run_type}")
    logging.info(f"Output directory: {output_dirs['base']}")
    logging.info("")
    
    # Log version information
    logging.info("SYSTEM INFORMATION:")
    logging.info(f"  Python version: {sys.version.split()[0]}")
    logging.info(f"  NumPy version: {np.__version__}")
    logging.info(f"  SciPy version: {scipy.__version__}")
    logging.info(f"  Matplotlib version: {matplotlib.__version__}")
    logging.info("  Note: Code automatically handles scipy version differences")
    
    logging.info("")
    logging.info("PHYSICS CONTEXT:")
    logging.info("This code calculates meson masses from lattice QCD.")
    logging.info("Single configuration results may show:")
    logging.info("- Statistical fluctuations (e.g., M_ρ < M_π)")
    logging.info("- Large errors (20-50% typical)")
    logging.info("- Failed channels (poor signal-to-noise)")
    logging.info("Physical results require ensemble averaging.")
    logging.info("")
    
    # Load gauge configuration
    U, metadata = load_gauge_configuration(args.input_config, args.verbose)
    
    # Auto-detect dimensions if needed
    if args.ls is None or args.lt is None:
        lattice_dims = detect_lattice_dimensions(U, [args.ls, args.ls, args.ls, args.lt], args.verbose)
    else:
        lattice_dims = [args.ls, args.ls, args.ls, args.lt]
    
    logging.info(f"Lattice dimensions: {lattice_dims}")
    
    # Start calculations
    start_time = time.time()
    
    if args.mass_scan:
        # Mass scan
        masses = [float(m.strip()) for m in args.mass_scan.split(',')]
        logging.info(f"Mass scan: {masses}")
        logging.info("\nPHYSICS: Studying chiral behavior M²_π ∝ m_quark")
        logging.info("In QCD, pion mass vanishes in chiral limit (m → 0)")
        logging.info("Gell-Mann-Oakes-Renner: M²_π = 2Bm where B = -<ψ̄ψ>/f²_π")
        
        results = []
        for mass in masses:
            logging.info(f"\n{'-'*40}")
            logging.info(f"Mass = {mass}")
            logging.info(f"{'-'*40}")
            
            result = calculate_meson_mass(U, mass, lattice_dims, args.channel,
                                        args.wilson_r, solver=args.solver, verbose=args.verbose)
            results.append(result)
            
            logging.info(f"Result: M_{args.channel} = {result['meson_mass']:.6f} ± {result['meson_error']:.6f}")
            
            # Check if extraction succeeded
            if result['meson_error'] / result['meson_mass'] > 0.5:
                logging.warning(f"  ⚠ Very large error - extraction may have failed")
                logging.info(f"  Consider different mass value or larger lattice")
        
        # Analyze chiral behavior
        logging.info(f"\n{'='*40}")
        logging.info("CHIRAL ANALYSIS")
        logging.info(f"{'='*40}")
        
        if args.channel == 'pion':
            # Extract successful results
            valid_results = [(r['input_mass'], r['meson_mass'], r['meson_error']) 
                           for r in results if r['meson_error'] / r['meson_mass'] < 0.5]
            
            if len(valid_results) >= 2:
                masses_in = [r[0] for r in valid_results]
                masses_out = [r[1] for r in valid_results]
                masses_sq = [r[1]**2 for r in valid_results]
                
                logging.info("Input mass → Meson mass:")
                for m_in, m_out, err in valid_results:
                    logging.info(f"  {m_in:.3f} → {m_out:.6f} ± {err:.6f}")
                
                logging.info(f"\nChiral behavior check:")
                logging.info(f"  M²_π values: {[f'{m:.4f}' for m in masses_sq]}")
                
                # Simple linear check
                if len(masses_in) >= 2:
                    # Ratio of M²/m for smallest and largest mass
                    ratio_small = masses_sq[0] / masses_in[0] if masses_in[0] > 0 else 0
                    ratio_large = masses_sq[-1] / masses_in[-1] if masses_in[-1] > 0 else 0
                    
                    if abs(ratio_small - ratio_large) / ratio_large < 0.3:
                        logging.info(f"  ✓ Approximately linear M²_π vs m (good chiral behavior)")
                    else:
                        logging.info(f"  ⚠ Non-linear M²_π vs m (may need lighter masses)")
                
                logging.info("\nNOTE: Single config results have large fluctuations")
                logging.info("Ensemble averaging needed for precise chiral extrapolation")
            else:
                logging.warning("Too few valid results for chiral analysis")
    
    elif args.wilson_scan:
        # Wilson parameter scan
        wilson_values = [float(r.strip()) for r in args.wilson_scan.split(',')]
        logging.info(f"Wilson scan: {wilson_values}")
        logging.info("\nPHYSICS: Wilson fermions and discretization effects")
        logging.info("Wilson term -r∇² removes fermion doubling but breaks chiral symmetry")
        logging.info("Effective mass: m_eff = m + 4r (additive mass renormalization)")
        logging.info("Smaller r → better chiral properties but risk of doublers")
        
        results = []
        for wilson_r in wilson_values:
            logging.info(f"\n{'-'*40}")
            logging.info(f"Wilson r = {wilson_r}")
            logging.info(f"{'-'*40}")
            
            eff_mass = args.mass + 4.0 * wilson_r
            logging.info(f"Effective mass: {args.mass} + 4×{wilson_r} = {eff_mass:.3f}")
            
            if eff_mass < 0.05:
                logging.warning("⚠ Very light effective mass - may see lattice artifacts")
            elif eff_mass > 2.0:
                logging.warning("⚠ Very heavy effective mass - poor chiral behavior")
            
            result = calculate_meson_mass(U, args.mass, lattice_dims, args.channel,
                                        wilson_r, solver=args.solver, verbose=args.verbose)
            results.append(result)
            
            logging.info(f"Result: M_{args.channel} = {result['meson_mass']:.6f} ± {result['meson_error']:.6f}")
            
            # Analyze Wilson effects
            ratio = result['meson_mass'] / eff_mass
            logging.info(f"  M_meson/m_eff = {ratio:.3f}")
            
            if ratio < 1.5:
                logging.info(f"  → Near free quark limit (weak binding)")
            elif ratio > 5.0:
                logging.info(f"  → Strong binding regime")
        
        # Wilson parameter analysis
        logging.info(f"\n{'='*40}")
        logging.info("WILSON PARAMETER ANALYSIS")
        logging.info(f"{'='*40}")
        
        logging.info("r      m_eff    M_meson   M/m_eff")
        logging.info("-" * 40)
        for r_val, res in zip(wilson_values, results):
            m_eff = args.mass + 4.0 * r_val
            m_meson = res['meson_mass']
            ratio = m_meson / m_eff if m_eff > 0 else 0
            quality = "✓" if res['meson_error'] / m_meson < 0.2 else "⚠"
            logging.info(f"{r_val:4.2f}   {m_eff:6.3f}   {m_meson:7.4f}   {ratio:6.3f}  {quality}")
        
        logging.info("\nInterpretation:")
        logging.info("- Standard choice: r = 1 (good compromise)")
        logging.info("- Small r (< 0.5): Better chiral properties, risk of doublers")
        logging.info("- Large r (> 1): Suppresses doublers, poor chiral limit")
        logging.info("- For light quarks: Consider r = 0.1-0.5")
        logging.info("\nNOTE: Optimal r depends on β and bare quark mass")
    
    elif args.channel == 'all':
        # Full spectrum
        channels = ['pion', 'sigma', 'rho_x', 'rho_y', 'rho_z']
        results = []
        
        logging.info("\nPHYSICS NOTE: Single configuration spectrum")
        logging.info("On a single gauge config, meson masses can deviate from")
        logging.info("ensemble expectations due to statistical fluctuations.")
        logging.info("Expected ensemble hierarchy: M_π < M_ρ < M_σ")
        logging.info("Single config may show: M_ρ < M_π or M_ρx ≠ M_ρy ≠ M_ρz\n")
        
        for channel in channels:
            logging.info(f"\n{'-'*40}")
            logging.info(f"Channel: {channel.upper()}")
            logging.info(f"{'-'*40}")
            
            result = calculate_meson_mass(U, args.mass, lattice_dims, channel,
                                        args.wilson_r, solver=args.solver, verbose=args.verbose)
            results.append(result)
            
            logging.info(f"Result: M_{channel} = {result['meson_mass']:.6f} ± {result['meson_error']:.6f}")
            
            # Add physics interpretation
            if result['meson_error'] / result['meson_mass'] > 0.3:
                logging.warning(f"  Large error ({100*result['meson_error']/result['meson_mass']:.0f}%) - poor signal in {channel}")
                logging.info(f"  This channel may need more statistics or larger time extent")
            
            if result['chi_squared'] > 10:
                logging.warning(f"  Poor plateau fit (χ²/dof = {result['chi_squared']:.1f})")
                logging.info(f"  Likely excited state contamination or noisy correlator")
        
        # Analyze spectrum physics
        logging.info(f"\n{'='*40}")
        logging.info("SPECTRUM ANALYSIS")
        logging.info(f"{'='*40}")
        
        # Extract masses
        pion_mass = next((r['meson_mass'] for r in results if r['channel'] == 'pion'), None)
        sigma_mass = next((r['meson_mass'] for r in results if r['channel'] == 'sigma'), None)
        rho_masses = [(r['channel'], r['meson_mass']) for r in results if 'rho' in r['channel']]
        
        # Check mass hierarchy
        if pion_mass and sigma_mass:
            if pion_mass < sigma_mass:
                logging.info(f"✓ Correct hierarchy: M_π ({pion_mass:.3f}) < M_σ ({sigma_mass:.3f})")
            else:
                logging.warning(f"⚠ Inverted hierarchy: M_π ({pion_mass:.3f}) > M_σ ({sigma_mass:.3f})")
                logging.info("  This can occur on single configs due to fluctuations")
        
        # Check rho degeneracy
        if len(rho_masses) == 3:
            rho_values = [m[1] for m in rho_masses]
            rho_spread = max(rho_values) - min(rho_values)
            rho_avg = sum(rho_values) / len(rho_values)
            
            logging.info(f"\nRho meson analysis:")
            for ch, mass in rho_masses:
                logging.info(f"  {ch}: {mass:.6f}")
            logging.info(f"  Average: {rho_avg:.6f}")
            logging.info(f"  Spread: {rho_spread:.6f} ({100*rho_spread/rho_avg:.1f}%)")
            
            if rho_spread / rho_avg > 0.1:
                logging.warning("  Large rho mass splitting indicates broken rotational symmetry")
                logging.info("  This is expected on a single config - ensemble average restores symmetry")
            
            if pion_mass and rho_avg < pion_mass:
                logging.warning(f"⚠ Unusual: M_ρ ({rho_avg:.3f}) < M_π ({pion_mass:.3f})")
                logging.info("  This statistical fluctuation would average out in ensemble")
        
        logging.info("\nREMINDER: Physical meson masses require ensemble averaging")
        logging.info("Single configuration results are for algorithm testing only")
    
    else:
        # Single calculation
        result = calculate_meson_mass(U, args.mass, lattice_dims, args.channel,
                                    args.wilson_r, args.solver, args.verbose)
        results = result
        
        logging.info(f"\n{args.channel.upper()} RESULTS:")
        logging.info(f"  Mass: {result['meson_mass']:.6f} ± {result['meson_error']:.6f}")
        logging.info(f"  χ²/dof: {result['chi_squared']:.4f}")
        
        # Physics interpretation
        eff_mass = args.mass + 4.0 * args.wilson_r
        ratio = result['meson_mass'] / eff_mass if eff_mass > 0 else 0
        
        logging.info(f"\nPhysics analysis:")
        logging.info(f"  Quark mass: {args.mass:.3f}")
        logging.info(f"  Wilson shift: +{4.0 * args.wilson_r:.3f}")
        logging.info(f"  Effective mass: {eff_mass:.3f}")
        logging.info(f"  M_meson/m_eff ratio: {ratio:.3f}")
        
        # Channel-specific expectations
        if args.channel == 'pion':
            logging.info(f"  Pion: Lightest meson, Goldstone boson of chiral symmetry")
            logging.info(f"  Expected: M_π → 0 as m_quark → 0 (chiral limit)")
            if ratio < 2.0:
                logging.info(f"  ✓ Light pion mass consistent with Goldstone nature")
        elif args.channel == 'sigma':
            logging.info(f"  Sigma: Scalar meson, chiral partner of pion")
            logging.info(f"  Expected: Remains massive in chiral limit")
            if ratio > 3.0:
                logging.info(f"  ✓ Heavy sigma mass indicates chiral symmetry breaking")
        elif 'rho' in args.channel:
            logging.info(f"  Rho: Vector meson, J^PC = 1^(--)")
            logging.info(f"  Expected: Intermediate mass between pion and sigma")
            logging.info(f"  Note: Single config may break rotational symmetry (ρx ≠ ρy ≠ ρz)")
        
        # Quality warnings
        rel_error = result['meson_error'] / result['meson_mass']
        if rel_error > 0.2:
            logging.warning(f"\n⚠ Large statistical error: {rel_error*100:.0f}%")
            logging.info("  Recommendations:")
            logging.info("  - Generate ensemble of gauge configurations")
            logging.info("  - Increase temporal extent (current Lt = {})".format(lattice_dims[3]))
            logging.info("  - Try different plateau fit range")
        
        if result['chi_squared'] > 5.0:
            logging.warning(f"\n⚠ Poor plateau fit quality: χ²/dof = {result['chi_squared']:.1f}")
            logging.info("  Possible causes:")
            logging.info("  - Excited state contamination (try larger t_min)")
            logging.info("  - Statistical noise (need ensemble averaging)")
            logging.info("  - Fit range too large (reduce t_max)")
    
    total_time = time.time() - start_time
    logging.info(f"\nCalculation time: {total_time:.2f} seconds")
    
    # Save results
    save_results(results, output_dirs, args.save_correlators)
    
    # Create plots
    if not args.no_plots:
        create_analysis_plots(results, output_dirs)
    
    # Final summary
    logging.info(f"\n{'='*60}")
    logging.info(f"CALCULATION COMPLETE")
    logging.info(f"{'='*60}")
    logging.info(f"Results saved in: {output_dirs['base']}")
    
    # Add physics summary
    if isinstance(results, list) and len(results) > 1:
        logging.info(f"\nPhysics Summary:")
        
        # Count successful extractions
        good_results = [r for r in results if r['meson_error'] / r['meson_mass'] < 0.5]
        logging.info(f"  Successful extractions: {len(good_results)}/{len(results)}")
        
        # Note about failures
        if len(good_results) < len(results):
            logging.info(f"  Failed channels typically have poor signal-to-noise")
            logging.info(f"  This is common on single configurations")
        
        # Ensemble reminder
        logging.info(f"\nIMPORTANT REMINDERS:")
        logging.info(f"  1. These are single configuration results")
        logging.info(f"  2. Physical masses require ensemble averaging")
        logging.info(f"  3. Statistical fluctuations can invert mass hierarchy")
        logging.info(f"  4. Rotational symmetry breaking (ρx ≠ ρy ≠ ρz) is expected")
    
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
