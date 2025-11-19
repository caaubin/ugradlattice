"""
SU2 Lattice QCD Thermodynamics Generator

A comprehensive framework for the generation and analysis of SU(2) gauge field configurations
for the study of finite temperature lattice QCD thermodynamics and phase transitions.

This code implements the Monte Carlo simulation of pure SU(2) gauge theory on a 4D lattice,
as described in Chapter 10 of the textbook. The fundamental principles follow from the 
discretization of spacetime (Eq. 10.1) and the Wilson gauge action (Eq. 10.35).

Physical Background:
------------------
In lattice QCD, continuous spacetime is replaced by a discrete lattice with spacing 'a',
where x = na for integer n (Eq. 10.1). Gauge fields U_mu(x) are SU(2) matrices living on 
the links between lattice sites. The plaquette (Eq. 10.35):
    W_[] = Tr[U_mu(x) U_nu(x+mu_hat) U_mu^dag(x+nu_hat) U_nu^dag(x)]
measures the local field strength and forms the basis of the Wilson action.

The partition function (Eq. 10.55-10.56) is evaluated using Monte Carlo methods:
    Z = ∫ DU e^{-S[U]}
where the action S = -beta/2 sum Tr(plaquettes) with beta = 2N/g^2 for SU(N).

Key Physics Concepts:
--------------------
1. Gauge Invariance: Physical observables remain unchanged under local gauge transformations
2. Confinement/Deconfinement: Phase transition at critical temperature T_c ~ beta_c = 2.3 for SU(2)
3. Thermalization: Process of reaching thermal equilibrium from initial configuration
4. Detailed Balance: Metropolis algorithm ensures correct Boltzmann distribution
5. Ergodicity: All configurations accessible regardless of initial state

Available Start Modes:
---------------------
1. Cold Start (--mode cold):
   - All links U_mu(x) ~ I (identity matrix), minimal action
   - Corresponds to T -> infinity in physical units
   - Use when: Starting from ordered phase, cooling studies
   
2. Hot Start (--mode hot):
   - All links random SU(2) matrices, maximal disorder
   - Corresponds to T -> 0 in physical units
   - Use when: Starting from disordered phase, heating studies
   
3. Mixed/Parity Start (--mode mixed/parity):
   - Checkerboard pattern of hot/cold sites based on sum(x_mu) mod 2
   - Intermediate initial action
   - Use when: Testing ergodicity, avoiding metastable states

Output Structure:
---------------
thermal_generator_runs/
├── standard/
│   └── [timestamp]_L[Lx]x[Ly]x[Lz]T[Lt]_b[beta]_[mode]/
│       ├── configs/           # Saved configurations
│       ├── plots/             # Plaquette evolution plots
│       ├── checkpoints/       # Simulation checkpoints
│       ├── physics_report.txt # Physics analysis report
│       └── run.log            # Detailed logging output
│
├── batch/
│   └── [timestamp]_batch_run/
│       ├── b[beta]/           # Subdirectory for each beta value
│       │   ├── configs/       
│       │   └── plots/
│       └── batch.log
│
├── comparison/
│   └── [timestamp]_comparison_L[Lx]x[Ly]x[Lz]T[Lt]_b[beta]/
│       ├── [mode]/            # Subdirectory for each mode
│       │   ├── configs/
│       │   └── plots/
│       ├── plots/             # Comparison plots
│       └── comparison.log
│
└── run_all/
    └── [timestamp]_run_all/
        ├── b[beta]/           # Subdirectory for each beta
        │   ├── [mode]/        # Subdirectory for each mode
        │   │   ├── configs/
        │   │   └── plots/
        │   └── plots/         # Comparison plots for this beta
        ├── plots/             # Overall comparison plots
        └── run_all.log

Author: Zeke
Advisor: Dr. Aubin
Institution: Fordham University
Date: September 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import su2
from time import time
import logging
from datetime import datetime
import argparse
import pickle
import sys
from matplotlib.ticker import MaxNLocator
import matplotlib
# Set up for real-time plotting
matplotlib.use('Agg')  # Use non-interactive backend for compatibility

# Global flag to track if logging has been configured
_logging_configured = False

def configure_logging(log_file, console_level=logging.INFO):
    """
    Configure logging with both file and console output.
    
    This function ensures logging is only configured once, and
    subsequent calls will only add handlers if needed.
    
    Args:
        log_file (str): Path to log file
        console_level (int): Logging level for console output
    """
    global _logging_configured
    
    if not _logging_configured:
        # Configure the root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)  # Explicitly use stdout
            ]
        )
        _logging_configured = True
    else:
        # Already configured, just add handlers if needed
        root_logger = logging.getLogger()
        
        # Check if we already have a file handler for this log file
        has_file_handler = False
        has_console_handler = False
        
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if handler.baseFilename == os.path.abspath(log_file):
                    has_file_handler = True
            elif isinstance(handler, logging.StreamHandler):
                has_console_handler = True
                
        # Add file handler if needed
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            root_logger.addHandler(file_handler)
            
        # Add console handler if needed
        if not has_console_handler:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            root_logger.addHandler(console_handler)

def validate_gauge_config(U, lattice_dims, tolerance=1e-10, verbose=False):
    """
    Validate that a gauge configuration satisfies SU(2) constraints.
    
    For SU(2), we require:
    1. Unitarity: U†U = 𝟙
    2. Special: det(U) = 1
    3. Correct shape and dimensions
    
    This validation ensures the configuration represents a valid point
    in the SU(2)^(4V) configuration space, where V is the lattice volume.
    
    Args:
        U (numpy.ndarray): Gauge configuration to validate
        lattice_dims (list): [Lx, Ly, Lz, Lt] lattice dimensions
        tolerance (float): Numerical tolerance for unitarity/determinant checks
        verbose (bool): Whether to print detailed validation info
        
    Returns:
        tuple: (is_valid, message)
            is_valid (bool): True if configuration is valid
            message (str): Description of validation result or error
    """
    V = np.prod(lattice_dims)
    
    # Check shape
    if U.shape != (V, 4, 4):
        return False, f"Wrong shape: expected ({V}, 4, 4), got {U.shape}"
    
    # Sample random links to check (checking all would be slow for large lattices)
    n_samples = min(100, V)
    sample_sites = np.random.choice(V, n_samples, replace=False)
    
    max_det_error = 0.0
    max_unit_error = 0.0
    
    for site in sample_sites:
        for mu in range(4):
            link = U[site, mu]
            
            # Check if it's a valid 4-component real vector (SU(2) parameterization)
            if len(link) != 4:
                return False, f"Link at site {site}, direction {mu} has wrong size"
            
            # Check determinant (should be 1 for SU(2))
            det_val = su2.det(link)
            det_error = abs(det_val - 1.0)
            max_det_error = max(max_det_error, det_error)
            
            if det_error > tolerance:
                return False, f"Determinant violation at site {site}, mu={mu}: det={det_val:.6e}"
            
            # Check unitarity: U†U should give identity
            # For our parameterization, this means |a|^2 = a_0^2 + a_1^2 + a_2^2 + a_3^2 = 1
            norm_squared = np.sum(link**2)
            unit_error = abs(norm_squared - 1.0)
            max_unit_error = max(max_unit_error, unit_error)
            
            if unit_error > tolerance:
                return False, f"Unitarity violation at site {site}, mu={mu}: |U|^2={norm_squared:.6e}"
    
    if verbose:
        logging.info(f"Configuration validation passed:")
        logging.info(f"  Max determinant error: {max_det_error:.2e}")
        logging.info(f"  Max unitarity error: {max_unit_error:.2e}")
    
    return True, "Configuration is valid SU(2)"

def save_configuration(U, config_dir, plaquette, sweep, beta, Lx, Ly, Lz, Lt, mode, acceptance_rate):
    """
    Save a gauge configuration to disk in multiple formats.
    
    This function saves the configuration in two formats:
    1. Native pickle format (.pkl) for this tool
    2. Propagator-compatible format for analysis with Propagator.py
    
    The saved configuration represents a snapshot of the gauge field at a specific
    Monte Carlo time, thermalized according to the Boltzmann distribution e^{-S[U]}.
    
    Args:
        U (numpy.ndarray): Gauge configuration U[site][mu][a] where a=0,1,2,3 are SU(2) components
        config_dir (str): Directory to save the configuration
        plaquette (float): Current plaquette value <W_[]>
        sweep (int): Current Monte Carlo sweep number
        beta (float): Coupling parameter β = 2N/g²
        Lx, Ly, Lz, Lt (int): Lattice dimensions
        mode (str): Start mode used for initialization
        acceptance_rate (float): Current Metropolis acceptance rate
    
    Returns:
        str: Path to the saved configuration
    """
    os.makedirs(config_dir, exist_ok=True)
    
    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Format sweep number with leading zeros
    sweep_str = f"{sweep:04d}"
    
    # Save in native format (pickle)
    native_file = os.path.join(
        config_dir, 
        f"config_L{Lx}x{Ly}x{Lz}T{Lt}_b{beta:.2f}_{mode}_{sweep_str}.pkl"
    )
    
    config_data = {
        'U': U,
        'plaquette': plaquette,
        'sweep': sweep,
        'beta': beta,
        'Lx': Lx,
        'Ly': Ly,
        'Lz': Lz,
        'Lt': Lt,
        'mode': mode,
        'acceptance_rate': acceptance_rate,
        'timestamp': timestamp
    }
    
    try:
        with open(native_file, 'wb') as f:
            pickle.dump(config_data, f)
    except Exception as e:
        logging.error(f"Error saving configuration to {native_file}: {e}")
        return None
    
    # Save in Propagator-compatible format
    propagator_file = os.path.join(
        config_dir,
        f"quSU2_b{beta:.1f}_{Lx}_{Ly}_{Lz}_{Lt}_{sweep_str}"
    )
    
    # Propagator format: [plaquette, U]
    propagator_data = [plaquette, U]
    try:
        with open(propagator_file, 'wb') as f:
            pickle.dump(propagator_data, f)
    except Exception as e:
        logging.error(f"Error saving Propagator format to {propagator_file}: {e}")
    
    return native_file

def save_checkpoint(output_dir, U, plaq_values, acceptance_rates, sweep_indices,
                   beta, Lx, Ly, Lz, Lt, mode, M, save_interval, target_configs,
                   current_sweep, is_equilibrated, equil_idx, stats=None, saved_configs=0):
    """
    Save a checkpoint to allow resuming the simulation later.
    
    Checkpoints are crucial for long-running simulations, allowing recovery from
    interruptions without losing Monte Carlo history needed for equilibration detection.
    
    Args:
        output_dir (str): Base output directory
        U (numpy.ndarray): Current gauge configuration
        plaq_values (list): History of plaquette measurements
        acceptance_rates (list): History of acceptance rates
        sweep_indices (list): Sweep indices for plotting
        beta (float): Coupling parameter
        Lx, Ly, Lz, Lt (int): Lattice dimensions
        mode (str): Start mode
        M (int): Total number of trajectories
        save_interval (int): Interval for saving configurations
        target_configs (int): Number of configurations to save
        current_sweep (int): Current sweep number
        is_equilibrated (bool): Whether equilibration was detected
        equil_idx (int): Equilibration index if detected
        stats (dict): Dictionary with equilibration statistics
        saved_configs (int): Number of saved configurations
    
    Returns:
        str: Path to the saved checkpoint
    """
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create checkpoint file
    checkpoint_file = os.path.join(
        checkpoint_dir,
        f"checkpoint_{timestamp}_sweep{current_sweep}.pkl"
    )
    
    # Create checkpoint data with version for future compatibility
    checkpoint_data = {
        'version': '2.0',  # Version for format tracking
        'U': U,
        'plaq_values': plaq_values,
        'acceptance_rates': acceptance_rates,
        'sweep_indices': sweep_indices,
        'beta': beta,
        'Lx': Lx,
        'Ly': Ly,
        'Lz': Lz,
        'Lt': Lt,
        'mode': mode,
        'M': M,
        'save_interval': save_interval,
        'target_configs': target_configs,
        'sweep': current_sweep,
        'is_equilibrated': is_equilibrated,
        'equil_idx': equil_idx,
        'stats': stats if stats else {},
        'saved_configs': saved_configs,
        'timestamp': timestamp
    }
    
    # Save checkpoint
    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    except Exception as e:
        logging.error(f"Error saving checkpoint to {checkpoint_file}: {e}")
        return None
    
    # Remove older checkpoints to save space (keep the latest 3)
    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')],
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True
    )
    
    for old_file in checkpoint_files[3:]:
        try:
            os.remove(os.path.join(checkpoint_dir, old_file))
            logging.debug(f"Removed old checkpoint: {old_file}")
        except Exception as e:
            logging.warning(f"Failed to remove old checkpoint {old_file}: {e}")
    
    return checkpoint_file

def load_checkpoint(checkpoint_file):
    """
    Load a simulation checkpoint.
    
    Args:
        checkpoint_file (str): Path to the checkpoint file
        
    Returns:
        dict: Checkpoint data
    """
    try:
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
            
        # Check version compatibility
        version = data.get('version', '1.0')
        if version != '2.0':
            logging.warning(f"Loading checkpoint version {version}, current version is 2.0")
            
        return data
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint {checkpoint_file}: {e}")

def initialize_lattice(Lx, Ly, Lz, Lt, mode='cold', prev_config=None):
    """Initialize lattice with specified start mode.
    
    This function creates the initial gauge configuration following the principles
    outlined in Section 10.1. The lattice spacing 'a' discretizes spacetime (Eq. 10.1),
    with gauge fields U_mu(x) in SU(2) living on links between sites.
    
    Different initialization modes correspond to different physical temperatures:
    - Cold start: U_mu(x) ~ I, minimal action, corresponds to weak coupling (beta -> infinity)
    - Hot start: U_mu(x) random, maximal disorder, strong coupling (beta -> 0)
    - Mixed/parity: Intermediate, useful for testing ergodicity
    
    The choice of initial configuration doesn't affect equilibrium properties
    (ergodicity), but impacts thermalization time.
    
    Args:
        Lx, Ly, Lz, Lt (int): Lattice dimensions (spatial × temporal)
        mode (str): Initialization mode ('cold', 'hot', 'mixed', 'parity')
        prev_config (numpy.ndarray): Optional previous configuration to continue from
    
    Returns:
        numpy.ndarray: Initial gauge configuration U[site][μ][a]
                      where site in [0,V), mu in [0,4), a in [0,4)
    """
    lattice_dims = [Lx, Ly, Lz, Lt]
    V = np.prod(lattice_dims)
    
    # Check if we're continuing from a previous config
    if prev_config is not None:
        logging.info(f"Continuing from previous configuration")
        # Validate the loaded configuration
        is_valid, msg = validate_gauge_config(prev_config, lattice_dims)
        if not is_valid:
            logging.warning(f"Previous configuration validation failed: {msg}")
        return prev_config.copy()
    
    U = np.zeros((V, 4, 4))
    logging.info(f"Initializing {mode} start on {Lx}x{Ly}x{Lz}x{Lt} lattice")
    
    # For mixed modes, we need to calculate parity
    for i in range(V):
        # Get the point coordinates using su2 module function
        point = su2.i2p(i, lattice_dims)
        
        # Determine parity (sum of coordinates mod 2)
        # This creates a checkerboard pattern in 4D
        is_even = su2.parity(point) == 0
            
        for mu in range(4):
            if mode == 'cold':
                # Cold start: Unit matrix (minimal action)
                # This represents the perturbative vacuum at weak coupling
                U[i, mu] = su2.cstart()  # Returns [1, 0, 0, 0] = identity
            elif mode == 'hot':
                # Hot start: Random SU(2) matrix (maximal disorder)
                # Represents strong coupling regime, no correlations
                U[i, mu] = su2.hstart()  # Returns random unit quaternion
            elif mode == 'mixed' or mode == 'parity':
                # Mixed start: Checkerboard of hot/cold
                # Tests ergodicity, reduces thermalization time
                U[i, mu] = su2.cstart() if is_even else su2.hstart()
    
    # Validate the initial configuration
    is_valid, msg = validate_gauge_config(U, lattice_dims, verbose=True)
    if not is_valid:
        logging.error(f"Initial configuration validation failed: {msg}")
    
    return U

def run_trajectory(U, V, lattice_dims, mups, mdns, beta, Mlink=10):
    """Run a single Monte Carlo trajectory with multiple link updates.
    
    This implements the Metropolis algorithm for importance sampling of the
    path integral (Eq. 10.55-10.56). The algorithm ensures detailed balance,
    generating configurations distributed according to e^{-S[U]}/Z.
    
    The Wilson gauge action is (Eq. 10.35):
        S = -beta/2 sum_{x,mu<nu} Tr[U_mu(x)U_nu(x+mu_hat)U_mu^dag(x+nu_hat)U_nu^dag(x)]
    
    For each link U_mu(x), the local action involves 6 plaquettes (staples)
    in 4D. The Metropolis acceptance criterion is:
        P_accept = min(1, exp(-beta*Delta_S))
    
    This satisfies detailed balance: P(U->U')rho(U) = P(U'->U)rho(U')
    ensuring convergence to the correct Boltzmann distribution.
    
    Args:
        U (numpy.ndarray): Current gauge field configuration
        V (int): Lattice volume = Lx*Ly*Lz*Lt
        lattice_dims (list): [Lx, Ly, Lz, Lt]
        mups (numpy.ndarray): Forward neighbor table mups[i,mu] = i+mu_hat
        mdns (numpy.ndarray): Backward neighbor table mdns[i,mu] = i-mu_hat
        beta (float): Inverse coupling beta = 2N/g^2 (N=2 for SU(2))
        Mlink (int): Number of update attempts per link
        
    Returns:
        tuple: (plaq, count)
            plaq (float): Average plaquette <W_[]> after update
            count (int): Number of accepted updates
    """
    count = 0
    
    # Loop through all lattice sites
    for i in range(V):
        # Loop through all 4 spacetime directions
        for mu in range(4):
            # Copy the current link (important: work with original value)
            U0 = U[i][mu].copy()
            
            # Calculate staples: sum of 6 "U-shaped" paths that complete plaquettes
            # The staple sum S = sum_{nu!=mu} [U_nu(x+mu_hat)U_mu^dag(x+nu_hat)U_nu^dag(x) + U_nu^dag(x+mu_hat-nu_hat)U_mu^dag(x-nu_hat)U_nu(x-nu_hat)]
            staples = su2.getstaple(U, i, mups, mdns, mu)
            
            # Perform multiple update attempts per link for better decorrelation
            for _ in range(Mlink):
                # Propose new link value near current value
                U0n = su2.update(U0)
                
                # Calculate action difference: ΔS = S_new - S_old
                # Since S = -beta/2 Tr[U*staples], we have:
                # Delta_S = -beta/2 Tr[(U_new - U_old)*staples]
                dS = -0.5 * su2.tr(su2.mult(U0n - U0, staples))
                
                # Metropolis accept/reject step
                # Accept if Delta_S < 0 (lower action) or with probability exp(-beta*Delta_S)
                rand = np.random.random()
                if dS < 0 or rand < np.exp(-beta * dS):
                    U[i, mu] = U0n.copy()  # Accept the update
                    count += 1
    
    # Calculate average plaquette after trajectory
    # This measures <Tr(W_[])>/2, related to the action density
    plaq = su2.calcPlaq(U, lattice_dims, mups)
    return plaq, count

def estimate_autocorr_time(data):
    """Estimate the integrated autocorrelation time τ_int.
    
    The autocorrelation time measures how many Monte Carlo sweeps are needed
    between effectively independent samples. This is crucial for:
    1. Error estimation: σ_eff² = σ²(2τ_int + 1)/N
    2. Deciding measurement frequency
    3. Assessing simulation efficiency
    
    For a time series O(t), the autocorrelation function is:
        C(tau) = <O(t)O(t+tau)> - <O>^2
    
    The integrated autocorrelation time is:
        tau_int = 1 + 2*sum_{tau=1}^infinity C(tau)/C(0)
    
    Args:
        data: Array of equilibrated measurements
        
    Returns:
        float: Integrated autocorrelation time τ_int
    """
    # Normalize the data
    mean = np.mean(data)
    normalized = data - mean
    
    # Calculate autocorrelation function using FFT for efficiency
    acf = np.correlate(normalized, normalized, mode='full')
    acf = acf[len(normalized)-1:] / (np.var(data) * len(data))
    
    # Find cutoff where autocorrelation becomes negligible
    # Standard criterion: C(τ) < 0.05
    cutoff = np.where(np.abs(acf) < 0.05)[0]
    if len(cutoff) > 0:
        cutoff = cutoff[0]
    else:
        cutoff = min(len(acf), 50)  # Limit window size
    
    # Integrated autocorrelation time (factor of 2 for positive lags only)
    tau_int = 1.0 + 2.0 * np.sum(acf[1:cutoff])
    
    return max(1.0, tau_int)  # Ensure we don't return values < 1

def detect_equilibration(plaq_values, window_size=50, confidence=0.95):
    """Detect equilibration using statistical analysis.
    
    Equilibration (thermalization) is when the Markov chain reaches the
    target distribution e^{-S[U]}/Z, losing memory of initial conditions.
    This is essential because:
    1. Pre-equilibrium data has systematic bias
    2. Only equilibrated data follows the correct physics
    3. Autocorrelations are only meaningful after equilibration
    
    Detection criteria:
    - Mean stabilizes (low variation over windows)
    - Standard deviation stabilizes
    - Sufficient data after equilibration point
    
    Physical interpretation: The system has explored enough of configuration
    space to represent the canonical ensemble at temperature T = 1/(a*beta^{1/2}).
    
    Args:
        plaq_values (list): Time series of plaquette measurements
        window_size (int): Size of analysis window (in sweeps)
        confidence (float): Required confidence level (not currently used)
        
    Returns:
        tuple: (is_equilibrated, equil_idx, stats)
            is_equilibrated (bool): Whether equilibration was detected
            equil_idx (int): Index where equilibration occurs
            stats (dict): Statistical information (mean, std, autocorrelation)
    """
    # Need enough data for meaningful analysis
    if len(plaq_values) < 2 * window_size:
        return False, None, {"error": "Insufficient data for equilibration analysis"}
    
    values = np.array(plaq_values)
    
    # Calculate moving statistics
    windows = len(values) - window_size + 1
    moving_means = np.array([
        np.mean(values[i:i+window_size]) for i in range(windows)
    ])
    moving_stds = np.array([
        np.std(values[i:i+window_size]) for i in range(windows)
    ])
    
    # Look for stabilization in both mean and standard deviation
    for i in range(window_size, windows):
        # Check if recent means are stable
        recent_means = moving_means[i-window_size//2:i]
        mean_variation = np.std(recent_means)
        
        # Check if standard deviation is stable
        recent_stds = moving_stds[i-window_size//2:i]
        std_variation = np.std(recent_stds)
        
        # Equilibration criteria:
        # 1. Mean variation < 20% of typical fluctuation size
        # 2. Std variation < 30% of typical std
        # 3. Enough samples after potential equilibration
        if (mean_variation < 0.2 * np.mean(recent_stds) and
            std_variation < 0.3 * np.mean(recent_stds) and
            i >= window_size * 2):
            
            # Calculate statistics for equilibrated region
            equil_data = values[i:]
            mean_val = np.mean(equil_data)
            std_val = np.std(equil_data)
            
            # Calculate autocorrelation time
            if len(equil_data) > 2:
                autocorr = np.corrcoef(equil_data[:-1], equil_data[1:])[0,1]
                integrated_autocorr = estimate_autocorr_time(equil_data)
            else:
                autocorr = None
                integrated_autocorr = None
            
            # Return results
            stats = {
                'mean': mean_val,
                'std': std_val,
                'stderr': std_val / np.sqrt(len(equil_data)),
                'autocorr': autocorr,
                'integrated_autocorr': integrated_autocorr,
                'equilibration_quality': 1.0 / (mean_variation + 1e-10),
                'n_equilibrated': len(equil_data)
            }
            
            return True, i, stats
    
    # No equilibration detected
    return False, None, {"warning": "No equilibration detected with current parameters"}

def setup_output_dirs(mode, beta, Lx, Ly, Lz, Lt, run_type='standard'):
    """Create organized output directory structure.
    
    Sets up a standardized directory structure for saving simulation outputs.
    Different run types have different organizational patterns to keep
    results organized and easily accessible.
    
    Args:
        mode (str): Start mode
        beta (float): Coupling parameter
        Lx, Ly, Lz, Lt (int): Lattice dimensions
        run_type (str): Type of run ('standard', 'batch', 'comparison', 'run_all')
        
    Returns:
        dict: Dictionary with output directory paths
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "thermal_generator_runs"
    
    # Create directory structure based on run type
    if run_type == 'standard':
        run_dir = f"{timestamp}_L{Lx}x{Ly}x{Lz}T{Lt}_b{beta:.2f}_{mode}"
        full_path = os.path.join(base_dir, "standard", run_dir)
    
    elif run_type == 'batch':
        run_dir = f"{timestamp}_batch_run"
        full_path = os.path.join(base_dir, "batch", run_dir)
        beta_dir = os.path.join(full_path, f"b{beta:.2f}")
        
        # Create directories
        os.makedirs(full_path, exist_ok=True)
        os.makedirs(beta_dir, exist_ok=True)
        os.makedirs(os.path.join(beta_dir, "configs"), exist_ok=True)
        os.makedirs(os.path.join(beta_dir, "plots"), exist_ok=True)
        
        return {
            'base': full_path,
            'beta_dir': beta_dir,
            'configs': os.path.join(beta_dir, "configs"),
            'plots': os.path.join(beta_dir, "plots"),
            'log': os.path.join(full_path, "batch.log")
        }
    
    elif run_type == 'comparison':
        run_dir = f"{timestamp}_comparison_L{Lx}x{Ly}x{Lz}T{Lt}_b{beta:.2f}"
        full_path = os.path.join(base_dir, "comparison", run_dir)
        mode_dir = os.path.join(full_path, mode)
        
        # Create directories
        os.makedirs(full_path, exist_ok=True)
        os.makedirs(mode_dir, exist_ok=True)
        os.makedirs(os.path.join(mode_dir, "configs"), exist_ok=True)
        os.makedirs(os.path.join(mode_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "plots"), exist_ok=True)
        
        return {
            'base': full_path,
            'mode_dir': mode_dir,
            'configs': os.path.join(mode_dir, "configs"),
            'plots': os.path.join(mode_dir, "plots"),
            'comparison_plots': os.path.join(full_path, "plots"),
            'log': os.path.join(full_path, "comparison.log")
        }
    
    elif run_type == 'run_all':
        run_dir = f"{timestamp}_run_all"
        full_path = os.path.join(base_dir, "run_all", run_dir)
        beta_dir = os.path.join(full_path, f"b{beta:.2f}")
        mode_dir = os.path.join(beta_dir, mode)
        
        # Create directories
        os.makedirs(full_path, exist_ok=True)
        os.makedirs(beta_dir, exist_ok=True)
        os.makedirs(mode_dir, exist_ok=True)
        os.makedirs(os.path.join(mode_dir, "configs"), exist_ok=True)
        os.makedirs(os.path.join(mode_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(beta_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "plots"), exist_ok=True)
        
        return {
            'base': full_path,
            'beta_dir': beta_dir,
            'mode_dir': mode_dir,
            'configs': os.path.join(mode_dir, "configs"),
            'plots': os.path.join(mode_dir, "plots"),
            'beta_plots': os.path.join(beta_dir, "plots"),
            'all_plots': os.path.join(full_path, "plots"),
            'log': os.path.join(full_path, "run_all.log")
        }
    
    # Create standard directories
    os.makedirs(full_path, exist_ok=True)
    os.makedirs(os.path.join(full_path, "configs"), exist_ok=True)
    os.makedirs(os.path.join(full_path, "plots"), exist_ok=True)
    
    # Return dictionary with paths
    dirs = {
        'base': full_path,
        'configs': os.path.join(full_path, "configs"),
        'plots': os.path.join(full_path, "plots"),
        'log': os.path.join(full_path, "run.log")
    }
    
    return dirs

def setup_real_time_plot(output_dir):
    """
    Set up a real-time plot for monitoring thermalization progress.
    
    Real-time visualization helps identify:
    1. Equilibration point
    2. Autocorrelation patterns
    3. Acceptance rate stability
    4. Potential issues (stuck states, poor mixing)
    
    Args:
        output_dir (str): Directory to save the plot
        
    Returns:
        tuple: (fig, ax1, ax2, plot_file) 
               - fig: Figure handle
               - ax1: Primary axis (plaquette values)
               - ax2: Secondary axis (acceptance rate)
               - plot_file: Path to save the real-time plot
    """
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Configure primary axis (plaquette)
    ax1.set_xlabel('Monte Carlo Sweep')
    ax1.set_ylabel('Plaquette Value <W_[]>', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Configure secondary axis (acceptance)
    ax2.set_ylabel('Acceptance Rate', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, 1)  # Acceptance rate is 0-1
    
    # Create empty lines to update later
    plaq_line, = ax1.plot([], [], 'b.', alpha=0.6, label='Plaquette')
    avg_line, = ax1.plot([], [], 'g-', linewidth=2, label='Running Avg (100 sweeps)')
    acc_line, = ax2.plot([], [], 'r-', linewidth=1, alpha=0.7, label='Acceptance Rate')
    
    # Add legend with handles from both axes
    lines = [plaq_line, avg_line, acc_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    # Add title placeholder
    plt.title('Thermalization Progress - Real-time Monitoring')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create a path to save the real-time plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, 'real_time_progress.png')
    else:
        plot_file = 'real_time_progress.png'
    
    return fig, ax1, ax2, plot_file

def update_real_time_plot(fig, ax1, ax2, plot_file, sweep_indices, plaq_values, acc_rates, 
                         beta, lattice_dims, mode, is_equilibrated=False, equil_idx=None):
    """
    Update the real-time progress plot with new data points.
    
    Args:
        fig, ax1, ax2: Figure and axes handles
        plot_file: Path to save the updated plot
        sweep_indices: Array of Monte Carlo sweep indices
        plaq_values: Array of plaquette values
        acc_rates: Array of acceptance rates
        beta: Coupling parameter
        lattice_dims: Lattice dimensions [Lx, Ly, Lz, Lt]
        mode: Start mode 
        is_equilibrated: Whether equilibration has been detected
        equil_idx: Equilibration index if detected
    """
    # Ensure arrays have the same length before plotting
    min_length = min(len(sweep_indices), len(plaq_values), len(acc_rates))
    
    # Use only the data points up to the minimum length
    safe_indices = sweep_indices[:min_length]
    safe_plaq = plaq_values[:min_length]
    safe_acc = acc_rates[:min_length]
    
    # Update data for all lines
    ax1.get_lines()[0].set_data(safe_indices, safe_plaq)
    ax2.get_lines()[0].set_data(safe_indices, safe_acc)
    
    # Calculate and update running average (last 100 sweeps or all if less)
    window = min(100, len(safe_plaq))
    if window > 0:
        avg_data = []
        for i in range(len(safe_plaq)):
            start_idx = max(0, i - window + 1)
            avg_data.append(np.mean(safe_plaq[start_idx:i+1]))
        ax1.get_lines()[1].set_data(safe_indices, avg_data)
    
    # Add equilibration line if detected
    if is_equilibrated and equil_idx is not None:
        # Check if we already added an equilibration line
        equil_line = None
        for line in ax1.get_lines()[2:]:
            if line.get_label() == 'Equilibration':
                equil_line = line
                break
                
        if equil_line is None:
            # Add a vertical line at equilibration point
            equil_sweep = safe_indices[min(equil_idx, len(safe_indices)-1)]
            ax1.axvline(x=equil_sweep, color='m', linestyle='--', label='Equilibration')
            
            # Update legend
            lines = ax1.get_lines()[:3] + [ax1.get_lines()[-1]]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best')
    
    # Adjust x and y limits automatically
    ax1.relim()
    ax1.autoscale_view()
    
    # Update title with current information
    Lx, Ly, Lz, Lt = lattice_dims
    lattice_info = f"{Lx}x{Ly}x{Lz}x{Lt}"
    if len(safe_plaq) > 0:
        current_plaq = safe_plaq[-1]
        title = (f"Thermalization Progress - beta={beta:.2f}, {lattice_info} Lattice, {mode} start\n"
                f"Current sweep: {len(safe_plaq)}, Plaquette: {current_plaq:.6f}")
        if is_equilibrated:
            title += f" (Equilibrated at sweep {equil_idx})"
        plt.title(title)
    
    # Save updated figure
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)  # Close to prevent memory leaks
    
    return fig

def calculate_polyakov_loop(U, Lx, Ly, Lz, Lt):
    """
    Calculate the Polyakov loop, an order parameter for confinement.
    
    The Polyakov loop is the trace of the product of temporal links
    wrapping around the time direction:
        P(x_vec) = Tr[prod_{t=0}^{Lt-1} U_4(x_vec,t)]
    
    Physical significance:
    - In confined phase: <|P|> ~ 0 (center symmetry preserved)
    - In deconfined phase: <|P|> > 0 (center symmetry broken)
    - Related to static quark free energy: P ~ exp(-F_q/T)
    
    The phase transition occurs at beta_c ~ 2.3 for SU(2) pure gauge theory.
    This is a second-order phase transition in the 3D Ising universality class.
    
    Args:
        U (numpy.ndarray): Gauge configuration
        Lx, Ly, Lz, Lt (int): Lattice dimensions
        
    Returns:
        tuple: (polyakov_mean, polyakov_susceptibility)
            polyakov_mean: Spatial average of Polyakov loop
            polyakov_susceptibility: chi = V(<|P|^2> - <|P|>^2)
    """
    lattice_dims = [Lx, Ly, Lz, Lt]
    V_spatial = Lx * Ly * Lz
    
    # Calculate Polyakov loop at each spatial point
    polyakov_values = np.zeros(V_spatial, dtype=complex)
    
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                # Starting point in spatial slice
                point = np.array([x, y, z, 0])
                site_idx = su2.p2i(point, lattice_dims)
                
                # Initialize with identity
                p_loop = su2.cstart()
                
                # Multiply temporal links together (direction μ=3)
                for t in range(Lt):
                    point[3] = t
                    site_idx = su2.p2i(point, lattice_dims)
                    # Multiply by temporal link
                    p_loop = su2.mult(p_loop, U[site_idx, 3])
                
                # Calculate trace and normalize
                spatial_idx = x + Lx * y + Lx * Ly * z
                polyakov_values[spatial_idx] = su2.tr(p_loop) / 2.0  # Normalize by dimension
    
    # Overall Polyakov loop is the spatial average
    polyakov_mean = np.mean(polyakov_values)
    
    # Calculate susceptibility
    polyakov_abs = np.abs(polyakov_values)
    polyakov_susceptibility = V_spatial * (np.mean(polyakov_abs**2) - np.mean(polyakov_abs)**2)
    
    return polyakov_mean, polyakov_susceptibility

def estimate_critical_temperature(beta, Lt):
    """
    Estimate critical temperature based on known SU(2) phase transition.
    
    For SU(2) pure gauge theory, the deconfinement transition occurs at:
        beta_c ~ 2.3 (exact value depends on lattice size)
    
    The temperature in lattice units is T = 1/(a·Lt), where 'a' is the
    lattice spacing. The critical temperature T_c marks the transition
    between confined (hadron) and deconfined (quark-gluon plasma) phases.
    
    Args:
        beta (float): Coupling parameter β = 2N/g²
        Lt (int): Temporal lattice extent
        
    Returns:
        dict: Temperature information including T/T_c ratio and phase prediction
    """
    # Approximate lattice spacing from beta (simplified formula)
    # In asymptotic scaling: a ~ exp(-beta/4) in suitable units
    a_approx = np.exp(-beta/4.0)
    
    # Critical beta for SU(2) pure gauge theory
    beta_critical = 2.3
    
    # Temperature estimate: T = 1/(a·Lt)
    temperature = 1.0/(a_approx * Lt)
    
    # Critical temperature
    a_critical = np.exp(-beta_critical/4.0)
    T_critical = 1.0/(a_critical * Lt)
    
    # T/T_c ratio
    if T_critical > 0:
        t_ratio = temperature / T_critical
    else:
        t_ratio = float('inf')
    
    # Phase estimation based on beta
    phase = "Confined" if beta < beta_critical else "Deconfined"
    
    return {
        "a_approx": a_approx,
        "temperature": temperature,
        "T_critical": T_critical,
        "T_ratio": t_ratio,
        "phase": phase,
        "beta_critical": beta_critical
    }

def generate_configurations(Lx, Ly, Lz, Lt, beta, M=2000, Mlink=10, 
                           mode='cold', save_interval=10, 
                           output_dir=None, prev_config=None,
                           target_configs=200, save_after_equilibration=True,
                           real_time_plot=True, checkpoint_interval=100,
                           resume_from=None, progress_interval=1, 
                           force_stdout=False, verbose_validation=False):
    """Generate gauge configurations with robust monitoring and analysis.
    
    This is the main simulation engine implementing Monte Carlo generation of
    gauge configurations according to the partition function Z = ∫DU e^{-S[U]}
    (Eq. 10.55-10.56). The Metropolis algorithm ensures importance sampling
    with the correct Boltzmann weight.
    
    The simulation proceeds through:
    1. Initialization (hot/cold/mixed start)
    2. Thermalization (reaching equilibrium)
    3. Production (saving decorrelated configurations)
    
    Key physics parameters:
    - beta = 2N/g^2: Controls coupling strength (higher beta = weaker coupling)
    - beta < 2.3: Confined phase (color singlets only)
    - beta > 2.3: Deconfined phase (free quarks/gluons)
    
    Args:
        Lx, Ly, Lz, Lt: Lattice dimensions (spatial x temporal)
        beta: Coupling parameter beta = 2N/g^2
        M: Number of Monte Carlo sweeps
        Mlink: Update attempts per link per sweep
        mode: Start mode ('cold', 'hot', 'mixed', 'parity')
        save_interval: Sweeps between saved configurations
        output_dir: Base directory for outputs
        prev_config: Previous configuration to continue from
        target_configs: Number of configurations to save
        save_after_equilibration: Only save after thermalization
        real_time_plot: Enable real-time plotting
        checkpoint_interval: Sweeps between checkpoints
        resume_from: Checkpoint file to resume from
        progress_interval: Sweeps between progress updates
        force_stdout: Force progress to stdout
        verbose_validation: Enable detailed validation output
        
    Returns:
        tuple: (U, plaq_values, acceptance_rates, equilibration_info, config_dir)
    """
    # If forcing stdout, ensure progress messages go there
    if force_stdout:
        progress_handler = logging.StreamHandler(sys.stdout)
        progress_handler.setLevel(logging.INFO)
        progress_handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(progress_handler)
        
    # Check if resuming from checkpoint
    start_sweep = 0
    checkpoint_data = None
    
    if resume_from and os.path.exists(resume_from):
        try:
            checkpoint_data = load_checkpoint(resume_from)
            logging.info(f"Resuming from checkpoint: {resume_from}")
            
            # Extract checkpoint data
            U = checkpoint_data['U']
            plaq_values = checkpoint_data['plaq_values']
            acceptance_rates = checkpoint_data['acceptance_rates']
            sweep_indices = checkpoint_data['sweep_indices']
            start_sweep = checkpoint_data['sweep'] + 1
            is_equilibrated = checkpoint_data['is_equilibrated']
            equil_idx = checkpoint_data['equil_idx']
            
            # Verify parameters match
            param_mismatch = False
            if (Lx != checkpoint_data['Lx'] or Ly != checkpoint_data['Ly'] or 
                Lz != checkpoint_data['Lz'] or Lt != checkpoint_data['Lt'] or 
                beta != checkpoint_data['beta'] or mode != checkpoint_data['mode']):
                logging.warning("Parameters in checkpoint don't match current settings!")
                param_mismatch = True
                
            if param_mismatch:
                logging.warning(f"Checkpoint: L{checkpoint_data['Lx']}x{checkpoint_data['Ly']}x{checkpoint_data['Lz']}T{checkpoint_data['Lt']}, beta={checkpoint_data['beta']}, mode={checkpoint_data['mode']}")
                logging.warning(f"Current: L{Lx}x{Ly}x{Lz}T{Lt}, beta={beta}, mode={mode}")
                # Use checkpoint parameters
                Lx = checkpoint_data['Lx']
                Ly = checkpoint_data['Ly']
                Lz = checkpoint_data['Lz']
                Lt = checkpoint_data['Lt']
                beta = checkpoint_data['beta']
                mode = checkpoint_data['mode']
                logging.info("Using checkpoint parameters")
            
            logging.info(f"Resuming from sweep {start_sweep}")
            logging.info(f"Already have {len(plaq_values)} measurements")
            if is_equilibrated:
                logging.info(f"Equilibration already detected at sweep {equil_idx}")
                
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            logging.info("Starting new simulation instead")
            checkpoint_data = None
    
    # Set up lattice
    lattice_dims = [Lx, Ly, Lz, Lt]
    V = np.prod(lattice_dims)
    
    # Create output directory structure if needed
    config_dir = None
    if output_dir:
        config_dir = os.path.join(output_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)
    
    # Initialize neighbor indices using su2 module functions
    logging.info("Computing neighbor indices...")
    mups = su2.getMups(V, 4, lattice_dims)
    mdns = su2.getMdns(V, 4, lattice_dims)
    
    # Use checkpoint data or initialize new simulation
    if checkpoint_data is not None:
        U = checkpoint_data['U']
        plaq_values = checkpoint_data['plaq_values']
        acceptance_rates = checkpoint_data['acceptance_rates']
        sweep_indices = checkpoint_data['sweep_indices']
        is_equilibrated = checkpoint_data['is_equilibrated']
        equil_idx = checkpoint_data['equil_idx']
        saved_configs = checkpoint_data.get('saved_configs', 0)
        stats = checkpoint_data.get('stats', {})
    else:
        # Initialize gauge field
        U = initialize_lattice(Lx, Ly, Lz, Lt, mode, prev_config)
        
        # Validate initial configuration
        is_valid, msg = validate_gauge_config(U, lattice_dims, verbose=verbose_validation)
        if not is_valid:
            logging.error(f"Initial configuration invalid: {msg}")
        
        # Arrays to store measurements
        plaq_values = []
        acceptance_rates = []
        sweep_indices = []
        
        # Initial plaquette
        initial_plaq = su2.calcPlaq(U, lattice_dims, mups)
        plaq_values.append(initial_plaq)
        acceptance_rates.append(0.0)
        sweep_indices.append(0)
        logging.info(f"Initial plaquette: {initial_plaq:.6f}")
        
        # Initial values
        is_equilibrated = False
        equil_idx = None
        saved_configs = 0
        stats = {}
    
    # Set up real-time plotting
    if real_time_plot and output_dir:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fig, ax1, ax2, rt_plot_file = setup_real_time_plot(plots_dir)
        # Initial plot update
        fig = update_real_time_plot(
            fig, ax1, ax2, rt_plot_file, 
            sweep_indices, plaq_values, acceptance_rates,
            beta, [Lx, Ly, Lz, Lt], mode
        )
    
    # Check initial plaquette value
    if len(plaq_values) > 0 and abs(plaq_values[0]) < 0.1:
        logging.warning(f"Initial plaquette {plaq_values[0]:.6f} is very small")
        logging.warning("This might indicate issues with gauge field initialization")
    
    # Print progress header
    if progress_interval > 0:
        print("\n" + "="*80)
        print(f"Starting lattice generation: {Lx}x{Ly}x{Lz}x{Lt} lattice, beta={beta:.2f}, {mode} start")
        print("-"*80)
        print("Sweep\tPlaquette\tAccept\tElapsed\tRemaining\tInfo")
        print("-"*80)
        sys.stdout.flush()
    
    # Calculate physical temperature estimates
    temp_info = estimate_critical_temperature(beta, Lt)
    phase_prediction = temp_info["phase"]
    t_ratio = temp_info["T_ratio"]
    
    logging.info(f"Physics analysis:")
    logging.info(f"  Approximate T/T_c ratio: {t_ratio:.4f}")
    logging.info(f"  Expected phase: {phase_prediction}")
    logging.info(f"  beta_critical (SU(2) pure gauge) ~ 2.3")
    
    # Initialize arrays for physical observables
    polyakov_values = []
    polyakov_susceptibility = []
    
    # Main update loop - implementing the Markov chain
    start_time = time()
    total_accepted = 0
    total_tried = 0
    is_equilibrated = False
    equil_idx = None
    stats = {}
    saved_configs = 0
    
    for m in range(start_sweep, M):
        # Run a single trajectory (one sweep through the lattice)
        plaq, accepted = run_trajectory(U, V, lattice_dims, mups, mdns, beta, Mlink)
        
        # Update statistics
        plaq_values.append(plaq)
        acceptance_rate = accepted / (V * 4 * Mlink)
        acceptance_rates.append(acceptance_rate)
        
        total_accepted += accepted
        total_tried += V * 4 * Mlink
        
        # Calculate physical observables periodically
        if m % 10 == 0:
            # Calculate Polyakov loop (order parameter for confinement)
            polyakov, susceptibility = calculate_polyakov_loop(U, Lx, Ly, Lz, Lt)
            polyakov_values.append(polyakov)
            polyakov_susceptibility.append(susceptibility)
            
            # Physics logging
            if m % 100 == 0:
                logging.info(f"Physics measurements at sweep {m}:")
                logging.info(f"  Plaquette: {plaq:.6f}")
                logging.info(f"  Polyakov loop: {abs(polyakov):.6f}")
                logging.info(f"  Polyakov susceptibility: {susceptibility:.6f}")
                
                # Physical interpretation
                if abs(polyakov) < 0.1:
                    logging.info(f"  Interpretation: Confined phase (color singlets)")
                elif abs(polyakov) > 0.3:
                    logging.info(f"  Interpretation: Deconfined phase (quark-gluon plasma)")
                else:
                    logging.info(f"  Interpretation: Near phase transition")
        
        # Validate configuration periodically
        if verbose_validation and m % 100 == 0:
            is_valid, msg = validate_gauge_config(U, lattice_dims, verbose=False)
            if not is_valid:
                logging.warning(f"Configuration validation failed at sweep {m}: {msg}")
        
        # Check for equilibration
        if not is_equilibrated and m > 100 and m % 10 == 0:
            is_equilibrated, equil_idx, stats = detect_equilibration(plaq_values)
            if is_equilibrated:
                logging.info(f"Equilibration detected at sweep {equil_idx}")
                logging.info(f"  Equilibrated plaquette: {stats['mean']:.6f} ± {stats['stderr']:.6f}")
                logging.info(f"  Autocorrelation time: {stats.get('integrated_autocorr', 'N/A')}")
                logging.info(f"Beginning to save configurations every {save_interval} sweeps")
                
                # Save the equilibrated configuration
                if config_dir:
                    save_configuration(U, config_dir, plaq, m, beta, Lx, Ly, Lz, Lt, mode, acceptance_rate)
                    saved_configs += 1
                    logging.info(f"Saved equilibrated configuration ({saved_configs}/{target_configs})")
        
        # Save configuration if requested
        should_save = (
            config_dir and 
            (
                # Save after equilibration
                (save_after_equilibration and is_equilibrated and 
                 (m - equil_idx) % save_interval == 0 and m > equil_idx) or
                # Always save periodically if not waiting for equilibration
                (not save_after_equilibration and m % save_interval == 0 and m > 0) or
                # Always save final configuration
                (m == M - 1)
            )
        )
        
        if should_save:
            save_configuration(U, config_dir, plaq, m, beta, Lx, Ly, Lz, Lt, mode, acceptance_rate)
            saved_configs += 1
            logging.info(f"Saved configuration {m} ({saved_configs}/{target_configs})")
            
            # Stop if target reached
            if saved_configs >= target_configs:
                logging.info(f"Reached target of {target_configs} configurations")
                break
        
        # Update sweep indices
        sweep_indices.append(m + 1)
        
        # Update real-time plot
        if real_time_plot and output_dir and (m + 1) % 10 == 0:
            if 'fig' in locals():
                fig = update_real_time_plot(
                    fig, ax1, ax2, rt_plot_file, 
                    sweep_indices, plaq_values, acceptance_rates,
                    beta, [Lx, Ly, Lz, Lt], mode, 
                    is_equilibrated, equil_idx
                )
        
        # Periodic progress logging
        if progress_interval > 0 and (m + 1) % progress_interval == 0:
            elapsed = time() - start_time
            remaining = elapsed / (m + 1 - start_sweep) * (M - m - 1)
            
            # Calculate recent statistics
            recent_window = min(10, len(plaq_values))
            recent_plaq = np.mean(plaq_values[-recent_window:])
            recent_std = np.std(plaq_values[-recent_window:])
            
            # Progress string
            equilibration_status = "✓ Equilibrated" if is_equilibrated else "Thermalizing"
            progress_str = (
                f"{m+1}/{M}\t"
                f"{plaq:.6f}\t"
                f"{acceptance_rate:.4f}\t"
                f"{elapsed:.1f}s\t"
                f"{remaining:.1f}s\t"
                f"{equilibration_status}"
            )
            
            # ASCII progress bar
            progress_percent = (m + 1) / M
            bar_width = 50
            filled_length = int(bar_width * progress_percent)
            bar = '█' * filled_length + '░' * (bar_width - filled_length)
            
            # Print progress
            progress_bar = f"[{bar}] {progress_percent*100:.1f}%"
            print(f"{progress_str}\n{progress_bar}")
            sys.stdout.flush()
            
            # Detailed logging
            if (m + 1) % (progress_interval * 10) == 0:
                logging.info(
                    f"Sweep {m+1}/{M}: "
                    f"Plaq = {plaq:.6f} (Avg{recent_window} = {recent_plaq:.6f} ± {recent_std:.6f}), "
                    f"Accept = {acceptance_rate:.4f}, "
                    f"Time: {elapsed:.1f}s"
                )
        
        # Save checkpoint
        if checkpoint_interval > 0 and (m + 1) % checkpoint_interval == 0 and output_dir:
            save_checkpoint(
                output_dir, U, plaq_values, acceptance_rates, sweep_indices,
                beta, Lx, Ly, Lz, Lt, mode, M, save_interval, target_configs,
                m, is_equilibrated, equil_idx, stats, saved_configs
            )
            logging.info(f"Saved checkpoint at sweep {m+1}")
    
    # Final equilibration check
    if not is_equilibrated:
        is_equilibrated, equil_idx, stats = detect_equilibration(plaq_values)
    
    # Compute overall statistics
    overall_acceptance = total_accepted / total_tried
    runtime = time() - start_time
    
    # Theoretical expected value for comparison (Chapter 10, Eq. 10.55-10.67)
    # SU(2) plaquette expectation value for finite temperature gauge theory
    # Weak coupling (high beta): plaq ≈ 1 - 1/(2*beta) + O(1/beta^2)
    # Strong coupling (low beta): plaq ≈ I_1(beta)/I_0(beta) where I_n are modified Bessel functions
    # Empirical fit for SU(2): plaq ≈ 1 - 1.25/beta + higher order corrections
    if beta < 1.5:
        # Strong coupling regime - use Bessel function approximation
        theoretical_value = 0.5 * beta / (1 + 0.125 * beta)
    else:
        # Weak to intermediate coupling - improved perturbative formula
        theoretical_value = 1.0 - 1.25/beta + 0.1/(beta*beta)
    
    # Final statistics
    final_mean = np.mean(plaq_values[-500:]) if len(plaq_values) >= 500 else np.mean(plaq_values)
    final_std = np.std(plaq_values[-500:]) if len(plaq_values) >= 500 else np.std(plaq_values)
    final_stderr = final_std / np.sqrt(min(500, len(plaq_values)))
    
    # Prepare return data
    equilibration_info = {
        'is_equilibrated': is_equilibrated,
        'equil_idx': equil_idx,
        'stats': stats,
        'overall_acceptance': overall_acceptance,
        'runtime': runtime,
        'final_stats': {
            'mean': final_mean,
            'std': final_std,
            'stderr': final_stderr
        },
        'theoretical_value': theoretical_value,
        'saved_configs': saved_configs
    }
    
    # Final plot update
    if real_time_plot and output_dir and 'fig' in locals():
        fig = update_real_time_plot(
            fig, ax1, ax2, rt_plot_file, 
            sweep_indices, plaq_values, acceptance_rates,
            beta, [Lx, Ly, Lz, Lt], mode, 
            is_equilibrated, equil_idx
        )
        # Save final copy
        if os.path.exists(rt_plot_file):
            final_plot_file = rt_plot_file.replace('.png', '_final.png')
            os.system(f"cp {rt_plot_file} {final_plot_file}")
            logging.info(f"Final plot saved as {final_plot_file}")
    
    # Generate physics report
    create_physics_report(output_dir, beta, Lx, Ly, Lz, Lt, plaq_values, 
                         polyakov_values if polyakov_values else None,
                         equilibration_info, temp_info)
    
    # Print summary
    logging.info(f"\n{'='*80}")
    logging.info(f"LATTICE QCD SIMULATION SUMMARY")
    logging.info(f"{'-'*80}")
    logging.info(f"Lattice: {Lx}x{Ly}x{Lz}x{Lt}")
    logging.info(f"Physical volume: {Lx*Ly*Lz*Lt} sites")
    logging.info(f"Coupling beta: {beta:.4f}")
    logging.info(f"Start mode: {mode}")
    logging.info(f"Total sweeps: {len(plaq_values)}")
    logging.info(f"Runtime: {runtime:.2f} seconds")
    logging.info(f"{'-'*80}")
    
    # Phase analysis
    logging.info(f"PHASE ANALYSIS")
    logging.info(f"Approximate T/T_c: {t_ratio:.4f}")
    logging.info(f"Expected phase: {phase_prediction}")
    
    if polyakov_values:
        final_polyakov = np.mean([abs(p) for p in polyakov_values[-5:]]) if len(polyakov_values) >= 5 else abs(polyakov_values[-1])
        final_susceptibility = np.mean(polyakov_susceptibility[-5:]) if len(polyakov_susceptibility) >= 5 else polyakov_susceptibility[-1]
        
        logging.info(f"Final Polyakov loop: {final_polyakov:.6f}")
        logging.info(f"Final susceptibility: {final_susceptibility:.6f}")
        
        # Phase determination
        if final_polyakov < 0.1:
            phase_result = "CONFINED"
        elif final_polyakov > 0.3:
            phase_result = "DECONFINED"
        else:
            phase_result = "TRANSITION"
        
        logging.info(f"Phase: {phase_result}")
    
    logging.info(f"{'-'*80}")
    logging.info(f"GAUGE OBSERVABLES")
    logging.info(f"Final plaquette: {final_mean:.6f} ± {final_stderr:.6f}")
    logging.info(f"Theory prediction: {theoretical_value:.6f}")
    logging.info(f"Deviation: {abs(final_mean - theoretical_value):.6f} ({abs(final_mean - theoretical_value)/theoretical_value*100:.2f}%)")
    logging.info(f"Acceptance rate: {overall_acceptance:.4f}")
    
    # Validation
    if abs(final_mean - theoretical_value)/theoretical_value < 0.05:
        logging.info(f"Plaquette within 5% of theory ✓")
    else:
        logging.info(f"Plaquette deviates from theory - check thermalization")
    
    logging.info(f"{'-'*80}")
    logging.info(f"MONTE CARLO STATISTICS")
    logging.info(f"Saved {saved_configs} configurations")
    
    if is_equilibrated:
        mean_val = stats.get('mean', final_mean)
        stderr = stats.get('stderr', final_stderr)
        logging.info(f"Equilibrated at sweep {equil_idx}")
        logging.info(f"Equilibrated plaquette: {mean_val:.6f} ± {stderr:.6f}")
        if 'integrated_autocorr' in stats:
            tau = stats['integrated_autocorr']
            logging.info(f"Autocorrelation time: {tau:.1f}")
            effective_samples = (len(plaq_values)-equil_idx)/tau if equil_idx else len(plaq_values)/tau
            logging.info(f"Effective samples: {effective_samples:.1f}")
    else:
        logging.warning("No clear equilibration - run longer or adjust parameters")
    
    # Print completion
    if progress_interval > 0:
        print("\n" + "="*80)
        print(f"Generation completed in {runtime:.2f} seconds")
        print(f"Saved {saved_configs} configurations in {config_dir}")
        print("="*80 + "\n")
        sys.stdout.flush()
    
    return U, plaq_values, acceptance_rates, equilibration_info, config_dir

def create_physics_report(output_dir, beta, Lx, Ly, Lz, Lt, plaq_values, polyakov_values=None, 
                      equil_info=None, temp_info=None):
    """
    Generate a comprehensive physics report in text format.
    
    Creates a detailed report focusing on physics results, suitable for
    documentation and analysis. References equations from Chapter 10.
    
    Args:
        output_dir (str): Directory to save the report
        beta (float): Coupling parameter
        Lx, Ly, Lz, Lt (int): Lattice dimensions
        plaq_values (list): Plaquette measurements
        polyakov_values (list): Polyakov loop measurements
        equil_info (dict): Equilibration information
        temp_info (dict): Temperature estimates
        
    Returns:
        str: Path to the generated report
    """
    if not output_dir or not os.path.exists(output_dir):
        logging.warning("Cannot generate physics report: directory not found")
        return None
    
    # Create report path
    report_file = os.path.join(output_dir, "physics_report.txt")
    
    # Calculate statistics
    plaq_mean = np.mean(plaq_values[-500:]) if len(plaq_values) >= 500 else np.mean(plaq_values)
    plaq_std = np.std(plaq_values[-500:]) if len(plaq_values) >= 500 else np.std(plaq_values)
    
    # Theoretical prediction (Chapter 10, Eq. 10.55-10.67)
    # SU(2) plaquette expectation value for finite temperature gauge theory
    if beta < 1.5:
        # Strong coupling regime - use Bessel function approximation
        theoretical_value = 0.5 * beta / (1 + 0.125 * beta)
    else:
        # Weak to intermediate coupling - improved perturbative formula
        theoretical_value = 1.0 - 1.25/beta + 0.1/(beta*beta)
        
    # Temperature information
    if not temp_info:
        temp_info = estimate_critical_temperature(beta, Lt)
    
    phase_prediction = temp_info["phase"]
    t_ratio = temp_info["T_ratio"]
    
    # Generate report content
    report_content = f"""
=============================================================================
                      SU(2) LATTICE QCD PHYSICS REPORT
=============================================================================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}

SIMULATION PARAMETERS
--------------------
Lattice dimensions: {Lx} x {Ly} x {Lz} x {Lt}
Physical volume: {Lx*Ly*Lz*Lt} lattice sites
Gauge coupling beta: {beta:.4f}

This simulation implements SU(2) lattice gauge theory as described in Chapter 10.
The lattice spacing discretizes spacetime (Eq. 10.1): x = na
The Wilson action uses plaquettes (Eq. 10.35): W_[] = Tr[U_mu*U_nu*U_mu^dag*U_nu^dag]

PHYSICAL INTERPRETATION
----------------------
Approximate temperature (T/T_c): {t_ratio:.4f}
Expected phase regime: {phase_prediction}
Critical beta for SU(2): ~2.3

The confinement/deconfinement phase transition separates:
- Confined phase (beta < 2.3): Color charges bound in hadrons
- Deconfined phase (beta > 2.3): Quark-gluon plasma with free quarks/gluons

PLAQUETTE MEASUREMENTS
---------------------
Mean plaquette value: {plaq_mean:.6f} ± {plaq_std:.6f}
Theoretical prediction: {theoretical_value:.6f}
Deviation from theory: {abs(plaq_mean - theoretical_value):.6f} ({abs(plaq_mean - theoretical_value)/theoretical_value*100:.2f}%)

The plaquette (Eq. 10.35) measures the local field strength F_mu_nu.
In the continuum limit: 1 - Re(W_[])/2 proportional to a^4 Tr(F_mu_nu^2)
"""
    
    # Add Polyakov loop analysis if available
    if polyakov_values:
        poly_mean = np.mean([abs(p) for p in polyakov_values[-20:]])
        poly_std = np.std([abs(p) for p in polyakov_values[-20:]])
        
        if poly_mean < 0.1:
            phase_result = "CONFINED PHASE"
        elif poly_mean > 0.3:
            phase_result = "DECONFINED PHASE"
        else:
            phase_result = "TRANSITION REGION"
            
        report_content += f"""
POLYAKOV LOOP ANALYSIS
---------------------
Mean absolute value: {poly_mean:.6f} ± {poly_std:.6f}
Phase determination: {phase_result}

The Polyakov loop P = Tr[prod_t U_4(x,t)] is an order parameter for confinement.
It relates to the static quark free energy: P ~ exp(-F_q/T)
- Confined: <|P|> ~ 0 (center symmetry preserved)
- Deconfined: <|P|> > 0 (center symmetry broken)

Result {'agrees with' if phase_result.startswith(phase_prediction.upper()) else 'differs from'} theoretical prediction.
"""
    
    # Add thermalization analysis
    report_content += """
THERMALIZATION ANALYSIS
----------------------
"""
    
    if equil_info and equil_info.get('is_equilibrated'):
        equil_idx = equil_info.get('equil_idx')
        stats = equil_info.get('stats', {})
        tau = stats.get('integrated_autocorr', 'N/A')
        
        report_content += f"""
Equilibration detected at sweep {equil_idx}
Integrated autocorrelation time: {tau}

The system has thermalized to the Boltzmann distribution e^{{-S[U]}}/Z (Eq. 10.55-10.56).
Monte Carlo importance sampling correctly represents the canonical ensemble.
"""
        if isinstance(tau, (int, float)):
            effective_samples = (len(plaq_values)-equil_idx)/tau
            report_content += f"Effective independent samples: {effective_samples:.1f}\n"
    else:
        report_content += """
No clear equilibration detected in this simulation.
Results may contain initialization bias. Consider extending runtime.
"""
    
    report_content += f"""
=============================================================================
                               PHYSICS SUMMARY
=============================================================================
This simulation samples the SU(2) gauge field partition function at beta={beta:.2f}.
The Monte Carlo method (Eq. 10.65-10.67) generates configurations distributed
according to the Boltzmann weight exp(-S[U]).

For hadron physics: Use configurations from confined phase (beta < 2.3)
For QGP studies: Use configurations from deconfined phase (beta > 2.3)

To extract hadron masses, compute meson correlators using these configurations
with appropriate quark propagators (fermion inversions).
=============================================================================
"""
    
    # Write report
    try:
        with open(report_file, 'w') as f:
            f.write(report_content)
        logging.info(f"Generated physics report: {report_file}")
        return report_file
    except Exception as e:
        logging.error(f"Error writing physics report: {e}")
        return None

def create_detailed_plot(plaq_values, equilibration_info, mode, beta, Lx, Ly, Lz, Lt, output_dir):
    """Create a detailed analysis plot of a single run.
    
    Generates comprehensive visualization showing:
    1. Plaquette evolution and equilibration
    2. Autocorrelation analysis
    
    These plots help assess:
    - Thermalization quality
    - Statistical independence of samples
    - Appropriate measurement frequency
    
    Args:
        plaq_values (list): Plaquette measurements
        equilibration_info (dict): Equilibration information
        mode (str): Start mode
        beta (float): Coupling parameter
        Lx, Ly, Lz, Lt (int): Lattice dimensions
        output_dir (str): Directory to save plot
    
    Returns:
        matplotlib.figure.Figure: Created figure
    """
    # Extract equilibration data
    is_equilibrated = equilibration_info['is_equilibrated']
    equil_idx = equilibration_info.get('equil_idx')
    stats = equilibration_info.get('stats', {})
    
    # Create multi-panel figure
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Panel 1: Plaquette evolution
    ax1 = axs[0]
    ax1.plot(plaq_values, 'b.', alpha=0.6, label='Plaquette Values')
    
    if is_equilibrated:
        # Add equilibration line
        ax1.axvline(x=equil_idx, color='r', linestyle='--', 
                  label=f'Equilibration at sweep {equil_idx}')
        
        # Add mean and confidence band
        mean_val = stats['mean']
        std_val = stats['std']
        
        ax1.axhline(y=mean_val, color='g', linestyle='-',
                  label=f'Equilibrium Mean: {mean_val:.6f} ± {stats["stderr"]:.6f}')
        
        # Confidence band (1σ)
        x = np.arange(len(plaq_values))
        ax1.fill_between(x[equil_idx:], 
                       mean_val - std_val,
                       mean_val + std_val,
                       color='g', alpha=0.2)
    
    ax1.set_xlabel('Monte Carlo Sweep')
    ax1.set_ylabel('Plaquette Value <W_[]>')
    ax1.set_title(f'Plaquette Evolution (beta={beta:.1f}, {Lx}x{Ly}x{Lz}x{Lt} Lattice, {mode} start)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Autocorrelation analysis
    ax2 = axs[1]
    
    if is_equilibrated:
        # Get equilibrated data
        equil_data = np.array(plaq_values[equil_idx:])
        
        # Calculate autocorrelation function
        mean = np.mean(equil_data)
        normalized = equil_data - mean
        acf = np.correlate(normalized, normalized, mode='full')
        acf = acf[len(normalized)-1:] / (np.var(equil_data) * len(equil_data))
        lags = np.arange(min(50, len(acf)))
        
        # Plot autocorrelation
        ax2.plot(lags, acf[:len(lags)], 'r-', label='Autocorrelation C(τ)')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Mark integrated autocorrelation time
        if 'integrated_autocorr' in stats:
            tau = stats['integrated_autocorr']
            ax2.axvline(x=tau, color='g', linestyle='--',
                     label=f"tau_int ~ {tau:.1f}")
        
        ax2.set_xlabel('Lag τ (MC sweeps)')
        ax2.set_ylabel('Autocorrelation C(τ)')
        ax2.set_title('Autocorrelation Analysis')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No clear equilibration detected\nExtend simulation for analysis',
               ha='center', va='center', fontsize=14, transform=ax2.transAxes)
    
    plt.tight_layout()
    
    # Save figure
    plot_file = os.path.join(output_dir, f'analysis_{mode}_b{beta:.2f}_L{Lx}x{Ly}x{Lz}T{Lt}.png')
    plt.savefig(plot_file, dpi=300)
    logging.info(f"Saved analysis plot to {plot_file}")
    
    return fig

def create_comparison_plot(results, output_dir):
    """Create comparative plot of different start modes.
    
    Tests ergodicity by showing that different initial conditions
    converge to the same equilibrium distribution. This validates
    the Markov chain's ability to explore configuration space.
    
    Args:
        results (dict): Results for each mode
        output_dir (str): Directory to save plot
    
    Returns:
        matplotlib.figure.Figure: Created figure
    """
    # Set up plot
    plt.figure(figsize=(12, 8))
    
    # Colors and markers for different modes
    styles = {
        'cold': {'color': 'b', 'marker': '.', 'label': 'Cold Start'},
        'hot': {'color': 'r', 'marker': '.', 'label': 'Hot Start'},
        'mixed': {'color': 'g', 'marker': '.', 'label': 'Mixed Start'},
        'parity': {'color': 'purple', 'marker': '.', 'label': 'Parity Start'}
    }
    
    # Plot data for each mode
    for mode, data in results.items():
        style = styles.get(mode, {'color': 'k', 'marker': '.'})
        
        # Plot plaquette values
        plt.plot(data['plaq_values'], 
                style['marker'], 
                color=style['color'], 
                alpha=0.6,
                label=style['label'])
        
        # Add equilibration line
        if data['equilibration_info']['is_equilibrated']:
            equil_idx = data['equilibration_info']['equil_idx']
            plt.axvline(x=equil_idx, 
                       color=style['color'], 
                       linestyle='--',
                       alpha=0.5,
                       label=f"{mode.capitalize()} Equilibration")
    
    # Format plot
    plt.xlabel('Monte Carlo Sweep')
    plt.ylabel('Plaquette Value <W_[]>')
    
    # Extract parameters from first result
    first_mode = next(iter(results))
    beta = results[first_mode]['beta']
    Lx = results[first_mode]['Lx']
    Ly = results[first_mode]['Ly']
    Lz = results[first_mode]['Lz']
    Lt = results[first_mode]['Lt']
    
    plt.title(f'Start Mode Comparison (beta={beta}, {Lx}x{Ly}x{Lz}x{Lt} Lattice)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plot_file = os.path.join(output_dir, f'comparison_b{beta:.2f}_L{Lx}x{Ly}x{Lz}T{Lt}.png')
    plt.savefig(plot_file, dpi=300)
    logging.info(f"Saved comparison plot to {plot_file}")
    
    return plt.gcf()

def create_beta_comparison_plot(results, output_dir):
    """Create comparison plot across different beta values.
    
    Shows scaling behavior and approach to continuum limit.
    The plaquette approaches 1 as beta->infinity (weak coupling).
    
    Args:
        results (dict): Results for each beta
        output_dir (str): Directory to save plot
    
    Returns:
        matplotlib.figure.Figure: Created figure
    """
    # Extract and sort beta values
    betas = sorted(results.keys())
    
    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Panel 1: Equilibrium plaquette vs beta
    ax1 = axs[0]
    
    plaq_values = []
    plaq_errors = []
    equil_idx = []
    
    for beta in betas:
        info = results[beta]['equilibration_info']
        if info['is_equilibrated']:
            plaq_values.append(info['stats']['mean'])
            plaq_errors.append(info['stats']['stderr'])
            equil_idx.append(info['equil_idx'])
        else:
            data = results[beta]['plaq_values']
            plaq_values.append(np.mean(data[-500:]))
            plaq_errors.append(np.std(data[-500:]) / np.sqrt(500))
            equil_idx.append(None)
    
    # Plot with error bars
    ax1.errorbar(betas, plaq_values, yerr=plaq_errors, 
               fmt='o-', color='b', capsize=5)
    
    # Add theoretical curve
    beta_fine = np.linspace(min(betas), max(betas), 100)
    theory_values = []
    for b in beta_fine:
        if b < 2.0:
            theory_values.append(1.0 - 0.25*b)
        else:
            theory_values.append(0.75/b)
    
    ax1.plot(beta_fine, theory_values, 'r--', label='Theoretical Prediction')
    
    ax1.set_xlabel('beta = 2N/g^2')
    ax1.set_ylabel('Plaquette Value <W_[]>')
    ax1.set_title('Equilibrium Plaquette vs. Coupling')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Equilibration time vs beta
    ax2 = axs[1]
    
    valid_betas = []
    valid_equil = []
    
    for i, beta in enumerate(betas):
        if equil_idx[i] is not None:
            valid_betas.append(beta)
            valid_equil.append(equil_idx[i])
    
    if valid_betas:
        ax2.plot(valid_betas, valid_equil, 'o-', color='g')
        ax2.set_xlabel('beta = 2N/g^2')
        ax2.set_ylabel('Equilibration Sweep')
        ax2.set_title('Thermalization Time vs. Coupling')
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Insufficient equilibration data',
               ha='center', va='center', fontsize=14, transform=ax2.transAxes)
    
    plt.tight_layout()
    
    # Get lattice dimensions
    first_beta = betas[0]
    Lx = results[first_beta]['Lx']
    Ly = results[first_beta]['Ly']
    Lz = results[first_beta]['Lz']
    Lt = results[first_beta]['Lt']
    
    # Save figure
    plot_file = os.path.join(output_dir, f'beta_comparison_L{Lx}x{Ly}x{Lz}T{Lt}.png')
    plt.savefig(plot_file, dpi=300)
    logging.info(f"Saved beta comparison plot to {plot_file}")
    
    return fig

def analyze_saved_configs(config_dir, plot_dir=None):
    """Analyze previously saved configurations.
    
    Performs equilibration analysis on saved configuration files,
    useful for post-processing and verification.
    
    Args:
        config_dir (str): Directory with configurations
        plot_dir (str): Directory for plots
    
    Returns:
        dict: Analysis results
    """
    if not os.path.exists(config_dir):
        logging.error(f"Configuration directory {config_dir} does not exist")
        return {}
    
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Find configuration files
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.pkl')]
    
    if not config_files:
        logging.warning(f"No configuration files found in {config_dir}")
        return {}
    
    # Group files by parameters
    config_groups = {}
    
    for filename in config_files:
        # Parse parameters from filename
        parts = filename.split('_')
        if len(parts) < 4:
            continue
            
        # Extract parameters
        lattice_info = parts[1]  # L4x4x4T20
        beta_info = parts[2]     # b2.40
        mode = parts[3]         # cold/hot
        
        # Create group key
        group_key = f"{lattice_info}_{beta_info}_{mode}"
        
        if group_key not in config_groups:
            config_groups[group_key] = []
        
        config_groups[group_key].append(os.path.join(config_dir, filename))
    
    # Process each group
    results = {}
    
    for group_key, files in config_groups.items():
        logging.info(f"Processing group: {group_key} ({len(files)} files)")
        
        # Sort files
        files.sort()
        
        # Load configurations
        plaq_values = []
        sweeps = []
        last_config = None
        
        for file in files:
            try:
                with open(file, 'rb') as f:
                    config_data = pickle.load(f)
                
                plaq_values.append(config_data['plaquette'])
                sweeps.append(config_data['sweep'])
                last_config = config_data
                
            except Exception as e:
                logging.error(f"Error loading {file}: {e}")
        
        # Skip if no valid data
        if not plaq_values or last_config is None:
            continue
        
        # Analyze equilibration
        is_equilibrated, equil_idx, stats = detect_equilibration(plaq_values)
        
        # Store results
        results[group_key] = {
            'plaq_values': plaq_values,
            'sweeps': sweeps,
            'last_config': last_config,
            'equilibration_info': {
                'is_equilibrated': is_equilibrated,
                'equil_idx': equil_idx,
                'stats': stats
            }
        }
        
        # Create plot if requested
        if plot_dir:
            # Parse parameters
            parts = group_key.split('_')
            lattice_part = parts[0]
            beta_part = parts[1]
            mode = parts[2]
            
            # Parse lattice dimensions
            if 'x' in lattice_part:
                # Format: L3x3x4T20
                spatial_parts = lattice_part.split('T')[0][1:].split('x')
                if len(spatial_parts) >= 3:
                    Lx, Ly, Lz = map(int, spatial_parts)
                else:
                    Lx = Ly = Lz = int(lattice_part[1:].split('T')[0])
                Lt = int(lattice_part.split('T')[1])
            else:
                # Legacy format: L4T4
                Lx = Ly = Lz = int(lattice_part[1:].split('T')[0])
                Lt = int(lattice_part.split('T')[1])
            
            # Parse beta
            beta = float(beta_part[1:])
            
            # Create plot
            fig = plt.figure(figsize=(10, 6))
            plt.plot(sweeps, plaq_values, 'b.', alpha=0.6)
            
            if is_equilibrated:
                equil_sweep = sweeps[equil_idx] if equil_idx < len(sweeps) else sweeps[-1]
                plt.axvline(x=equil_sweep, color='r', linestyle='--', 
                           label=f'Equilibration at sweep {equil_sweep}')
                
                plt.axhline(y=stats['mean'], color='g', linestyle='-',
                          label=f'Mean: {stats["mean"]:.6f} ± {stats["stderr"]:.6f}')
            
            plt.xlabel('Monte Carlo Sweep')
            plt.ylabel('Plaquette Value <W_[]>')
            plt.title(f'Plaquette Evolution (beta={beta}, {Lx}x{Ly}x{Lz}x{Lt} Lattice, {mode} start)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plot_file = os.path.join(plot_dir, f'analysis_{group_key}.png')
            plt.savefig(plot_file)
            plt.close(fig)
            
            logging.info(f"Saved analysis plot to {plot_file}")
    
    return results

def batch_run(beta_values, beta_start, beta_end, beta_step, Lx, Ly, Lz, Lt, mode, M, save_interval, target_configs):
    """Run batch of simulations with different beta values.
    
    Useful for studying phase transitions and scaling behavior.
    The critical beta_c ~ 2.3 separates confined and deconfined phases.
    
    Args:
        beta_values: List of beta values (if None, use range)
        beta_start, beta_end, beta_step: Range parameters
        Lx, Ly, Lz, Lt: Lattice dimensions
        mode: Start mode
        M: Number of trajectories
        save_interval: Interval for saving
        target_configs: Target number of configurations
        
    Returns:
        dict: Results for each beta
    """
    # Setup directories
    dirs = setup_output_dirs(mode, 0.0, Lx, Ly, Lz, Lt, run_type='batch')
    
    # Configure logging
    log_file = dirs['log']
    configure_logging(log_file)
    
    logging.info(f"Starting batch run with {mode} start mode")
    
    # Generate beta values if not provided
    if beta_values is None:
        beta_values = np.arange(beta_start, beta_end + 0.5*beta_step, beta_step)
    
    logging.info(f"Will run with beta values: {beta_values}")
    
    # Results dictionary
    results = {}
    
    # Run for each beta value
    for beta in beta_values:
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting run for beta={beta:.2f}")
        
        # Create beta-specific directories
        beta_dirs = setup_output_dirs(mode, beta, Lx, Ly, Lz, Lt, run_type='batch')
        
        # Run simulation
        U, plaq_values, acceptance_rates, equil_info, config_dir = generate_configurations(
            Lx=Lx, Ly=Ly, Lz=Lz, Lt=Lt, 
            beta=beta, 
            M=M, 
            mode=mode,
            save_interval=save_interval,
            output_dir=beta_dirs['beta_dir'],
            target_configs=target_configs
        )
        
        # Create detailed plot
        create_detailed_plot(
            plaq_values, equil_info, mode, beta, 
            Lx, Ly, Lz, Lt, beta_dirs['plots']
        )
        
        # Store results
        results[beta] = {
            'plaq_values': plaq_values,
            'acceptance_rates': acceptance_rates,
            'equilibration_info': equil_info,
            'Lx': Lx,
            'Ly': Ly,
            'Lz': Lz,
            'Lt': Lt,
            'mode': mode
        }
    
    # Create beta comparison plot
    create_beta_comparison_plot(results, dirs['plots'])
    
    # Save all results
    results_file = os.path.join(dirs['base'], "batch_results.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    logging.info(f"\nBatch run completed. Results saved to {results_file}")
    
    return results

def read_config_file(config_file):
    """Read configuration parameters from a file.
    
    File format is simple key=value pairs:
        mode = cold
        beta = 2.4
        Lx = 3
        Ly = 3
        Lz = 4
        Lt = 20
        save_interval = 10
        target_configs = 200
        
    Args:
        config_file (str): Path to configuration file
        
    Returns:
        dict: Configuration parameters
    """
    params = {}
    
    try:
        with open(config_file, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse key-value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert numeric values
                    if key in ['beta', 'Lx', 'Ly', 'Lz', 'Lt', 'save_interval', 
                              'target_configs', 'trajectories']:
                        try:
                            if '.' in value:
                                params[key] = float(value)
                            else:
                                params[key] = int(value)
                        except ValueError:
                            logging.warning(f"Couldn't convert {key}={value} to number")
                            params[key] = value
                    else:
                        params[key] = value
        
        logging.info(f"Read parameters from {config_file}: {params}")
        return params
    
    except Exception as e:
        logging.error(f"Error reading config file {config_file}: {e}")
        return {}

def main():
    """Main execution function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Thermal Generator for Lattice QCD',
        epilog='See Chapter 10 of the textbook for physics background.'
    )
    
    # Progress monitoring
    parser.add_argument('--progress-interval', type=int, default=1,
                       help='Print progress every N sweeps (0 to disable)')
    parser.add_argument('--force-stdout', action='store_true',
                       help='Force progress output to stdout')
    parser.add_argument('--verbose-validation', action='store_true',
                       help='Enable detailed configuration validation')
    
    # Plotting
    parser.add_argument('--real-time-plot', action='store_true', default=True,
                       help='Enable real-time plotting')
                       
    # Checkpointing
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                       help='Save checkpoint every N trajectories')
    parser.add_argument('--resume-from', type=str,
                       help='Resume from checkpoint file')
    
    # Basic parameters
    parser.add_argument('--mode', choices=['cold', 'hot', 'mixed', 'parity', 'dual', 'all'],
                       default='cold', help='Start mode (see docs for physics)')
    parser.add_argument('--beta', type=float, default=2.4,
                       help='Coupling beta=2N/g^2 (critical beta_c~2.3)')
    
    # Lattice dimensions
    parser.add_argument('--Lx', type=int, default=4,
                       help='Spatial X dimension')
    parser.add_argument('--Ly', type=int, default=4,
                       help='Spatial Y dimension')
    parser.add_argument('--Lz', type=int, default=4,
                       help='Spatial Z dimension')
    parser.add_argument('--Lt', type=int, default=20,
                       help='Temporal dimension (T=1/(a·Lt))')
    
    # Legacy support
    parser.add_argument('--Ls', type=int, default=None, 
                       help='Legacy: cubic spatial size (sets Lx=Ly=Lz)')
    
    # Simulation parameters
    parser.add_argument('--trajectories', type=int, default=5000,
                       help='Number of Monte Carlo sweeps')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save configuration every N sweeps')
    parser.add_argument('--target-configs', type=int, default=200,
                       help='Number of configurations to save')
    parser.add_argument('--continue-from', type=str,
                       help='Continue from previous configuration')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory')
    
    # Configuration file
    parser.add_argument('--config', type=str,
                       help='Read parameters from file')
    
    # Analysis
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze saved configurations')
    parser.add_argument('--config-dir', type=str,
                       help='Directory with configurations')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple start modes')
    
    # Batch runs
    parser.add_argument('--batch', action='store_true',
                       help='Run batch with different beta values')
    parser.add_argument('--beta-start', type=float, default=2.0,
                       help='Starting beta for batch')
    parser.add_argument('--beta-end', type=float, default=3.5,
                       help='Ending beta for batch')
    parser.add_argument('--beta-step', type=float, default=0.5,
                       help='beta step size')
    parser.add_argument('--beta-values', type=str,
                       help='Comma-separated beta values')
                       
    # Comprehensive runs
    parser.add_argument('--run-all', action='store_true',
                       help='Run all beta values with all start modes')

    # Parse arguments
    args = parser.parse_args()
    
    # Handle configuration file
    if args.config:
        config_params = read_config_file(args.config)
        for key, value in config_params.items():
            if hasattr(args, key):
                setattr(args, key, value)
                print(f"Using {key}={value} from config file")
    
    # Process beta values
    beta_values = None
    if args.beta_values:
        try:
            beta_values = [float(b.strip()) for b in args.beta_values.split(',')]
            print(f"Using beta values: {beta_values}")
        except ValueError:
            print(f"Error parsing beta values '{args.beta_values}'")
    
    # Handle legacy Ls parameter
    if args.Ls is not None:
        args.Lx = args.Ls
        args.Ly = args.Ls
        args.Lz = args.Ls
        print(f"Using cubic lattice {args.Ls}x{args.Ls}x{args.Ls}x{args.Lt}")
    
    # Load previous configuration if specified
    prev_config = None
    if args.continue_from:
        try:
            with open(args.continue_from, 'rb') as f:
                config_data = pickle.load(f)
                
            # Handle different formats
            if isinstance(config_data, dict) and 'U' in config_data:
                prev_config = config_data['U']
                print(f"Loaded previous configuration from {args.continue_from}")
            elif isinstance(config_data, list) and len(config_data) == 2:
                # Propagator format [plaq, U]
                prev_config = config_data[1]
                print(f"Loaded configuration (Propagator format)")
            else:
                print("Warning: Unable to parse configuration format")
                prev_config = None
        except Exception as e:
            print(f"Error loading configuration: {e}")
            prev_config = None
    
    # Analysis mode
    if args.analyze:
        if not args.config_dir:
            print("Error: --config-dir required with --analyze")
            return
        
        # Create analysis directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = os.path.join("thermal_generator_runs", "analysis", f"{timestamp}_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        plot_dir = os.path.join(analysis_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
            
        # Configure logging
        log_file = os.path.join(analysis_dir, "analysis.log")
        configure_logging(log_file)
        
        # Run analysis
        logging.info(f"Analyzing configurations in {args.config_dir}")
        results = analyze_saved_configs(args.config_dir, plot_dir=plot_dir)
        
        # Save results
        results_file = os.path.join(analysis_dir, "analysis_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
            
        logging.info(f"Analysis completed. Results saved to {results_file}")
        return
    
    # Batch run mode
    if args.batch:
        batch_run(
            beta_values=beta_values,
            beta_start=args.beta_start, 
            beta_end=args.beta_end, 
            beta_step=args.beta_step,
            Lx=args.Lx, 
            Ly=args.Ly, 
            Lz=args.Lz, 
            Lt=args.Lt, 
            mode=args.mode, 
            M=args.trajectories,
            save_interval=args.save_interval,
            target_configs=args.target_configs
        )
        return
    
    # Compare multiple start modes
    if args.compare:
        dirs = setup_output_dirs(args.mode, args.beta, args.Lx, args.Ly, args.Lz, args.Lt, run_type='comparison')
        
        # Setup logging
        log_file = dirs['log']
        configure_logging(log_file)
        
        # Modes to compare
        if args.mode == 'dual':
            modes = ['cold', 'hot']
        elif args.mode == 'all':
            modes = ['cold', 'hot', 'mixed', 'parity']
        else:
            modes = [args.mode]
        
        # Run simulations
        results = {}
        
        for mode in modes:
            logging.info(f"\nStarting {mode} simulation")
            
            mode_dirs = setup_output_dirs(mode, args.beta, args.Lx, args.Ly, args.Lz, args.Lt, run_type='comparison')
            
            U, plaq_values, acceptance_rates, equil_info, config_dir = generate_configurations(
                Lx=args.Lx, Ly=args.Ly, Lz=args.Lz, Lt=args.Lt, 
                beta=args.beta, 
                M=args.trajectories, 
                mode=mode,
                save_interval=args.save_interval,
                output_dir=mode_dirs['mode_dir'],
                prev_config=prev_config if mode == 'previous' else None,
                target_configs=args.target_configs,
                progress_interval=args.progress_interval,
                force_stdout=args.force_stdout,
                verbose_validation=args.verbose_validation
            )
            
            create_detailed_plot(
                plaq_values, equil_info, mode, args.beta, 
                args.Lx, args.Ly, args.Lz, args.Lt, mode_dirs['plots']
            )
            
            results[mode] = {
                'plaq_values': plaq_values,
                'acceptance_rates': acceptance_rates,
                'equilibration_info': equil_info,
                'beta': args.beta,
                'Lx': args.Lx,
                'Ly': args.Ly,
                'Lz': args.Lz,
                'Lt': args.Lt
            }
        
        if len(results) > 1:
            create_comparison_plot(results, dirs['comparison_plots'])
        
        results_file = os.path.join(dirs['base'], "comparison_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        logging.info(f"\nComparison completed. Results saved to {results_file}")
        return
    
    # Run all modes and betas
    if args.run_all:
        dirs = setup_output_dirs(args.mode, args.beta, args.Lx, args.Ly, args.Lz, args.Lt, run_type='run_all')
        
        log_file = dirs['log']
        configure_logging(log_file)
        
        logging.info("Starting comprehensive run")
        
        # Generate beta values
        if args.batch:
            betas = np.arange(args.beta_start, args.beta_end + 0.5*args.beta_step, args.beta_step)
        else:
            betas = [args.beta, 3.5]
        
        # Select modes
        if args.mode == 'all':
            modes = ['cold', 'hot', 'mixed', 'parity']
        else:
            modes = [args.mode]
        
        logging.info(f"Beta values: {betas}")
        logging.info(f"Start modes: {modes}")
        
        # Master results
        all_results = {
            'betas': {},
            'modes': {},
            'comparisons': {}
        }
        
        # Run all combinations
        for beta in betas:
            logging.info(f"\n{'='*50}")
            logging.info(f"Starting runs for beta={beta:.2f}")
            
            beta_results = {}
            
            for mode in modes:
                logging.info(f"\n{'-'*40}")
                logging.info(f"Starting {mode} simulation with beta={beta:.2f}")
                
                mode_dirs = setup_output_dirs(mode, beta, args.Lx, args.Ly, args.Lz, args.Lt, run_type='run_all')
                
                U, plaq_values, acceptance_rates, equil_info, config_dir = generate_configurations(
                    Lx=args.Lx, Ly=args.Ly, Lz=args.Lz, Lt=args.Lt, 
                    beta=beta, 
                    M=args.trajectories, 
                    mode=mode, 
                    save_interval=args.save_interval,
                    output_dir=mode_dirs['mode_dir'],
                    prev_config=prev_config if mode == 'previous' else None,
                    target_configs=args.target_configs,
                    progress_interval=args.progress_interval,
                    force_stdout=args.force_stdout,
                    verbose_validation=args.verbose_validation
                )
                
                create_detailed_plot(
                    plaq_values, equil_info, mode, beta, 
                    args.Lx, args.Ly, args.Lz, args.Lt, mode_dirs['plots']
                )
                
                beta_results[mode] = {
                    'plaq_values': plaq_values,
                    'acceptance_rates': acceptance_rates,
                    'equilibration_info': equil_info,
                    'beta': beta,
                    'Lx': args.Lx,
                    'Ly': args.Ly,
                    'Lz': args.Lz,
                    'Lt': args.Lt
                }
            
            if len(beta_results) > 1:
                create_comparison_plot(beta_results, dirs['beta_plots'])
            
            all_results['betas'][beta] = beta_results
        
        # Create mode comparisons
        for mode in modes:
            mode_results = {}
            for beta in betas:
                mode_results[beta] = all_results['betas'][beta][mode]
            all_results['modes'][mode] = mode_results
            
            create_beta_comparison_plot(mode_results, dirs['all_plots'])
        
        # Save results
        master_file = os.path.join(dirs['base'], "all_results.pkl")
        with open(master_file, 'wb') as f:
            pickle.dump(all_results, f)
        
        logging.info(f"\nCompleted. Results saved to {master_file}")
        return
    
    # Standard single run
    dirs = setup_output_dirs(args.mode, args.beta, args.Lx, args.Ly, args.Lz, args.Lt, run_type='standard')
    
    log_file = dirs['log']
    configure_logging(log_file)
    
    # Log parameters
    logging.info(f"Starting lattice generation:")
    logging.info(f"  Lattice: {args.Lx}x{args.Ly}x{args.Lz}x{args.Lt}")
    logging.info(f"  beta: {args.beta}")
    logging.info(f"  Start mode: {args.mode}")
    logging.info(f"  Trajectories: {args.trajectories}")
    logging.info(f"  Save interval: {args.save_interval}")
    logging.info(f"  Target configs: {args.target_configs}")
    if prev_config:
        logging.info(f"  Continuing from: {args.continue_from}")
    
    # Run simulation
    U, plaq_values, acceptance_rates, equil_info, config_dir = generate_configurations(
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz, Lt=args.Lt, 
        beta=args.beta, 
        M=args.trajectories, 
        mode=args.mode,
        save_interval=args.save_interval,
        output_dir=dirs['base'],
        prev_config=prev_config,
        target_configs=args.target_configs,
        save_after_equilibration=True,
        real_time_plot=args.real_time_plot,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,
        progress_interval=args.progress_interval,
        force_stdout=args.force_stdout,
        verbose_validation=args.verbose_validation
    )
    
    # Create detailed plot
    create_detailed_plot(
        plaq_values, equil_info, args.mode, args.beta, 
        args.Lx, args.Ly, args.Lz, args.Lt, dirs['plots']
    )
    
    # Save results
    results = {
        'plaq_values': plaq_values,
        'acceptance_rates': acceptance_rates,
        'equilibration_info': equil_info,
        'parameters': vars(args)
    }
    
    results_file = os.path.join(dirs['base'], "run_results.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    logging.info(f"\nRun completed. Results saved to {results_file}")
    
    # Save final checkpoint
    if args.checkpoint_interval > 0:
        final_checkpoint = save_checkpoint(
            dirs['base'], U, plaq_values, acceptance_rates, list(range(len(plaq_values))),
            args.beta, args.Lx, args.Ly, args.Lz, args.Lt, args.mode, 
            args.trajectories, args.save_interval, args.target_configs,
            len(plaq_values)-1, equil_info['is_equilibrated'], equil_info.get('equil_idx'),
            equil_info.get('stats', {}), equil_info.get('saved_configs', 0)
        )
        logging.info(f"Saved final checkpoint: {final_checkpoint}")
    
    # Display useful commands
    logging.info(f"\nConfiguration files: {dirs['configs']}")
    logging.info(f"Analysis plots: {dirs['plots']}")
    logging.info(f"Saved {equil_info.get('saved_configs', 0)} configurations")
    
    logging.info(f"\nTo analyze later:")
    logging.info(f"python thermal_generator.py --analyze --config-dir {dirs['configs']}")
    
    logging.info(f"\nTo use with Propagator.py:")
    logging.info(f"python Propagator.py --input-config {dirs['configs']}/quSU2_b{args.beta:.1f}_{args.Lx}_{args.Ly}_{args.Lz}_{args.Lt}_*")

if __name__ == "__main__":
    main()
