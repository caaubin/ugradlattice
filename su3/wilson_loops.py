"""
================================================================================
Wilson Loop Calculator for String Tension Extraction
================================================================================

This module computes Wilson loops on lattice QCD gauge configurations to
extract the static quark-antiquark potential and string tension, providing
direct evidence for quark confinement.

PHYSICS BACKGROUND:
-------------------
In QCD, quarks are confined - they cannot exist as free particles but are
always bound inside hadrons (mesons, baryons). The Wilson loop provides a
theoretical probe of this confinement mechanism.

THE WILSON LOOP:
---------------
A Wilson loop W(R,T) is the path-ordered product of gauge links around a
closed rectangular path of spatial extent R and temporal extent T:

    W(R,T) = Tr[U_path]  (trace over color indices)

Physically, this represents:
    - A static quark-antiquark pair created at time 0
    - Separated by distance R
    - Propagating for time T
    - Then annihilated

The expectation value decays exponentially:
    <W(R,T)> ~ exp(-V(R) × T)

where V(R) is the static quark-antiquark potential.

THE STATIC POTENTIAL:
--------------------
For a confining theory, the potential has the "Cornell" form:

    V(R) = σR - α/R + c

where:
    - σ = STRING TENSION (linear confinement term)
    - α/R = Coulomb-like term at short distances
    - c = constant (self-energy)

The linear term σR dominates at large R, meaning it costs infinite energy
to separate quarks to infinity. This is CONFINEMENT!

AREA LAW:
---------
For large Wilson loops in a confining theory:
    <W(R,T)> ~ exp(-σ × Area)

where Area = R × T. This "area law" is the signature of confinement.
(Contrast with "perimeter law" for non-confining theories.)

STRING TENSION VALUES:
---------------------
Physical QCD: σ ≈ (440 MeV)² ≈ 0.18 GeV²
In lattice units: σa² ~ 0.05 at β = 6.0

EXTRACTION METHOD:
------------------
1. Compute <W(R,T)> for various R and T
2. Extract V(R) from: V(R) = -ln[W(R,T)/W(R,T-1)] (large T)
3. Fit V(R) = σR + c (or Cornell potential for more accuracy)
4. String tension σ determines the confinement scale

REFERENCE:
----------
K.G. Wilson, "Confinement of Quarks", Phys. Rev. D 10, 2445 (1974).
M. Creutz, "Monte Carlo Study of Quantized SU(2) Gauge Theory",
Phys. Rev. D 21, 2308 (1980).

Author: Zeke Mohammed
Advisor: Dr. Aubin
Institution: Fordham University
Date: January 2026
================================================================================
"""

import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt

# Import lattice navigation functions from su2 module
# These provide periodic boundary condition handling
sys.path.insert(0, '..')  # Adjust path as needed
import su2

# ============================================================================
#  WILSON LOOP CALCULATION
# ============================================================================

def compute_wilson_loop(U, lattice_dims, R, T, start_site=0):
    """
    Compute a single Wilson loop W(R,T) starting from a given lattice site.

    The Wilson loop is the trace of the path-ordered product of gauge links
    around a closed rectangular contour in the x-t plane:

    PATH GEOMETRY:
    --------------
        (0,T) ←------ (R,T)
          |             ↑
          |  AREA=R×T   |
          |             |
          ↓             |
        (0,0) ------→ (R,0)

    The path proceeds:
    1. R steps in +x direction (spatial)
    2. T steps in +t direction (temporal)
    3. R steps in -x direction (backward spatial)
    4. T steps in -t direction (backward temporal)

    PHYSICS INTERPRETATION:
    -----------------------
    - Creates qq̄ pair at origin
    - Separates them by distance R
    - Propagates for Euclidean time T
    - Annihilates the pair

    Parameters:
    -----------
    U : ndarray, shape (V, 4, n_c, n_c)
        Gauge configuration with V = lattice volume, 4 directions, n_c colors
    lattice_dims : list of int
        [Lx, Ly, Lz, Lt] lattice dimensions
    R : int
        Spatial extent of the Wilson loop
    T : int
        Temporal extent of the Wilson loop
    start_site : int
        Starting lattice site index (default: origin)

    Returns:
    --------
    w : float
        Wilson loop value: (1/N_c) Re Tr[W]
        Normalized by number of colors for comparison across SU(N)

    GAUGE INVARIANCE:
    -----------------
    The Wilson loop is gauge invariant because it is a closed path.
    Under gauge transformation U_μ(x) → Ω(x) U_μ(x) Ω†(x+μ),
    the transformation matrices cancel around the closed loop.

    IMPLEMENTATION NOTE:
    --------------------
    We use the convention that:
    - mu=0 is the x-direction (spatial)
    - mu=3 is the t-direction (temporal)
    Other spatial directions (y,z) could also be used; averaging over
    orientations improves statistics.
    """
    Lx, Ly, Lz, Lt = lattice_dims
    n_colors = U.shape[2]  # SU(2): 2, SU(3): 3

    # Initialize Wilson loop as identity matrix in color space
    # W will accumulate the product of all link matrices around the path
    W = np.eye(n_colors, dtype=complex)

    current_site = start_site

    # -------------------------------------------------------------------------
    # STEP 1: Go R steps in x-direction (mu=0)
    # -------------------------------------------------------------------------
    # Multiplying by U_μ(x) transports color charge in the +μ direction
    for _ in range(R):
        W = W @ U[current_site, 0]  # Multiply by link matrix
        current_site = su2.mupi(current_site, 0, lattice_dims)  # Move to next site

    # -------------------------------------------------------------------------
    # STEP 2: Go T steps in t-direction (mu=3)
    # -------------------------------------------------------------------------
    for _ in range(T):
        W = W @ U[current_site, 3]
        current_site = su2.mupi(current_site, 3, lattice_dims)

    # -------------------------------------------------------------------------
    # STEP 3: Go R steps back in -x direction
    # -------------------------------------------------------------------------
    # Going backward requires U†_μ(x-μ), so we first move back then use dagger
    for _ in range(R):
        current_site = su2.mdowni(current_site, 0, lattice_dims)  # Move back first
        W = W @ U[current_site, 0].conj().T  # Multiply by U†

    # -------------------------------------------------------------------------
    # STEP 4: Go T steps back in -t direction
    # -------------------------------------------------------------------------
    for _ in range(T):
        current_site = su2.mdowni(current_site, 3, lattice_dims)
        W = W @ U[current_site, 3].conj().T

    # Return normalized trace: (1/N_c) Re Tr[W]
    # Real part is taken because W should be real for averaged configurations
    # (complex phase averages to zero due to CP symmetry)
    return np.real(np.trace(W)) / n_colors


def compute_wilson_loop_average(U, lattice_dims, R, T):
    """
    Compute Wilson loop averaged over all starting positions on the lattice.

    Translation invariance of the gauge action implies that Wilson loops
    at different positions have the same expectation value. Averaging
    over all positions reduces statistical fluctuations:

        <W(R,T)> = (1/V) Σ_x W(R,T; x)

    Parameters:
    -----------
    U : ndarray
        Gauge configuration
    lattice_dims : list of int
        Lattice dimensions
    R : int
        Spatial extent
    T : int
        Temporal extent

    Returns:
    --------
    avg_w : float
        Volume-averaged Wilson loop value

    STATISTICAL NOTE:
    -----------------
    The variance decreases as 1/V with averaging. For a 4³×8 lattice,
    this gives ~500 measurements per configuration, significantly
    reducing statistical noise.
    """
    V = np.prod(lattice_dims)  # Total lattice volume

    loop_sum = 0.0
    for site in range(V):
        loop_sum += compute_wilson_loop(U, lattice_dims, R, T, site)

    return loop_sum / V


def compute_all_wilson_loops(U, lattice_dims, R_max=None, T_max=None):
    """
    Compute Wilson loops for all R and T values up to specified maxima.

    This provides the full data needed for potential extraction.
    Returns a dictionary indexed by (R,T) tuples.

    Parameters:
    -----------
    U : ndarray
        Gauge configuration
    lattice_dims : list of int
        Lattice dimensions [Lx, Ly, Lz, Lt]
    R_max : int, optional
        Maximum spatial extent (default: Lx/2)
    T_max : int, optional
        Maximum temporal extent (default: Lt/2)

    Returns:
    --------
    loops : dict
        Dictionary mapping (R,T) → <W(R,T)>

    FINITE SIZE NOTE:
    -----------------
    R_max and T_max are limited to half the lattice extent to avoid
    wrap-around effects from periodic boundary conditions.

    COMPUTATIONAL COST:
    -------------------
    For each (R,T) combination, we compute V Wilson loops (one per site).
    Total operations ~ V × R_max × T_max × (R + T) matrix multiplications.
    This scales as O(L^7) for an L^4 lattice!
    """
    Lx, Ly, Lz, Lt = lattice_dims

    # Default: go up to half the lattice extent
    if R_max is None:
        R_max = Lx // 2
    if T_max is None:
        T_max = Lt // 2

    loops = {}

    print(f"Computing Wilson loops up to R={R_max}, T={T_max}...")

    for R in range(1, R_max + 1):
        for T in range(1, T_max + 1):
            W = compute_wilson_loop_average(U, lattice_dims, R, T)
            loops[(R, T)] = W
            print(f"  W({R},{T}) = {W:.6f}")

    return loops


# ============================================================================
#  STATIC POTENTIAL EXTRACTION
# ============================================================================

def extract_static_potential(loops, T_fit=None):
    """
    Extract static quark-antiquark potential V(R) from Wilson loops.

    The potential is extracted from the asymptotic decay of Wilson loops:

        <W(R,T)> ~ exp(-V(R) × T)    for large T

    Taking the ratio at consecutive T values:
        V(R) = -ln[W(R,T+1) / W(R,T)]

    This ratio method cancels the unknown normalization constant.

    Parameters:
    -----------
    loops : dict
        Dictionary of Wilson loops: (R,T) → <W(R,T)>
    T_fit : int, optional
        Time slice to use for extraction (default: T_max - 1)
        Using T_max-1 allows ratio with T_max.

    Returns:
    --------
    R_arr : ndarray
        Array of R values
    V_arr : ndarray
        Array of V(R) values at each R

    SYSTEMATIC EFFECTS:
    -------------------
    - Small T: Excited state contamination (higher mass states contribute)
    - Large T: Statistical noise grows (signal decays exponentially)
    - Optimal: Intermediate T where plateau in effective potential forms

    For accurate extraction, one should check that V(R) is independent of T_fit
    within statistical errors (plateau behavior).
    """
    # Find available R and T values from the loops dictionary
    R_values = sorted(set(r for r, t in loops.keys()))
    T_values = sorted(set(t for r, t in loops.keys()))

    # Default: use second-to-last T value
    if T_fit is None:
        T_fit = max(T_values) - 1

    potentials = {}

    for R in R_values:
        # Need both W(R,T_fit) and W(R,T_fit+1) for the ratio
        if (R, T_fit) in loops and (R, T_fit + 1) in loops:
            W_T = loops[(R, T_fit)]
            W_T1 = loops[(R, T_fit + 1)]

            # Only compute if both are positive (required for logarithm)
            if W_T > 0 and W_T1 > 0:
                # V(R) = -ln[W(R,T+1)/W(R,T)] = ln[W(R,T)] - ln[W(R,T+1)]
                V = -np.log(W_T1 / W_T)
                potentials[R] = V

    R_arr = np.array(list(potentials.keys()))
    V_arr = np.array(list(potentials.values()))

    return R_arr, V_arr


def fit_string_tension(R_values, V_values):
    """
    Extract string tension σ from linear fit to static potential.

    At large distances, the potential is dominated by the confining term:
        V(R) = σR + c

    A simple linear fit extracts:
        - σ (slope): string tension in lattice units
        - c (intercept): self-energy constant

    For more precision, use the full Cornell potential:
        V(R) = σR + c - α/R

    Parameters:
    -----------
    R_values : ndarray
        Spatial separation values
    V_values : ndarray
        Potential values V(R)

    Returns:
    --------
    sigma : float or None
        String tension σa² in lattice units
    const : float or None
        Constant term c

    LATTICE UNITS:
    --------------
    The returned σ is in lattice units: σa² where 'a' is the lattice spacing.
    To convert to physical units:
        σ_physical = σa² / a²

    At β=6.0 with a ≈ 0.1 fm:
        σa² ≈ 0.05 corresponds to σ ≈ (440 MeV)²

    LIMITATIONS:
    ------------
    - Need R ≥ 2 for meaningful fit (minimum 2 points)
    - Linear fit ignores short-distance Coulomb term
    - Finite-size effects if R approaches L/2
    """
    if len(R_values) < 2:
        print("Warning: Need at least 2 R values for fit")
        return None, None

    # Simple linear fit: V = σR + c
    coeffs = np.polyfit(R_values, V_values, 1)
    sigma = coeffs[0]  # Slope = string tension
    const = coeffs[1]  # Intercept = constant

    return sigma, const


# ============================================================================
#  MAIN ANALYSIS FUNCTIONS
# ============================================================================

def analyze_confinement(config_file, output_prefix='wilson'):
    """
    Perform complete confinement analysis from a single gauge configuration.

    This function:
    1. Loads gauge configuration from pickle file
    2. Computes Wilson loops W(R,T) for range of R and T
    3. Extracts static potential V(R)
    4. Fits string tension σ
    5. Creates diagnostic plots

    Parameters:
    -----------
    config_file : str
        Path to pickle file containing gauge configuration
    output_prefix : str
        Prefix for output files (plots, data)

    Returns:
    --------
    loops : dict
        Dictionary of Wilson loop values
    sigma : float
        Fitted string tension σa²
    const : float
        Fitted constant term

    OUTPUT FILES:
    -------------
    {output_prefix}_b{beta}.png : Two-panel plot showing:
        - Left: Wilson loops W(R,T) vs R for various T
        - Right: Static potential V(R) with linear fit

    USAGE EXAMPLE:
    --------------
    >>> loops, sigma, c = analyze_confinement('config_b6.00_L4x8_0001.pkl')
    >>> print(f"String tension: σa² = {sigma:.4f}")
    """
    print("=" * 70)
    print("WILSON LOOP ANALYSIS FOR STRING TENSION")
    print("=" * 70)

    # Load gauge configuration from pickle file
    with open(config_file, 'rb') as f:
        data = pickle.load(f)

    U = data['U']
    lattice_dims = [data['Lx'], data['Ly'], data['Lz'], data['Lt']]
    beta = data.get('beta', 6.0)

    print(f"Configuration: {config_file}")
    print(f"Lattice: {lattice_dims}")
    print(f"Beta: {beta}")
    print(f"Plaquette: {data.get('plaquette', 'N/A')}")

    # Compute Wilson loops
    # Limit R_max and T_max for computational efficiency
    R_max = min(lattice_dims[0] // 2, 4)
    T_max = min(lattice_dims[3] // 2, 4)

    loops = compute_all_wilson_loops(U, lattice_dims, R_max, T_max)

    # Extract static potential
    print("\n" + "-" * 50)
    print("STATIC POTENTIAL V(R):")
    print("-" * 50)

    R_arr, V_arr = extract_static_potential(loops, T_fit=T_max-1)

    for R, V in zip(R_arr, V_arr):
        print(f"  V({R}) = {V:.4f}")

    # Fit string tension
    if len(R_arr) >= 2:
        sigma, const = fit_string_tension(R_arr, V_arr)

        print("\n" + "-" * 50)
        print("STRING TENSION FIT:")
        print("-" * 50)
        print(f"Fit: V(R) = σR + c")
        print(f"String tension σa² = {sigma:.4f}")
        print(f"Constant c = {const:.4f}")

        # Expected values for reference
        print(f"\nExpected σa² at β={beta}: ~0.05")

        # =====================================================================
        # Create diagnostic plots
        # =====================================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # -----------------------------------------------------------------
        # Plot 1: Wilson loops vs R for various T
        # -----------------------------------------------------------------
        # Shows exponential decay with both R and T (area law)
        for T in range(1, T_max + 1):
            W_values = [loops.get((R, T), 0) for R in range(1, R_max + 1)]
            ax1.plot(range(1, R_max + 1), W_values, 'o-', label=f'T={T}')

        ax1.set_xlabel('R (spatial extent)', fontsize=12)
        ax1.set_ylabel('<W(R,T)>', fontsize=12)
        ax1.set_title('Wilson Loops', fontsize=14)
        ax1.legend()
        ax1.set_yscale('log')  # Log scale shows exponential decay as linear
        ax1.grid(True, alpha=0.3)

        # -----------------------------------------------------------------
        # Plot 2: Static potential with linear fit
        # -----------------------------------------------------------------
        ax2.plot(R_arr, V_arr, 'bo', markersize=10, label='Data')

        # Overlay the linear fit
        R_fit = np.linspace(min(R_arr), max(R_arr), 100)
        V_fit = sigma * R_fit + const
        ax2.plot(R_fit, V_fit, 'r--', linewidth=2, label=f'Fit: σa² = {sigma:.3f}')

        ax2.set_xlabel('R (lattice units)', fontsize=12)
        ax2.set_ylabel('V(R)', fontsize=12)
        ax2.set_title('Static Quark Potential', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = f'{output_prefix}_b{beta:.2f}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved: {plot_file}")
        plt.close()

        return loops, sigma, const

    return loops, None, None


def ensemble_wilson_analysis(config_dir, output_prefix='wilson_ensemble'):
    """
    Wilson loop analysis with ensemble averaging over multiple configurations.

    Averaging over independent gauge configurations reduces statistical errors.
    Uses standard error of the mean: σ_mean = σ / sqrt(N)

    Parameters:
    -----------
    config_dir : str
        Directory containing gauge configuration pickle files
    output_prefix : str
        Prefix for output files

    Returns:
    --------
    avg_loops : dict
        Dictionary of (mean, error) tuples for each (R,T)

    ENSEMBLE AVERAGING:
    -------------------
    For N configurations:
        <W(R,T)> = (1/N) Σ_i W_i(R,T)
        δW = σ / √N

    where σ is the standard deviation across configurations.

    ERROR ANALYSIS:
    ---------------
    The statistical error in the potential is:
        δV(R) ≈ δW / (W × T)

    Since W decays exponentially with area, errors grow rapidly for large R,T.
    This limits practical extraction to moderate loop sizes.
    """
    import glob

    # Find all configuration files
    config_files = sorted(glob.glob(f"{config_dir}/config_*.pkl"))
    n_configs = len(config_files)

    if n_configs == 0:
        print("No configurations found!")
        return

    print("=" * 70)
    print(f"ENSEMBLE WILSON LOOP ANALYSIS ({n_configs} configs)")
    print("=" * 70)

    # Load first configuration to get dimensions
    with open(config_files[0], 'rb') as f:
        data = pickle.load(f)
    lattice_dims = [data['Lx'], data['Ly'], data['Lz'], data['Lt']]
    beta = data.get('beta', 6.0)

    R_max = min(lattice_dims[0] // 2, 3)
    T_max = min(lattice_dims[3] // 2, 3)

    # Accumulate Wilson loops from all configurations
    all_loops = {}

    for cfg_idx, config_file in enumerate(config_files):
        print(f"\nConfig {cfg_idx + 1}/{n_configs}...")

        with open(config_file, 'rb') as f:
            data = pickle.load(f)
        U = data['U']

        loops = compute_all_wilson_loops(U, lattice_dims, R_max, T_max)

        # Store each configuration's values for later averaging
        for key, val in loops.items():
            if key not in all_loops:
                all_loops[key] = []
            all_loops[key].append(val)

    # =========================================================================
    # Compute ensemble averages and errors
    # =========================================================================
    print("\n" + "=" * 70)
    print("ENSEMBLE AVERAGED WILSON LOOPS:")
    print("=" * 70)

    avg_loops = {}
    for key in sorted(all_loops.keys()):
        values = np.array(all_loops[key])
        mean = np.mean(values)
        err = np.std(values) / np.sqrt(len(values))  # Standard error of mean
        avg_loops[key] = (mean, err)
        print(f"  <W{key}> = {mean:.6f} ± {err:.6f}")

    # Extract potential from averaged loops
    loops_mean = {k: v[0] for k, v in avg_loops.items()}
    R_arr, V_arr = extract_static_potential(loops_mean, T_fit=T_max-1)

    if len(R_arr) >= 2:
        sigma, const = fit_string_tension(R_arr, V_arr)
        print(f"\nString tension σa² = {sigma:.4f}")

    return avg_loops


# ============================================================================
#  COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Wilson Loop Analysis for String Tension Extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PHYSICS BACKGROUND:
    Wilson loops W(R,T) measure the static quark-antiquark potential V(R).
    For a confining theory: V(R) = σR + c, where σ is the string tension.
    Non-zero σ proves CONFINEMENT - quarks cannot be separated to infinity!

EXAMPLES:
    # Analyze single configuration
    python wilson_loops.py --config config_b6.00_L4x8_0001.pkl

    # Analyze ensemble of configurations
    python wilson_loops.py --config_dir production_configs/

    # Custom output name
    python wilson_loops.py --config config.pkl --output my_analysis
        """
    )
    parser.add_argument('--config', type=str, help='Single config file')
    parser.add_argument('--config_dir', type=str, help='Config directory for ensemble')
    parser.add_argument('--output', type=str, default='wilson', help='Output prefix')

    args = parser.parse_args()

    if args.config:
        analyze_confinement(args.config, args.output)
    elif args.config_dir:
        ensemble_wilson_analysis(args.config_dir, args.output)
    else:
        print("Provide --config or --config_dir")
        print("Run with --help for usage information")
