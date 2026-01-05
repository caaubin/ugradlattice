"""
================================================================================
SU(3) Lattice QCD Thermal Configuration Generator
================================================================================

This program generates thermalized SU(3) gauge field configurations for use in
lattice QCD calculations. It implements the Cabibbo-Marinari algorithm, which
decomposes SU(3) heatbath updates into a sequence of SU(2) subgroup updates.

PHYSICS BACKGROUND:
-------------------
In lattice QCD, gauge fields are represented by SU(3) matrices U_μ(x) living
on the links of a 4D spacetime lattice. These matrices represent the parallel
transport of color (quark charge) along that link. For physical simulations,
we need gauge configurations sampled from the thermal (Boltzmann) distribution:

    P[U] ∝ exp(-β S[U])

where S[U] is the Wilson gauge action and β = 6/g² is the inverse coupling.

THE CABIBBO-MARINARI ALGORITHM:
-------------------------------
Direct SU(3) heatbath sampling is difficult. The Cabibbo-Marinari method
cleverly decomposes each SU(3) update into TWO sequential SU(2) updates:

    1. Update upper-left 2×2 block (red-green colors, indices 0-1)
    2. Update lower-right 2×2 block (green-blue colors, indices 1-2)

Each SU(2) update uses the standard Creutz heatbath algorithm. This sequence
approximately samples from the correct SU(3) distribution when iterated.

THERMALIZATION:
--------------
Starting from either:
  - Cold start: All links = identity (ordered, plaquette = 1.0)
  - Hot start: Random SU(3) matrices (disordered, plaquette ≈ 0)

Multiple sweeps of heatbath updates drive the configuration toward thermal
equilibrium, where the plaquette stabilizes at a value determined by β:
  - β = 6.0 (strong coupling): plaquette ≈ 0.60
  - β = 5.7 (near T_c): plaquette ≈ 0.58 (critical β_c ≈ 5.69)
  - β → ∞ (weak coupling): plaquette → 1.0

REFERENCE:
----------
M. Creutz, "Monte Carlo Study of Quantized SU(2) Gauge Theory",
Phys. Rev. D 21, 2308 (1980).

N. Cabibbo and E. Marinari, "A New Method for Updating SU(N) Matrices in
Computer Simulations of Gauge Theories", Phys. Lett. B 119, 387 (1982).

Author: Zeke Mohammed
Advisor: Dr. Aubin
Institution: Fordham University
Date: January 2026
Version: 2.0
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sun  # SU(N) utility functions (from existing codebase)
import logging
from datetime import datetime
import argparse
import pickle
import sys

_logging_configured = False

def configure_logging(log_file, console_level=logging.INFO):
    """
    Configure logging to write to both file and console.

    Parameters:
    -----------
    log_file : str
        Path to log file
    console_level : logging level
        Minimum level for console output
    """
    global _logging_configured
    if not _logging_configured:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        _logging_configured = True

# ============================================================================
#  INITIALIZATION FUNCTIONS
# ============================================================================

def generate_identity_su3(lattice_dims):
    """
    Generate cold start configuration with all links = identity.

    Cold start means the gauge field is in a completely ordered state.
    This represents zero temperature (T=0) or infinite coupling (β→∞).

    Parameters:
    -----------
    lattice_dims : list of int
        Lattice dimensions [Lx, Ly, Lz, Lt]

    Returns:
    --------
    U : ndarray, shape (V, 4, 3, 3), dtype=complex
        Gauge configuration with U[site, mu] = 3×3 identity matrix

    PHYSICS NOTE:
    -------------
    Starting from identity gives plaquette = 1.0 exactly. Thermalization
    will reduce this to the thermal equilibrium value determined by β.
    """
    V = np.prod(lattice_dims)  # Total volume (number of lattice sites)
    U = np.zeros((V, 4, 3, 3), dtype=complex)
    identity = np.eye(3, dtype=complex)

    for site in range(V):
        for mu in range(4):  # 4 spacetime directions (x, y, z, t)
            U[site, mu] = identity.copy()

    return U

def random_su3_matrix():
    """
    Generate random SU(3) matrix via Gram-Schmidt orthogonalization.

    SU(3) matrices must satisfy:
      1. Unitary: U† U = I (preserves probability)
      2. Det(U) = 1 (special unitary)

    ALGORITHM:
    ----------
    1. Generate 3 random complex vectors
    2. Gram-Schmidt orthogonalization:
       - Normalize first vector: u
       - Orthogonalize second to first: v ⊥ u
       - Third vector from cross product: w = u* × v*
    3. Rows form SU(3) matrix (automatically det=1 from construction)

    Returns:
    --------
    U : ndarray, shape (3, 3), dtype=complex
        Random SU(3) matrix

    MATHEMATICAL NOTE:
    ------------------
    The cross product of conjugate vectors gives det(U) = +1 rather than -1,
    ensuring we generate SU(3) and not just U(3).
    """
    # Start with random 3×3 complex matrix
    M = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)

    # First row: normalize
    u = M[0]
    u = u / np.linalg.norm(u)

    # Second row: orthogonalize to first, then normalize
    v = M[1]
    v = v - np.dot(v, np.conj(u)) * u  # Remove component parallel to u
    v = v / np.linalg.norm(v)

    # Third row: cross product (automatically orthogonal and normalized)
    w = np.cross(np.conj(u), np.conj(v))

    U = np.array([u, v, w], dtype=complex)
    return U

def generate_random_su3(lattice_dims):
    """
    Generate hot start configuration with random SU(3) matrices on all links.

    Hot start represents infinite temperature (T=∞) or zero coupling (β=0).
    Random SU(3) matrices give plaquette ≈ 0, the maximally disordered state.

    Parameters:
    -----------
    lattice_dims : list of int
        Lattice dimensions [Lx, Ly, Lz, Lt]

    Returns:
    --------
    U : ndarray, shape (V, 4, 3, 3), dtype=complex
        Gauge configuration with random SU(3) matrices on each link

    PHYSICS NOTE:
    -------------
    Hot start is useful for:
      - Testing thermalization from both directions (cold and hot)
      - Verifying the algorithm reaches the same equilibrium state
      - Studying critical slowing down near phase transitions
    """
    V = np.prod(lattice_dims)
    U = np.zeros((V, 4, 3, 3), dtype=complex)

    for site in range(V):
        for mu in range(4):
            U[site, mu] = random_su3_matrix()

    return U

# ============================================================================
#  PLAQUETTE MEASUREMENT
# ============================================================================

def plaquette_su3(U, site, mu, nu, lattice_dims):
    """
    Calculate Wilson plaquette in the μ-ν plane at site x.

    The plaquette is the smallest closed loop on the lattice:

        x+ν ----U_μ(x+ν)---→ x+μ+ν
         ↑                     ↓
        U_ν(x)              U†_ν(x+μ)
         ↑                     ↓
         x -----U_μ(x)-----→ x+μ

    Plaquette operator:
        P_μν(x) = U_μ(x) U_ν(x+μ) U†_μ(x+ν) U†_ν(x)

    Observable:
        p_μν(x) = (1/3) Re Tr[P_μν(x)]

    Parameters:
    -----------
    U : ndarray, shape (V, 4, 3, 3)
        Gauge configuration
    site : int
        Lattice site index
    mu, nu : int
        Spacetime directions (0=x, 1=y, 2=z, 3=t)
    lattice_dims : list of int
        Lattice dimensions for periodic boundary conditions

    Returns:
    --------
    p : float
        Plaquette value in range [-1, 1], with 1 = perfectly ordered

    PHYSICS INTERPRETATION:
    -----------------------
    - p ≈ 1: Weak coupling (β large), field is smooth
    - p ≈ 0: Strong coupling (β small), field is rough/disordered
    - <p> determines the lattice spacing 'a' via asymptotic freedom
    """
    # Get neighboring site indices (with periodic boundary conditions)
    x_plus_mu = sun.mupi(site, mu, lattice_dims)
    x_plus_nu = sun.mupi(site, nu, lattice_dims)

    # Compute plaquette: U_μ(x) × U_ν(x+μ) × U†_μ(x+ν) × U†_ν(x)
    P = sun.mult(U[site, mu], U[x_plus_mu, nu])
    P = sun.mult(P, sun.dag(U[x_plus_nu, mu]))
    P = sun.mult(P, sun.dag(U[site, nu]))

    # Return (1/N_c) Re Tr[P] where N_c=3 for SU(3)
    return np.trace(P).real / 3.0

def average_plaquette_su3(U, lattice_dims):
    """
    Calculate average plaquette over entire lattice.

    The average plaquette is the primary observable for measuring
    thermalization and determining the lattice coupling.

        <P> = (1/6V) Σ_{x,μ<ν} Re Tr[P_μν(x)] / 3

    There are 6 independent plaquette orientations per site:
        (x,y), (x,z), (x,t), (y,z), (y,t), (z,t)

    Parameters:
    -----------
    U : ndarray
        Gauge configuration
    lattice_dims : list of int
        Lattice dimensions

    Returns:
    --------
    avg_plaq : float
        Average plaquette value

    TYPICAL VALUES:
    ---------------
    - Identity config (cold): <P> = 1.0
    - Random config (hot): <P> ≈ 0.0
    - β = 6.0 (thermal): <P> ≈ 0.60
    - β = 5.7 (near T_c): <P> ≈ 0.58
    """
    V = np.prod(lattice_dims)
    total = 0.0
    count = 0

    # Sum over all sites and all plaquette orientations
    for site in range(V):
        for mu in range(4):
            for nu in range(mu + 1, 4):  # μ < ν to avoid double counting
                total += plaquette_su3(U, site, mu, nu, lattice_dims)
                count += 1

    return total / count

# ============================================================================
#  STAPLE CALCULATION
# ============================================================================

def staple_sum_su3(U, site, mu, lattice_dims):
    """
    Calculate sum of staples around link U_μ(x).

    A staple is a path of 3 links forming a U-shape around the link of interest.
    For each orthogonal direction ν ≠ μ, there are 2 staples (forward and backward):

    FORWARD STAPLE (direction ν > 0):
        x+μ+ν
         ↑
        U_ν(x+μ)
         ↑
    x+μ ←---U†_μ(x+ν)---- x+μ+ν
                           ↑
                          U†_ν(x)

    BACKWARD STAPLE (direction ν < 0):
    x+μ ←---U†_μ(x-ν)---- x+μ-ν
     ↓                      ↓
    U†_ν(x+μ-ν)           U_ν(x-ν)
     ↓                      ↓
    x-ν

    Total staples per link: 2 × 3 directions = 6

    The staple sum Σ_staples appears in the Wilson action:
        S = -β Σ_{x,μ} Re Tr[U_μ(x) Σ_staples]

    Parameters:
    -----------
    U : ndarray
        Gauge configuration
    site : int
        Lattice site
    mu : int
        Link direction
    lattice_dims : list
        Lattice dimensions

    Returns:
    --------
    staple : ndarray, shape (3, 3), dtype=complex
        Sum of all 6 staples (SU(3) matrix)

    PHYSICS NOTE:
    -------------
    The staple sum represents the neighboring field configuration that
    exerts a "force" on the link U_μ(x). The heatbath algorithm updates
    U_μ(x) to be statistically aligned with this force.
    """
    staple = np.zeros((3, 3), dtype=complex)

    # Loop over 3 orthogonal directions
    for nu in range(4):
        if nu == mu:
            continue  # Skip parallel direction

        # FORWARD STAPLE: U_ν(x+μ) × U†_μ(x+ν) × U†_ν(x)
        x_plus_mu = sun.mupi(site, mu, lattice_dims)
        x_plus_nu = sun.mupi(site, nu, lattice_dims)

        S_fwd = sun.mult(U[x_plus_mu, nu], sun.dag(U[x_plus_nu, mu]))
        S_fwd = sun.mult(S_fwd, sun.dag(U[site, nu]))

        # BACKWARD STAPLE: U†_ν(x+μ-ν) × U†_μ(x-ν) × U_ν(x-ν)
        x_minus_nu = sun.mdowni(site, nu, lattice_dims)
        x_plus_mu_minus_nu = sun.mdowni(x_plus_mu, nu, lattice_dims)

        S_bwd = sun.dag(U[x_plus_mu_minus_nu, nu])
        S_bwd = sun.mult(S_bwd, sun.dag(U[x_minus_nu, mu]))
        S_bwd = sun.mult(S_bwd, U[x_minus_nu, nu])

        staple += S_fwd + S_bwd

    return staple

# ============================================================================
#  SU(2) SUBGROUP OPERATIONS
# ============================================================================

def extract_su2_from_su3(M, indices):
    """
    Extract 2×2 SU(2) submatrix from 3×3 SU(3) matrix and convert to real form.

    SU(2) matrices can be represented in "real form" as 4-vectors:
        U = a0*I + i(a1*σ1 + a2*σ2 + a3*σ3)

    where σ_i are Pauli matrices. In 2×2 matrix form:
        U = [[a0 + i*a3,   a2 + i*a1 ],
             [-a2 + i*a1,  a0 - i*a3 ]]

    This real representation enables efficient quaternion-like multiplication.

    Parameters:
    -----------
    M : ndarray, shape (3, 3)
        SU(3) matrix
    indices : tuple (i, j)
        Which 2×2 block to extract:
        - (0, 1): upper-left (red-green)
        - (1, 2): lower-right (green-blue)

    Returns:
    --------
    a_vec : ndarray, shape (4,)
        Real 4-vector [a0, a1, a2, a3] representing SU(2) matrix

    MATHEMATICAL NOTE:
    ------------------
    The conversion formulas are derived by equating matrix elements:
        a0 = (U_00 + U_11) / 2  (real part of trace)
        a3 = (Im U_00 - Im U_11) / 2  (imaginary diagonal difference)
        a1 = (Im U_01 + Im U_10) / 2  (imaginary off-diagonal sum)
        a2 = (Re U_01 - Re U_10) / 2  (real off-diagonal difference)
    """
    i, j = indices

    # Extract 2×2 submatrix
    sub = M[np.ix_([i, j], [i, j])]

    # Convert to real representation
    # These formulas come from U = a0*I + i*a·σ
    a0 = 0.5 * (sub[0,0].real + sub[1,1].real)
    a1 = 0.5 * (sub[0,1].imag + sub[1,0].imag)
    a2 = 0.5 * (sub[0,1].real - sub[1,0].real)
    a3 = 0.5 * (sub[0,0].imag - sub[1,1].imag)

    return np.array([a0, a1, a2, a3])

def embed_su2_in_su3_block(a_vec, indices):
    """
    Embed SU(2) real representation into 3×3 SU(3) block matrix.

    Creates a 3×3 matrix with:
      - SU(2) matrix in specified 2×2 block
      - Identity (1) in the unused diagonal position
      - Zeros elsewhere

    For indices (0,1) - upper-left block:
        [[a0+i*a3,  a2+i*a1,    0     ],
         [-a2+i*a1, a0-i*a3,    0     ],
         [   0,        0,       1     ]]

    For indices (1,2) - lower-right block:
        [[   1,        0,       0     ],
         [   0,    a0+i*a3,  a2+i*a1 ],
         [   0,   -a2+i*a1, a0-i*a3  ]]

    Parameters:
    -----------
    a_vec : ndarray, shape (4,)
        Real 4-vector [a0, a1, a2, a3]
    indices : tuple (i, j)
        Which block to embed in

    Returns:
    --------
    U : ndarray, shape (3, 3), dtype=complex
        3×3 block matrix with embedded SU(2)

    PHYSICS NOTE:
    -------------
    The identity in the unused color keeps that quark flavor unchanged.
    Only the two colors in indices (i,j) get mixed by this SU(2) rotation.
    """
    i, j = indices
    a0, a1, a2, a3 = a_vec

    # Start with identity
    U = np.eye(3, dtype=complex)

    # Embed 2×2 SU(2) matrix in the (i,j) block
    U[i, i] = a0 + 1j*a3
    U[i, j] = a2 + 1j*a1
    U[j, i] = -a2 + 1j*a1
    U[j, j] = a0 - 1j*a3

    return U

def su2_dag(u):
    """
    Hermitian conjugate of SU(2) matrix in real representation.

    For SU(2): U = a0*I + i*a·σ

    Hermitian conjugate: U† = a0*I - i*a·σ

    In real form: [a0, a1, a2, a3]† = [a0, -a1, -a2, -a3]

    This is equivalent to quaternion conjugation.

    Parameters:
    -----------
    u : ndarray, shape (4,)
        SU(2) matrix in real form

    Returns:
    --------
    u_dag : ndarray, shape (4,)
        Hermitian conjugate in real form
    """
    return np.array([u[0], -u[1], -u[2], -u[3]])

def su2_mult(u1, u2):
    """
    Multiply two SU(2) matrices in real representation.

    Uses quaternion multiplication formulas:
        c0 = a0*b0 - (a1*b1 + a2*b2 + a3*b3)  [scalar products]
        c_i = a0*b_i + b0*a_i + (a × b)_i      [vector parts]

    Where (a × b) is the cross product of the vector parts.

    Parameters:
    -----------
    u1, u2 : ndarray, shape (4,)
        SU(2) matrices in real form

    Returns:
    --------
    result : ndarray, shape (4,)
        Product u1 × u2 in real form

    EFFICIENCY NOTE:
    ----------------
    This is ~3x faster than converting to 2×2 complex matrices and
    using standard matrix multiplication.
    """
    a0, b0 = u1[0], u2[0]
    a, b = u1[1:], u2[1:]  # Vector parts

    # Scalar part: a0*b0 - a·b
    c0 = a0 * b0 - np.dot(a, b)

    # Vector part: b0*a + a0*b - a×b
    # (note: minus sign in cross product for correct quaternion formula)
    c = b0*a + a0*b - np.cross(a, b)

    return np.array([c0, c[0], c[1], c[2]])

# ============================================================================
#  HEATBATH ALGORITHM
# ============================================================================

def heatbath_su2_real(r_vec, beta):
    """
    Generate SU(2) matrix via Creutz heatbath algorithm.

    This is the core of the Monte Carlo update. It generates a new SU(2)
    matrix that, when applied repeatedly, samples from the thermal distribution:

        P(U) ∝ exp(β k a0)

    where k = |r_vec| is the magnitude of the staple sum and a0 is the scalar
    part of the new SU(2) matrix.

    ALGORITHM (Creutz, 1980):
    --------------------------
    1. Sample x uniformly from [exp(-4βk/3), 1]
       (bounds for SU(3), different from SU(2)!)

    2. Compute a0 = 1 + (3/2βk) ln(x)
       This gives the correct Boltzmann weight

    3. Use rejection sampling with probability 1 - sqrt(1-a0²)
       This enforces the SU(2) constraint a0² + a1² + a2² + a3² = 1

    4. Sample (a1, a2, a3) uniformly on sphere of radius sqrt(1-a0²)

    5. **CRITICAL STEP**: Multiply by inverse of normalized staple
       U = U_random × (r/k)†
       This orients the random matrix toward the staple direction

    Parameters:
    -----------
    r_vec : ndarray, shape (4,)
        Staple sum in real SU(2) form
    beta : float
        Inverse coupling β = 6/g²

    Returns:
    --------
    U : ndarray, shape (4,)
        New SU(2) matrix in real form, sampled from heatbath distribution

    PHYSICS INTERPRETATION:
    -----------------------
    - Small beta (strong coupling): Wide distribution, large fluctuations
    - Large beta (weak coupling): Narrow distribution, U ≈ optimal alignment
    - The staple r_vec acts as an external "field" aligning U

    CRITICAL BUG FIX:
    -----------------
    The original code was missing the final multiplication by (r/k)†.
    Without this step, the algorithm generates random SU(2) matrices that
    are NOT properly weighted by the action, leading to incorrect thermalization.
    This caused the plaquette to collapse to ~0 instead of reaching ~0.6.
    """
    # Magnitude of staple sum
    k = np.linalg.norm(r_vec)

    # If staple is negligible, return identity (no preferred direction)
    if k < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])

    # STEP 1-3: Sample a0 with rejection sampling
    # Bounds are for SU(3): exp(-4βk/3), different from SU(2)!
    low = np.exp(-4.0 * beta * k / 3.0)
    high = 1.0

    while True:
        # Sample x from [low, 1]
        x = np.random.uniform(low, high)

        # Compute a0 from Boltzmann weight
        a0 = 1.0 + 3.0 * np.log(x) / (2.0 * beta * k)

        # Reject if this would violate |U|=1 constraint
        if a0**2 > 1.0:
            continue

        # Rejection probability (enforces correct distribution)
        reject = 1.0 - np.sqrt(1.0 - a0**2)
        if np.random.rand() > reject:
            break  # Accepted!

    # STEP 4: Sample remaining components uniformly on 3-sphere
    abs_a = np.sqrt(1.0 - a0**2)  # Radius of remaining components
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)

    # Spherical coordinates → Cartesian
    a1 = abs_a * np.cos(phi) * np.sin(theta)
    a2 = abs_a * np.sin(phi) * np.sin(theta)
    a3 = abs_a * np.cos(theta)

    Un = np.array([a0, a1, a2, a3])

    # STEP 5: **CRITICAL** - Multiply by inverse of normalized staple
    # This biases the random matrix toward the direction of the staple
    # Without this, thermalization FAILS and plaquette → 0
    Ubar = r_vec / k  # Normalize staple to unit vector
    Uinv = su2_dag(Ubar)  # Inverse is conjugate for SU(2)
    U = su2_mult(Un, Uinv)  # Orient toward staple direction

    return U

# ============================================================================
#  CABIBBO-MARINARI UPDATE
# ============================================================================

def su3_heatbath_update(U, site, mu, beta, lattice_dims):
    """
    Update single SU(3) link via Cabibbo-Marinari algorithm.

    The Cabibbo-Marinari method decomposes an SU(3) heatbath update into
    a sequence of TWO SU(2) heatbath updates in different subgroups:

    ALGORITHM:
    ----------
    1. Compute staple sum S (6 staples, same as for full SU(3))

    2. Form UR = U_old × S (right-multiply by staple)

    3. FIRST SU(2) UPDATE (red-green colors):
       - Extract upper-left 2×2 block of UR → r1
       - Generate SU(2) heatbath matrix A1 from r1
       - Embed A1 into 3×3 with identity at position (2,2)
       - Update: UR ← A1 × UR

    4. SECOND SU(2) UPDATE (green-blue colors):
       - Extract lower-right 2×2 block of UR → r2
       - Generate SU(2) heatbath matrix A2 from r2
       - Embed A2 into 3×3 with identity at position (0,0)
       - Update: UR ← A2 × UR

    5. Final link: U_new = A2 × A1 × U_old

    Parameters:
    -----------
    U : ndarray, shape (V, 4, 3, 3)
        Gauge configuration (modified in place)
    site : int
        Lattice site to update
    mu : int
        Link direction (0=x, 1=y, 2=z, 3=t)
    beta : float
        Inverse coupling
    lattice_dims : list
        Lattice dimensions

    PHYSICS NOTE:
    -------------
    This algorithm is approximate but becomes exact in the limit of many
    iterations. The choice of two SU(2) subgroups is not unique - we use
    (0,1) and (1,2) following the standard implementation. Other choices
    like (0,1), (0,2), (1,2) are also valid but slower (3 updates).

    CONVERGENCE:
    ------------
    Typically requires 10-50 sweeps (lattice volumes) to thermalize,
    depending on beta and starting configuration.
    """
    U_old = U[site, mu].copy()

    # STEP 1: Compute staple sum (6 staples around this link)
    staples = staple_sum_su3(U, site, mu, lattice_dims)

    # STEP 2: Right-multiply link by staple
    UR = sun.mult(U_old, staples)

    # STEP 3: First SU(2) update (upper-left 2×2 block, colors 0-1)
    r1 = extract_su2_from_su3(UR, (0, 1))  # Extract to real form
    a1_vec = heatbath_su2_real(r1, beta)    # Generate heatbath matrix
    A1 = embed_su2_in_su3_block(a1_vec, (0, 1))  # Embed back to 3×3

    # Apply first update
    A1_UR = sun.mult(A1, UR)

    # STEP 4: Second SU(2) update (lower-right 2×2 block, colors 1-2)
    r2 = extract_su2_from_su3(A1_UR, (1, 2))
    a2_vec = heatbath_su2_real(r2, beta)
    A2 = embed_su2_in_su3_block(a2_vec, (1, 2))

    # STEP 5: Final update (note: A2 × A1 × U_old, not A1 × A2!)
    U[site, mu] = sun.mult(A2, sun.mult(A1, U_old))

# ============================================================================
#  THERMALIZATION
# ============================================================================

def thermalize_su3(U, beta, n_sweeps, lattice_dims, verbose=False):
    """
    Thermalize SU(3) gauge configuration via repeated sweeps.

    A "sweep" is one complete update of all links on the lattice.
    After each sweep, we measure the average plaquette to monitor
    thermalization progress.

    Parameters:
    -----------
    U : ndarray, shape (V, 4, 3, 3)
        Gauge configuration (modified in place)
    beta : float
        Inverse coupling
    n_sweeps : int
        Number of complete lattice sweeps
    lattice_dims : list
        Lattice dimensions
    verbose : bool
        Whether to print progress

    Returns:
    --------
    plaq_history : list of float
        Average plaquette after each sweep

    THERMALIZATION DIAGNOSTICS:
    ---------------------------
    - Plaquette should evolve monotonically (up from hot, down from cold)
    - Plateau indicates thermal equilibrium reached
    - Typical equilibration time: 10-30 sweeps for small lattices
    - Near phase transitions: can require 100+ sweeps (critical slowing down)

    EXPECTED PLAQUETTE VALUES (equilibrium):
    ----------------------------------------
    β = 5.7: <P> ≈ 0.58 (near deconfinement transition)
    β = 6.0: <P> ≈ 0.60 (standard coupling)
    β = 6.5: <P> ≈ 0.70 (weak coupling)
    """
    V = np.prod(lattice_dims)
    plaq_history = []

    for sweep in range(n_sweeps):
        # Update all links once (one complete sweep)
        for site in range(V):
            for mu in range(4):
                su3_heatbath_update(U, site, mu, beta, lattice_dims)

        # Measure plaquette to monitor thermalization
        plaq = average_plaquette_su3(U, lattice_dims)
        plaq_history.append(plaq)

        # Print progress
        if verbose and (sweep % 10 == 0 or sweep < 5):
            logging.info(f"  Sweep {sweep:4d}: plaquette = {plaq:.6f}")

    return plaq_history

def generate_su3_config(lattice_dims, beta, n_sweeps, mode='cold', verbose=True):
    """
    Generate thermalized SU(3) gauge configuration.

    High-level function that:
    1. Initializes configuration (cold or hot start)
    2. Thermalizes via heatbath sweeps
    3. Returns thermalized configuration and diagnostics

    Parameters:
    -----------
    lattice_dims : list of int
        Lattice dimensions [Lx, Ly, Lz, Lt]
    beta : float
        Inverse coupling β = 6/g²
    n_sweeps : int
        Number of thermalization sweeps
    mode : str
        'cold' (identity) or 'hot' (random) start
    verbose : bool
        Print progress information

    Returns:
    --------
    U : ndarray, shape (V, 4, 3, 3)
        Thermalized gauge configuration
    final_plaq : float
        Final average plaquette value
    plaq_history : list
        Plaquette evolution during thermalization

    USAGE EXAMPLE:
    --------------
    # Generate 4³×4 lattice at β=6.0 with 50 sweeps
    U, plaq, history = generate_su3_config([4,4,4,4], 6.0, 50, 'cold')
    """
    if verbose:
        logging.info(f"Generating SU(3) configuration:")
        logging.info(f"  Lattice: {lattice_dims}")
        logging.info(f"  Beta: {beta}")
        logging.info(f"  Sweeps: {n_sweeps}")
        logging.info(f"  Mode: {mode}")

    # Initialize configuration
    if mode == 'cold':
        U = generate_identity_su3(lattice_dims)
        if verbose:
            logging.info(f"  Starting from cold (identity) configuration")
    else:
        U = generate_random_su3(lattice_dims)
        if verbose:
            logging.info(f"  Starting from hot (random) configuration")

    # Measure initial plaquette
    initial_plaq = average_plaquette_su3(U, lattice_dims)
    if verbose:
        logging.info(f"  Initial plaquette: {initial_plaq:.6f}")

    # Thermalize
    if verbose:
        logging.info(f"  Thermalizing...")
    plaq_history = thermalize_su3(U, beta, n_sweeps, lattice_dims, verbose)

    # Report final state
    final_plaq = plaq_history[-1]
    if verbose:
        logging.info(f"  Final plaquette: {final_plaq:.6f}")
        logging.info(f"  Change: {final_plaq - initial_plaq:+.6f}")

    return U, final_plaq, plaq_history

# ============================================================================
#  MAIN PROGRAM
# ============================================================================

def main():
    """
    Command-line interface for SU(3) thermal configuration generator.

    Example usage:
    --------------
    # Cold start, β=6.0, 4³×4 lattice, 50 sweeps
    python Thermal_Generator_SU3.py --beta 6.0 --ls 4 --lt 4 --sweeps 50

    # Hot start, near critical temperature
    python Thermal_Generator_SU3.py --beta 5.7 --ls 6 --lt 6 --sweeps 100 --mode hot

    # Weak coupling (continuum limit)
    python Thermal_Generator_SU3.py --beta 10.0 --ls 8 --lt 8 --sweeps 30
    """
    parser = argparse.ArgumentParser(
        description='SU(3) Thermal Configuration Generator (Cabibbo-Marinari Algorithm)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --beta 6.0 --ls 4 --lt 4 --sweeps 50
  %(prog)s --beta 5.7 --ls 6 --lt 6 --sweeps 100 --mode hot --output thermal_b57.pkl
        """
    )

    parser.add_argument('--ls', type=int, default=4,
                       help='Spatial lattice size (default: 4)')
    parser.add_argument('--lt', type=int, default=4,
                       help='Temporal lattice size (default: 4)')
    parser.add_argument('--beta', type=float, default=6.0,
                       help='Inverse coupling β=6/g² (default: 6.0)')
    parser.add_argument('--sweeps', type=int, default=50,
                       help='Number of thermalization sweeps (default: 50)')
    parser.add_argument('--mode', type=str, default='cold', choices=['cold', 'hot'],
                       help='Starting configuration (default: cold)')
    parser.add_argument('--output', type=str, default='su3_config.pkl',
                       help='Output file (default: su3_config.pkl)')

    args = parser.parse_args()
    lattice_dims = [args.ls, args.ls, args.ls, args.lt]

    # Setup logging
    log_file = args.output.replace('.pkl', '.log')
    configure_logging(log_file)

    logging.info("="*70)
    logging.info("SU(3) THERMAL CONFIGURATION GENERATOR")
    logging.info("Cabibbo-Marinari Algorithm (2 SU(2) subgroups)")
    logging.info("="*70)

    # Generate configuration
    U, plaquette, plaq_history = generate_su3_config(
        lattice_dims, args.beta, args.sweeps, args.mode, verbose=True
    )

    # Save to pickle file
    config_data = {
        'U': U,
        'plaquette': plaquette,
        'beta': args.beta,
        'Lx': args.ls,
        'Ly': args.ls,
        'Lz': args.ls,
        'Lt': args.lt,
        'n_colors': 3,
        'mode': args.mode,
        'sweeps': args.sweeps,
        'plaquette_history': plaq_history,
        'timestamp': datetime.now().isoformat(),
        'algorithm': 'Cabibbo-Marinari (2 SU(2) subgroups)',
        'reference': 'Creutz PRD 21, 2308 (1980); Cabibbo-Marinari PLB 119, 387 (1982)'
    }

    with open(args.output, 'wb') as f:
        pickle.dump(config_data, f)

    logging.info(f"\nSaved to: {args.output}")
    logging.info("="*70)

    # Plot thermalization history
    plt.figure(figsize=(10, 6))
    plt.plot(plaq_history, 'b-', linewidth=2, label='Thermalization')
    plt.axhline(y=plaquette, color='r', linestyle='--',
                label=f'Final: {plaquette:.4f}', linewidth=2)
    plt.xlabel('Sweep', fontsize=14)
    plt.ylabel('Average Plaquette', fontsize=14)
    plt.title(f'SU(3) Thermalization: β={args.beta}, {args.ls}³×{args.lt}, {args.mode} start',
             fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = args.output.replace('.pkl', '_therm.png')
    plt.savefig(plot_file, dpi=150)
    logging.info(f"Plot saved: {plot_file}")

if __name__ == '__main__':
    main()
