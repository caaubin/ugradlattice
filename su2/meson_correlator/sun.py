"""
SU(N) Lattice Gauge Theory Implementation
==========================================

This module implements SU(N) gauge theory on a discrete spacetime lattice for
Lattice QCD simulations, generalizing the SU(2) implementation to arbitrary
number of colors N_c.

Key Physics Concepts:
- Gauge fields U_μ(x) are N×N unitary matrices with det(U) = 1
- For N=2: Recovers SU(2) gauge theory (toy model)
- For N=3: Real QCD with three color charges (red, green, blue)
- Plaquettes (1×1 Wilson loops) measure the field strength
- Path integrals are evaluated using Monte Carlo methods

Mathematical Representation:
- SU(N) matrices are N×N complex matrices: U†U = UU† = 𝟙, det(U) = 1
- For N=2: Can use efficient quaternion representation (see su2.py)
- For N≥3: Use standard complex matrix representation

This generalization enables:
- Real QCD simulations with N_c=3
- Large-N limit studies (N→∞)
- Testing SU(2) as a special case

Author: Zeke Mohammed
Advisor: Dr. Christopher Aubin
Institution: Fordham University
Date: October 2025
"""

import math
import numpy as np

# ============================================================================
# FUNDAMENTAL CONSTANTS AND MATRICES
# ============================================================================

# Standard 4×4 identity matrix (used in Dirac space - independent of N_c)
eye4 = np.eye(4)

# Dirac gamma matrices in the Dirac representation
# These 4×4 matrices satisfy the Clifford algebra: {γ_μ, γ_ν} = 2g_μν
# Used to couple fermions to gauge fields in the Dirac equation
# NOTE: These are independent of the gauge group!
g0 = np.array([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]])  # γ⁰ (timelike)
g1 = np.array([[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]])      # γ¹ (x-direction)
g2 = np.array([[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]])  # γ² (y-direction)
g3 = np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])        # γ³ (z-direction)
gammas = np.array((g0,g1,g2,g3))

# ============================================================================
# SU(N) MATRIX OPERATIONS
# ============================================================================

def dagger(u):
    """Hermitian conjugate of a complex matrix (†-operation)

    In quantum field theory, U† represents the inverse gauge transformation.
    For SU(N), U† = U⁻¹ since these matrices are unitary.

    Parameters
    ----------
    u : array_like
        Complex N×N matrix representing a gauge field link

    Returns
    -------
    numpy.ndarray
        Hermitian conjugate U† = (U*)ᵀ

    Physics Note
    ------------
    The dagger operation reverses the direction of parallel transport.
    If U_μ(x) transports from x to x+μ̂, then U†_μ(x) transports backwards.

    Works for any gauge group dimension N.
    """
    return np.conjugate(u).T


def dag(U):
    """Hermitian conjugate for SU(N) matrices

    Alias for dagger() to maintain compatibility with su2.py naming.
    For SU(N), this is standard Hermitian conjugate of complex matrices.

    Parameters
    ----------
    U : array_like
        SU(N) matrix (N×N complex)

    Returns
    -------
    numpy.ndarray
        U† = (U*)ᵀ
    """
    return dagger(U)


def mult(U1, U2):
    """Multiply two SU(N) matrices

    Standard complex matrix multiplication.
    For SU(N), the product of two group elements is another group element:
    - Unitary: (U1·U2)†(U1·U2) = 𝟙
    - Unit determinant: det(U1·U2) = det(U1)·det(U2) = 1

    Parameters
    ----------
    U1, U2 : array_like
        SU(N) matrices (N×N complex)

    Returns
    -------
    numpy.ndarray
        Product U1·U2 (N×N complex)

    Physics Note
    ------------
    This multiplication represents composition of gauge transformations
    or parallel transport along consecutive links.
    """
    return np.dot(U1, U2)


def tr(UU):
    """Trace of an SU(N) matrix

    The trace appears in the Wilson action:
    S = β Σ_{plaq} (1 - Re[Tr(U_plaq)]/N)

    For SU(N), Tr(𝟙) = N.

    Parameters
    ----------
    UU : array_like
        SU(N) matrix (N×N complex)

    Returns
    -------
    complex
        Tr(U) = Σᵢ Uᵢᵢ

    Physics Note
    ------------
    The trace is gauge-invariant under cyclic permutations.
    For plaquettes: Tr(U₁U₂U₃U₄) measures the field strength.
    """
    return np.trace(UU)


def det(UU):
    """Determinant of an SU(N) matrix

    For SU(N), det(U) = 1 by definition (special unitary).
    This function is mainly for checking unitarity.

    Parameters
    ----------
    UU : array_like
        SU(N) matrix (N×N complex)

    Returns
    -------
    complex
        det(U) (should be 1 for valid SU(N) matrices)
    """
    return np.linalg.det(UU)


# ============================================================================
# LATTICE VOLUME AND NAVIGATION (GAUGE-GROUP INDEPENDENT)
# ============================================================================

def vol(La):
    """Calculate the total volume (number of sites) of the lattice

    For a lattice with dimensions [Lx, Ly, Lz, Lt], the volume is
    V = Lx × Ly × Lz × Lt, which equals the total number of lattice points.
    This appears in path integral normalizations.

    Parameters
    ----------
    La : array_like
        Lattice dimensions [Lx, Ly, Lz, Lt] in lattice units

    Returns
    -------
    int
        Total number of lattice sites V

    Note
    ----
    Independent of gauge group (same for SU(2), SU(3), etc.)
    """
    product = 1
    for x in range(len(La)):
        product *= La[x]
    return product


def dim(La):
    """Convert lattice dimensions array to dictionary format

    Utility function for easier access to individual dimensions.

    Parameters
    ----------
    La : array_like
        Lattice dimensions [Lx, Ly, Lz, Lt]

    Returns
    -------
    dict
        {0: Lx, 1: Ly, 2: Lz, 3: Lt}
    """
    D = {}
    for x in range(len(La)):
        D.update({x:La[x]})
    return D


def p2i(point, La):
    """Convert lattice coordinates to linear index

    Maps a 4D lattice point (x,y,z,t) to a single index for array storage.
    Uses row-major ordering: index = x + Lx·y + Lx·Ly·z + Lx·Ly·Lz·t

    Parameters
    ----------
    point : array_like
        Lattice coordinates [x, y, z, t]
    La : array_like
        Lattice dimensions [Lx, Ly, Lz, Lt]

    Returns
    -------
    int
        Linear index for array storage
    """
    return (La[2] * La[1] * La[0] * point[3]) + (La[1] * La[0] * point[2]) + (La[0] * point[1]) + (point[0])


def i2p(ind, La):
    """Convert linear index back to lattice coordinates

    Inverse of p2i function. Essential for identifying the physical
    location of lattice sites in calculations.

    Parameters
    ----------
    ind : int
        Linear index of lattice site
    La : array_like
        Lattice dimensions [Lx, Ly, Lz, Lt]

    Returns
    -------
    numpy.ndarray
        Lattice coordinates [x, y, z, t]
    """
    v = La[0] * La[1] * La[2]
    a = La[0] * La[1]
    l = La[0]
    t = divmod(ind, v)
    z = divmod(t[1], a)
    y = divmod(z[1], l)
    x = divmod(y[1], 1)
    return np.array([x[0], y[0], z[0], t[0]])


def parity(pt):
    """Determine the parity (even/odd sublattice) of a lattice point

    The lattice divides into two sublattices (checkerboard pattern).
    Parity = (x + y + z + t) mod 2

    Parameters
    ----------
    pt : array_like
        Lattice point [x, y, z, t]

    Returns
    -------
    int
        0 for even parity, 1 for odd parity
    """
    return np.sum(pt) % 2


def mupi(ind, mu, La):
    """Get neighbor index in positive μ direction with periodic BC

    Implements periodic boundary conditions on the lattice.
    If x_μ = L_μ - 1, then x_μ + 1 wraps to 0.

    Parameters
    ----------
    ind : int
        Current site index
    mu : int
        Direction (0=t, 1=x, 2=y, 3=z)
    La : array_like
        Lattice dimensions [Lx, Ly, Lz, Lt]

    Returns
    -------
    int
        Index of neighbor in +μ direction
    """
    pt = i2p(ind, La)
    pt[mu] = (pt[mu] + 1) % La[mu]
    return p2i(pt, La)


def mdowni(ind, mu, La):
    """Get neighbor index in negative μ direction with periodic BC

    Implements periodic boundary conditions on the lattice.
    If x_μ = 0, then x_μ - 1 wraps to L_μ - 1.

    Parameters
    ----------
    ind : int
        Current site index
    mu : int
        Direction (0=t, 1=x, 2=y, 3=z)
    La : array_like
        Lattice dimensions [Lx, Ly, Lz, Lt]

    Returns
    -------
    int
        Index of neighbor in -μ direction
    """
    pt = i2p(ind, La)
    pt[mu] = (pt[mu] - 1) % La[mu]
    return p2i(pt, La)


def getMups(V, numdim, La):
    """Precompute all forward neighbor indices

    Creates lookup table for efficiency in Monte Carlo updates.

    Parameters
    ----------
    V : int
        Lattice volume
    numdim : int
        Number of dimensions (typically 4)
    La : array_like
        Lattice dimensions

    Returns
    -------
    numpy.ndarray
        Array[site, direction] of neighbor indices
    """
    mups = np.zeros((V, numdim), dtype=int)
    for i in range(V):
        for mu in range(numdim):
            mups[i, mu] = mupi(i, mu, La)
    return mups


def getMdns(V, numdim, La):
    """Precompute all backward neighbor indices

    Creates lookup table for efficiency in Monte Carlo updates.

    Parameters
    ----------
    V : int
        Lattice volume
    numdim : int
        Number of dimensions (typically 4)
    La : array_like
        Lattice dimensions

    Returns
    -------
    numpy.ndarray
        Array[site, direction] of neighbor indices
    """
    mdns = np.zeros((V, numdim), dtype=int)
    for i in range(V):
        for mu in range(numdim):
            mdns[i, mu] = mdowni(i, mu, La)
    return mdns


# ============================================================================
# SU(N) GAUGE FIELD INITIALIZATION
# ============================================================================

def identity_SU_N(N_c):
    """Generate N×N identity matrix for SU(N)

    Returns the identity element of SU(N).
    Used for cold start configurations (U_μ(x) = 𝟙 everywhere).

    Parameters
    ----------
    N_c : int
        Number of colors (gauge group dimension)

    Returns
    -------
    numpy.ndarray
        N×N identity matrix (complex)
    """
    return np.eye(N_c, dtype=complex)


def random_SU_N(N_c, method='gram-schmidt'):
    """Generate a random SU(N) matrix

    Creates a random element of SU(N) uniformly distributed according
    to the Haar measure. Used for hot start configurations and
    Monte Carlo updates.

    Methods:
    - 'gram-schmidt': QR decomposition with phase correction
    - 'exponential': Matrix exponential of traceless Hermitian matrix

    Parameters
    ----------
    N_c : int
        Number of colors (gauge group dimension)
    method : str
        Generation method ('gram-schmidt' or 'exponential')

    Returns
    -------
    numpy.ndarray
        Random N×N SU(N) matrix (complex)

    Physics Note
    ------------
    For hot starts, gauge field links are initialized randomly.
    The system then evolves to thermal equilibrium via Monte Carlo.
    """
    if method == 'gram-schmidt':
        # Generate random complex matrix
        M = np.random.randn(N_c, N_c) + 1j * np.random.randn(N_c, N_c)

        # QR decomposition gives orthonormal columns
        Q, R = np.linalg.qr(M)

        # Correct phases to ensure det(Q) = 1
        # Extract diagonal phases from R
        Lambda = np.diag(R) / np.abs(np.diag(R))
        Q = Q @ np.diag(Lambda)

        # Ensure det = 1 (project to SU(N) from U(N))
        Q = Q / (np.linalg.det(Q) ** (1.0 / N_c))

        return Q

    elif method == 'exponential':
        # Generate random traceless Hermitian matrix (Lie algebra)
        H = np.random.randn(N_c, N_c) + 1j * np.random.randn(N_c, N_c)
        H = (H + H.conj().T) / 2  # Make Hermitian
        H = H - np.trace(H) * np.eye(N_c) / N_c  # Make traceless

        # Exponentiate to get SU(N) element
        U = np.linalg.matrix_exp(1j * H)

        # Ensure det = 1 (should be automatic, but enforce numerically)
        U = U / (np.linalg.det(U) ** (1.0 / N_c))

        return U
    else:
        raise ValueError(f"Unknown method: {method}. Use 'gram-schmidt' or 'exponential'")


def project_to_SU_N(M, N_c):
    """Project a matrix to SU(N)

    Takes an arbitrary matrix and projects it to the nearest SU(N) element.
    Ensures unitarity (U†U = 𝟙) and det(U) = 1.

    Used to:
    - Restore exact unitarity after numerical errors
    - Convert approximate updates to valid gauge matrices

    Parameters
    ----------
    M : array_like
        N×N complex matrix
    N_c : int
        Number of colors

    Returns
    -------
    numpy.ndarray
        Nearest SU(N) matrix to M

    Algorithm
    ---------
    1. Polar decomposition: M = U·P where U is unitary
    2. Adjust phase to ensure det(U) = 1
    """
    # Polar decomposition using SVD: M = U @ diag(s) @ Vh
    U_svd, s, Vh = np.linalg.svd(M)

    # Unitary part is U @ Vh (removes scaling)
    U = U_svd @ Vh

    # Ensure det(U) = 1 (project from U(N) to SU(N))
    U = U / (np.linalg.det(U) ** (1.0 / N_c))

    return U


def hstart(N_c):
    """Generate a random SU(N) matrix for hot start

    Wrapper for random_SU_N with default parameters.
    Maintains compatibility with su2.py naming convention.

    Parameters
    ----------
    N_c : int
        Number of colors

    Returns
    -------
    numpy.ndarray
        Random N×N SU(N) matrix
    """
    return random_SU_N(N_c)


def cstart(N_c):
    """Generate identity matrix for cold start

    Returns the identity element for cold start configurations.
    Maintains compatibility with su2.py naming convention.

    Parameters
    ----------
    N_c : int
        Number of colors

    Returns
    -------
    numpy.ndarray
        N×N identity matrix
    """
    return identity_SU_N(N_c)


# ============================================================================
# PLAQUETTE AND WILSON LOOP CALCULATIONS
# ============================================================================

def plaq(U, site, mups, mu, nu, N_c):
    """Calculate a single plaquette (1×1 Wilson loop)

    Computes the plaquette in the μ-ν plane at lattice site 'site':

    P_μν(x) = U_μ(x) · U_ν(x+μ̂) · U†_μ(x+ν̂) · U†_ν(x)

    The plaquette measures the field strength (lattice curl of gauge field).

    Parameters
    ----------
    U : array_like
        Gauge field configuration [site, direction, N×N matrix]
    site : int
        Starting lattice site index
    mups : array_like
        Forward neighbor lookup table
    mu, nu : int
        Directions for the plaquette (0-3)
    N_c : int
        Number of colors

    Returns
    -------
    numpy.ndarray
        N×N plaquette matrix P_μν(x)

    Physics Note
    ------------
    In the continuum limit: Tr(P) → 1 + ia²F_μν + O(a⁴)
    where F_μν is the field strength tensor.
    """
    if mu == nu:
        return identity_SU_N(N_c)

    site_p_mu = mups[site, mu]
    site_p_nu = mups[site, nu]

    # P = U_μ(x) · U_ν(x+μ) · U†_μ(x+ν) · U†_ν(x)
    P = U[site, mu]
    P = mult(P, U[site_p_mu, nu])
    P = mult(P, dag(U[site_p_nu, mu]))
    P = mult(P, dag(U[site, nu]))

    return P


def calcPlaq(U, La, mups, N_c):
    """Calculate average plaquette over entire lattice

    Computes the average plaquette:
    ⟨P⟩ = (1/6V) Σ_{x,μ<ν} Re[Tr(P_μν(x))] / N

    This measures the average field strength and appears in the action.

    Parameters
    ----------
    U : array_like
        Gauge field configuration
    La : array_like
        Lattice dimensions
    mups : array_like
        Forward neighbor lookup table
    N_c : int
        Number of colors

    Returns
    -------
    float
        Average plaquette value

    Physics Note
    ------------
    ⟨P⟩ → 1 as β → ∞ (classical limit, pure gauge)
    ⟨P⟩ → 0 as β → 0 (quantum fluctuations dominate)
    """
    V = vol(La)
    plaq_sum = 0.0

    for site in range(V):
        for mu in range(4):
            for nu in range(mu + 1, 4):
                P = plaq(U, site, mups, mu, nu, N_c)
                plaq_sum += np.real(tr(P)) / N_c

    # Average over all plaquettes (6 per site: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3))
    return plaq_sum / (6.0 * V)


def getstaple(U, site, mups, mdns, mu, N_c):
    """Calculate staple for Monte Carlo updates

    Computes the sum of staples around a link for the Metropolis algorithm.
    The staple is the sum of the three sides of plaquettes that don't
    include the link U_μ(x).

    Used in computing the change in action for link updates.

    Parameters
    ----------
    U : array_like
        Gauge field configuration
    site : int
        Lattice site
    mups, mdns : array_like
        Neighbor lookup tables
    mu : int
        Link direction
    N_c : int
        Number of colors

    Returns
    -------
    numpy.ndarray
        N×N staple matrix
    """
    staple = np.zeros((N_c, N_c), dtype=complex)

    for nu in range(4):
        if nu == mu:
            continue

        # Forward staple: U_ν(x+μ) · U†_μ(x+ν) · U†_ν(x)
        site_p_mu = mups[site, mu]
        site_p_nu = mups[site, nu]

        S = U[site_p_mu, nu]
        S = mult(S, dag(U[site_p_nu, mu]))
        S = mult(S, dag(U[site, nu]))
        staple += S

        # Backward staple: U†_ν(x+μ-ν) · U†_μ(x-ν) · U_ν(x-ν)
        site_m_nu = mdns[site, nu]
        site_p_mu_m_nu = mdns[site_p_mu, nu]

        S = dag(U[site_p_mu_m_nu, nu])
        S = mult(S, dag(U[site_m_nu, mu]))
        S = mult(S, U[site_m_nu, nu])
        staple += S

    return staple


# ============================================================================
# COMPATIBILITY FUNCTIONS
# ============================================================================

def showU(U, mu, i):
    """Display a gauge link (for debugging)

    Print the gauge field matrix at site i in direction μ.

    Parameters
    ----------
    U : array_like
        Gauge field configuration
    mu : int
        Direction
    i : int
        Site index
    """
    print(f"U[{i}, {mu}] =")
    print(U[i, mu])
    print()
