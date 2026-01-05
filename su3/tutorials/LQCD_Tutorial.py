#!/usr/bin/env python3
"""
================================================================================
                    LATTICE QCD TUTORIAL FOR BEGINNERS
================================================================================

Welcome! This tutorial teaches the fundamentals of Lattice QCD through
hands-on examples. Each section builds on the previous one.

TABLE OF CONTENTS:
------------------
1. Introduction to Lattice QCD
2. Gauge Fields and SU(N) Matrices
3. The Wilson Action and Plaquettes
4. Monte Carlo and Thermalization
5. Quark Propagators and the Dirac Equation
6. Meson Correlators and Mass Extraction
7. The GMOR Relation and Chiral Physics
8. Wilson Loops and Confinement
9. Putting It All Together

PREREQUISITES:
--------------
- Python 3 with NumPy, SciPy, matplotlib
- Basic quantum mechanics and field theory
- Linear algebra (matrices, eigenvalues)

Author: Zeke Mohammed
Advisor: Dr. Aubin
Institution: Fordham University Physics Department
Date: January 2026
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')  # Adjust path as needed for your installation

print("""
================================================================================
                    LATTICE QCD TUTORIAL
================================================================================
""")

# ==============================================================================
# SECTION 1: INTRODUCTION TO LATTICE QCD
# ==============================================================================

print("""
================================================================================
SECTION 1: INTRODUCTION TO LATTICE QCD
================================================================================

What is Lattice QCD?
--------------------
Quantum Chromodynamics (QCD) is the theory of the strong nuclear force.
It describes how quarks interact via gluons to form protons, neutrons, and
other hadrons.

The Problem:
- QCD is "strongly coupled" at low energies
- Perturbation theory doesn't work!
- We need non-perturbative methods

The Solution: Lattice QCD
- Discretize spacetime onto a 4D grid (lattice)
- Replace continuous fields with variables on lattice sites/links
- Use Monte Carlo methods to compute path integrals numerically

Key Concepts:
1. Gauge fields (gluons) live on LINKS between lattice sites
2. Quark fields live on lattice SITES
3. The lattice spacing 'a' provides a UV cutoff
4. Continuum physics is recovered as a → 0

Why SU(3)?
- QCD has 3 "colors" of quarks: red, green, blue
- Gluons are described by SU(3) matrices
- SU(2) is simpler for learning (2 colors)

================================================================================
""")

# ==============================================================================
# SECTION 2: GAUGE FIELDS AND SU(N) MATRICES
# ==============================================================================

print("""
================================================================================
SECTION 2: GAUGE FIELDS AND SU(N) MATRICES
================================================================================

What are SU(N) Matrices?
------------------------
SU(N) = Special Unitary group of N×N matrices with:
  - U†U = I (unitary)
  - det(U) = 1 (special)

For SU(2): 2×2 complex matrices with 3 real parameters
For SU(3): 3×3 complex matrices with 8 real parameters

On the lattice, a gauge link U_μ(x) is an SU(N) matrix on the link
from site x to site x+μ̂ (where μ = 0,1,2,3 for t,x,y,z).

Let's create some SU(N) matrices:
""")

def create_su2_matrix():
    """Create a random SU(2) matrix using quaternion parametrization."""
    # SU(2) can be written as: U = a0*I + i*(a1*σ1 + a2*σ2 + a3*σ3)
    # with a0² + a1² + a2² + a3² = 1

    # Random point on 3-sphere
    vec = np.random.randn(4)
    vec = vec / np.linalg.norm(vec)
    a0, a1, a2, a3 = vec

    # Pauli matrices
    sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    U = a0*I + 1j*(a1*sigma1 + a2*sigma2 + a3*sigma3)
    return U

def create_su3_matrix():
    """Create a random SU(3) matrix using Gram-Schmidt."""
    # Start with random complex matrix
    M = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)

    # Gram-Schmidt orthogonalization
    Q, R = np.linalg.qr(M)

    # Fix determinant to be 1
    det = np.linalg.det(Q)
    Q = Q * (det.conj() ** (1/3))

    return Q

# Demonstrate SU(2)
print("Creating random SU(2) matrix:")
U2 = create_su2_matrix()
print(f"U = \n{U2}")
print(f"U†U = \n{U2.conj().T @ U2}")
print(f"det(U) = {np.linalg.det(U2):.6f}")
print(f"Should be: I and det=1")

print("\nCreating random SU(3) matrix:")
U3 = create_su3_matrix()
print(f"U†U - I max error: {np.max(np.abs(U3.conj().T @ U3 - np.eye(3))):.2e}")
print(f"|det(U) - 1| = {abs(np.linalg.det(U3) - 1):.2e}")

# ==============================================================================
# SECTION 3: THE WILSON ACTION AND PLAQUETTES
# ==============================================================================

print("""

================================================================================
SECTION 3: THE WILSON ACTION AND PLAQUETTES
================================================================================

The Plaquette:
--------------
The simplest gauge-invariant quantity is the PLAQUETTE - a 1×1 Wilson loop:

    P_μν(x) = U_μ(x) × U_ν(x+μ) × U†_μ(x+ν) × U†_ν(x)

This is a product of 4 links around a square. The average plaquette:

    <P> = (1/N_c) Re Tr[P_μν]  averaged over all squares

Physical Meaning:
- <P> = 1: Free field (all links = identity)
- <P> ≈ 0.6: Typical interacting field at β ≈ 6
- <P> → 0: Strong coupling (disordered)

The Wilson Action:
------------------
S = β Σ_plaquettes (1 - P_μν/N_c)

where β = 2N_c/g² is the inverse coupling.

At strong coupling (small β): <P> ≈ β/(2N_c)
At weak coupling (large β): <P> → 1 - const/β

""")

# ==============================================================================
# SECTION 4: MONTE CARLO AND THERMALIZATION
# ==============================================================================

print("""
================================================================================
SECTION 4: MONTE CARLO AND THERMALIZATION
================================================================================

The Goal:
---------
Generate gauge configurations distributed according to:
    P[U] ∝ exp(-S[U])

This is the Boltzmann distribution! We use Monte Carlo methods.

Algorithms:
-----------
1. METROPOLIS:
   - Propose random change to a link
   - Accept with probability min(1, exp(-ΔS))
   - Simple but slow

2. HEATBATH:
   - Sample new link directly from conditional distribution
   - Always accepts (more efficient)
   - More complex to implement

For SU(3), we use CABIBBO-MARINARI:
   - Decompose SU(3) update into SU(2) subgroup updates
   - Apply heatbath to upper-left 2×2 block
   - Apply heatbath to lower-right 2×2 block

Thermalization:
---------------
Starting from cold (U=I) or hot (random U) configuration:
1. Run many "sweeps" (update all links once)
2. Monitor plaquette until it stabilizes
3. Configuration is "thermalized" when equilibrium reached

Typical thermalization: 20-100 sweeps for small lattices

""")

print("Let's thermalize a small lattice...")

# Import our thermal generator
try:
    from Thermal_Generator_SU3_v2 import generate_su3_config

    print("\nGenerating 4³×4 SU(3) configuration at β=6.0...")
    U, plaq, history = generate_su3_config([4,4,4,4], 6.0, 30, 'cold', verbose=False)

    print(f"Initial plaquette (cold): 1.0")
    print(f"Final plaquette: {plaq:.4f}")
    print(f"Expected at β=6.0: ~0.60")

    # Plot thermalization
    plt.figure(figsize=(8, 5))
    plt.plot(history, 'b-', linewidth=2)
    plt.xlabel('Sweep', fontsize=12)
    plt.ylabel('Average Plaquette', fontsize=12)
    plt.title('Thermalization of SU(3) Gauge Field', fontsize=14)
    plt.axhline(y=0.6, color='r', linestyle='--', label='Expected equilibrium')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('tutorial_thermalization.png', dpi=150)
    plt.close()
    print("\nPlot saved: tutorial_thermalization.png")

except ImportError:
    print("(Thermal generator not available for demo)")

# ==============================================================================
# SECTION 5: QUARK PROPAGATORS
# ==============================================================================

print("""

================================================================================
SECTION 5: QUARK PROPAGATORS AND THE DIRAC EQUATION
================================================================================

The Dirac Equation:
-------------------
In continuum QCD: (γ_μ D_μ + m)ψ = 0

On the lattice, we use WILSON FERMIONS:
    D_W = m + 4r - (1/2) Σ_μ [(1-γ_μ)U_μ T_+μ + (1+γ_μ)U†_μ T_-μ]

where:
- m = bare quark mass
- r = Wilson parameter (usually 0.5)
- T_±μ = translation operators
- The "4r" term is the Wilson mass (prevents fermion doubling)

The Quark Propagator:
---------------------
S(x,y) = D⁻¹(x,y) = ⟨ψ(x)ψ̄(y)⟩

This tells us how a quark propagates from y to x through the gluon field.

Computing the Propagator:
1. Create point source: b_α,a(x) = δ(x-x₀)δ_α,α₀ δ_a,a₀
2. Solve: D × S = b (large sparse linear system!)
3. Repeat for all source colors and spins

For SU(2): 8 inversions (2 colors × 4 spins)
For SU(3): 12 inversions (3 colors × 4 spins)

""")

# ==============================================================================
# SECTION 6: MESON CORRELATORS
# ==============================================================================

print("""
================================================================================
SECTION 6: MESON CORRELATORS AND MASS EXTRACTION
================================================================================

What is a Meson?
----------------
A meson is a quark-antiquark bound state: M = q̄Γq

Different Γ matrices give different mesons:
- Γ = γ₅: PION (π), J^PC = 0⁻⁺, pseudoscalar
- Γ = γᵢ: RHO (ρ), J^PC = 1⁻⁻, vector
- Γ = I:  SIGMA (σ), J^PC = 0⁺⁺, scalar

The Correlator:
---------------
C(t) = ⟨M(t) M†(0)⟩ = Σ_x ⟨(q̄Γq)(x,t) (q̄Γq)†(0,0)⟩

Using Wick contractions (quenched approximation):
    C(t) = -Σ_x Tr[Γ S(0;x,t) Γ S†(0;x,t)]

For large t, the correlator decays exponentially:
    C(t) ~ A exp(-M_meson × t)

Mass Extraction:
----------------
The EFFECTIVE MASS:
    M_eff(t) = ln[C(t)/C(t+1)]

For large t, M_eff(t) → M_meson (plateau)

IMPORTANT: The pion is the GOLDSTONE BOSON of chiral symmetry breaking!
As quark mass → 0, pion mass → 0 (unlike other hadrons)

""")

# ==============================================================================
# SECTION 7: THE GMOR RELATION
# ==============================================================================

print("""
================================================================================
SECTION 7: THE GMOR RELATION AND CHIRAL PHYSICS
================================================================================

The Gell-Mann-Oakes-Renner Relation:
-------------------------------------
    m²_π = 2B(m_q + m_crit)

where:
- m_π = pion mass
- m_q = bare quark mass
- m_crit = critical mass (additive renormalization)
- B = parameter related to chiral condensate

Physical Meaning:
-----------------
1. The pion mass SQUARED is linear in quark mass
2. At m_q = -m_crit, the pion becomes MASSLESS (chiral limit)
3. For Wilson fermions, m_crit is NEGATIVE

Critical Mass:
--------------
Wilson fermions have "additive mass renormalization":
    m_eff = m_q + 4r

The critical mass is where the physical pion mass vanishes.
Typical values (quenched, Wilson):
- β = 5.7: m_crit ≈ -0.90
- β = 6.0: m_crit ≈ -0.80
- β = 6.2: m_crit ≈ -0.70

Extracting m_crit:
------------------
1. Compute m_π at several quark masses
2. Plot m²_π vs m_q
3. Linear fit: m²_π = slope × m_q + intercept
4. m_crit = -intercept/slope

""")

# ==============================================================================
# SECTION 8: WILSON LOOPS AND CONFINEMENT
# ==============================================================================

print("""
================================================================================
SECTION 8: WILSON LOOPS AND CONFINEMENT
================================================================================

What is Confinement?
--------------------
Quarks cannot exist as free particles - they are always confined
inside hadrons. This is one of the most important features of QCD!

The Wilson Loop:
----------------
W(R,T) = Tr[U_path] around an R×T rectangle

This represents a static quark-antiquark pair separated by distance R
existing for time T.

The expectation value:
    ⟨W(R,T)⟩ ~ exp(-V(R) × T)

where V(R) is the static quark potential.

The Static Potential:
---------------------
For a confining theory:
    V(R) = σR + c - α/R  (Cornell potential)

where:
- σ = STRING TENSION (confinement!)
- c = constant
- α/R = Coulomb-like term at short distances

String Tension:
---------------
- σ ≈ (440 MeV)² ≈ 0.18 GeV² in physical QCD
- In lattice units: σa² ~ 0.05 at β = 6.0
- Non-zero σ proves CONFINEMENT!

Area Law:
---------
For large loops: ⟨W(R,T)⟩ ~ exp(-σ × Area)
This is the AREA LAW - signature of confinement.

""")

# ==============================================================================
# SECTION 9: PUTTING IT ALL TOGETHER
# ==============================================================================

print("""
================================================================================
SECTION 9: PUTTING IT ALL TOGETHER
================================================================================

Complete Workflow for Lattice QCD:
----------------------------------

1. GENERATE CONFIGURATIONS
   - Choose lattice size (e.g., 8³×16)
   - Choose β (determines lattice spacing)
   - Thermalize using heatbath algorithm
   - Generate ensemble (10-100 independent configs)

2. MEASURE OBSERVABLES
   For each configuration, compute:
   - Plaquette (basic check)
   - Meson correlators (propagator inversions)
   - Wilson loops (for string tension)

3. ENSEMBLE AVERAGE
   - Average observables over all configurations
   - Use jackknife/bootstrap for errors

4. EXTRACT PHYSICS
   - Fit correlators to get hadron masses
   - Fit GMOR relation to get m_crit
   - Fit Wilson loops to get string tension
   - Convert to physical units using lattice spacing

5. CONTINUUM EXTRAPOLATION
   - Repeat at multiple β values
   - Extrapolate to a → 0 (continuum limit)

Code Files in This Project:
---------------------------
- Thermal_Generator_SU3_v2.py : Generate gauge configurations
- MesonIntegration.py : Meson mass calculations
- gmor_ensemble_analysis.py : GMOR relation study
- wilson_loops.py : String tension extraction
- generate_ensemble.py : Production ensemble generation

Example Commands:
-----------------
# Generate 10 configurations
python generate_ensemble.py --beta 6.0 --ls 4 --lt 8 --configs 10

# Run GMOR analysis
python gmor_ensemble_analysis.py --config_dir production_configs

# Compute Wilson loops
python wilson_loops.py --config production_configs/config_b6.00_L4x8_0000.pkl

================================================================================
                    END OF TUTORIAL
================================================================================

For more information:
- Creutz, "Quarks, Gluons, and Lattices" (1983)
- Rothe, "Lattice Gauge Theories: An Introduction" (2005)
- DeGrand & DeTar, "Lattice Methods for QCD" (2006)
================================================================================
""")

print("\nTutorial complete! Check tutorial_thermalization.png for the plot.")
