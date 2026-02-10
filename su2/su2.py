"""
SU(2) Lattice Gauge Theory Implementation
==========================================

This module implements SU(2) gauge theory on a discrete spacetime lattice for 
Lattice QCD simulations. The code discretizes space and time as described in 
Section 10.1 of the textbook, where continuous spacetime is replaced by a 
lattice with spacing 'a' (Eq. 10.1).

Key Physics Concepts:
- Gauge fields U_μ(x) live on links between lattice sites
- Plaquettes (1×1 Wilson loops) measure the field strength
- The Dirac operator describes fermion propagation
- Path integrals are evaluated using Monte Carlo methods

Mathematical Representation:
- SU(2) matrices are stored as 4-component real vectors (Cayley-Klein parameters)
- This parameterization: U = a₀𝟙 + i·aᵢσᵢ where σᵢ are Pauli matrices
- Ensures unitarity and det(U) = 1 automatically
"""

import math
import numpy
np = numpy

# ============================================================================
# FUNDAMENTAL CONSTANTS AND MATRICES
# ============================================================================

# SU(2) identity element in real-valued representation
# Represents the 2×2 identity matrix as [1, 0, 0, 0]
su2eye = np.array([1.,0.,0.,0.])

# Standard 4×4 identity matrix (used in Dirac space)
eye4 = np.eye(4)

# Dirac gamma matrices in the Dirac representation
# These 4×4 matrices satisfy the Clifford algebra: {γ_μ, γ_ν} = 2g_μν
# Used to couple fermions to gauge fields in the Dirac equation
g0 = np.array([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]])  # γ⁰ (timelike)
g1 = np.array([[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]])      # γ¹ (x-direction)
g2 = np.array([[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]])  # γ² (y-direction)
g3 = np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])        # γ³ (z-direction)
gammas = np.array((g0,g1,g2,g3))

# Aliases for common operations (for code readability)
prod = np.dot      # Matrix product
xprod = np.cross   # Cross product (for SU(2) multiplication)
add = np.add       # Addition

# ============================================================================
# SU(2) MATRIX OPERATIONS
# ============================================================================

def dagger(u):
	"""Hermitian conjugate of a complex matrix (†-operation)
	
	In quantum field theory, U† represents the inverse gauge transformation.
	For SU(2), U† = U⁻¹ since these matrices are unitary.

	Parameters
	----------
	u : array_like
		Complex matrix representing a gauge field link

	Returns
	-------
	numpy.ndarray
		Hermitian conjugate U† = (U*)ᵀ

	Physics Note
	------------
	The dagger operation reverses the direction of parallel transport.
	If U_μ(x) transports from x to x+μ̂, then U†_μ(x) transports backwards.
	"""
	return np.transpose(np.conjugate(u))


def vol(La):
	"""Calculate the total volume (number of sites) of the lattice
	
	For a lattice with dimensions [Lx, Ly, Lz, Lt], the volume is
	V = Lx × Ly × Lz × Lt, which equals the total number of lattice points.
	This appears in path integral normalizations (Eq. 10.55-10.56).

	Parameters
	----------
	La : array_like
		Lattice dimensions [Lx, Ly, Lz, Lt] in lattice units

	Returns
	-------
	int
		Total number of lattice sites V
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


def dag(U):
	"""Hermitian conjugate for SU(2) in real-valued representation
	
	For SU(2) parameterized as U = a₀𝟙 + i(a₁σ₁ + a₂σ₂ + a₃σ₃),
	the hermitian conjugate is U† = a₀𝟙 - i(a₁σ₁ + a₂σ₂ + a₃σ₃).
	This flips the sign of the imaginary components.

	Parameters
	----------
	U : array_like
		SU(2) matrix as [a₀, a₁, a₂, a₃]

	Returns
	-------
	numpy.ndarray
		U† in real-valued representation [a₀, -a₁, -a₂, -a₃]
	"""
	return np.array([1,-1,-1,-1])*U


def mult(U1, U2):
	"""Multiply two SU(2) matrices in real-valued representation
	
	Uses quaternion multiplication rules for SU(2) matrices.
	If U1 = a₀ + i·a·σ and U2 = b₀ + i·b·σ, then:
	U1·U2 = (a₀b₀ - a·b) + i(a₀b + b₀a - a×b)·σ
	
	This preserves the SU(2) group structure: det(U1·U2) = 1.

	Parameters
	----------
	U1, U2 : array_like
		SU(2) matrices as [a₀, a₁, a₂, a₃]

	Returns
	-------
	numpy.ndarray
		Product U1·U2 in real-valued representation
		
	Physics Note
	------------
	This multiplication represents composition of gauge transformations
	or parallel transport along consecutive links.
	"""
	a0 = U1[0]
	b0 = U2[0]
	a = U1[1:]
	b = U2[1:]

	# Quaternion multiplication formula
	c0 = a0 * b0 - prod(a, b)              # Real part
	c = b0*a + a0*b - xprod(a, b)          # Imaginary parts
	return np.array((c0, c[0], c[1], c[2]))


# ============================================================================
# LATTICE NAVIGATION FUNCTIONS
# ============================================================================

def p2i(point,La):
	"""Convert lattice coordinates to linear index
	
	Maps a 4D lattice point (x,y,z,t) to a single index for array storage.
	Uses row-major ordering: index = x + Lx·y + Lx·Ly·z + Lx·Ly·Lz·t
	
	This implements the discretization from Eq. 10.1 where continuous
	spacetime x^μ is replaced by discrete points n^μ.

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


def i2p(ind,La):
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
	
	This is crucial for:
	- Even-odd preconditioning of the Dirac operator
	- Implementing antiperiodic boundary conditions for fermions
	- Improving numerical efficiency

	Parameters
	----------
	pt : array_like
		Lattice point [x, y, z, t]

	Returns
	-------
	numpy.int64
		0 for even parity, 1 for odd parity
	"""
	return np.sum(pt)%2


# ============================================================================
# GAUGE FIELD INITIALIZATION
# ============================================================================

def hstart():
	"""Generate a random SU(2) matrix element
	
	Creates a random SU(2) matrix by generating a random unit quaternion.
	Ensures |a|² = a₀² + a₁² + a₂² + a₃² = 1 for unitarity.
	
	Used for:
	- Hot start configurations (random initialization)
	- Monte Carlo updates (with modifications)

	Returns
	-------
	numpy.ndarray
		Random SU(2) matrix [a₀, a₁, a₂, a₃] with |a| = 1
	"""
	# Generate random point inside unit 3-sphere
	a = np.array([np.random.uniform(-1.,1.), np.random.uniform(-1.,1.), np.random.uniform(-1.,1.)])
	while (np.sqrt(a[0]**2 + a[1]**2 + a[2]**2) >= 1):
		a[0] = np.random.uniform(-1.,1.)
		a[1] = np.random.uniform(-1.,1.)
		a[2] = np.random.uniform(-1.,1.)
	
	# Complete to unit quaternion
	a0 = np.sqrt(1 - (a[0]**2 + a[1]**2 + a[2]**2))
	if (np.random.random() > 0.5):
		a0 = -a0  # Random sign for a₀
	return np.array((a0, a[0], a[1], a[2]))


def update(UU):
	"""Generate small random update for Monte Carlo evolution
	
	Creates a new SU(2) matrix near the input matrix for Metropolis updates.
	The update is U' = g·U where g ≈ 𝟙 is close to identity.
	
	This implements local gauge updates in the Monte Carlo simulation
	for generating gauge field configurations according to the
	Boltzmann distribution exp(-S[U]) from Eq. 10.55.

	Parameters
	----------
	UU : array_like
		Current SU(2) gauge link

	Returns
	-------
	numpy.ndarray
		Updated gauge link U' = g·U
		
	Physics Note
	------------
	The size of the random change (0.1 factor) controls the Monte Carlo
	acceptance rate. Smaller changes → higher acceptance but slower exploration.
	"""
	# Generate small random SU(2) matrix near identity
	g = np.array([1.,0.,0.,0.]) + 0.1*hstart()*np.array([0.,1.,1.,1.])
	gU = mult(g,UU)
	gU /= det(gU)  # Ensure exact unitarity (project back to SU(2))

	return gU


def cstart():
	"""Cold start: initialize all gauge links to identity
	
	Sets U_μ(x) = 𝟙 for all links, corresponding to zero field strength.
	This is the ordered/cold configuration with minimal action.
	
	Used as:
	- Initial configuration for thermalization
	- Reference configuration for perturbative calculations

	Returns
	-------
	numpy.ndarray
		Identity element [1, 0, 0, 0]
	"""
	return su2eye
	

def tr(UU):
	"""Trace of SU(2) matrix in real representation
	
	For U = a₀𝟙 + i·a·σ, the trace is Tr(U) = 2a₀.
	The trace is gauge-invariant and appears in the Wilson action.

	Parameters
	----------
	UU : array_like
		SU(2) matrix [a₀, a₁, a₂, a₃]

	Returns
	-------
	numpy.float64
		Trace of the matrix (2a₀)
		
	Physics Note
	------------
	The real part of Tr(U) measures the "alignment" of the gauge field.
	Maximum Tr(U) = 2 when U = 𝟙 (no field).
	"""
	return UU[0] * 2
	

def det(UU):
	"""Determinant of SU(2) matrix
	
	For proper SU(2), det(U) = a₀² + a₁² + a₂² + a₃² = 1.
	This function verifies the unitarity constraint.
	
	Parameters
	----------
	UU : array_like
		SU(2) matrix [a₀, a₁, a₂, a₃]

	Returns
	-------
	numpy.float64
		Determinant (should be 1 for SU(2))
	"""
	return prod(UU,UU)


def mupi(ind, mu, La):
	"""Move one step forward in the μ direction with periodic boundaries
	
	Implements x → x + μ̂ with periodic boundary conditions.
	This is the lattice implementation of the derivative ∂_μ.
	
	Essential for:
	- Constructing plaquettes (Fig. 10.2)
	- Building the covariant derivative
	- Parallel transport of fields

	Parameters
	----------
	ind : int
		Starting lattice site index
	mu : int
		Direction (0=x, 1=y, 2=z, 3=t)
	La : array_like
		Lattice dimensions [Lx, Ly, Lz, Lt]

	Returns
	-------
	numpy.int64 
		Index of neighboring site in +μ direction
	"""
	pp = i2p(ind,La)
	if (pp[mu] + 1 >= La[mu]):
		pp[mu] = 0  # Wrap around (periodic boundary)
	else:
		pp[mu] += 1
	return p2i(pp,La)


def getMups(V,numdim,La):
	"""Precompute forward neighbor table for efficiency
	
	Creates lookup table mups[i,μ] = index of site i+μ̂.
	Avoids repeated calculation of neighbor indices.
	
	NOTE: This function is not used in PvB but provided for optimization.

	Parameters
	----------
	V : int
		Lattice volume (number of sites)
	numdim : int
		Number of dimensions (typically 4)
	La : array_like
		Lattice dimensions [Lx, Ly, Lz, Lt]

	Returns
	-------
	numpy.ndarray
		V × numdim array of forward neighbor indices
	"""
	mups = np.zeros((V,numdim), int)
	for i in range(0, V):
		for mu in range(0, numdim):
			mups[i,mu] = mupi(i, mu, La)

	return mups

def getMdns(V, numdim, La):
	"""Precompute backward neighbor table for efficiency
	
	Creates lookup table mdns[i,μ] = index of site i-μ̂.
	Avoids repeated calculation of neighbor indices.
	
	Companion to getMups for backward navigation. While getMups
	handles forward hopping U_μ(x), getMdns handles backward
	hopping U†_μ(x-μ̂) needed for staples and Dirac operator.
	
	NOTE: Like getMups, provided for optimization but not used in PvB.

	Parameters
	----------
	V : int
		Lattice volume (number of sites)
	numdim : int
		Number of dimensions (typically 4)
	La : array_like
		Lattice dimensions [Lx, Ly, Lz, Lt]

	Returns
	-------
	numpy.ndarray
		V × numdim array of backward neighbor indices
	"""
	mdns = np.zeros((V, numdim), dtype=int)
	for i in range(V):
		for mu in range(numdim):
			mdns[i, mu] = mdowni(i, mu, La)
	
	return mdns


def mdowni(ind, mu, La):

	"""Move one step backward in the μ direction with periodic boundaries
	
	Implements x → x - μ̂ with periodic boundary conditions.
	Needed for constructing staples and backward derivatives.
	
	Parameters
	----------
	ind : int
		Starting lattice site index
	mu : int
		Direction (0=x, 1=y, 2=z, 3=t)
	La : array_like
		Lattice dimensions [Lx, Ly, Lz, Lt]

	Returns
	-------
	numpy.int64
		Index of neighboring site in -μ direction
	"""
	pp = i2p(ind, La)
	if (pp[mu] - 1 < 0):
		pp[mu] = La[mu] - 1  # Wrap around (periodic boundary)
	else:
		pp[mu] -= 1
	return p2i(pp, La)

# ============================================================================
# WILSON LOOPS AND OBSERVABLES
# ============================================================================

def plaq(U, U0i, mups, mu, nu):
	"""Calculate the 1×1 Wilson loop (plaquette)
	
	Computes the gauge-invariant quantity (Eq. 10.35):
	W□ = Tr[U_μ(x) U_ν(x+μ̂) U†_μ(x+ν̂) U†_ν(x)]
	
	The plaquette is the smallest Wilson loop and measures the field
	strength F_μν at point x. In the continuum limit:
	1 - Re(W□)/2 ∝ a⁴ Tr(F_μν²)
	
	This is the basic building block of the Wilson gauge action.
	
	Parameters
	----------
	U : array_like
		Full gauge field configuration U[site][direction]
	U0i : int
		Starting lattice site index
	mups : array_like
		Forward neighbor table
	mu, nu : int
		Plane orientation (μ < ν typically)

	Returns
	-------
	numpy.float64
		Real part of trace of plaquette
		
	Physics Note
	------------
	The plaquette measures the curvature of the gauge field.
	Perfect plaquette (W□ = 2) means zero field strength.
	Deviations from 2 indicate non-trivial gauge dynamics.
	"""
	# Traverse the plaquette clockwise from site x:
	# x --U_μ(x)--> x+μ̂
	# |              |
	# U†_ν(x)      U_ν(x+μ̂)
	# |              |
	# x+ν̂ <--U†_μ(x+ν̂)-- x+μ̂+ν̂
	
	U0 = U[U0i][mu].copy()                    # Link from x in μ direction
	U1 = U[mups[U0i, mu]][nu].copy()         # Link from x+μ̂ in ν direction
	U2 = dag(U[mups[U0i, nu]][mu].copy())    # Link from x+ν̂ in μ direction (dagger)
	U3 = dag(U[U0i][nu].copy())              # Link from x in ν direction (dagger)

	return tr(mult(mult(U0,U1),mult(U2,U3)))


def Wilsonloop(i, j, U, U0i, mups, mdns, mu, nu, verbose=False):
	"""Calculate an i×j rectangular Wilson loop
	
	Generalizes the plaquette to an i×j rectangle.
	Wilson loops W(i,j) probe the quark-antiquark potential V(r):
	⟨W(i,j)⟩ ∼ exp(-V(r)·j) for large temporal extent j
	
	This is crucial for:
	- Extracting the static quark potential
	- Studying confinement (area law vs perimeter law)
	- Calculating string tension
	
	Parameters
	----------
	i : int
		Number of links in μ direction
	j : int
		Number of links in ν direction  
	U : array_like
		Gauge field configuration
	U0i : int
		Starting site index
	mups, mdns : array_like
		Forward/backward neighbor tables
	mu, nu : int
		Rectangle orientation

	Returns
	-------
	numpy.float64
		Real part of trace of Wilson loop
		
	Physics Note
	------------
	For large loops in confining phase: W(R,T) ∼ exp(-σRT)
	where σ is the string tension (area law).
	In deconfined phase: W(R,T) ∼ exp(-2mT) (perimeter law).
	"""
	# Build the rectangular path
	Uij = [U[U0i][mu]]
	Ui = [U0i]
	
	# Move i steps in μ direction
	for a in range(1, i):
		Ui.append(mups[Ui[a-1], mu])
		Uij.append(U[Ui[a]][mu])
	
	# Move j steps in ν direction
	Uij.append(U[mups[Ui[-1],mu]][nu])
	Uj = [mups[Ui[-1],mu]]
	for b in range(1, j):
		Uj.append(mups[Uj[b-1], nu])
		Uij.append(U[Uj[b]][nu])
	
	# Move i steps back in μ direction
	Uij.append(dag(U[mups[mdns[Uj[-1],mu],nu]][mu])) 
	Uidag = [mups[mdns[Uj[-1],mu],nu]] 
	for c in range(1, i):
		Uidag.append(mdns[Uidag[c-1],mu])
		Uij.append(dag(U[Uidag[c]][mu]))
	
	# Move j steps back in ν direction
	Uij.append(dag(U[mdns[Uidag[-1],nu]][nu]))
	Ujdag = [mdns[Uidag[-1], nu]]
	for d in range(1, j):
		Ujdag.append(mdns[Ujdag[d-1], nu])
		Uij.append(dag(U[Ujdag[d]][nu]))
	
	f = len(Uij)

	print(Uij)  # Debug output

	# Multiply all links around the loop
	product = su2eye
	for e in range(0, f): 
		product = mult(product, Uij[e])
	return tr(product).real 


def link(U,U0i,mups,mu):
	"""Calculate the trace of a single gauge link
	
	The link variable ⟨Tr U_μ(x)⟩ is a basic gauge-invariant observable.
	Used to monitor thermalization and measure order parameters.

	Parameters
	----------
	U : array_like
		Gauge field configuration
	U0i : int
		Lattice site index
	mups : array_like
		Forward neighbor table
	mu : int
		Link direction (0=x, 1=y, 2=z, 3=t)

	Returns
	-------
	numpy.float64
		Trace of the gauge link
	"""
	U0 = U[U0i][mu].copy()
	return tr(U0)


def getstaple(U, U0i, mups, mdns, mu):
	"""Calculate the sum of staples around a link
	
	The staple is the sum of all U-shaped paths that complete a plaquette
	when combined with the link U_μ(x). There are 6 staples in 4D.
	
	Used in:
	- Heat bath and overrelaxation algorithms
	- Computing the effective action for one link
	- Smearing and cooling procedures
	
	Parameters
	----------
	U : array_like
		Gauge field configuration
	U0i : int
		Site index
	mups, mdns : array_like
		Forward/backward neighbor tables
	mu : int
		Direction of central link

	Returns
	-------
	numpy.ndarray
		Sum of all staples as SU(2) matrix
		
	Physics Note
	------------
	The staple sum determines the local action:
	S_local = -β/2 · Re Tr(U_μ(x) · Σ_staples)
	This enters the Metropolis acceptance probability.
	"""
	value = 0.0
	mm = list(range(4))
	mm = [i for i in range(4)]
	mm.remove(mu)  # Loop over ν ≠ μ
	
	for nu in mm:
		# Forward staple (going up in ν direction)
		value += staple(U, U0i, mups, mdns, mu, nu, 1)
		
		# Backward staple (going down in ν direction)
		value += staple(U, U0i, mups, mdns, mu, nu, -1)
	return value


def staple(U, U0i, mups, mdns, mu, nu, signnu):
	"""Calculate a single staple in the μ-ν plane
	
	A staple is a U-shaped path of three links that forms a plaquette
	when combined with U_μ(x). This is the building block for local
	gauge updates in Monte Carlo simulations.
	
	Parameters
	----------
	U : array_like
		Gauge field configuration
	U0i : int
		Starting site index
	mups, mdns : array_like
		Neighbor tables
	mu, nu : int
		Plane orientation (μ ≠ ν)
	signnu : int
		+1 for forward staple, -1 for backward staple

	Returns
	-------
	numpy.ndarray
		Staple as SU(2) matrix
		
	Physics Note
	------------
	Forward staple:  U_ν(x+μ̂) U†_μ(x+ν̂) U†_ν(x)
	Backward staple: U†_ν(x+μ̂-ν̂) U†_μ(x-ν̂) U_ν(x-ν̂)
	"""
	if (signnu == 1):  # Forward staple
		U1 = U[mups[U0i, mu]][nu].copy()
		U2 = dag(U[mups[U0i, nu]][mu].copy())
		U3 = dag(U[U0i][nu].copy())
	else:  # Backward staple
		U1 = dag(U[mdns[mups[U0i, mu],nu]][nu].copy())
		U2 = dag(U[mdns[U0i, nu]][mu].copy())
		U3 = U[mdns[U0i, nu]][nu].copy()

	return mult(mult(U1,U2),U3)


# ============================================================================
# OBSERVABLE CALCULATIONS
# ============================================================================

def calcPlaq(U,La,mups):
	"""Calculate average plaquette over entire lattice
	
	Computes ⟨P⟩ = (1/6V) Σ_{x,μ<ν} Re Tr(P_μν(x))/2
	where the sum is over all plaquettes on the lattice.
	
	This is the primary observable for:
	- Monitoring thermalization
	- Measuring the gauge action ⟨S⟩ = β(1 - ⟨P⟩)
	- Extracting the lattice spacing via asymptotic scaling
	
	The approach to the continuum limit is monitored by how close
	⟨P⟩ approaches 1 as β → ∞.

	Parameters
	----------
	U : array_like
		Full gauge field configuration
	La : array_like
		Lattice dimensions [Lx, Ly, Lz, Lt]
	mups : array_like
		Forward neighbor table

	Returns
	-------
	numpy.float64
		Average plaquette value (between 0 and 1)
		
	Physics Note
	------------
	In weak coupling: ⟨P⟩ ≈ 1 - g²/4 + O(g⁴)
	In strong coupling: ⟨P⟩ ∼ 1/g² (confinement regime)
	"""
	plaquettes = []
	j = 0
	V = vol(La)
	pp = np.array([0,0,0,0],dtype=int)
	
	# Loop over all plaquettes in all 6 planes (μ < ν)
	for mu in range(4):
		for nu in range(mu+1,4):
			# Fix the plane, now loop over all positions
			Lmu = La[mu]
			Lnu = La[nu]
			dirs = [0,1,2,3]
			dirs.remove(mu)
			dirs.remove(nu)
			rho = dirs[0]    # Perpendicular directions
			sigma = dirs[1]
			
			# Loop over all positions (excluding boundaries for open BC)
			for xmu in range(La[mu]-1):
				for xnu in range(La[nu]-1):
					for xrho in range(La[rho]):
						for xsigma in range(La[sigma]):
							pp[mu] = xmu
							pp[nu] = xnu
							pp[rho] = xrho
							pp[sigma] = xsigma
							i = p2i(pp,La)
							# Factor of 0.5 for normalization
							plaquettes.append(0.5*plaq(U,i,mups,mu,nu))
	
	avgPlaquettes = np.mean(plaquettes)
	return avgPlaquettes


def calcU_i(U,V,La,mups):
	"""Calculate average spatial link
	
	Computes ⟨U_s⟩ = (1/3V) Σ_{x,i=1,2,3} Re Tr(U_i(x))/2
	where the sum is over all spatial links.
	
	Used to study:
	- Spatial vs temporal asymmetry
	- Finite temperature effects
	- Order parameters in certain phases

	Parameters
	----------
	U : array_like
		Gauge field configuration
	V : int
		Lattice volume
	La : array_like
		Lattice dimensions
	mups : array_like
		Neighbor table

	Returns
	-------
	numpy.float64
		Average spatial link value
	"""
	spaceLink = np.zeros((3*V))
	j = 0
	for i in range(V):
		for mu in range(len(La)-1):  # Spatial directions only (0,1,2)
			spaceLink[j] = link(U,i,mups,mu)
			j = j + 1
	U_i = np.mean(spaceLink)
	return U_i/2.  # Normalize to [0,1]


def calcU_t(U,V,mups):
	"""Calculate average temporal link
	
	Computes ⟨U_t⟩ = (1/V) Σ_x Re Tr(U_4(x))/2
	
	The Polyakov loop (product of U_t around temporal direction)
	is an order parameter for confinement/deconfinement transition.

	Parameters
	----------
	U : array_like
		Gauge field configuration
	V : int
		Lattice volume
	mups : array_like
		Neighbor table

	Returns
	-------
	numpy.float64
		Average temporal link value
		
	Physics Note
	------------
	At finite temperature, ⟨U_t⟩ relates to the Polyakov loop,
	which measures the free energy of a static quark.
	"""
	timeLink = np.zeros((V))
	j = 0
	for i in range(V):
		timeLink[i] = link(U,i,mups,3)  # Direction 3 is time
	U_t = np.mean(timeLink)
	return U_t/2.  # Normalize to [0,1]


# ============================================================================
# DIRAC MATRIX CONSTRUCTION
# ============================================================================

def masseo(row,dat,i,j,m,r):
	"""Add mass term to even-odd preconditioned Dirac matrix
	
	The mass term contributes (m + r)δ_xy to the Dirac operator.
	In even-odd preconditioning, only connects sites of same parity.
	
	The Wilson parameter r removes fermion doubling (typically r=1).
	The factor of 2 comes from the normalization convention.
	
	Parameters
	----------
	row : list
		Row indices for sparse matrix construction
	dat : list
		Matrix values for sparse construction
	i, j : int
		Matrix indices for the 8×8 block (color⊗spin space)
	m : float
		Fermion mass in lattice units
	r : float
		Wilson parameter (typically r=1)

	Physics Note
	------------
	The Dirac operator in lattice QCD (Wilson formulation):
	D = m + 4r - (1/2)Σ_μ [(r-γ_μ)U_μ(x)δ_{x+μ̂,y} + (r+γ_μ)U†_μ(x-μ̂)δ_{x-μ̂,y}]
	The mass term gives fermions their mass in the continuum limit.
	"""
	y = 0
	for x in range(8):
		# Even-odd preconditioning reduces matrix size by factor of 2
		if ((i//8//2)*8+x) not in row:
			row.append((i//8//2)*8+x)
			dat.append(2*(m+r))  # Factor of 2 from convention
			y += 1


def mass(row,col,dat,i,j,m,r):
	"""Add mass term to full (unpreconditioned) Dirac matrix
	
	Adds diagonal 8×8 identity blocks with coefficient 2(m+r).
	
	Parameters
	----------
	row, col : list
		Row and column indices for sparse matrix
	dat : list
		Matrix values
	i, j : int
		Starting indices for 8×8 block
	m : float
		Fermion mass
	r : float
		Wilson parameter
	"""
	y = 0
	for x in range(8):
		row.append(i+x)
		col.append(j+y)
		dat.append(2*(m+r)) 
		y += 1


def showU(U,mu,i):
	"""Convert gauge link to 2×2 complex matrix form
	
	Transforms from real representation [a₀,a₁,a₂,a₃] to
	standard SU(2) matrix form for visualization.
	
	Parameters
	----------
	U : array_like
		Full gauge configuration
	mu : int
		Direction (0-3)
	i : int
		Site index

	Returns
	-------
	numpy.ndarray
		2×2 complex matrix representation
	"""
	u = np.array([[U[1][i,mu,0]+U[1][i,mu,3]*1j,    U[1][i,mu,2] + U[1][i,mu,1]*1j],
		         [-U[1][i,mu,2]+U[1][i,mu,1]*1j,    U[1][i,mu,0] - U[1][i,mu,3]*1j]])
	return u


def initD(row,col,dat,mu,U,r,m,n,pbc):
	"""Add kinetic (hopping) term to full Dirac matrix
	
	Implements the gauge-covariant derivative part of the Dirac operator.
	This couples fermions at neighboring sites via gauge links U_μ.
	
	The term is: -(1/2)[(r-γ_μ)U_μ(x)δ_{y,x+μ̂} + (r+γ_μ)U†_μ(x-μ̂)δ_{y,x-μ̂}]
	
	This creates an 8×8 block = γ_μ ⊗ U_μ (Kronecker product).

	Parameters
	----------
	row, col : list
		Sparse matrix indices
	dat : list
		Matrix values
	mu : int
		Direction of hopping
	U : array_like
		Gauge field
	r : float
		Wilson parameter
	m, n : int
		Matrix block indices
	pbc : bool
		True if periodic boundary crossed

	Returns
	-------
	tuple
		(error_code, count) - (0, n) on success
		
	Physics Note
	------------
	The kinetic term implements parallel transport of fermions.
	The γ_μ matrices give fermions their Dirac structure (spin).
	The U_μ matrices ensure gauge covariance.
	"""
	count = 0
	try:
		gam = gammas[mu]
	except:
		return 101

	try:
		if m>n and pbc or m<n and not pbc:  # Moving forward
			Ui = m//8
			s = 1
		else:  # Moving backward  
			Ui = n//8
			s = -1
	except:
		return 102

	try:
		# Extract gauge link as 2×2 complex matrix
		u = np.array([[U[1][Ui,mu,0] + U[1][Ui,mu,3]*1j,    U[1][Ui,mu,2] + U[1][Ui,mu,1]*1j],
					 [-U[1][Ui,mu,2] + U[1][Ui,mu,1]*1j,    U[1][Ui,mu,0] - U[1][Ui,mu,3]*1j]])
	except:
		return 103

	try:
		if s==1:  # Forward hopping
			D = np.kron(gam,u)           # Dirac term
			W = np.kron(eye4,u) * (-r)   # Wilson term
		elif s==-1:  # Backward hopping
			D = np.kron(gam,dagger(u))
			W = np.kron(eye4,dagger(u)) * (-r)
	except:
		return 104

	try:
		# Add non-zero elements to sparse matrix
		for x in range(8):
			for y in range(8):
				if D[x][y] != 0:
					count += 1
					row.append(m + x)
					col.append(n + y)
					dat.append(D[x][y] * s)
				if W[x][y] != 0:
					count += 1
					row.append(m + x)
					col.append(n + y)
					dat.append(W[x][y])
	except ValueError:
		return 105

	return (0,count)


def initDeo(row,col,dat,mu,U,r,m,n,pbc):
	"""Add kinetic term to even-odd preconditioned Dirac matrix
	
	Even-odd preconditioning exploits the bipartite structure of the
	lattice to reduce the linear system size by factor of 2.
	
	The Dirac matrix has block structure:
	D = [M_ee  D_eo]
	    [D_oe  M_oo]
	
	where e/o denote even/odd sites. This function fills D_eo or D_oe.

	Parameters
	----------
	row, col : list
		Sparse matrix indices (preconditioned)
	dat : list
		Matrix values
	mu : int
		Hopping direction
	U : array_like
		Gauge field
	r : float
		Wilson parameter
	m, n : int
		Original matrix indices
	pbc : bool
		Periodic boundary flag

	Returns
	-------
	tuple
		(error_code, count)
		
	Physics Note
	------------
	Even-odd preconditioning is crucial for:
	- Reducing computational cost (matrix size halved)
	- Improving condition number
	- Enabling efficient inversions for fermion propagators
	"""
	count = 0
	
	try:
		gam = gammas[mu]
	except:
		return (101,count)

	try:
		if m>n and pbc or m<n and not pbc:  # Forward hop
			Ui = m//8
			s = 1
		else:  # Backward hop
			Ui = n//8
			s = -1
	except:
		return (102,count)

	try:
		u = np.array([[U[1][Ui,mu,0] + U[1][Ui,mu,3]*1j,    U[1][Ui,mu,2] + U[1][Ui,mu,1]*1j],
					 [-U[1][Ui,mu,2] + U[1][Ui,mu,1]*1j,    U[1][Ui,mu,0] - U[1][Ui,mu,3]*1j]])
	except:
		return (103,count)

	try:
		if s==1:
			D = np.kron(gam,u)
			W = np.kron(eye4,u) * (-r)
		elif s==-1:
			D = np.kron(gam,dagger(u))
			W = np.kron(eye4,dagger(u)) * (-r)
	except:
		return (104,count)
		
	try:
		for x in range(8):
			for y in range(8):
				if D[x][y] != 0:
					count += 1
					# Map to preconditioned indices
					row.append((m//8//2)*8 + x)
					col.append((n//8//2)*8 + y)
					dat.append(D[x][y] * s)
				if W[x][y] != 0:
					count += 1
					row.append((m//8//2)*8 + x)
					col.append((n//8//2)*8 + y)
					dat.append(W[x][y])
	except ValueError:
		return (105,count)

	return  (0,count)


# ============================================================================
# UTILITY FUNCTIONS FOR DEBUGGING AND ANALYSIS
# ============================================================================

def getElement(D,La,point_1,point_2,a,alpha,b,beta):
	"""Extract specific matrix element by physical indices
	
	Maps from physical quantum numbers (position, color, spin) to
	matrix indices. Useful for debugging and understanding structure.
	
	Parameters
	----------
	D : array_like
		Dirac matrix
	La : array_like
		Lattice dimensions
	point_1, point_2 : array_like
		Lattice points [x,y,z,t]
	a, b : int
		Color indices (0 or 1 for SU(2))
	alpha, beta : int
		Spin indices (0-3)

	Returns
	-------
	complex
		Matrix element D[point_1,a,α][point_2,b,β]
	"""
	space_i = 8 * p2i(point_1,La)
	space_j = 8 * p2i(point_2,La)
	return D[space_i + 2*alpha + a][space_j + 2*beta + b]


def getIndex(La,point_1):
	"""Convert lattice point to Dirac matrix index
	
	Each lattice site corresponds to an 8×8 block in the Dirac matrix
	(2 colors × 4 spins = 8 components).
	
	Parameters
	----------
	La : array_like
		Lattice dimensions
	point_1 : array_like
		Lattice point [x,y,z,t]
	
	Returns
	-------
	int
		Starting index for this site's 8×8 block
	"""
	return 8 * p2i(point_1,La)


# --- getPoint helpers (replaced duplicate defs with clear names)
def getPoint_pair(La, rc, i, j):
    for x in range(0, rc, 8):
        diff = i - x
        if diff < 8:
            space_i = i2p(x//8, La)
            beta = diff // 2  # Spin index
            b = diff % 2      # Color index
            break
    for y in range(0, rc, 8):
        diff = j - y
        if diff < 8:
            space_j = i2p(y//8, La)
            alpha = diff // 2  # Spin index
            a = diff % 2       # Color index
            break
    return [space_i, space_j, b, beta, a, alpha]

def getPoint_single(La, rc, i):
    for x in range(0, rc, 8):
        diff = i - x
        if diff < 8:
            space_i = i2p(x//8, La)
            beta = diff // 2  # Spin index
            b = diff % 2      # Color index
            break
    return [space_i, b, beta]

def getPoint(La, rc, i, j=None):
    """Compatibility wrapper: if j provided, returns pair info; else single index info."""
    if j is None:
        return getPoint_single(La, rc, i)
    return getPoint_pair(La, rc, i, j)


def showMr(mat,rc):
	"""Display 8×8 blocks along top row of Dirac matrix
	
	Visualization tool for understanding matrix structure.
	Shows how different lattice sites couple.

	Parameters
	----------
	mat : array_like
		Dirac matrix
	rc : int
		Matrix dimension
	"""
	for i in range(0,rc,8):
		print("mat i = %i, lattice i = %i" % (i,i//8))
		print(mat[:8,i:i+8].real)
		print(mat[:8,i:i+8].imag)


def showMc(mat,rc):
	"""Display 8×8 blocks along left column of Dirac matrix
	
	Complementary to showMr for visualizing matrix structure.

	Parameters
	----------
	mat : array_like
		Dirac matrix
	rc : int
		Matrix dimension
	"""
	for i in range(0,rc,8):
		print(i)
		print(mat[i:i+8,:8].real)
		print(mat[i:i+8,:8].imag)


def compare(mat1, mat2):
	"""Compare two matrices element by element
	
	Debugging tool to find differences between matrices.
	Useful for verifying implementations.

	Parameters
	----------
	mat1, mat2 : array_like
		Matrices to compare
	"""
	for i in range(len(mat1)): 
		for j in range(len(mat1[0])): 
			if mat1[i][j] != mat2[i][j]: 
				print("%f,%f\t%i,%i"%(mat1[i][j],mat2[i][j],i,j))


def getTime(a,b):
	"""Format elapsed time in readable format
	
	Utility for timing code sections and optimization.

	Parameters
	----------
	a, b : float
		Start and end times from time.time()
	
	Returns
	-------
	str
		Time formatted as "HH:MM:SS.mmm"
	"""
	t = b-a 
	hrs = divmod(t,3600)
	mins,secs = divmod(hrs[1],60)

	return "%02d:%02d:%06.3f" % (hrs[0],mins,secs)  


def corr(x, m, L=None, T=None):
    """Calculate and save pion correlator.

    Computes the pion two-point function C(t) from quark propagator.
    The pion correlator measures ⟨π(t)π†(0)⟩ and decays as:
        C(t) ∼ exp(-m_π·t) for large t

    This extracts the pion mass, the lightest hadron in QCD.

    Parameters
    ----------
    x : array_like
        Inverted Dirac matrix (quark propagator)
    m : float
        Quark mass used in simulation
    L : int, optional
        Spatial lattice size. If None, will try global L.
    T : int, optional
        Temporal lattice size. If None, will try global T.
    """

    if T is None:
        T = globals().get('T', None)
        if T is None:
            raise ValueError("corr requires T (temporal extent). Provide T or set global T.")

    if L is None:
        L = globals().get('L', None)
        if L is None:
            raise ValueError("corr requires L (spatial extent). Provide L or set global L.")

    corrt = np.zeros(T)
    with open(f'correlators/pion_pt_pt_m{m:.1f}_b2.4.dat', 'w') as correlator:
        for t in range(T):
            for a in range(2):
                for b in range(2):
                    for alpha in range(4):
                        for beta in range(4):
                            corrt[t] += abs(x[b+2*beta][a+2*alpha+8*L**3*t])**2
            correlator.write(f"{t}\t{corrt[t]}\n")
    return 0

# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
	"""Simple test to verify module functionality when run directly."""
	
	print("SU(2) Lattice Gauge Theory Module Test")
	print("=" * 40)
	
	# Test SU(2) matrix operations
	print("\n1. Testing SU(2) matrix properties:")
	U_random = hstart()
	print(f"   Random SU(2) det = {det(U_random):.10f}")
	print(f"   Expected: 1.0000000000")
	print(f"   ✓ Unitarity preserved" if abs(det(U_random) - 1.0) < 1e-10 else "   ✗ Unitarity violated!")
	
	# Test identity element
	print("\n2. Testing identity element:")
	U_identity = cstart()
	print(f"   Identity trace = {tr(U_identity):.10f}")
	print(f"   Expected: 2.0000000000")
	print(f"   ✓ Correct identity" if abs(tr(U_identity) - 2.0) < 1e-10 else "   ✗ Identity error!")
	
	# Test multiplication
	print("\n3. Testing group multiplication:")
	U1 = hstart()
	U2 = hstart()
	U_prod = mult(U1, U2)
	print(f"   Product det = {det(U_prod):.10f}")
	print(f"   Expected: 1.0000000000")
	print(f"   ✓ Group closure" if abs(det(U_prod) - 1.0) < 1e-10 else "   ✗ Group closure violated!")
	
	# Test inverse property
	print("\n4. Testing inverse property:")
	U = hstart()
	U_dag = dag(U)
	U_identity_test = mult(U, U_dag)
	print(f"   U·U† trace = {tr(U_identity_test):.10f}")
	print(f"   Expected: 2.0000000000 (identity)")
	print(f"   ✓ Inverse property" if abs(tr(U_identity_test) - 2.0) < 1e-10 else "   ✗ Inverse property failed!")
	
	# Test lattice navigation
	print("\n5. Testing lattice navigation:")
	La = [4, 4, 4, 8]  # Small test lattice
	test_point = [2, 3, 1, 5]
	idx = p2i(test_point, La)
	recovered_point = i2p(idx, La)
	print(f"   Original point: {test_point}")
	print(f"   Index: {idx}")
	print(f"   Recovered: {list(recovered_point)}")
	print(f"   ✓ Bijection works" if all(test_point[i] == recovered_point[i] for i in range(4)) else "   ✗ Bijection failed!")
	
	print("\n" + "=" * 40)
	print("All tests completed successfully!" if all([
		abs(det(U_random) - 1.0) < 1e-10,
		abs(tr(U_identity) - 2.0) < 1e-10,
		abs(det(U_prod) - 1.0) < 1e-10,
		abs(tr(U_identity_test) - 2.0) < 1e-10,
		all(test_point[i] == recovered_point[i] for i in range(4))
	]) else "Some tests failed - check implementation!")
	print("\nModule ready for use in lattice QCD simulations.")