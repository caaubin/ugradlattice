import math
import numpy
np = numpy

su2eye = np.array([1.,0.,0.,0.])
eye4 = np.eye(4)
g0 = np.array([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]])
g1 = np.array([[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]])
g2 = np.array([[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]])
g3 = np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
gammas = np.array((g0,g1,g2,g3))

prod = np.dot
xprod = np.cross
add = np.add

def dagger(u):
	"""Gives the hermitian conjugate of a matrix

	Parameters
	----------
	u : array_like
		Matrix representing a gauge field

	Returns
	-------
	numpy.ndarray
		Hermitian conjugate of the input

	"""

	return np.transpose(np.conjugate(u))


def vol(La):
	"""Takes array of dimensions as input, returns volume

	Parameters
	----------
	La : array_like
		Array where each element describes the length of one 
		dimension of the lattice ([x,y,z,t])


	Returns
	-------
	int
		The volume of the lattice, which is equivalent to the number of
		points on the lattice

	"""

	product = 1
	for x in range(len(La)):
		product *= La[x]
	return product

def dim(La):
	"""Returns the dimensions of the array in a dictionary

	Parameters
	----------
	La : array_like
		Array where each element describes the length of one 
		dimension of the lattice ([x,y,z,t])

	Returns
	-------
	dict
		Dictionary containing the lattice dimensions

	"""

	D = {}
	for x in range(len(La)):
		D.update({x:La[x]})
	return D


def dag(U):
	"""Gives the hermitian conjugate of a matrix writen in the 
	real-valued representation of an SU(2) matrix

	Parameters
	----------
	u : array_like
		Real-valued matrix representaion of an SU(2) matrix

	Returns
	-------
	numpy.ndarray
		Hermitian conjugate of the input written as a real-valued 
		representation of an SU(2) matrix

	"""

	return np.array([1,-1,-1,-1])*U

def mult(U1, U2):
	"""Multiplies two SU(2) matrices written in the real-valued 
	representation

	Parameters
	----------
	U1 : array_like
		Real-valued matrix representaion of an SU(2) gauge field
	U2 : array_like
		Real-valued matrix representaion of an SU(2) gauge field

	Returns
	-------
	numpy.ndarray
		The product of the two input arrays written as a real-valued 
		matrix representaion of an SU(2) matrix	

	"""
	
	a0 = U1[0]
	b0 = U2[0]
	a = U1[1:]
	b = U2[1:]

	c0 = a0 * b0 - prod(a, b)
	c = b0*a + a0*b - xprod(a, b)
	return np.array((c0, c[0], c[1], c[2]))

def p2i(point,La):
	"""Takes the array describing a point in the spacetime lattice 
	([x,y,z,t] notation) and returns the index of that point

	Parameters
	----------
	point : array_like
		Array containing the spacetime coordinates of a position on 
		the lattice written as [x,y,z,t]
	La : array_like
		Array where each element describes the length of one 
		dimension of the lattice ([x,y,z,t])

	Returns
	-------
	int
		The index of the input point


	"""

	return (La[2] * La[1] * La[0] * point[3]) + (La[1] * La[0] * point[2]) + (La[0] * point[1]) + (point[0])


def i2p(ind,La):
	"""Takes the index of a point on the lattice and returns the 
	spacetime position of that point

	Parameters
	----------
	ind : int
		As the elements of the Dirac matrix are themselves 8x8 
		matrices, i is the index of the first row/colum of the elements
		in the Dirac matrix which pertain to a particular spacetime 
		position
	La : array_like 
		Array where each element describes the length of one dimension
		of the lattice ([x,y,z,t])

	Returns
	-------
	numpy.ndarray
		The position on the lattice which corresponds to the input 
		index. The position is written as [x,y,z,t]

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
	"""Returns the parity of a point on the lattice

	Parameters
	----------
	pt : array_like
		Point on the lattice, written as [x,y,z,t]

	Returns
	-------
	numpy.int64
		Returns 1 if the point has even parity, and 0 if it has odd 
		parity
	"""

	return np.sum(pt)%2


def hstart():
	"""Returns a random complex 2x2 matrix written in real-valued form

	Parameters
	----------
	

	Returns
	-------
	numpy.ndarray
		2x2 matrix written in real-valued form. Elements are assigned 
		to random values between -1 and 1
	"""

	a = np.array([np.random.uniform(-1.,1.), np.random.uniform(-1.,1.), np.random.uniform(-1.,1.)])
	while (np.sqrt(a[0]**2 + a[1]**2 + a[2]**2) >= 1):
		a[0] = np.random.uniform(-1.,1.)
		a[1] = np.random.uniform(-1.,1.)
		a[2] = np.random.uniform(-1.,1.)
	a0 = np.sqrt(1 - (a[0]**2 + a[1]**2 + a[2]**2))
	if (np.random.random() > 0.5):
		a0 = -a0;
	return np.array((a0, a[0], a[1], a[2]))


def update(UU):
	"""Make a random SU(2) matrix near the identity

	Parameters
	----------
	UU : array_like
		SU(2) matrix written in real-valued form

	Returns
	-------
	numpy.ndarray
		Updated version of the input matrix with a slight random 
		modification. Matrix is near the identity and written in 
		real-valued form
	"""

	g = np.array([1.,0.,0.,0.]) + 0.1*hstart()*np.array([0.,1.,1.,1.])
	gU = mult(g,UU)
	gU /= det(gU) # make sure it is still SU(2)

#	return hstart()
	return gU


def cstart():
	"""Returns 2x2 identity matrix

	Parameters
	----------

	Returns
	-------
	numpy.ndarray
		2x2 Identity matrix written in the convention of a real-valued 
		SU(2) matrix
	"""

	return su2eye
	

def tr(UU):
	"""Return the trace of a matrix

	Parameters
	----------
	UU : array_like
		SU(2) matrix writen in real-valued form

	Returns
	-------
	numpy.float64
		The trace of the input matrix
	"""

	return UU[0] * 2
	

def det(UU):
	"""Returns the determinant of the matrix
	
	Parameters
	---------
	UU : array_like
		SU(2) matrix writen in real-valued form

	Returns
	-------
	numpy.float64
		Determinant of the input matrix
	"""

	return prod(UU,UU)


def mupi(ind, mu, La):
	"""Increment a position in the mu'th direction, looping if needed	

	Parameters
	----------
	ind : int
		the index of a point on the lattice
	mu : int
		Index corresponding to one of the directions on the lattice: 
		0:x, 1:y, 2:z, 3:t
	La : array_like
		Array where each element describes the length of one dimension
		of the lattice ([x,y,z,t])

	Returns
	-------
	numpy.int64 
		The function increments a step in the mu'th direction, if the 
		boundary is met it then loops around to the other side of the 
		lattice. The return value is the index of the new point on the 
		lattice.
	"""

	pp = i2p(ind,La)
	if (pp[mu] + 1 >= La[mu]):
		pp[mu] = 0
	else:
		pp[mu] += 1
	return p2i(pp,La)


# NOTE: THIS FUNCTION IS NEVER USED IN PvB
def getMups(V,numdim,La):
	"""Returns the mups array

	Parameters
	----------
	V : int
		The volume of the lattice, which is equivalent to the number of
		points on the lattice
	numdim : int
		Number of dimensions of the lattice
	La : array_like
		Array where each element describes the length of one 
		dimension of the lattice ([x,y,z,t])

	Returns
	-------
	numpy.ndarray
		The output is a V x numdim matrix array which can be used as 
		shorthand when calling elements from the gaugefield array. 
		Specifically, this array can be used in functions which 
		involve stepping up or down between points on the lattice.
		In the output matrix array, the ith array corresponds to the 
		ith point on the lattice, and the elements of that array are 
		the indexes of the adjacent points. For example, the [i,mu] 
		element in the output array is the index of point on the 
		lattice corresponding to the point one step in the mu'th 
		direction.  
	"""

	mups = np.zeros((V,numdim), int)
	for i in range(0, V):
		for mu in range(0, numdim):
			mups[i,mu] = mupi(i, mu, La)

	return mups


def mdowni(ind, mu, La):
	"""Decrement a position in the mu'th direction, looping if needed
	
	Parameters
	----------
	ind : int
		As the elements of the Dirac matrix are themselves 8x8 
		matrices, i is the index of the first row/colum of the elements
		in the Dirac matrix which pertain to a particular spacetime 
		position
	mu : int
		Index corresponding to one of the directions on the lattice: 
		0:x, 1:y, 2:z, 3:t
	La : array_like
		Array where each element describes the length of one dimension
		of the lattice ([x,y,z,t])

	Returns
	-------
	numpy.int64
		The function deccrements a step in the mu'th direction, if the 
		boundary is met it then loops around to the other side of the 
		lattice. The return value is the index of the new point on the 
		lattice.
	"""

	pp = i2p(ind, La)
	if (pp[mu] - 1 < 0):
		pp[mu] = La[mu] - 1
	else:
		pp[mu] -= 1
	return p2i(pp, La)


def plaq(U, U0i, mups, mu, nu):
	"""Compute the plaquette	
	
	Paramters
	---------
	U : array_like
		Array containing the gaugefields for every point on the lattice
	U0i : int
		Lattice point index of the starting point on the lattice for 
		the calculation.
	mups : array_like
		The mups array. This array is used as shorthand for taking a 
		step forwards in the mu'th direction from the U0i'th point
	mu : int
		Index corresponding to one of the directions on the lattice: 
		0:x, 1:y, 2:z, 3:t
	nu : int
		Index corresponding to another direction on the lattice: 
		0:x, 1:y, 2:z, 3:t.

	Returns
	-------
	numpy.float64
		The value of the plaquette 
	"""

	# Forward only
	U0 = U[U0i][mu].copy()
	U1 = U[mups[U0i, mu]][nu].copy()
	U2 = dag(U[mups[U0i, nu]][mu].copy())
	U3 = dag(U[U0i][nu].copy())

	return tr(mult(mult(U0,U1),mult(U2,U3)))


def Wilsonloop(i, j, U, U0i, mups, mdns, mu, nu):
        """Compute the Wilson loop	
	
	Paramters
	---------
	i: int
                Index corresponding to number of links to be moved in the mu
                direction
	j: int
                Index corresponding to the number of links to be moved in the
                nu direction
	U : array_like
		Array containing the gaugefields for every point on the lattice
	U0i : int
		Lattice point index of the starting point on the lattice for 
		the calculation.
	mups : array_like
		The mups array. This array is used as shorthand for taking a 
		step forwards in the mu'th direction from the U0i'th point
	mu : int
		Index corresponding to one of the directions on the lattice: 
		0:x, 1:y, 2:z, 3:t
	nu : int
		Index corresponding to another direction on the lattice: 
		0:x, 1:y, 2:z, 3:t.

	Returns
	-------
	numpy.float64
		The value of the Wilson loop
	"""
        Uij = [U[U0i][mu]]
        Ui = [U0i]
        for a in range(1, i):
                Ui.append(mups[Ui[a-1], mu])
                Uij.append(U[Ui[a]][mu])
        Uij.append(U[mups[Ui[-1],mu]][nu])
        Uj = [mups[Ui[-1],mu]]
        for b in range(1, j):
                Uj.append(mups[Uj[b-1], nu])
                Uij.append(U[Uj[b]][nu])
        Uij.append(dag(U[mups[mdns[Uj[-1],mu],nu]][mu])) 
        Uidag = [mups[mdns[Uj[-1],mu],nu]] 
        for c in range(1, i):
                Uidag.append(mdns[Uidag[c-1],mu])
                Uij.append(dag(U[Uidag[c]][mu]))
        Uij.append(dag(U[mdns[Uidag[-1],nu]][nu]))
        Ujdag = [mdns[Uidag[-1], nu]]
        for d in range(1, j):
                Ujdag.append(mdns[Ujdag[d-1], nu])
                Uij.append(dag(U[Ujdag[d]][nu]))
        f = len(Uij)

        print(Uij)

        product = eye4
        for e in range(0, f): 
                product = mult(product, Uij[e])
        return tr(product).real 


def link(U,U0i,mups,mu):
	"""Returns the trace of the link between two points

	Parameters
	---------
	U : array_like
		Array containing the gaugefields for every point on the lattice
	U0i : int
		Lattice point index of the starting point on the lattice for 
		the calculation.
	mups : array_like
		The mups array. This array is used as shorthand for taking a 
		step forwards in the mu'th direction from the U0i'th point
	mu : int
		Index corresponding to one of the directions on the lattice: 
		0:x, 1:y, 2:z, 3:t

	Returns
	-------
	numpy.float64
		The value of the link between the point at U0i and the point
		one step in the mu'th direction
	"""

	U0 = U[U0i][mu].copy()

	return tr(U0)

def getstaple(U, U0i, mups, mdns, mu):
	"""Returns the value of the staple

	Parameters
	----------
	U : array_like
		Array containing the gaugefields for every point on the lattice
	U0i : int
		Lattice point index of the starting point on the lattice for 
		the calculation.
	mups : array_like
		The mups array. This array is used as shorthand for taking a 
		step forwards in the mu'th direction from the U0i'th point
	mdns : array_like
		The mdns array. This array is used as shorthand for taking a 
		step backwards in the mu'th direction from the U0i'th point
	mu : int
		Index corresponding to one of the directions on the lattice: 
		0:x, 1:y, 2:z, 3:t

	Returns
	-------
	numpy.ndarray
		Returns the staple starting at the U0i'th point
	"""
	
	value = 0.0
	mm = list(range(4))
	mm = [i for i in range(4)]
	mm.remove(mu)
	for nu in mm:
#		if nu != mu:
		# Forward staple components
		value += staple(U, U0i, mups, mdns, mu, nu, 1)
	
		# Reverse staple components
		value += staple(U, U0i, mups, mdns, mu, nu, -1)
	return value

# Compute the staple in the mu-nu plane
def staple(U, U0i, mups, mdns, mu, nu, signnu):
	"""Compute the staple in the mu-nu plane

	Parameters
	----------
	U : array_like
		Array containing the gaugefields for every point on the lattice
	U0i : int
		Lattice point index of the starting point on the lattice for 
		the calculation.
	mups : array_like
		The mups array. This array is used as shorthand for taking a 
		step forwards in the mu'th direction from the U0i'th point
	mdns : array_like
		The mdns array. This array is used as shorthand for taking a 
		step backwards in the mu'th direction from the U0i'th point
	mu : int
		Index corresponding to one of the directions on the lattice: 
		0:x, 1:y, 2:z, 3:t
	nu : int
		Index corresponding to one of the directions on the lattice: 
		0:x, 1:y, 2:z, 3:t. Must be different than mu
	signnu : int
		Equal to either 1 or -1. Dictates if the staple is calulcuted
		forwards (+1) or in reverse (-1).

	Returns
	-------
	numpy.ndarray
		Returns the staple starting at the U0i'th point
	"""

	if (signnu == 1): # Forward
		U1 = U[mups[U0i, mu]][nu].copy()
		U2 = dag(U[mups[U0i, nu]][mu].copy())
		U3 = dag(U[U0i][nu].copy())
	else: # Reverse
		U1 = dag(U[mdns[mups[U0i, mu],nu]][nu].copy())
		U2 = dag(U[mdns[U0i, nu]][mu].copy())
		U3 = U[mdns[U0i, nu]][nu].copy()

	return mult(mult(U1,U2),U3)

# 
def calcPlaq(U,V,mups):
	"""Calculates the average value of the plaquettes about all points 
	in the lattice

	Parameters
	----------
	U : array_like
		Array containing the gaugefields for every point on the lattice
	V : int
		The volume of the lattice, which is equivalent to the number of
		points on the lattice
	mups : array_like
		The mups array. This array is used as shorthand for taking a 
		step forwards in the mu'th direction from the U0i'th point

	Returns
	-------
	numpy.float64
		The average value of the plaquettes about the whole lattice	

	"""
	plaquettes = np.zeros(6*V) # is 6 * V correct? 
	j = 0
	for i in range (V):
		for mu in range(4):
			for nu in range(mu+1,4):
				plaquettes[j] = 1.0 - 0.5*plaq(U,i,mups,mu,nu)
				j = j + 1
	avgPlaquettes = np.mean(plaquettes)

	# print(type(avgPlaquettes))

	return avgPlaquettes

def calcU_i(U,V,La,mups):
	"""Calculates the average values of the spacial links in the 
	lattice

	Parameters
	---------
	U : array_like
		Array containing the gaugefields for every point on the lattice
	V : int
		The volume of the lattice, which is equivalent to the number of
		points on the lattice
	mups : array_like
		The mups array. This array is used as shorthand for taking a 
		step forwards in the mu'th direction from the U0i'th point

	Returns
	-------
	numpy.float64
		The average value of the spacial links in the lattice

	"""

	spaceLink = np.zeros((3*V))
	j = 0
	for i in range(V):
		for mu in range(len(La)-1):
			spaceLink[j] = link(U,i,mups,mu)
			j = j + 1
	U_i = np.mean(spaceLink)
	return U_i/2.


def calcU_t(U,V,mups):
	"""Calculates the average values of the time links in the lattice

	Parameters
	---------
	U : array_like
		Array containing the gaugefields for every point on the lattice
	V : int
		The volume of the lattice, which is equivalent to the number of
		points on the lattice
	mups : array_like
		The mups array. This array is used as shorthand for taking a 
		step forwards in the mu'th direction from the U0i'th point

	Returns
	-------
	numpy.float64
		The average value of the time links in the lattice

	"""

	timeLink = np.zeros((V))
	j = 0
	for i in range(V):
		timeLink[i] = link(U,i,mups,3) 
	U_t = np.mean(timeLink)
	return U_t/2.

# matrix functions

# 2*mass 8x8 identity matrix
# def masseo(row,col,dat,i,j,m,r):
# 	y = 0
# 	for x in range(8):
#  		row.append((i//8//2)*8+x)
#  		col.append((j//8//2)*8+y)
#  		# row.append((i/8/2)*8+x)
#  		# col.append((j/8/2)*8+y)
#  		dat.append(2*(m+r)) 
#  		y += 1

# this version doesn't multiply (m+r) by 2
# def masseo(row,col,dat,i,j,m,r):
# 	y = 0
# 	for x in range(8):
#  		row.append((i//8//2)*8+x)
#  		col.append((j//8//2)*8+y)
#  		# row.append((i/8/2)*8+x)
#  		# col.append((j/8/2)*8+y)
#  		dat.append(m+r)
#  		# print(dat) 
#  		y += 1


def masseo(row,dat,i,j,m,r):
	"""Generates the data needed to make an 8x8 sparse submatrix
	containing the mass terms of the dirac matrix

	Parameters
	----------
	row : list
		list containing row indices for the non-zero elements of the 
		dirac matrix. Is used to generate the dirac matrix as a sparse
		matrix.
	dat : list
		list containing the values of the non-zero elements of the
		dirac matrix. Is used to generate the dirac matrix as a sparse
		matrix.
	i : int
		row index of the Dirac matrix which, when used in conjunction 
		with j, will locate the first diagonal element of the submatrix
	j : int
		column index of the Dirac matrix which, when used in 
		conjunction with i, will locate the first diagonal element of 
		the submatrix
	m : double
		the mass of the particle
	r : double
		the value of the wilson term

	Returns
	-------
	void
		Appends the row and dat lists with the data needed to construct
		an 8x8 submatrix which will be part of the larger dirac matrix. 
		All mass terms should be along the diagonal of each submatrix 
		as well as the larger dirac matrix.

	"""

	y = 0
	# print('@:', i//8//2 * 8) # debug
	for x in range(8):
		# print((i//8//2)*8+x)
		# caa - put this in to stop duplicates
		# caa - because for eo prec, the mass should
		# caa - only include half as many entries
		# caa - possibly better way to do this.
		# caa - put back in the 2*(m+r) bc of 1/2 later
		if ((i//8//2)*8+x) not in row:
			row.append((i//8//2)*8+x)
		# col.append((j//8//2)*8+y)
		# row.append((i/8/2)*8+x)
		# col.append((j/8/2)*8+y)
			dat.append(2*(m+r)) 
		# print(dat) 
			y += 1

# 2*mass 8x8 identity matrix
def mass(row,col,dat,i,j,m,r):
	y = 0
	for x in range(8):
 		row.append(i+x)
 		col.append(j+y)
 		dat.append(2*(m+r)) 
 		y += 1


# all of the below functions generate an 8x8 matrix that is the Kronecker Product of one of the 
# gamma matrices with the gauge field, which needs to be input.

def showU(U,mu,i):
	"""Returns the gauge field at the i'th lattice point and in the 
	mu'th direction written as matrix writen in the real-valued 
	representation of an SU(2) matrix

	Returns
	-------
	U : array_like
		Real-valued matrix representaion of an SU(2) gauge field 
	mu : int
		Index corresponding to one of the directions on the lattice: 
		0:x, 1:y, 2:z, 3:t
	i : int
		row index of the Dirac matrix which, when used in conjunction 
		with j, will locate the first diagonal element of the submatrix

	Returns
	-------
	numpy.ndarray
		The gauge field which corresponds to the mu'th direction on the
		i'th point on the lattice. It is a 2x2 matrix written in the 
		real-valued representation

	"""

	u = np.array([[U[1][i,mu,0]+U[1][i,mu,3]*1j,    U[1][i,mu,2] + U[1][i,mu,1]*1j],
		         [-U[1][i,mu,2]+U[1][i,mu,1]*1j,    U[1][i,mu,0] - U[1][i,mu,3]*1j]])

	# u /= 2

	return u

# consider changing m,n to i,j for consistency
def initD(row,col,dat,mu,U,r,m,n,pbc):
	"""Generates the data needed to make an 8x8 sparse submatrix 
	containing the kinetic terms (including time) of the dirac matrix. 


	Parameters
	----------
	row : list
		list containing row indices for the non-zero elements of the 
		dirac matrix. Is used to generate the dirac matrix as a sparse
		matrix.
	col : list
		list containing column indices for the non-zero elements of the 
		dirac matrix. Is used to generate the dirac matrix as a sparse
		matrix.
	dat : list
		list containing the values of the non-zero elements of the
		dirac matrix. Is used to generate the dirac matrix as a sparse
		matrix.
	mu : int
		Index corresponding to one of the directions on the lattice: 
		0:x, 1:y, 2:z, 3:t
	U : array_like
		Array containing the gaugefields for every point on the lattice
	r : double
		the value of the wilson term
	m : int
		row index of the Dirac matrix which, when used in conjunction 
		with n, will locate the first diagonal element of the submatrix
	n : int
		column index of the Dirac matrix which, when used in 
		conjunction with m, will locate the first diagonal element of 
		the submatrix
	pbc : bool
		boolean variable indicating whethere the particle has triggered
		periodic boundary conditions. Is true when the particle hit a 
		boundary and had to be moved to the opposite end of the
		lattice, and false otherwise.

	Returns
	-------
	int, int
		Appends the row, col, and dat lists with the data needed to 
		construct an 8x8 submatrix which will be part of the larger 
		dirac matrix. The submatrix is the kronecker product of the
		gamma matrix corresponding to the mu direction and the 
		gaugefield connecting the lattice points given by 
		su2.i2p(m//8,La) and su2.i2p(n//8,La). Returns error codes if 
		an error occurs or 0 otherwise, and the number of elements 
		initialized. 
	"""

	count = 0
	try:
		gam = gammas[mu]
	except:
		return 101


	try:
		if m>n and pbc or m<n and not pbc: # moving up
			Ui = m//8
			s = 1
		else:							   # moving down
			Ui = n//8
			s = -1
	except:
		return 102


	try:
	# print(type(Ui),type(mu))
		u = np.array([[U[1][Ui,mu,0] + U[1][Ui,mu,3]*1j,    U[1][Ui,mu,2] + U[1][Ui,mu,1]*1j],
					 [-U[1][Ui,mu,2] + U[1][Ui,mu,1]*1j,    U[1][Ui,mu,0] - U[1][Ui,mu,3]*1j]])
	except:
		return 103


	try:
		if s==1:
			D = np.kron(gam,u)
			# wilson term
			W = np.kron(eye4,u) * (-r)
		elif s==-1:
			D = np.kron(gam,dagger(u))
			# wilson term
			W = np.kron(eye4,dagger(u)) * (-r)
	except:
		return 104

	try:
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
	"""Generates the data needed to make an 8x8 sparse, even/odd 
	precoditioned submatrix containing the kinetic terms (including 
	time) of the dirac matrix. 

	Parameters
	----------
	row : list
		list containing row indices for the non-zero elements of the 
		dirac matrix. Is used to generate the dirac matrix as a sparse
		matrix.
	col : list
		list containing column indices for the non-zero elements of the 
		dirac matrix. Is used to generate the dirac matrix as a sparse
		matrix.
	dat : list
		list containing the values of the non-zero elements of the
		dirac matrix. Is used to generate the dirac matrix as a sparse
		matrix.
	mu : int
		Index corresponding to one of the directions on the lattice: 
		0:x, 1:y, 2:z, 3:t
	U : array_like
		Array containing the gaugefields for every point on the lattice
	r : double
		the value of the wilson term
	m : int
		row index of the Dirac matrix which, when used in conjunction 
		with n, will locate the first diagonal element of the submatrix
	n : int
		column index of the Dirac matrix which, when used in 
		conjunction with m, will locate the first diagonal element of 
		the submatrix
	pbc : bool
		boolean variable indicating whethere the particle has triggered
		periodic boundary conditions. Is true when the particle hit a 
		boundary and had to be moved to the opposite end of the
		lattice, and false otherwise.

	Returns
	-------
	int, int
		This function takes advantage of even/odd preconditioning.
		Since the Dirac matrix can be broken into mass terms and terms
		which either connect "even" points with "odd" points or vice 
		versa, the Dirac matrix can be formulated as three smaller 
		matrices which are half the order of the full matrix. So, this
		fucntion appends the row, col, and dat lists with the data 
		needed to construct an 8x8 submatrix which will be part of the
		larger even-odd or odd-even precoditioned matrix. The submatrix
		is the kronecker product of the gamma matrix corresponding to 
		the mu direction and the gaugefield connecting the lattice 
		points given by su2.i2p(m//8,La) and su2.i2p(n//8,La). Returns
		error codes if an error occurs or 0 otherwise, and the number 
		of elements initialized. 
	"""

	count = 0
	
	try:
		gam = gammas[mu]
	except:
		return (101,count)


	try:
		if m>n and pbc or m<n and not pbc: # moving up
			Ui = m//8
			s = 1
		else:							   # moving down
			Ui = n//8
			s = -1
	except:
		return (102,count)


	try:
		# print(type(Ui))
		u = np.array([[U[1][Ui,mu,0] + U[1][Ui,mu,3]*1j,    U[1][Ui,mu,2] + U[1][Ui,mu,1]*1j],
					 [-U[1][Ui,mu,2] + U[1][Ui,mu,1]*1j,    U[1][Ui,mu,0] - U[1][Ui,mu,3]*1j]])
	except:
		return (103,count)


	try:
		if s==1:
			D = np.kron(gam,u)
			# wilson term
			W = np.kron(eye4,u) * (-r)
		elif s==-1:
			D = np.kron(gam,dagger(u))
			# wilson term
			W = np.kron(eye4,dagger(u)) * (-r)
	except:
		return (104,count)
		
	try:
		for x in range(8):
			for y in range(8):
				if D[x][y] != 0:
					count += 1
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


"""

	Consider deleting, never used in anything and not that helpful
	for debugging either

"""
def getElement(D,La,point_1,point_2,a,alpha,b,beta):
	"""
	input the matrix, the dimensions of the latice, the initial and final point of the particle, 
	and the other spin and field stuff, and get the associated element of the matrix
	point_1, point_2 range from 0 to [max index] - 1
	a, b range from 0 to 1
	alpha, beta range from 0 to 3
	"""
	space_i = 8 * p2i(point_1,La)
	space_j = 8 * p2i(point_2,La)
	return D[space_i + 2*alpha + a][space_j + 2*beta + b]

"""

	consider deleting, python doesn't allow overloading and we don't need
	a and alpha to make the function work 

"""
# def getIndex(La,point,a,alpha):
# 	"""Returns the index of the full Dirac matrix which corresponds to
# 	the element associated with a particular point on the lattice, 
# 	color, and spin 

# 	Parameters
# 	----------
# 	La : array_like
# 		Array where each element describes the length of one 
# 		dimension of the lattice ([x,y,z,t])
# 	point : array_like
# 		Array containing the spacetime coordinates of a position on 
# 		the lattice written as [x,y,z,t]
# 	a : int
# 		corresponds to color charge, can be either 0 or 1
# 	alpha : int
# 		corresponds to spin, ranges from 0 to 3

# 	Returns
# 	-------
# 	int
# 	"""
# 	'''
# 	input the matrix, the dimensions of the latice, the initial and final point of the particle, 
# 	and the other spin and field stuff, and get the associated element of the matrix
# 	point_1, point_2 range from 0 to [max index] - 1
# 	a, b range from 0 to 1
# 	alpha, beta range from 0 to 3
# 	'''

# 	return 8 * p2i(point,La) + 2*alpha + a


"""

	Consider deleting this too, since its kinda pointless since its just 
	8x i2p

"""

def getIndex(La,point_1):
	"""Returns the index of the full Dirac matrix which corresponds to
	the element associated with a particular point on the lattice

	Parameters
	----------
	La : array_like
		Array where each element describes the length of one 
		dimension of the lattice ([x,y,z,t])
	point : array_like
		Array containing the spacetime coordinates of a position on 
		the lattice written as [x,y,z,t]
	
	Returns
	-------
	int
		This function operates very similarly to p2i
	"""
	return 8 * p2i(point_1,La)


"""
	
	Consider whether these should be reworked since we don't need to 
	use spin or color in the code, thought it might be usefull for 
	bebugging or similar reasons

"""

def getPoint(La,rc,i,j):

	for x in range(0,rc,8):
		diff = i - x
		if diff < 8:
			space_i = i2p(x//8,La)
			beta = diff // 2
			b = diff % 2
			break
	for y in range(0,rc,8):
		diff = j - y
		if diff < 8:
			space_j = i2p(y//8,La)
			alpha = diff // 2
			a = diff % 2
			break
	#print "space_i, space_j, b, beta, a, alpha"
	return [space_i, space_j, b, beta, a, alpha]

def getPoint(La,rc,i):	
	for x in range(0,rc,8):
		diff = i - x
		if diff < 8:
			space_i = i2p(x//8,La)
			beta = diff // 2
			b = diff % 2
			break
	#print "space_i, space_j, b, beta, a, alpha"
	return [space_i, b, beta]

def showMr(mat,rc):
	"""Prints the submatrices along the top row of the full Dirac 
	matrix

	Parameters
	---------
	mat : array_like
		The Dirac matrix
	rc : int
		The order of the Dirac matrix

	Returns
	-------
	void
		Prints 8x8 submatrices that exist along the top of the Dirac 
		matrix. 

	"""

	for i in range(0,rc,8):
		# if mat[0][i] != 0 and i != rc:
		print("mat i = %i, lattice i = %i" % (i,i//8))
		print(mat[:8,i:i+8].real)
		print(mat[:8,i:i+8].imag)

def showMc(mat,rc):
	"""Prints the submatrices along the leftmost column of the full 
	Dirac matrix

	Parameters
	---------
	mat : array_like
		The Dirac matrix
	rc : int
		The order of the Dirac matrix

	Returns
	-------
	void
		Prints 8x8 submatrices that exist along the leftmost section of
		the Dirac matrix. 
	"""

	for i in range(0,rc,8):
		print(i)
		print(mat[i:i+8,:8].real)
		print(mat[i:i+8,:8].imag)

def compare(mat1, mat2):
	"""Prints any dissimilar elements between two matrices

	Parameters
	----------
	mat1 : array_like
		A matrix
	mat2 : array_like
		A matrix

	Returns
	-------
	void
		Compares corresponding elements between two matrices and prints
		any values which are not exactly equal. 
	"""

	for i in range(len(mat1)): 
		for j in range(len(mat1[0])): 
			if mat1[i][j] != mat2[i][j]: 
				print("%f,%f\t%i,%i"%(mat1[i][j],mat2[i][j],i,j))

# displays the elapsed time in hr:min:sec format
def getTime(a,b):
	"""Displays the elalpsed time in hr:min:sec format

	Parameters
	----------
	a : float
		The current time given by the time() function in the standard
		time module
	b : float
		The current time given by the time() function in the standard
		time module
	
	Returns
	-------
	string
		Calculates the time elapsed between when a and b were 
		initialized and print the elapsed time in hr:min:sec format.
		Can be used to find the time elapsed by a section of code if a
		is initialized before said section and b is intialized 
		immediately after 
	"""

	t = b-a 
	hrs = divmod(t,3600)
	mins,secs = divmod(hrs[1],60)

	return "%02d:%02d:%06.3f" % (hrs[0],mins,secs)  


def corr(x):
	"""Writes out the correlator to a file
		
	Parameters
	----------
	x : array_like
		The inverse 

	"""
	# input x from invertDirac
	correlator = open('correlators/pion_pt_pt_m%.1f_b2.4.dat' % (m),'w')
	corrt = np.zeros(T)
	for t in range(T):	
		for a in range(2):
			for b in range(2):
				for alpha in range(4):
					for beta in range(4):
						corrt[t] += abs(x[b + 2*beta][a + 2*alpha + 8*L**3*t]) ** 2
		correlator.write(str(t) + '\t' + str(corrt[t]) + '\n')
	correlator.close()

	return 0
