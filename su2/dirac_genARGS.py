# MIGHT NOT WANT AT ALL

import numpy as np
from scipy import linalg
from time import time
# import newsu2 as su2
import su2
import sys
import scipy.sparse
import scipy.sparse.linalg
import pickle
import checks
import argparse

# caa - made m, r floats
m = 1.
r = 1.
L = T = 2
V = L*L*L*T 
La = [L,L,L,T]
p = [L-1,L-1,L-1,T-1]
rowCol = 2 * (4 * (su2.p2i(p,La) + 1))

fmass = open("datMass.dat","w")

def gen_dirac(U, m, spaceLength, timeLength, r=1.):
# def genDirac():
	"""Generates a set of preconditioned matrices which comprise a 
	Dirac matrix

	Parameters
	----------

	Returns
	-------
	De : array_like
		A preconditioned matrix containing the elements which connect
		an "even" point on the lattice with an "odd" point. Is half the
		order of the full Dirac matrix
	Do : array_like
		A preconditioned matrix containing the elements which connect
		an "odd" point on the lattice with an "even" point. All 
		elements are kinetic terms and De is half the order of the full
		Dirac matrix 
	Dm : array_like
		A preconditioned matrix containing the elements containing the 
		mass terms. Is half the order of the full Dirac matrix.
	"""

# 	# gauge fields
# 	if(L==4):
# #		U = pickle.load(open('Configs/Ucold.0','rb'), encoding='latin1')
# 		U = pickle.load(open('Configs/Uhot_030520.pkl','rb'), encoding='latin1')
# 	if(L==2):
# 		U = pickle.load(open('Configs/Ucold_2_2_2_2','rb'), encoding='latin1')
# 	# U = pickle.load(open('Configs/quSU2_b0.4_4_4.100','rb'), encoding='latin1')
	# U = pickle.load(open('Configs/Uhot.0','rb'), encoding='latin1')

	L = spaceLength
	T = timeLength
	V = L*L*L*T
	La = [L,L,L,T]
	p = [L-1,L-1,L-1,T-1]
	rowCol = 2 * (4 * (su2.p2i(p,La) + 1))


	numdim = 4
	print("L = %i, T = %i" % (L,T))
	print("m = %.2f, r = %.2f" % (m,r))
	print()

	# check plaq
	mups = su2.getMups(V,numdim,La)

	Uplaq = su2.calcPlaq(U[1],V,mups)

	diff = Uplaq - U[0]
	if np.abs(diff) > 1e-8:
		print("Gauge fields failed to read in successfully")
		print(Uplaq, U[0])
	else:
		print("Gauge fields read in successfully, diff = ", diff)
		

	# lists get filled up with positions and data to create a sparse matrix
	rowEven = []
	colEven = []
	datEven = []

	rowOdd = []
	colOdd = []
	datOdd = []

	rowMass = []
	# colMass = []
	datMass = []

	row = []
	col = []
	dat = []


	tSgen = time()

	counts = 0

	# assigns the proper values for the matrix
	for i in range(0,rowCol,8):# runs over xyzt
		ec = 0

		space_i, b, beta = su2.getPoint(La,rowCol,i)

		# diagonal is all m + r diagonal matrices
		# su2.masseo(rowMass,colMass,datMass,i,i,m,r)
		su2.masseo(rowMass,datMass,i,i,m,r)
		fmass.write("%.1f\t%i,%i\n" % (m+r,i,i))
		# print('called at i = ' + str(i) + '\t(i//8//2)*8 = ', (i//8//2)*8)
		# su2.mass(row,col,dat,i,i,m,r)


		# kinetic terms
		for mu in range(4):
			space_j = space_i.copy()
			if (space_i[mu]+1) == La[mu]:
				pbc = True
			else:
				pbc = False

			# print(b, beta)
			space_j[mu] = (space_i[mu]+1)%La[mu] # mod (%) is PBC
			# j = su2.getIndex(La,space_j,b,beta) #consider getting rid of this line in favor of the next
			# j = su2.getIndex(La,space_j)
			j = 8 * su2.p2i(space_j,La)

			ie = su2.parity(space_i)
			je = su2.parity(space_j)

			# (ec_norm,cc3) = su2.initD(row,col,dat,mu,U,r,i,j,pbc)	
			# (ec_norm,cc3) = su2.initD(row,col,dat,mu,U,r,j,i,pbc)	

			# if ec_norm != 0:
			# 	print(ec_norm, i, j)

			if ie==0 and je==1:
				(ec, cc1) = su2.initDeo(rowEven,colEven,datEven,mu,U,r,i,j,pbc)
				(ec, cc2) = su2.initDeo(rowOdd,colOdd,datOdd,mu,U,r,j,i,pbc)		

			counts += cc1 + cc2
	# print()
	# print('i= ', i)
	# print('rowCol= ', rowCol)
	# print('rowCol//2= ', rowCol//2)
	# print(rowMass)

	# Note: if this try fails, then you'll probably get an error that De,Do,Dm referenced before assignment
	# try:
		# De = D_eo in notes

	De = scipy.sparse.csc_matrix((datEven,(rowEven,colEven)), shape = (rowCol//2,rowCol//2))
	De = De/2.0

	# Do = D_oe in notes
	Do = scipy.sparse.csc_matrix((datOdd,(rowOdd,colOdd)), shape = ((rowCol//2),(rowCol//2)))
	Do = Do/2.0

	# mass 
	# Dm = scipy.sparse.csc_matrix((datMass,(rowMass,colMass)), shape = ((rowCol//2),(rowCol//2)))
	Dm = scipy.sparse.csc_matrix((datMass,(rowMass,rowMass)), shape = ((rowCol//2),(rowCol//2)))
	Dm = Dm/2.0 #for some reason mass terms in Dfull are 2* mass in Dm
	
	# tyring to make an un-preconditioned matrix for testing
	Dfull = scipy.sparse.csc_matrix((dat,(row,col)), shape = (rowCol,rowCol))
	Dfull = Dfull/2.0


	sparseSuccess = True
	# except:
	# 	print('Error converting to sparse')
	# 	sparseSuccess = False

	tEgen = time()

	if ec==0:
		print('Created Dirac matrix successfully')
		print("time gen :", su2.getTime(tSgen,tEgen))
	else:
		print('Failure creating Diract matrix, error code: ',ec)

	return De,Do,Dm

def invert_dirac(De,Do,Dm, sizeofinverse=8):
	"""Takes in the even/odd preconditioned Dirac matrices and finds 
	the inverse of the full Dirac matrix

	Parameters
	----------
	De : array_like
		A preconditioned matrix containing the elements which connect
		an "even" point on the lattice with an "odd" point. Is half the
		order of the full Dirac matrix
	Do : array_like
		A preconditioned matrix containing the elements which connect
		an "odd" point on the lattice with an "even" point. All 
		elements are kinetic terms and De is half the order of the full
		Dirac matrix 
	Dm : array_like
		A preconditioned matrix containing the elements containing the 
		mass terms. Is half the order of the full Dirac matrix.
	
	Returns
	-------
	numpy.ndarray
		The inverse of the Dirac matrix. By default only the first 8 
		rows of the inverse are calculated.

	Other Parameters
	----------------
	sizeofinverse : int
		The number of rows of the inverse that will be calculated.
	"""
	
	# imput De, Do, and Dm from getDirac to invert 
	Me = Dm**2 - De.dot(Do)
	b_o = np.zeros((rowCol//2,1))

	# inversion
	B = scipy.sparse.identity(rowCol//2).tocsc()
	x_e = []
	x_o = []
	ncount = 0

	total_s = time()
	for n in range(sizeofinverse):
		s = time()
		b_e = np.transpose(B[n].toarray())

		b_ep = Dm.dot(b_e) - De.dot(b_o) # messes up the dimensions 
		inv = scipy.sparse.linalg.bicgstab(Me,b_ep)
		print('\tInverse code =',inv[1],'for n =',n)
		x_e.append(inv[0])
		x_oi = (b_o[:,0] - Do.dot(x_e[ncount])) / (m+r)

		# print(x_oi)
		x_o.append(x_oi)

		ncount += 1

		e = time()


	total_e = time()
	print("time inv :", su2.getTime(total_s,total_e))

	# except:
	# 	print "Error inverting"

	# setting up x
	x = np.zeros((sizeofinverse,rowCol),dtype=complex)
	x_e = np.array(x_e)
	x_o = np.array(x_o)


	for a in range(2):
		for alpha in range(4):
			for t in range(T):
				if t % 2 == 0:
					x[:,a + 2*alpha + 8*L**3*t] = x_e[:,a + 2*alpha + 8*L**3*(t//2)]
				else:
					x[:,a + 2*alpha + 8*L**3*t] = x_o[:,a + 2*alpha + 8*L**3*(t//2)]

	return x


def corr(x):
	# writes out the correlator to a file
	# input x from invertDirac
	correlator = open('/Users/anthonygirardi/LQCDResearch/Code/Correlators/pion_pt_pt_m%.1f_b2.4.dat' % (m),'w')
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


# De,Do,Dm = genDirac()
# x = invertDirac(De,Do,Dm)
# Dd,Dinv = checks.denseInvertDirac(De,Do,Dm,rowCol) 
# Dinv = checks.fullInvertDirac(Dfull,rowCol)
# checks.checkInverse(Dinv,x,rowCol,1e-3)
# print(np.sum(Dinv[0,:] - x[0]))
# corr(x)
# checks.checkDirac(De,Do)
# fmass.close()


'''

	I'm adding this next bit for the case where someone has lattices already
	generated and just wants to invert them


'''
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='SU(2) Dirac Matrix Generator and Inverter')

	parser.add_argument('--fname','-f', required=True,
						help='Name of the pickled file that contains an SU(2) matrix',
						type=str)
	parser.add_argument('--spaceLength','-sl', required=True,
						help='Sets the length of the spacial dimensions of the lattice',
						type=int)
	parser.add_argument('--timeLength','-tl', required=True,
						help='Sets the length of the time dimension of the lattice',
						type=int)
	parser.add_argument('--mass','-m', required=True,
						help='Mass of the particle',
						type=float)
	parser.add_argument('--wilson','-w',
						help='Value of the wilson term',
						default=1.0,
						type=float)

	parser.parse_args()
	args = parser.parse_args()

	fname = args.fname
	spaceLength = args.spaceLength
	timeLength = args.timeLength
	m = args.mass
	U = pickle.load(open(fname,'rb'),encoding='latin1')

	De,Do,Dm = gen_dirac(U,m,spaceLength,timeLength,r)
	x = invert_dirac(De,Do,Dm)
	corr(x)
