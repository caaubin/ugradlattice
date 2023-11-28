import numpy as np
from scipy import linalg
from time import time
import su2
import scipy.sparse
import scipy.sparse.linalg
import pickle
import checks
import argparse
import os

def gen_dirac(U, m, spaceLength, timeLength, r=1.):
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

	print('\nGenerating the Dirac matrix...\n')

	# initializing variables
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

	# check if lattice was read in correctly
	mups = su2.getMups(V,numdim,La)

	# not sure which one is actually correct
	# Uplaq = 1 - su2.calcPlaq(U[1],V,mups)
	Uplaq = su2.calcPlaq(U[1],V,mups)

	diff = Uplaq - U[0]
	if np.abs(diff) > 1e-8:
		print("Gauge fields failed to read in successfully")
		print(Uplaq, U[0])
	else:
		print("Gauge fields read in successfully, diff = ", diff)

	print()

	row = []
	col = []
	dat = []

	# lists get filled up with positions and data to create a sparse matrix
	rowEven = []
	colEven = []
	datEven = []

	rowOdd = []
	colOdd = []
	datOdd = []

	rowMass = []
	datMass = []

	tSgen = time()

	counts = 0
	cc1 = 0
	cc2 = 0
	# start to initialize the dirac matrix
	for i in range(0,rowCol,8):
		
		ec = 0

		space_i, b, beta = su2.getPoint(La,rowCol,i)

		# diagonal is all m + r diagonal matrices
		# print(i)
		su2.masseo(rowMass,datMass,i,i,m,r)
		su2.mass(row,col,dat,i,i,m,r)
		# print(rowCol)
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

			su2.initD(row,col,dat,mu,U,r,i,j,pbc)
			su2.initD(col,row,dat,mu,U,r,j,i,pbc)
			# print(row[-8:],col[-8:])
			# print('------------------')

			if ie==0 and je==1:
				# print('made it')
				(ec, cc1) = su2.initDeo(rowEven,colEven,datEven,mu,U,r,i,j,pbc)
				(ec, cc2) = su2.initDeo(rowOdd,colOdd,datOdd,mu,U,r,j,i,pbc)		

			counts += cc1 + cc2

		# end of mu loop

	# end of i loop
	
	# Note: if this try fails, then you'll probably get an error that De,Do,Dm referenced before assignment
	# try:
		
	# De = D_eo in notes
	De = scipy.sparse.csc_matrix((datEven,(rowEven,colEven)), shape = (rowCol,rowCol))
	De = De/2.0

	# Do = D_oe in notes
	Do = scipy.sparse.csc_matrix((datOdd,(rowOdd,colOdd)), shape = ((rowCol),(rowCol)))
	Do = Do/2.0

	# mass 
	# Dm = scipy.sparse.csc_matrix((datMass,(rowMass,colMass)), shape = ((rowCol//2),(rowCol//2)))

	# trying to fix the problem of mass values overstepping bounds of Dm
	if len(rowMass) > rowCol//2:
		for i in range(rowCol//2,len(rowMass)):
			rowMass.pop()
			datMass.pop()
		
	Dm = scipy.sparse.csc_matrix((datMass,(rowMass,rowMass)), shape = ((rowCol),(rowCol)))
	Dm = Dm/2.0 #for some reason mass terms in Dfull are 2* mass in Dm
	

	# print(len(row),len(col),len(dat))

	D = scipy.sparse.csc_matrix((dat,(row,col)), shape = ((rowCol),(rowCol)))
	D = D/2.0

	sparseSuccess = True

	# except:

	# 	print('Error converting to sparse')
	# 	sparseSuccess = False

	tEgen = time()

	if ec==0:
		print('Created Dirac matrix successfully')
		print("time :", su2.getTime(tSgen,tEgen))
	else:
		print('Failure creating Diract matrix, error code: ',ec)

	return De,Do,Dm,D

def invert_dirac(De,Do,Dm,L,T,m,r,sizeofinverse=8):
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

	print("\nInverting...\n")
	
	rowCol = De.shape[0] * 2

	# imput De, Do, and Dm from getDirac to invert 
	Me = Dm**2 - De.dot(Do)
	b_o = np.zeros((rowCol//2,1)) # Why?? because we are working at the origin which is even


	# inversion
	B = scipy.sparse.identity(rowCol//2).tocsc() #check dim rowcol
	x_e = []
	x_o = []
	ncount = 0

	total_s = time()
	for n in range(sizeofinverse): 
		
		s = time()
		b_e = np.transpose(B[n].toarray())
		


		print('\nfor n =',n, '\nDm =\n', Dm,'\nb_e =\n', b_e,'\nb_o =\n', b_o,'\n')
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

	#x_e = np.reshape(x_e,(sizeofinverse,sizeofinverse))
	#x_o = np.reshape(x_o,(sizeofinverse,sizeofinverse))

	print('\nxe =\n',x_e, '\nx_o =\n', x_o,'\nx =\n', x)

	for a in range(2):
		for alpha in range(4):
			for t in range(T):
				if t % 2 == 0:
					x[:,a + 2*alpha + 8*L**3*t] = x_e[:,a + 2*alpha + 8*L**3*(t//2)]
				else:
					x[:,a + 2*alpha + 8*L**3*t] = x_o[:,a + 2*alpha + 8*L**3*(t//2)]

	print('\nx =\n', x)
	print('Inverted successfully')	

	return x




# can probably remove the fname bit since I just used it to do a bunch of 
# correlators all at once
def corr(x,L,T,beta,m):
	
	print("\nCalculating the correlator...\n")

	foundCorrDir = False
	# try: 
	cwd = os.getcwd()
	plaqd = cwd + '/Correlators/'
	os.chdir(plaqd)
	foundCorrDir = True
	# except:
	# 	print("\nCouldn't find Correlators directory\n")

	correlator = open('pion_pt_pt_m%.1f_b%.3f.dat' % (m,beta),'w')
	
	# try:	
	corrt = np.zeros(T)
	for t in range(T):	
		for a in range(2):
			for b in range(2):
				for alpha in range(4):
					for beta in range(4):
						corrt[t] += abs(x[b + 2*beta][a + 2*alpha + 8*L**3*t]) ** 2
		correlator.write(str(t) + '\t' + str(corrt[t]) + '\n')
		print('\tt = %i:\t%f' % (t,corrt[t]))
	correlator.close()
	print()

	os.chdir('../')

	# except:
	# 	print("\nError in calculating the correlator\n")
	# 	return -1 

	print("Correlator calculated successfully")

	return 0


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='SU(2) Dirac Matrix Generator and Inverter')

	parser.add_argument('fname',
						help='Name of the pickled file that contains an SU(2) gauge configuration')
	parser.add_argument('--mass','-m', required=True,
						help='Mass of the particle',
						type=float)
	parser.add_argument('--wilson','-w',
						help='Value of the wilson term',
						default=1.0,
						type=float)
	# not sure if the ninv option is necessary, to be honest
	parser.add_argument('--noInv','-ninv',
						help='Will not invert the dirac matrix',
						action='store_true')
	parser.add_argument('--noCorr','-ncorr',
						help='Will not calculate the correlator',
						action='store_true')

	parser.parse_args()
	args = parser.parse_args()

	fname = args.fname

	bee = 0
	firstund = secondund = thirdund = dot = 0
	for i in range(len(fname)):

		if fname[i] == 'b':
			bee = i
		if fname[i] == '_' and firstund == 0:
			firstund = i
		if fname[i] == '_' and i > firstund and firstund != 0 and secondund == 0:
			secondund = i
		if fname[i] == '_' and i > secondund and secondund != 0:
			thirdund = i
		if fname[i] =='.' and i > thirdund and thirdund != 0:
			dot =  i

	beta = float(fname[bee+1:secondund])
	spaceLength = int(fname[secondund+1:thirdund])
	timeLength = int(fname[thirdund+1:dot])

	m = args.mass
	r = args.wilson
	noInv = args.noInv
	noCorr = args.noCorr
	U = pickle.load(open(fname,'rb'),encoding='latin1')

	# De,Do,Dm = gen_dirac(U,m,spaceLength,timeLength,r)
	De,Do,Dm,D = gen_dirac(U,m,spaceLength,timeLength,r)


	'''
		Probably wanna make this stuff write out to a file, I don't 
		think it would be super helpful to write out the dirac matrix
		to a file though. Since qcd_su2 also would want this stuff 
		written out to a file, it would probably be best to do the 
		file stuff as part of the functions themselves and not just 
		in main
	'''

	if noInv == False:
		
		inverse = invert_dirac(De, Do, Dm, spaceLength, timeLength, m, r)

		if noCorr == False:
			corr = corr(inverse,spaceLength,timeLength,beta,m)
			
