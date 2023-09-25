import numpy as np
from scipy import linalg
from time import time
# import newsu2 as su2
import su2
import sys
import scipy.sparse
import scipy.sparse.linalg
import pickle,string

from scipy.io import mmread, mmwrite

def read_gaugefield(Ufile,La,fmt):
	start = time()
	l = La[0]
	t = La[3]
	V = l*l*l*t

	numdim = len(La)
	mups = np.zeros((V,numdim), int)
	for i in range(0, V):
		for mu in range(0, numdim):
			mups[i,mu] = su2.mupi(i, mu, La)


	if fmt=='pkl':
		print 'Reading from Python pickle format.'

		th = pickle.load(open(Ufile,'rb'))

		# check plaq
		Uplaq = su2.calcPlaq(th[1],V,mups)
		diff = Uplaq - th[0]
		if np.abs(diff) > 1e-8:
			print Uplaq, th[0]
			print 'Plaquettes differ, not sure if read in correctly.'
		else:
			print 'Plaquettes agree, difference =',diff

	elif fmt=='mathematica':
		print 'Reading from Mathematica Table format.'
		U = np.zeros((V, len(La), 4))

		fin = open(Ufile,'r')
		x = fin.readlines()
		fin.close()

		for i,xx in enumerate(x):
			y = string.split(xx)
			z=[]
			for zz in y:
				z.append(float(zz))
			U[i][0] = z[:4]
			U[i][1] = z[4:8]
			U[i][2] = z[8:12]
			U[i][3] = z[12:]

		avgplaq = su2.calcPlaq(U,V,mups)
		th = [avgplaq,U]

	else:
		print "Format not recognized. Valid formats are 'pkl'"
		print "or 'mathematica'."
		return -1
	end = time()
	print 'Reading in gaugefield took',end-start,'seconds...'
	return th



def write_gaugefield(Ufile, UU,fmt):
	""" Takes a gauge field in our format and writes it to 
	a pickle file (if fmt =pkl)
	or
		Takes in (U) python format and 
	outputs to regular "table" format from mathematica (Ufile).
	"""
	start = time()
	if fmt=='pkl':
		f = open(Ufile,"wb")
		pickle.dump(UU,f)
		f.close()
		end = time()
		print 'Writing out gaugefield took',end-start,'seconds...'
		return 0
	
	elif fmt=='mathematica':
		plaq = UU[0]
		U = UU[1]
		(V,numdim,nummu) = U.shape

		fout = open(Ufile,'w')
		for i in range(V):
			xx = ''
			for d in range(3):
				xx += str(U[i,d,0])+'  '+str(U[i,d,1])+'  '+str(U[i,d,2])+'  '+str(U[i,d,3])+ '\t'
			xx += str(U[i,3,0])+'  '+str(U[i,3,1])+'  '+str(U[i,3,2])+'  '+str(U[i,3,3])+ '\n'
			fout.write(xx)

		fout.close()
		end = time()
		print 'Writing out gaugefield took',end-start,'seconds...'
		return 0
	else:
		print "Format not recognized. Valid formats are 'pkl'"
		print "or 'mathematica'."
		return -1


def readmtx(MTXfile):
	A = mmread(MTXfile) 
	return A


def writemtx(MTXfile,U):
	# only use the actual matrix, not the weird form unfortunately
	A = mmwrite(MTXfile,U) 
	return A
