import math
import numpy
import matplotlib.pyplot as plt
import pickle,sys
import itertools
import su2
np = numpy
from time import time
import os
import argparse
from tqdm import tqdm # consider just ipmorting tqdm
import jackknife

def lattice(bS, spaceLength, timeLength, Uinput='na',
			M=1000, Mlink=10, Msep=10, Mtherm=100,
			bRange=0.,bI=.1, hotStart=False, makePlot=True,progBar=False):
	"""Generates a lattice or set of lattices utilizing SU(2) gauge
	fields

	Parameters
	----------
	bS : float
		The value of beta that will be used to generate a lattice. If
		a set of lattices generated over a range of betas is desired,
		then bS is the first value of beta evaluated over the range.
	spaceLength : int
		The length of the three spacial dimensions of the lattice
	timeLength : int
		The length of the time dimension of the lattice

	Returns
	-------
	th : numpy.ndarray
		Numpy array where the fist element is the average value of 
		the plaquettes of the lattice, and the second element is the 
		lattice. While the function only returns the final update of 
		the lattice, the lattice is written out into the configs 
		directory each time a measurement is made. 

	Other Parameters
	----------------
	M : int 
		The number of times the lattice is updated. An update of the 
		lattice is performed by iterating through each lattice point
		and each link extending from that point and updating those 
		links. If enough updates are performed on the lattice it  will 
		converge to some configuration. Set ot 1000 by default.
	Mlink : int
		The number of times each link is updated per lattice update. A 
		link is updated by creating a random SU(2) matrix near the 
		identity, then calculating the action of the new matrix. If the
		action is decreased or by some random probability the new 
		matrix will be assigned to the link, otherwise the link does
		not change. This process is repeated Mlink times. Set to 10 by
		default
	Msep : int
		The separation of measurements. While the lattice is being 
		updated M times, a measurement of the average value of the 
		plaquettes in the lattice will be taken every Msep updates.
		Each time a measurement is taken, the lattice will also be 
		written out to a pickled file.
		(E.g. if Msep = 10, measurements will be taken every 10 
		updates). Set to 10 by default
	Mtherm : int
		The "thermalization" of the lattice. Indicates the number of
		lattice updates that need to be performed before any 
		measurements of the average value of the plaquettes in the 
		lattice. Set to 100 by default.
	bRange : float
		The difference between the greatest and smallest value of beta
		that will be used to generate a lattice. A bRange of 0 means 
		only one value of beta will be used to make a lattice. Set to 
		0.0 by default.
	bI : float
		If it is desired to generate a set of lattices using a range of
		betas, bI is how much beta will increment between values of 
		beta. Set to 0.1 by default
	hotStart : bool
		Setting this to True will initialize the fields of the lattice
		with a hot start (randomly generated values) rather than the 
		default cold start (SU(2) matrices set to the identity). Set to
		False by default.
	makePlot : bool
		When True, plots of the plaquette vs. updates will be 
		generated. No plots generated when makePlot is set to False.
		Set to True by default.

	"""

	print("\nGenerating gauge fields...\n")

	foundConfd = False
	try:
		cwd = os.getcwd()
		confd = cwd + '/Configs'
		os.chdir(confd)
		foundConfd = True
	except:
		print("\nCouldn't find Configs directory\n")


	'''If the thermalization is accidentally set to be more than the 
	number of updates, thermalization is adjusted to be half the 
	number of updates by default '''
	if Mtherm > M - 1:
		Mtherm = M // 2

	bE = bS + bRange

	Lx = Ly = Lz = spaceLength
	Lt = timeLength
	La = [Lx,Ly,Lz,Lt]

	#D = {0:Lx, 1:Ly, 2:Lz, 3:Lt}


	# ***********************
	D = su2.dim(La) 
	# ***********************


	# planes of rotation
	# in 4D there are six planes

	# ***********************
	planes = len(D)*(len(D)-1)/2
	# ***********************

	# Volume
	V = su2.vol(La)

	# U matrices, (i[0-LxLyLzT], mu[x,y,z,t], U[0-3])
	
	# ***********************
	if Uinput == 'na':
		U = np.zeros((V, len(D), 4))
	else:
		Utemp = pickle.load(open(Uinput,'rb'),encoding='latin1')
		U = Utemp[1]
	# ***********************

	# Seed
	# 42442
	np.random.seed(42442)
			
	# *****************************
	mups = np.zeros((V,len(D)), int)
	mdns = np.zeros((V,len(D)), int)
	# *****************************


	beta = bS

	fpbeta = open("plaq_vs_beta.dat","w")

	Ti = time()

	pbs = []
	pbe = []
	betas = []
	while(beta<=bE):
		print("beta = " , beta)

		for i in range(0, V):

			# ******************************
			for mu in range(0, len(D)):
			# ******************************

				if Uinput == 'na':
					if hotStart == True:
						U[i][mu] = su2.hstart()
					else:
						U[i][mu] = su2.cstart()
				''' elif hotStart == False:
						read in a file
				'''
				mups[i,mu] = su2.mupi(i, mu, La)
				mdns[i,mu] = su2.mdowni(i, mu, La)

		# Update M times
		avgPlaquettes = np.zeros(M)
		avgLinks = np.zeros((M,4,4))
		U_i = np.zeros(M)
		U_t = np.zeros(M)
		count = 0

		plaqs = []

		if progBar == True:
			idek = tqdm(range(M))
		else:
			idek = range(M)

		# for m in range(M): 
		# for m in tqdm(range(M)): # tqdm is for progress bar stuff 
		for m in idek: # tqdm is for progress bar stuff 

			# Loop through each site
			for i in range(V):
				# Loop through each direction

				# *********************
				for mu in D:
				# *********************

					# Define the link
					U0 = U[i][mu].copy()
					
					# Determine the staples for the link in the mu'th direction
					staples = su2.getstaple(U,i,mups,mdns,mu)
										
					# Update the sites
					for mm in range(Mlink):
						U0n = su2.update(U0)

						# Compute dS
						dS =  - 0.5 * su2.tr(su2.mult(U0n-U0, staples))
				
						# Save the update if dS < 0 and the probability is < e^b*dS
						rand = np.random.random()
						if dS<0 or (rand < np.exp(-beta * dS)):
							U[i][mu] = U0n
							count +=1

			avgPlaquettes[m] = su2.calcPlaq(U,V,mups)
			# U_i[m] = su2.calcU_i(U,V,La,mups)
			# U_t[m] = su2.calcU_t(U,V,mups)

			'''makes measurements and writes out lattice every Msep 
			updates after there have already been Mtherm updates'''
			if (m>=Mtherm and m%Msep==0):

				# writes out to a Configs directory, if possible
				cwd = os.getcwd()
				if foundConfd == True and cwd[-7:] != 'Configs':
					os.chdir(confd)

				'''measure the average plaquettes and write out the 
				lattice as a pickle file'''
				plaqs.append(avgPlaquettes[m])
				th = [avgPlaquettes[m],U]
				completeName = "quSU2_b%.1f_%i_%i.%i" % (beta,Lx,Lt,m)
				f = open(completeName,"wb")
				pickle.dump(th,f)
				f.close()

				if os.getcwd() == confd:
					os.chdir('../')

				# end lattice write out

			if progBar == False and (m%100)==0:
				print('\tplaq = ',avgPlaquettes[m])


			# end m loop


		# calculate the average plaquette and error
		pbavg = np.mean(plaqs)
		pberr = np.std(plaqs)/np.sqrt(len(plaqs))
		print("\tfinal plaq = ", pbavg ," +/- ", pberr)

		jackknife.jack(avgPlaquettes)


		if (beta < 2):
			print("\t",1.0 - 0.25*beta)
		else:
			print("\t",0.75/beta)

		pbs.append(pbavg)
		pbe.append(pberr)
		fpbeta.write(str(beta) + '\t' + str(pbavg) + '\t' + str(pberr) + '\n')
		betas.append(beta)
		
		# *****************************************************
		print("\tAcceptance: " , count, "/", len(D)*M*Mlink*V)
		# ******************************************************


		'''Write out the plots of the plaquettes vs. beta. Attempts to
		write them into the PLaquettes directory, but writes them
		to the current working directory if this fails.'''
		foundPlaqd = False
		try: 
			cwd = os.getcwd()
			plaqd = cwd + '/Plaquettes'
			os.chdir(plaqd)
			foundPlaqd = True
		except:
			print("\nCouldn't find Plaquettes directory\n")

		fp = open("plaquettes_"+str(beta)+".dat","w")
		for i in range(len(avgPlaquettes)):
			fp.write(str(i) + '\t' + str(avgPlaquettes[i]) + '\n')
		fp.close()

		if makePlot == True:
			fig1 = plt.figure()
			ax1 = fig1.add_subplot(111)
			plt.plot(avgPlaquettes)
			plt.savefig("plaquettes_b" + str(beta)+ "_" + str(spaceLength) + "_" + str(timeLength) + ".png")
			plt.clf()

		

		if foundPlaqd == True:
			os.chdir("../")

		# end make plot

		beta = beta + bI 

		# end beta loop

	Te = time()
	totalTime = Te - Ti

	fpbeta.close()

	plt.plot(betas,pbs)
	plt.savefig('plaq_vs_beta.png')

	print("time = %f" % (totalTime))

	if makePlot == True:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.errorbar(betas,pbs,yerr=pbe)


	plt.close('all') # for memory issues, idk if this works though


	return th


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='SU(2) Lattice Generator')

	parser.add_argument('--beta','-b', required=True,
						help='Sets the value of beta which will be used to generate gauge fields',
						type=float)
	parser.add_argument('--spaceLength','-sl', required=True,
						help='Sets the length of the spacial dimensions of the lattice',
						type=int)
	parser.add_argument('--timeLength','-tl', required=True,
						help='Sets the length of the time dimension of the lattice',
						type=int)
	parser.add_argument('--updates','-u',
						help='The number of updates per lattice',
						default=1000,
						type=int)
	parser.add_argument('--links','-ln',
						help='The number of updates per link per lattice update',
						default=10,
						type=int)
	parser.add_argument('--sep','-s',
						help='The separation of measurements',
						default=10,
						type=int)
	parser.add_argument('--therm','-th',
						help='The thermalization of the lattice',
						default=100,
						type=int)
	parser.add_argument('--endBeta','-eb',
						help='If a set of lattices generated with a range of betas is desired, endBeta is the last value of beta in this range',
						default= -1,
						type=float)
	parser.add_argument('--incBeta','-ib',
						help='If a set of lattices generated with a range of betas is desired, incBeta is the ammount beta will be incremented between each lattice',
						default=0.1,
						type=float)
	parser.add_argument('--hotStart', '-hot',
						help='Lattice is initialized with a hot start (initialization is cold by default)',
						action='store_true')
	parser.add_argument('--noPlot', '-nplt',
						help='Will not generate any plots of plaquettes vs. beta (will make plots by default)',
						action='store_false')
	parser.add_argument('--progBar', '-prog',
						help='Display a progress bar in place periodic plaq output',
						action='store_true')
	# parser.add_argument('Uinput',
	# 					nargs='?',
	# 					type=argparse.FileType('r'),
	# 					default=sys.stdin,
	# 					help='Name of the pickled file that contains an existing SU(2) gauge configuration')
	parser.add_argument('Uinput',
						nargs='?',
						default='na',
						help='Name of the pickled file that contains an existing SU(2) gauge configuration')
	# add makeplot thing

	parser.parse_args()
	args = parser.parse_args()

	bS = args.beta
	spaceLength = args.spaceLength
	timeLength = args.timeLength
	U0 = args.Uinput
	M = args.updates
	Mlink = args.links
	Msep = args.sep
	Mtherm = args.therm
	if( args.endBeta != -1):
		bRange = args.endBeta - bS
	else:
		bRange = 0.
	bI = args.incBeta
	makePlot = args.noPlot
	hotStart = args.hotStart

	progBar = args.progBar

	lattice(bS, spaceLength, timeLength, U0,
			M, Mlink, Msep, Mtherm, bRange,bI,
			hotStart,makePlot,progBar)