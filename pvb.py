import math
import numpy
# import matplotlib 
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle , sys
import itertools
import su2
np = numpy
from time import time
import os.path

filesave = False

# Number of updates/lattice
M = 1000
# M = 110
# Number of updates per link per MC update
Mlink = 10

# sep of measurements
Msep = 50
# Thermalization
Mtherm = 100

# Starting/ending beta's
# bI is how much beta is incremented 
# bS = 2.4
# bE = 2.4
bS = 2.4
bE = 2.4
bI = 2.4

'''
Defined dimensions
can change the number of dimensions by changing the number of elements in La, 
so long as the length of a dimension isn't zero'''
Lx = Ly = Lz = 2
Lt = 2
La = [Lx,Ly,Lz,Lt]

#D = {0:Lx, 1:Ly, 2:Lz, 3:Lt}
D = su2.dim(La) 

# planes of rotation
# in 4D there are six planes
planes = len(D)*(len(D) - 1)//2

# Volume
V = su2.vol(La)

# U matrices, (i[0-LxLyLzT], mu[x,y,z,t], U[0-3])
U = np.zeros((V, len(D), 4))

# Seed
# 42442
np.random.seed(42442)
		
# # Generate the lattice & periodic BCs
mups = np.zeros((V,len(D)), int)
mdns = np.zeros((V,len(D)), int)

beta = bS

fpbeta = open("plaq_vs_beta.dat","w")

# here to test the time it takes for the program to run
# fname = "pvb_timing_%.1f_%i_%i" % (b)
# ftiming = open("pvb_timing_")
Ti = time()

pbs = []
pbe = []
betas = []
while(beta<=bE):
	print("beta = " , beta)
	
	fname = "pvb_timing_%.1f_%i_%i" % (beta,Lx,Lt)
	ftiming = open(fname,'w')

	for i in range(0, V):
		for mu in range(0, len(D)):
			U[i][mu] = su2.cstart()
			# U[i][mu] = su2.hstart()
			# mups[i,mu] = su2.mupi(i, mu, La, D)
			# mdns[i,mu] = su2.mdowni(i, mu, La, D)
			mups[i,mu] = su2.mupi(i, mu, La)
			mdns[i,mu] = su2.mdowni(i, mu, La)

	# Update M times
	avgPlaquettes = np.zeros(M)
	avgLinks = np.zeros((M,4,4))
	U_i = np.zeros(M)
	U_t = np.zeros(M)
	count = 0


	plaqs = []

	for m in range(M): 

		# Loop through each site
		for i in range(V):
			# Loop through each direction
			for mu in D:
			
				# Define the link
				U0 = U[i][mu].copy()
				
				# Determine the staples for the link in the mu'th direction
				staples = su2.getstaple(U,i,mups,mdns,mu)
				# print(type(staples))
									
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

		# avgPlaquettes[m] = su2.calcPlaq(U,V,D,planes,mups,M,m)
		avgPlaquettes[m] = su2.calcPlaq(U,V,mups)
		# U_i[m] = su2.calcU_i(U,V,D,mups)
		# U_t[m] = su2.calcU_t(U,V,D,mups)
		#print U_i[m], U_t[m]
		if (m>=Mtherm and m%Msep==0):
			plaqs.append(avgPlaquettes[m])
			th = [avgPlaquettes[m],U]
			#change the filename for whatever computer you're on
			# completeName = os.path.join("/Users/shannaford/Dropbox/SeanHannaford/Summer/Configs","quSU2_b%.1f_%i_%i.%i" % (beta,Lx,Lt,m))
			if filesave:
				completeName = os.path.join("C:/Users/shann/Dropbox/SeanHannaford/Summer","quSU2_b%.1f_%i_%i.%i" % (beta,Lx,Lt,m))
				f = open(completeName,"wb")
				pickle.dump(th,f)
				f.close()

		if (m%100)==0:
			print('\tplaq = ',avgPlaquettes[m])
	pbavg = np.mean(plaqs)
	pberr = np.std(plaqs)/np.sqrt(len(plaqs))
	print("\tfinal plaq = ", pbavg ," +/- ", pberr)

	if (beta < 2):
		print("\t",1.0 - 0.25*beta)
	else:
		print("\t",0.75/beta)

	pbs.append(pbavg)
	pbe.append(pberr)
	fpbeta.write(str(beta) + '\t' + str(pbavg) + '\t' + str(pberr) + '\n')
	betas.append(beta)

	print("\tAcceptance: " , count, "/", len(D)*M*Mlink*V)
	fp = open("plaquettes_"+str(beta)+".dat","w")
	for i in range(len(avgPlaquettes)):
		fp.write(str(i) + '\t' + str(avgPlaquettes[i]) + '\n')
	fp.close()
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	plt.plot(avgPlaquettes)
	plt.savefig("plaquettes_"+str(beta)+".png")
	plt.clf()
	beta = beta + bI 

Te = time()
totalTime = Te - Ti

fpbeta.close()

print("time = %f" % (totalTime))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.errorbar(betas,pbs,yerr=pbe)


