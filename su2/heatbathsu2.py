import math
import numpy
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle,sys
import itertools
import su2
np = numpy
# plt = matplotlib.pyplot
from time import time
import os.path
import statistics as stat
# Number of updates/lattice
M = 10
# M = 110

# sep of measurements
Msep = 1
# Thermalization
Mtherm = 1

# Starting/ending beta's
# bI is how much beta is incremented
# bS = 2.4
# bE = 2.4
bS = 6.0
bE = 6.0
bI = 2.4

'''
Defined dimensions
can change the number of dimensions by changing the number of elements in La,
so long as the length of a dimension isn't zero'''
Lx = Ly = Lz = 4
Lt = 4
La = [Lx,Ly,Lz,Lt]

#D = {0:Lx, 1:Ly, 2:Lz, 3:Lt}
D = 4

# planes of rotation
# in 4D there are six planes
planes = D*(D-1)/2

# Volume
V = su2.vol(La)

# U matrices, (i[0-LxLyLzT], mu[x,y,z,t], U[0-3])
U = np.zeros((V, D, 4))

# Seed
# 42442
np.random.seed(42446)

# # Generate the lattice & periodic BCs
mups = np.zeros((V,D), int)
mdns = np.zeros((V,D), int)

beta = float(input('beta: '))

# here to test the time it takes for the program to run
# fname = "pvb_timing_%.1f_%i_%i" % (b)
# ftiming = open("pvb_timing_")
Ti = time()

fname = "pvb_timing_%.1f_%i_%i" % (beta,Lx,Lt)
ftiming = open(fname,'w')

for i in range(0, V):
	for mu in range(0, D):
		U[i][mu] = su2.hstart()
		# mups[i,mu] = su2.mupi(i, mu, La, D)
		# mdns[i,mu] = su2.mdowni(i, mu, La, D)
		mups[i,mu] = su2.mupi(i, mu, La)
		mdns[i,mu] = su2.mdowni(i, mu, La)
	# Update M times
avgPlaquettes = []
avgLinks = np.zeros((M,4,4))
U_i = np.zeros(M)
U_t = np.zeros(M)
count = 0

plaqs = []

for m in range(M):

	# Loop through each site
    for i in range(V):
		# Loop through each direction
        for mu in range(D):
			# Define the link
            U0 = U[i][mu].copy()

			# Determine the staples for the link in the mu'th direction
            staples = su2.getstaple(U,i,mups,mdns,mu)
			# print(type(staples))

			# Update the sites
            
            k = np.sqrt(su2.det(staples))
            a = np.exp(-2*beta*k)
            b = 1
			
            boolean = False
            while boolean is False:	
                x = np.random.uniform(a,b)
                a0 = 1 + np.log(x)/(beta*k)
                newrand = np.random.rand()
                reject = 1-np.sqrt(1-a0*a0)
                if newrand > reject:
                    #print(a0)
                    #print(reject,newrand,'\n')
                    boolean = True
            absa0 = np.sqrt(1 - a0*a0)
            theta = np.random.uniform(0,math.pi)
            phi = np.random.uniform(0, 2*math.pi)
            a1 = absa0*np.cos(phi)*np.sin(theta)
            a2 = absa0*np.sin(phi)*np.sin(theta)
            a3 = absa0*np.cos(theta)
            
            U0n = np.array([a0,a1,a2,a3])
            Ubar = staples/k
            
            Uinv = su2.dag(Ubar)
            U[i][mu] = su2.mult(U0n, Uinv)


	# avgPlaquettes[m] = su2.calcPlaq(U,V,D,planes,mups,M,m)
    plaqs.append(su2.calcPlaq(U,V,mups))
    if m > Mtherm and m % Msep == 0:
        avgPlaquettes.append(plaqs[m])
        print(m, plaqs[m])
	# U_i[m] = su2.calcU_i(U,V,D,mups)
	# U_t[m] = su2.calcU_t(U,V,D,mups)
	#print U_i[m], U_t[m]

Ebar = stat.mean(avgPlaquettes)
print(Ebar)

x = []
for i in range(M):
	x.append(i)

Te = time()
totalTime = Te - Ti
print("time = %f" % (totalTime))

plt.scatter(x,plaqs,s=25,c=None)
plt.xlabel('Update')
plt.ylabel('AvgPlaq')
plt.title('Heat Bath PvU Beta = ' + str(beta))
plt.savefig('su2HB_' + 'b' + str(beta) + '.pdf')