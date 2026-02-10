import math
import numpy as np
import matplotlib.pyplot as plt
import su3
import su2
from time import time
import os.path
import statistics as stat
import sys 
import su3maps

# Number of updates/lattice
M = 200
# M = 110

# sep of measurements
Msep = 1
# Thermalization
Mtherm = 50

# Starting/ending beta's
# bI is how much beta is incremented
# bS = 2.4
# bE = 2.4
bS = 6.0
bE = 6.0
bI = 2.4
if len(sys.argv)<2:
    beta = float(input('Beta:'))
else:
    beta = float(sys.argv[1])
'''
Defined dimensions
can change the number of dimensions by changing the number of elements in La,
so long as the length of a dimension isn't zero'''
Lx = Ly = Lz = 4
Lt = 4
La = [Lx,Ly,Lz,Lt]

#D = {0:Lx, 1:Ly, 2:Lz, 3:Lt}
D = 4

def hbath(k, r, beta):
    #creates an SU(2) matrix via the heat bath method (see Creutz)
    #the matrix is written in real valued form
    low = np.exp(-4*beta*k/3)
    high = 1
    boolean = False
    while boolean is False:	
        x = np.random.uniform(low,high)
        a0 = 1. + 3.*np.log(x)/(beta*k*2.)
        reject = 1 - np.sqrt(1-a0*a0)

        if(1 - a0*a0 <= 0):
            print('BOUND ERROR')
            return np.zeros(4)

        newrand = np.random.rand()
        
        if newrand > reject:
            #print(a0, '\n')
            #print(reject,'\n')
            boolean = True	

    absa = np.sqrt(1 - a0*a0)
    theta = np.random.uniform(0,math.pi)
    phi = np.random.uniform(0, 2*math.pi)
    a1 = absa*np.cos(phi)*np.sin(theta)
    a2 = absa*np.sin(phi)*np.sin(theta)
    a3 = absa*np.cos(theta)
    Un = np.array([a0, a1, a2, a3])
    
    Ubar = r/k
    Uinv = su2.dag(Ubar)
    U = su2.mult(Un,Uinv)

    return U

# planes of rotation
# in 4D there are six planes
planes = D*(D-1)/2

# Volume
V = su3.vol(La)

#U matrices with shape (L^3 T, 4, 3, 3)
U = np.zeros((V, D, 3,3), dtype=complex )

# # Generate the lattice & periodic BCs
mups = np.zeros((V,D), int)
mdowns = np.zeros((V,D), int)



# here to test the time it takes for the program to run
# fname = "pvb_timing_%.1f_%i_%i" % (b)
# ftiming = open("pvb_timing_")
Ti = time()

fname = "pvb_timing_%.1f_%i_%i" % (beta,Lx,Lt)
ftiming = open(fname,'w')
print("Open timing file...")

if(beta <= 5.5):
    for i in range(0,V):
        for mu in range(0, D) :
            U[i][mu] = su3.hstart()
            mups[i,mu] = su3.mupi(i,mu,La)
            mdowns[i,mu] = su3.mudowni(i,mu,La)
elif(5.5 < beta < 9):
    for i in range(0,V):
        for mu in range(0, D) :
            U[i][mu] = su3.mstart()
            mups[i,mu] = su3.mupi(i,mu,La)
            mdowns[i,mu] = su3.mudowni(i,mu,La)
else:
    for i in range(0,V):
        for mu in range(0, D) :
            U[i][mu] = su3.cstart()
            mups[i,mu] = su3.mupi(i,mu,La)
            mdowns[i,mu] = su3.mudowni(i,mu,La)
print("Initialized lattices...")

avgLinks = np.zeros((M,4,4))
U_i = np.zeros(M)
U_t = np.zeros(M)
count = 0

plaqs = []
avgPlaq = []

print("Beginning heat bath at beta = ",beta)

for m in range(M):
	# Loop through each site
    for i in range(V):
		# Loop through each direction
        for mu in range(D):
            U0 = U[i][mu].copy()
            staples = su3.getstaple(U,i,mups,mdowns,mu)
            
            #the su(2) matrices must be generated according to a prob. distribution that depends on the product of the link and staples
            UR = su3.multi(U0,staples)
            #this next section creates the upper left sub matrix (a1) through heat bath method
            #r is the corresponding sub matrix of the product of the link and the staples
            r1 = su3.stapleleft(UR)
            k1 = np.sqrt((su2.det(r1)))
            su2r = r1/k1
            #print('k1: ',k1,'\n')
            #this if statement is just because I was having trouble with diverging values of k... feel free to delete
            if(k1 > 10):
                print(k1, '\n')
            a1 = hbath(k1,r1,beta)
            #have to use matrix representations to multiply
            a1 = su3.makeleft(a1)
            
            #multiply the generated matrix by UR
            aUR = su3.multi(a1,UR)

            #creates the lower right sub matrix (a2) through heat bath method
            r2 = su3.stapleright(aUR)
            k2 = np.sqrt((su2.det(r2)))
            #print('k2: ', k2, '\n')
            if(k2 > 10):
                print(k2, '\n')
            a2 = hbath(k2,r2,beta)
            a2 = su3.makeright(a2)
            #if a2.all == 0:
            #    break
            
            #Un is the updated link
            Un = su3.multi(a2,su3.multi(a1,U0))
            U[i][mu] = Un # su3.reunitarize(Un)
            #pp = su3old.calcPlaq(U,V,mups)
            #print(m,pp)
    pp = su3.calcPlaq(U,V,mups)
    plaqs.append(pp)
    if (m==Mtherm):
        print("We have thermalized (we think!)")
    if(m > Mtherm and m%Msep == 0):
        print(m,pp)
        avgPlaq.append(pp)

Ebar = stat.mean(avgPlaq)
x = []
for i in range(M):
	x.append(i)

bvE = open('bvE.dat', 'a')
bvE.write(str(beta) + '\t' + str(Ebar) + '\n')
bvE.close()

Te = time()
totalTime = Te - Ti
print("time = %f" % (totalTime))

print('Average Energy = ', Ebar)

plt.scatter(x,plaqs,s=25,c=None)
plt.xlabel('Update')
plt.ylabel('AvgPlaq')
plt.title('Heat Bath')
plt.savefig('su3heatbath_'+str(beta)+'.pdf')
