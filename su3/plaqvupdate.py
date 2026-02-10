import numpy as np

import matplotlib
matplotlib.use('agg')#'?')#'WebAgg') #'TKAgg')
import matplotlib.pyplot as plt
import pickle,sys
import itertools

import su3
#import jackknife

from time import time
import os.path
import timeit
import sys
from datetime import datetime
import statistics

#number of updates/lattice
M = 10

#number of updates/link/MC update proper values? ***
Mlink = 1 #just to make it run
#MLink testing !!! / Ch

#separation of measurements
Msep = 1

#thermalization
Mtherm = 1 #100

#defined dimensions ;; depends on number of elements in La ;; length of dimension != 0
Lx = Ly = Lz = 4
Lt = 4
La = [Lx, Ly, Lz, Lt]

#dictionary of dimensions
D = 4
#planes of rotation
planes = D*(D-1)//2
#volume
V = su3.vol(La)

#U matrices with shape (L^3 T, 4, 3, 3)
U = np.zeros((V, D, 3,3), dtype=complex )

#mups and mdns
mups = np.zeros((V, D), int)
mdowns = np.zeros((V, D), int)

beta = 5.5

#update the required number of times

avgPlaquettes = np.zeros(M)
avgLinks = np.zeros((M,4,4)) #should this be 4 and 4? ***

#U_i = np.zeros(M, dtype=complex)
#U_t = np.zeros(M, dtype=complex)
count = 0

plaqs = []
x = []

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

avgPlaq = [] #for calculating <E> for a given beta

for m in range(M):
    count = 0
        #loop through each site
    for i in range(V):
        #loop through each direction
        for mu in range(D):
            #define the link
            U0 = U[i][mu].copy()
            #determine the staples for link in mu-th dir
            staples = su3.getstaple(U,i,mups,mdowns,mu)

            #begin = timeit.default_timer()

            #update sites
            for mm in range(Mlink):
                #begin = timeit.default_timer()
#                U0n = su3.update(U0)
                U0n = su3.update(U0)
                #ds:
                dS =  (- beta * su3.trace(su3.multi(U0n-U0, staples))).real

#                print("ds = ",dS)

                # Save the update if dS < 0 and the probability is < e^b*dS
                rand = np.random.random() #note: what was this for?
                if dS<0 or (rand < np.exp(-beta * dS)):
                    U[i][mu] = U0n
                    count +=1

    if(m == 0.25*M):
        print('25% Done')
    elif(m == 0.5*M):
        print('50% Done')
    elif(m == 0.75*M):
        print('75% Done')
    elif(m == M):
        print('99% Done')

    #avgPlaquettes[m] = su3.calcPlaq(U,V,mups) # CASTING REAL TO IM
    #plaqs.append(avgPlaquettes[m])
    #x.append(m)
    if(m >= Mtherm and m%Msep == 0):
        avgPlaq.append(su3.calcPlaq(U,V,mups))

avgE = statistics.mean(avgPlaq)
f = open('betavavgplaq.txt', 'a')
f.write(str(beta))
f.write('\t')
f.write(str(avgE))
f.write('\n')
f.close()


#plt.scatter(x,plaqs,s=25,c=None)
#plt.xlabel('Update')
#plt.ylabel('Boxbar')
#plt.title('Beta = 6, mstart')
#plt.savefig('avgplaq6.pdf')
