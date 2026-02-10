import math
import numpy as np
import timeit
import su2

#import unitarize

prod = np.dot
tol = 1e-13 #work on this

def det(u):
    #determinant of matrix
    u_0 = u[0,0] * (u[1,1] * u[2,2] - u[1,2] * u[2,1])
    u_1 = u[0,1] * (u[1,2] * u[2,0] - u[1,0] * u[2,2])
    u_2 = u[0,2] * (u[1,0] * u[2,1] - u[1,1] * u[2,0])

    return u_0+u_1+u_2

def dagger(U):
    #hermitian conjugate

    return np.transpose(np.conjugate(U))

def vol(La):
    product = 1
    for x in range(len(La)):
        product *= La[x]
    return product

#dimensions of the array in a dictionary
def dim(La):
    D = {}
    for x in range(len(La)):
        D.update({x:La[x]})
    return D



def multi(U1, U2):
    #faster multiply two SU(3) matrices
    return np.dot(U1,U2)


def p2i(point, La):
    #from a point to an index
    #basically su(2)
    #point: [x,y,z,t] format
    #La: array-like; each element describes the length of the lattice dimension

    return ((La[2] * La[1] * La[0] * point[3]) + (La[1] * La[0] * point[2]) + (La[0] * point[1]) + (point[0]))


def i2p(ind,La):
    #from an index to a point
    #same comment as p2i
    #ind: index as int
    #La: array-like, same old

    v = La[0] * La[1] * La[2]
    a = La[0] * La[1]
    l = La[0]
    t = divmod(ind, v)
    z = divmod(t[1], a)
    y = divmod(z[1], l)
    x = divmod(y[1], 1)

    return np.array((x[0], y[0], z[0], t[0]))


def parity(pt):
    #parity of point on the lattice ; remainder==0, odd parity; ==1, even
    return np.sum(pt)%2

def hstart():
#doesn't this already happen in unitarize?
#    Uold = np.random.rand(3,3) + np.random.rand(3,3)*1j
#    U = unitarize.reunitarize(Uold)

    Uold = np.random.rand(3,3) + np.random.rand(3,3)*1j
    #would we need to differentiate between Uold and Unew like in unitarize.py?

    u = Uold[0]
    v = Uold[1]
    unorm = np.linalg.norm(u)
    uprime = u/unorm
    uprimestar = np.conj(uprime)

    v = v - (v.dot(uprimestar)) * uprime
    vnorm = np.linalg.norm(v)
    vprime = v/vnorm

    unew = uprime
    vnew = vprime
    w = np.cross(np.conj(unew),np.conj(vnew))
    Un = np.zeros((3,3),dtype = complex)
    Un[0] = unew
    Un[1] = vnew
    Un[2] = w

    U = Un

    return U

#def update(U):
    #sean's : make a random su(2) matrix near the identity
    #mine : didn't we just put in unitarize?
#    return unitarize.reunitarize(U)

def reunitarize(U):
     if(np.abs(np.linalg.det(U).real - 1)>tol or np.abs(np.linalg.det(U).imag - 1)>tol ):
         u = U[0]
         v = U[1]
         unorm = np.linalg.norm(u)
         uprime = u/unorm
         uprimestar = np.conj(uprime)

         v = v - (v.dot(uprimestar)) * uprime
         vnorm = np.linalg.norm(v)
         vprime = v/vnorm

         unew = uprime
         vnew = vprime
         w = np.cross(np.conj(unew),np.conj(vnew))

         Un = np.zeros((3,3),dtype = complex)
         Un[0] = unew
         Un[1] = vnew
         Un[2] = w

         U = Un
         #return Un
     else:
         print(np.linalg.det(U))
         print("Matrix is unitary.")
#         return U
     return U

def updateold(U):
    #update
    #a = np.identity(3) + 0.1*hstart()*np.identity(3)
    #a = np.array([1.,0.,0.,0.,0.,0.,0.,0.,0.]) + 0.1*hstart()*np.array([0.,1.,1.,1.,1.,1.,1.,1.,1.])
    #prod = multi(hstart(),np.identity(3, dtype = complex))
    #eye3 = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
#    eye3 = np.identity(3, dtype=complex)
    #prod = multi(hstart(),eye3)
#    prod = hstart() * eye3

#    Iden = np.identity(3, dtype=complex)
#    lamb = 0.1 * hstart()
#    prod = lamb * Iden
#    a = np.identity(3, dtype=complex) + prod #0.1*prod
    #hstart()*np.identity((3,3), dtype = complex)
    #a = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]) + 0.1*hstart()*np.array([[0.,1.,1.],[1.,0.,1.],[1.,1.,0.]])

    Iden = np.identity(3, dtype=complex)
    lamb = 0.1*hstart()
    prod = lamb * Iden

    a = np.identity(3, dtype=complex) + prod

    aU = a * U
#    aU = multi(a,U)

    aU /= det(aU)


    return aU


def update(U):
    return reunitarize(multi(U, hstart()))

#def update():

    #return hstart()

def cstart():
    # 3x3 identity matrix

    matrix = np.identity(3, dtype=complex)

    return(matrix)

def mstart():
    mrand = np.random.random()
    if(mrand >= 0.5):
        U = hstart()
    else:
        U = cstart()
    return U


def trace(U):
    ##trace of matrix

#    a = U[1,1]
#    b = U[2,2]
#    c = U[0,0]
#    return a+b+c

#    return U[1,1]+U[2,2]+U[0,0]

    return U.diagonal().sum()


def mupi(ind, mu, La):
    #increment a position in the mu-th direction
    #ind: index
    #mu: the direction of incrementation
    #La: array-like; elements describe length of lattice [x,y,z,t]

    pos = i2p(ind,La)
    if (pos[mu] + 1 >= La[mu]):
        pos[mu] = 0
    else:
        pos[mu] += 1
    return p2i(pos,La)

#apparently Sean doesn't use this function in PvB
def getMups(vol, numdim, La):
    #gets mups array
    #vol: volume/points on lattice
    #numdim: # dimensions on lattice
    #La: array-like

    mups = np.zeros((vol,numdim), int)
    for i in range(0, vol):
        for mu in range(0, numdim):
            mups[i,mu] = mupi(i,mu,La)

    return mups

def mudowni(ind, mu, La):
    #decrment a position in the mu-th direction
    #ind: index
    #mu: the direction of incremenatation
    #La: array-like; elements describe length of lattice [x,y,z,t]

    point = i2p(ind, La)
    if (point [mu] - 1 < 0):
        point[mu] = La[mu] - 1
    else:
        point[mu] -= 1
    return p2i(point, La)

def plaq(U, U0i, mups, mu, nu):
    #compute the plaquette ;; moving in forward dir only
    #copy is important bc pointers in python
    #U: La containing gauge fields for every point
    #U0i: int lattice pt index of starting point
    #mups: mups array; incrementing in muth dir from U0i
    #mu: int = one of pts on lattice, 0:x, 1:y, 2:z, 3:t
    #nu: int corresponding to another dir on lattice *see above*

    U0 = U[U0i][mu].copy()
    U1 = U[mups[U0i,mu]][nu].copy()
    U2 = dagger(U[mups[U0i,nu]][mu].copy())
    U3 = dagger(U[U0i][nu].copy())

#    prod1 = np.matmul(U0,U1)
#    prod2 = np.matmul(U2,U3)
#    prod3 = np.matmul(prod1,prod2)

#    return trace(prod3)
    return trace(multi(multi(U0,U1),multi(U2,U3))).real/3.


def link(U,U0i,mups,mu):
     #trace of link btwn 2 points
     #copy is important bc pointers in python
     #U: La containing gauge fields for every point
     #U0i: int lattice pt index of starting point
     #mups: mups array; incrementing in muth dir from U0i
     #mu: int = one of pts on lattice, 0:x, 1:y, 2:z, 3:t
     #nu: int corresponding to another dir on lattice *see above*

     U0 = U[U0i][mu].copy()

     return trace(U0)

def getstaple(U,U0i,mups,mudowns,mu):
    #returns value of staple
    #be careful about mups and mudowns, don't want it to be too close to the function names, but I prefer uniformity
    #U: La containing gauge fields for every point
    #U0i: int lattice pt index of starting point
    #mups: mups array; incrementing in muth dir from U0i in forward dir
    #mudowns: mups array; decrementing in muth dir from U0i in backward dir
    #mu: index corresponding to a direction 0:x, 1:y, 2:z, 3:t

    value = 0.0
    mm = list(range(4))
    mm = [i for i in range(4)]
    mm.remove(mu)
    for nu in mm:
        #if nu != mu:
        #forward staple
        value += staple(U, U0i, mups, mudowns, mu, nu, 1)

        #reverse staple
        value += staple(U, U0i, mups, mudowns, mu, nu, -1)
    return value

def staple(U,U0i,mups,mudowns,mu,nu,signnu):
    #compute staple in the mu-nu plane

    #U: La containing gauge fields for every point
    #U0i: int lattice pt index of starting point
    #mups: mups array; incrementing in muth dir from U0i in forward dir
    #mudowns: mups array; decrementing in muth dir from U0i in backward dir
    #mu: index corresponding to a direction 0:x, 1:y, 2:z, 3:t
    #nu: ^^^ but it's a new direction and must be different
    #signnu: either 1 or -1, corresponding to forward or backward respectively

    #forward
    if (signnu == 1):
        U1 = U[mups[U0i, mu]][nu].copy()
        U2 = dagger(U[mups[U0i, nu]][mu].copy())
        U3 = dagger(U[U0i][nu].copy())
    #backward
    else:
        U1 = dagger(U[mudowns[mups[U0i, mu],nu]][nu].copy())
        U2 = dagger(U[mudowns[U0i, nu]][mu].copy())
        U3 = U[mudowns[U0i, nu]][nu].copy()

    return multi(multi(U1,U2),U3)

def calcPlaq(U,vol,mups):
    #calc avg value of plaquettes about all lattice points
    #U: La containing gauge fields for every point
    #vol: int volume of lattice = # of points on lattice
    #mups: mups array

    plaquettes = np.zeros(6*vol) #6 directions on each point
    j = 0
    for i in range(vol):
        for mu in range(4):
            for nu in range(mu+1,4):
                plaquettes[j] = plaq(U,i,mups,mu,nu)
                j = j+1
    #this is the correct way to calculate <Sbox>, or average action per plaquette?
    avgPlaq = np.mean(plaquettes)

    return avgPlaq

def calcU_i(U,vol,D,mups):
    #calculates the average values of spacial links in lattice
    #U: La containing gauge fields for every point
    #vol: int volume of lattice = # of points on lattice
    #D: not entirely sure? for rn, assuming it's the D returned in dim(La), which is the dimensions of the array in a dictionary
    #new theory on D (talk to Sean about later); D=dimensions, bc putting in La works and gives a result -- but then why wouldn't you just call it La? idk it might be wrong, but it's better than the errors I was getting before
    #mups: mups array
    #as per Sean: D is from when I had dumbbrain and had weird ideas about how things would work. D should be a dictionary from dim(La).
    #BUT dim(La) didn't quite work for me, but since Sean said this never really came up after a point in the pvb code so maybe this isn't the end of the world? Idk I'll keep tinkering at it and come back to it

    spaceLink = np.zeros((3*vol), dtype=complex)
    j = 0
    for i in range(vol):
        for mu in range(D-1):
            #spaceLink[j] = link(U,i,mups,mu)
            spaceLink[j] = link(U,i,mups,mu)
            j = j+1
    U_i = np.mean(spaceLink)
    return U_i/2.

def calcU_t(U,vol,mups):
    #calculates the average values of time links in lattice
    #U: La containing gauge fields for every point
    #vol: int volume of lattice = # of points on lattice
    #mups: mups array

    timeLink = np.zeros((vol), dtype=complex)
    i = 0
    for i in range(vol):
        timeLink[i] = link(U,i,mups,3)

    U_t = np.mean(timeLink)
    return U_t/2.

def hbath(k, r, beta):
    #creates an SU(2) matrix via the heat bath method (see Creutz)
    #the matrix is written in real valued form
    low = np.exp(-4*beta*k/3)
    high = 1
    boolean = False
    while boolean is False:	
        x = np.random.uniform(low,high)
        a0 = 1. + 3.*np.log(x)/(beta*k)/2.
        reject = 1 - np.sqrt(1-a0*a0)

        if(1 - a0*a0 <= 0):
            print('BOUND ERROR')
            return np.zeros(4)

        newrand = np.random.rand()
        if newrand > reject:
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

def stapleleft(staples):
    #returns the real valued form for the upper left 2x2 submatrix of the staples
    #this process is necessary for the probability distribution
    # r = np.zeros(4)
    # r[0] = 0.5*(staples[0][0].real + staples[1][1].real)
    # r[1] = 0.5*(staples[0][1].imag + staples[1][0].imag)
    # r[2] = 0.5*(staples[0][1].real - staples[1][0].real)
    # r[3] = 0.5*(staples[0][0].imag - staples[1][1].imag)
    return staple_k(staples,0)#r

def stapleright(staples):
    #returns the real valued form for the bottom right 2x2 submatrix of the staples
    #this process is necessary for the probability distribution
    # r = np.zeros(4)
    # r[0] = 0.5*(staples[1][1].real + staples[2][2].real)
    # r[1] = 0.5*(staples[1][2].imag + staples[2][1].imag)
    # r[2] = 0.5*(staples[1][2].real - staples[2][1].real)
    # r[3] = 0.5*(staples[1][1].imag - staples[2][2].imag)
    return staple_k(staples,1)#r

def staple_k(staples,k):
    #returns the real valued form for the bottom right 2x2 submatrix of the staples
    #this process is necessary for the probability distribution
    r = np.zeros(4)
    r[0] = 0.5*(staples[k][k].real + staples[k+1][k+1].real)
    r[1] = 0.5*(staples[k][k+1].imag + staples[k+1][k].imag)
    r[2] = 0.5*(staples[k][k+1].real - staples[k+1][k].real)
    r[3] = 0.5*(staples[k][k].imag - staples[k+1][k+1].imag)
    return r

def stapleleft1(staples):
    #returns the real valued form for the upper left 2x2 submatrix of the staples
    #this process is necessary for the probability distribution
    # r = np.zeros(4)
    # r[0] = 0.5*(staples[0][0].real + staples[1][1].real)
    # r[1] = 0.5*(staples[0][1].imag + staples[1][0].imag)
    # r[2] = 0.5*(staples[0][1].real - staples[1][0].real)
    # r[3] = 0.5*(staples[0][0].imag - staples[1][1].imag)
    return staple_k1(staples,1)#r

def stapleright1(staples):
    #returns the real valued form for the bottom right 2x2 submatrix of the staples
    #this process is necessary for the probability distribution
    # r = np.zeros(4)
    # r[0] = 0.5*(staples[1][1].real + staples[2][2].real)
    # r[1] = 0.5*(staples[1][2].imag + staples[2][1].imag)
    # r[2] = 0.5*(staples[1][2].real - staples[2][1].real)
    # r[3] = 0.5*(staples[1][1].imag - staples[2][2].imag)
    return staple_k1(staples,1)#r

def staple_k1(staples,k):
    #returns the real valued form for the bottom right 2x2 submatrix of the staples
    #this process is necessary for the probability distribution
    r = np.zeros(4)
    r[0] = 0.5*(staples[k][k].real + staples[k+1][k+1].real)
    r[1] = 0.5*(staples[k][k+1].real + staples[k+1][k].real)
    r[2] = 0.5*(staples[k][k+1].real - staples[k+1][k].real)
    r[3] = 0.5*(staples[k][k].real - staples[k+1][k+1].real)
    
    return r

def makeleft(a):
    #converts real valued form to an upper left block matrix
    U = np.zeros((3,3),dtype=complex)
    U[0][0] = a[0] + 1.j*a[3]
    U[0][1] = a[2] + 1.j*a[1]
    U[1][0] = -a[2]+1.j*a[1]
    U[1][1] = a[0] - 1.j*a[3]
    U[2][2] = 1.+0.j   
    return U

def makecenter(a):
    #converts real valued form to a center block matrix
    U = np.zeros((3,3),dtype=complex)
    U[0][0] = a[0] + 1.j*a[3]
    U[0][2] = a[2] + 1.j*a[1]
    U[2][0] = -a[2]+1.j*a[1]
    U[2][2] = a[0] - 1.j*a[3]
    U[1][1] = 1.+0.j   
    return U

def makeright(a):
    #converts real valued form to a lower right block matrix
    U = np.zeros((3,3),dtype=complex)
    U[1][1] = a[0] + 1.j*a[3]
    U[1][2] = a[2] + 1.j*a[1]
    U[2][1] = -a[2]+1.j*a[1]
    U[2][2] = a[0] - 1.j*a[3]
    U[0][0] = 1.+0.j   
    return U


