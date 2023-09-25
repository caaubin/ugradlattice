import field_gen
import dirac_gen
import argparse

parser = argparse.ArgumentParser(description='SU(2) Quantum Chromodynamics')

parser.add_argument('--beta','-b', required=True,
					help='Sets the value of beta which will be used to generate gauge fields',
					type=float)
parser.add_argument('--spaceLength','-sl', required=True,
					help='Sets the length of the spacial dimensions of the lattice',
					type=int)
parser.add_argument('--timeLength','-tl', required=True,
					help='Sets the length of the time dimension of the lattice',
					type=int)
parser.add_argument('--mass','-m', required=True,
					help='Mass of the particle',
					type=float)
parser.add_argument('--updates','-u',
					help='The number of updates per lattice',
					default=1000,
					type=int)
parser.add_argument('--links','-ln',
					help='The number of updates per link per lattice update',
					default=10,
					type=int)
parser.add_argument('--wilson','-w',
					help='Value of the wilson term',
					default=1.0,
					type=float)
parser.add_argument('--hotStart', '-hot',
					help='Lattice is initialized with a hot start (initialization is cold by default)',
					action='store_true')

parser.parse_args()
args = parser.parse_args()

bS = args.beta
spaceLength = args.spaceLength
timeLength = args.timeLength
m = args.mass
M = args.updates
Mlink = args.links
Msep = 1
Mtherm = M - 1

bI = 1
r = args.wilson


if __name__ == '__main__':

	U = field_gen.lattice(bS, spaceLength, timeLength, 
				   		  M, Mlink, Msep, Mtherm, bRange,bI, hotStart)

	De,Do,Dm = dirac_gen.gen_dirac(U,m,spaceLength,timeLength,r)

	dirac_gen.invert_dirac(De,Do,Dm,spaceLength,timeLength)

	