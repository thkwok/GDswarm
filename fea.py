from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
import time
import math
import random
from scipy import integrate

# MAIN DRIVER
def fea(fixed,f,nelx,nely,passive):
	Emin=1e-9
	Emax=1.0
	# dofs:
	ndof = 2*(nelx+1)*(nely+1)
	# Allocate design variables (as array), initialize and allocate sens.
	x= np.ones(nely*nelx,dtype=float)
	x[passive]=1e-9
	xold=x.copy()
	xPhys=x.copy()
	g=0 # must be initialized to use the NGuyen/Paulino OC approach
	dc=np.zeros((nely,nelx), dtype=float)
	# FE: Build the index vectors for the for coo matrix format.
	KE=lk() #element stiffness matrix
	edofMat=np.zeros((nelx*nely,8),dtype=int)
	for elx in range(nelx):
		for ely in range(nely):
			el = ely+elx*nely
			n1=(nely+1)*elx+ely
			n2=(nely+1)*(elx+1)+ely
			edofMat[el,:]=np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])
	
	# node order of an element: bottom-left, bottom-right, top-right, top-bottom
	
	# Construct the index pointers for the coo format
	iK = np.kron(edofMat,np.ones((8,1))).flatten()
	jK = np.kron(edofMat,np.ones((1,8))).flatten()    
	# BC's and support
	dofs=np.arange(2*(nelx+1)*(nely+1))
	free=np.setdiff1d(dofs,fixed)
	# Solution and RHS vectors
	u=np.zeros((ndof,1))
	#column-based: i.e., left edge from bottom to top then move to right

	# Setup and solve FE problem
	sK=((KE.flatten()[np.newaxis]).T*xPhys).flatten(order='F')
	K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
	# Remove constrained dofs from matrix
	K = K[free,:][:,free]
	# Solve system 
	u[free,0]=spsolve(K,f[free,0])    

	return u, edofMat

#element stiffness matrix
def lk():
	E=1
	nu=0.33
	k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
	KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
	[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
	[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
	[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
	[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
	[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
	[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
	[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
	return (KE)

#Calculate principal stress and direction
#u is the deformation of 4 nodes (dim: 8)
#x,y is the barycentric coodinate, from 0 to 1 in the natural domain
def PrincipalStress(u, x, y): 
	if x>1:
		x = x%1
	if y>1:
		y = y%1

	DB = StressDisplacementMatrix(x, y);
	stress = DB@u
	if abs(stress[0]-stress[1])<1e-10 :
		print("Cannot calculate principal stress for element")

	#calculate principal direction
	theta = 0.5 * math.atan(2*stress[2] / (stress[0]-stress[1]))

	#calculate principal stresses
	c = math.cos(theta)
	s = math.sin(theta)
	cc=c*c
	ss=s*s
	cs=c*s
	ps1 = cc*stress[0] + 2*cs*stress[2] + ss*stress[1];
	ps2 = ss*stress[0] - 2*cs*stress[2] + cc*stress[1];

	if abs(ps2)>abs(ps1):
		tmp=ps1
		ps1=ps2
		ps2=tmp
		theta+=0.5*math.pi
	return theta, ps1, ps2

#return Strain-Displacement Matrix
def StrainDisplacementMatrix(x, y):
	B = np.array([
		[y-1, 0, 1-y, 0, y, 0, -y, 0],
		[0, x-1, 0, -x, 0, x, 0, 1-x],
		[x-1, y-1, -x, 1-y, x, y, 1-x, -y]
	]);
	return (B)

#return Stress-Displacement Matrix 
def StressDisplacementMatrix(x, y):
	E=1
	nu=0.33
	D = E/(1-nu*nu)*np.array(
		[[1, nu, 0],
		[nu, 1, 0],
		[0, 0, 0.5-0.5*nu]]);

	B = StrainDisplacementMatrix(x,y)

	return (D@B)


	DB = E/(1-nu*nu)*np.array([
		[y-1, nu*(x-1), 1-y, -nu*x, y, nu*x, -y, -nu*(x-1)],
		[nu*(y-1), x-1, -nu*(y-1), -x, nu*y, x, -nu*y, 1-x],
		[-0.5*(nu-1)*(x-1), -0.5*(nu-1)*(y-1), 0.5*x*(nu-1), 0.5*(nu-1)*(y-1), 
		 -0.5*x*(nu-1), -0.5*y*(nu-1), 0.5*(nu-1)*(x-1), 0.5*y*(nu-1)]
	]);
	return (DB)


# The real main driver    
if __name__ == "__main__":
	# Default input parameters
	nelx=120
	nely=80
	import sys
	if len(sys.argv)>1: nelx   =int(sys.argv[1])
	if len(sys.argv)>2: nely   =int(sys.argv[2])

	fea(nelx,nely)

	input("Press enter to exit program...")
