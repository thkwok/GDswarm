import topopt
import swarm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import time
from datetime import datetime
import math

def SymCantilever():
	nelx, nely = 120, 80
	ndof = 2*(nelx+1)*(nely+1)
	# BC's and support
	fixed=[0,1,2*nely,2*nely+1] #fix left top and bottom points
	# Set load
	f=np.zeros((ndof,1))
	f[2*nelx*(nely+1)+nely+1,0] = -1 #right-middle
	return fixed, f, nelx, nely, 0.4, []

def AsymCantilever():
	nelx, nely = 120, 80
	ndof = 2*(nelx+1)*(nely+1)
	# BC's and support
	fixed=[0,1,2*nely,2*nely+1] #fix left top and bottom points
	# Set load
	f=np.zeros((ndof,1))
	f[2*nelx*(nely+1)+1,0] = -1 #right-bottom
	return fixed, f, nelx, nely, 0.4, []


if __name__ == "__main__":
	volfrac, rmin, penal, ft = 0.4, 1.5, 3.0, 1
	# ft==0 -> sens, ft==1 -> dens

	folder = "./outputs/"
	fixed, f, nelx, nely, volfrac, passive = AsymCantilever()

	# for SIMP
	if True:
		density = volfrac * np.ones(nely*nelx,dtype=float)
		density, ce = topopt.simp(density,fixed,f,passive,nelx,nely,volfrac,penal,rmin,ft)
		filename = folder + "simp_%f.txt" % ce
		np.savetxt(filename, density.reshape((nelx,nely)).T, fmt='%f')
		print('Saved file : ', filename)

	np.random.seed()
	swarm_time=0
	opt_time=0
	elapsed = time.time()
	for loop in range(100):
		#swarm to set up the initial density
		L = math.sqrt(nelx*nely)
		ad, cd = np.random.uniform(0.06, 0.1, 2) * L
		sd = np.random.uniform(0.02, 0.06) * L
		w = np.random.uniform(0.5, 1.0, 4) #aw, cw, sw, psw
		aw, cw, sw, psw = w / np.sum(w)
		sampleN = np.random.randint(round(volfrac*L), round(L))
		print("sd %.2f, ad %.2f, cd %.2f, aw %.2f, cw %.2f, sw %.2f, psw %.2f, num %d" % (sd, ad, cd, aw, cw, sw, psw, sampleN))
		temp_time = time.time()
		density = swarm.swarm(fixed, f, nelx, nely, sampleN, sd, ad, cd, aw, cw, sw, psw, passive)
		swarm_time += (time.time() - temp_time)


		vol=np.sum(density)/(nelx*nely)
		iVol = vol;
		while vol<volfrac:
			# dilation
			dil = np.zeros(len(density))
			for i in range(nelx):
				for j in range(nely):
					ele = i*nely+j
					if density[ele]>0.5: continue
					if i>0 and density[(i-1)*nely+j]>0.5:
						dil[ele]=1.0
						continue
					if i<nelx-1 and density[(i+1)*nely+j]>0.5:
						dil[ele]=1.0
						continue
					if j>0 and density[i*nely+(j-1)]>0.5:
						dil[ele]=1.0
						continue
					if j<nely-1 and density[i*nely+(j+1)]>0.5:
						dil[ele]=1.0
						continue
			density += dil
			vol=np.sum(density)/(nelx*nely)


		#release a bit for optimization
		density[density > 0.9] = 0.99
		density[density < 0.1] = 0.01

		vol=np.sum(density)/(nelx*nely)
		if vol>volfrac:
			density = density * volfrac/vol


		#simp
		temp_time = time.time()
		density, ce = topopt.simp(density,fixed,f,passive,nelx,nely,volfrac,penal,rmin,ft)
		opt_time += (time.time() - temp_time)

		#export optimized result
		timestamp = datetime.now().strftime("%H%M")
		filename = folder + "%d_%s_%f.txt" % (loop,timestamp,ce)
		np.savetxt(filename, density.reshape((nelx,nely)).T, fmt='%f')
		print('Saved file : ', filename, '----------------------------')

		del density

	elapsed = (time.time() - elapsed)
	print("Total elapsed time is %.0f s. Swarm: %.0fs, TO: %.0fs" % (elapsed, swarm_time, opt_time))
	input("Press enter to exit program...")
