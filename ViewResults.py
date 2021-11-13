import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from os import listdir
from os.path import isfile, join

import math

# The real main driver    
if __name__ == "__main__":
	nelx, nely = 120, 80
	volfrac = 0.4

	mode = "outputs"
	import sys
	if len(sys.argv)>1: mode=sys.argv[1]
	folder = "./" + mode +"/"

	simpFolder = folder
	simpFile = ""
	simp_x=np.zeros(nelx*nely)
	simp_ce=0
	for file in listdir(simpFolder):
		if file.endswith(".txt") and file.startswith("simp"):
			simpFile = file
			break
	if len(simpFile)==0:
		print("cannot find simp result")
	else:
		s = simpFile.rfind('_')+1
		e = len(simpFile)-4
		simp_ce = float(simpFile[s:e])
		simp_x = np.loadtxt(simpFolder+simpFile)
		nely = len(simp_x)
		nelx = len(simp_x[0])
		volfrac = np.sum(simp_x)/(nelx*nely)
		print("SIMP result: compliance %.2f, volfrac %.3f" % (simp_ce, volfrac))


	filenames = []
	inputfiles = []
	for file in listdir(folder):
		if file.endswith(".txt") and "simp" not in file:
			filenames.append(file)


	num = len(filenames)
	gen_x = []
	gen_ce = []
	gen_sim = []

	for i in range(num):
		density = np.loadtxt(folder+filenames[i])
		gen_x.append(density)
		s = filenames[i].rfind('_')+1
		e = len(filenames[i])-4
		gen_ce.append(float(filenames[i][s:e]))
		gen_sim.append(np.linalg.norm(density - simp_x)/math.sqrt(nelx*nely*volfrac*2)*100)

	# print some direct statistics
	opt_count = 0
	for i in range(num):
		if gen_ce[i] < simp_ce:
			opt_count+=1
			print(filenames[i])
	print("%d designs have lower compliance than simp result" % opt_count)
	

	col = round(math.sqrt(num))
	row = math.ceil(num/col)
	print("num, row, col = ", num, row, col)

	################################################################################
	print("display all resutls without ordering")
	# Initialize plot and plot the initial design
	plt.ion() # Ensure that redrawing is possible
	fig, axs = plt.subplots(row, col, sharex=True, sharey=True)
	fig.canvas.manager.set_window_title('No ordering')
	for i in range(num):
		density = gen_x[i]

		m = int(i/col)
		n = int(i%col)
		im = axs[m,n].imshow(-density, origin='lower', cmap='gray',\
							interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
		title = filenames[i]
		axs[m,n].set_title(title, size=7, pad=2)
		axs[m,n].set_yticks([])
		axs[m,n].set_xticks([])

	# set the rest to unvisible
	for i in range(num,row*col):
		m = int(i/col)
		n = int(i%col)
		axs[m,n].set_visible(False)

	plt.get_current_fig_manager().window.state('zoomed')
	plt.subplots_adjust(left=0.0, right=1.0, top=0.99, bottom=0.0)
	fig.show()
	plt.show()
	plt.pause(0.01)

	################################################################################
	print("sort by dissimilarity")
	idx=[gen_sim.index(max(gen_sim))]
	for l in range(1,num):
		max_d, max_i = 0.0, -1
		for i in range(num): #check all elements that are not already sorted
			if i not in idx:
				dd = gen_sim[i]
				for j in idx: #compare will all elements in the sorted list
					dd += np.linalg.norm(gen_x[i] - gen_x[j])
				if dd>max_d:
					max_d, max_i = dd, i
		if max_i>=0:
			idx.append(max_i)


	gen_x = np.array(gen_x)[idx]
	gen_ce = np.array(gen_ce)[idx]
	gen_sim = np.array(gen_sim)[idx]
	filenames = np.array(filenames)[idx]

	# Initialize plot and plot the initial design
	plt.ion() # Ensure that redrawing is possible
	fig, axs = plt.subplots(row, col, sharex=True, sharey=True)
	fig.canvas.manager.set_window_title('Results (Sorted by diversity)')
	for i in range(num):
		density = gen_x[i]

		m = int(i/col)
		n = int(i%col)
		im = axs[m,n].imshow(-density, origin='lower', cmap='gray',\
							interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
		title = filenames[i]
		axs[m,n].set_title(title, size=7, pad=2)
		axs[m,n].set_yticks([])
		axs[m,n].set_xticks([])

	# set the rest to unvisible
	for i in range(num,row*col):
		m = int(i/col)
		n = int(i%col)
		axs[m,n].set_visible(False)

	plt.get_current_fig_manager().window.state('zoomed')
	plt.subplots_adjust(left=0.0, right=1.0, top=0.99, bottom=0.0)
	fig.show()
	plt.show()
	plt.pause(0.01)


	input("Press enter to exit program...")