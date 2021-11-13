# Swarm Intelligence

import numpy as np
import math
import random
from matplotlib import colors
import matplotlib.pyplot as plt
import fea
import time

nelx=120
nely=80
scale=1
width = 1
height = 1
u = []
edofMat = []
sSTD=0.0
sMAX=0.0
density=[]
#the distances defining neighbors
separateDistance = 2.5
alignmentDistance = 5.0
cohesionDistance = 5.0
#weight for the forces
alignmentWeight = 1.0
cohesionWeight = 1.0
separationWeight = 1.5
stressWeight = 1.0

def swarm(fixed, f, _nelx, _nely, sampleN=50, sd=2.5, ad=5.0, cd=5.0, aw=1.0, cw=1.0, sw=1.5, psw=1.0, passive=[]):
	global u, edofMat, sSTD, sMAX, width, height, nelx, nely, density
	global separateDistance, alignmentDistance, cohesionDistance, alignmentWeight, cohesionWeight, separationWeight, stressWeight

	separateDistance, alignmentDistance, cohesionDistance = sd, ad, cd
	alignmentWeight, cohesionWeight, separationWeight, stressWeight = aw, cw, sw, psw

	nelx, nely = _nelx, _nely
	width, height = nelx*scale, nely*scale


	#Do FEA to get deformations
	u, edofMat = fea.fea(fixed,f,nelx,nely,passive)
	sSTD, sMAX = GetMaxAndStdForPS(u, edofMat)

	#swarm
	swarm = Swarm()

	rList = np.random.rand(2, sampleN)
	rList[0] *= width
	rList[1] *= height

	for i in range(sampleN):
		swarm.addAgent(Agent(rList[0][i],rList[1][i]))


	density = np.zeros(nely*nelx,dtype=float)

	elapsed = time.time()
	for i in range(200):
		active = swarm.run()
		if not active:
			break
	elapsed = (time.time() - elapsed)*1000.0
	print("Swarm stopped at iter #%d: Elapsed time is %.3f ms." % (i, elapsed))

	del swarm

	return density



class Swarm:
	'''Swarm class'''
	def __init__(self):
		self.agentArray = []
		random.seed()

	def __del__(self):
		del self.agentArray

	def run(self):
		active = False
		for agent in self.agentArray:
			active = agent.run(self.agentArray) or active
		return active	
    
	def addAgent(self, agent):
		self.agentArray.append(agent)

	def render(self):
		x = []
		y = []
		for agent in self.agentArray:
			x.append(agent.position[0])
			y.append(agent.position[1])
		plt.clf()
		plt.xlim(0, width);
		plt.ylim(0, height);
		plt.scatter(x, y, marker='o')
		plt.show()
		plt.pause(0.000001)

	def drawPath(self):
		for agent in self.agentArray:
			x = []
			y = []
			for p in agent.history:
				x.append(p[0])
				y.append(p[1])
			plt.plot(x, y, '-')
		plt.show()
		plt.pause(0.001)

class Agent:
	'''Agent class'''
	r = 2.0 #size of agent
	maxspeed = 1.0 #Maximum speed - control step size
	maxforce = 0.1 #Maximum steering force - control smoothness

	def __init__(self, x, y, v1=0, v2=0):
		self.position = np.array([x,y])
		self.acceleration = np.array([0.0,0.0])
		self.velocity = np.array([v1,v2])
		if v1==0 and v2==0:
			angle = random.random()*math.pi*2
			self.velocity = np.array([math.cos(angle), math.sin(angle)])
		self.active = True
		self.history = [] #for storing the past positions

	def __del__(self):
		del self.position
		del self.acceleration
		del self.velocity
		del self.history


	def run(self, agentArray):
		if self.active == False:
			return False
		self.ACSrules(agentArray)
		self.PrincipalStressGuidence()
		self.updatePosition()
		self.checkTermination()
		return True
	
	def applyForce(self, force):
		# could add mass for a = F / m
		force = limitForce(force)
		self.acceleration += force

	global limitForce
	def limitForce(force):
		# Limit to maximum steering force
		f = np.sqrt(np.sum(force**2))
		if f>Agent.maxforce:
			force *= Agent.maxforce/f
		return force;

	# use principal stress direction to 
	def PrincipalStressGuidence(self):
		if stressWeight==0.0:
			return
		p = self.position / scale
		i = math.floor(p[0])
		if i>=nelx:
			i=nelx-1
		j = math.floor(p[1])
		if j>=nely:
			j=nely-1
		ele=i*nely+j
		theta, ps1, ps2 = fea.PrincipalStress(u[edofMat[ele]], p[0]-i, p[1]-j)
		v = np.array([math.cos(theta), math.sin(theta)])
		v2 = np.array([math.cos(theta+0.5*math.pi), math.sin(theta+0.5*math.pi)])
		d = np.dot(v, self.velocity)
		d2 = np.dot(v2, self.velocity)
		if abs(d)<abs(d2):
			v = v2
			d = d2
			ps1 = ps2
		if d<0:
			v = -v
		l = np.sqrt(np.sum(v**2))
		if l<=0 or np.isnan(l):
			return np.array([0.0,0.0])

		w = min(abs(ps1)/sSTD, 1)

		v *= w*Agent.maxspeed/l
		#Steering = Desired minus Velocity
		steer = v - self.velocity
		stressForce=steer

		self.applyForce(stressWeight*stressForce)

	# accumulate a new acceleration each time based on three rules
	def ACSrules(self, agentArray):
		# find neighbors
		aNeighbors, cNeighbors, sNeighbors = self.FindNeighbors(agentArray, alignmentDistance, cohesionDistance, separateDistance)
		if len(aNeighbors)==0:
			return 

		alignment = self.computeAlignment(aNeighbors);
		cohesion = self.computeCohesion(cNeighbors);
		separation = self.computeSeparation(sNeighbors);

		self.applyForce(alignmentWeight*alignment)
		self.applyForce(cohesionWeight*cohesion)
		self.applyForce(separationWeight*separation)

	def updatePosition(self):
		#save position
		self.history.append(self.position.copy())
		#Update velocity
		self.velocity += limitForce(self.acceleration)
		#limit velocity
		v = np.sqrt(self.velocity[0]**2+self.velocity[1]**2)
		if v > Agent.maxspeed:
			self.velocity *= Agent.maxspeed/v
		if v < 0.2:
			self.velocity *= 0.2/v
		self.position += self.velocity
		#reset acceleration
		self.acceleration = np.array([0.0,0.0])
		#update density
		p = self.position / scale
		i = math.floor(p[0])
		if i>=nelx: i=nelx-1
		if i<0: i=0
		j = math.floor(p[1])
		if j>=nely: j=nely-1
		if j<0: j=0
		ele=i*nely+j
		global density
		density[ele] = 1.0


	def FindNeighbors(self, agentArray, alignmentDistance, cohesionDistance, separateDistance):
		aNeighbors=[]
		cNeighbors=[] 
		sNeighbors=[]
		for agent in agentArray:
			if agent != self and agent.active:
				v = self.position - agent.position
				d = np.sqrt(np.sum(v**2))
				if d<=alignmentDistance:
					aNeighbors.append(agent)
				if d<=cohesionDistance:
					cNeighbors.append(agent)
				if d<=separateDistance:
					sNeighbors.append(agent)
		return aNeighbors, cNeighbors, sNeighbors;

	#Alignment - For every neighbor, calculate the average velocity
	def computeAlignment(self, neighbors):
		if len(neighbors)==0 or alignmentWeight==0.0:
			return np.array([0.0,0.0])
		v1 = self.velocity/np.sqrt(np.sum(self.velocity**2))
		v = np.array([0.0,0.0])
		for nAgent in neighbors:
			v2 = nAgent.velocity/np.sqrt(np.sum(nAgent.velocity**2))
			if np.dot(v1, v2)>0.2:
				v += nAgent.velocity
		#v /= len(neighbors);
		l = np.sqrt(np.sum(v**2))
		if l<=0:
			return np.array([0.0,0.0])
		v *= Agent.maxspeed/l
		#Steering = Desired minus Velocity
		steer = v - self.velocity
		return steer;

	#Cohesion - For the average position (i.e. center) of all neighbors, calculate steering vector towards that position
	def computeCohesion(self, neighbors):
		if len(neighbors)==0 or cohesionWeight==0.0:
			return np.array([0.0,0.0])
		v1 = self.velocity/np.sqrt(np.sum(self.velocity**2))
		p = np.array([0.0,0.0])
		count=0
		for nAgent in neighbors:
			v2 = nAgent.velocity/np.sqrt(np.sum(nAgent.velocity**2))
			if np.dot(v1, v2)>0.2:
				p += nAgent.position
				count+=1
		if count>0:
			p /= count
		# A vector pointing from the position to the target, Scale to maximum speed
		v = p - self.position
		l = np.sqrt(np.sum(v**2))
		if l<=0:
			return np.array([0.0,0.0])
		v *= Agent.maxspeed/l
		#Steering = Desired minus Velocity
		steer = v - self.velocity
		return steer;

	#Separation - checks for nearby neighbors and steers away
	def computeSeparation(self, neighbors):
		if len(neighbors)==0 or separationWeight==0.0:
			return np.array([0.0,0.0])
		v = np.array([0.0,0.0])
		for nAgent in neighbors:
			v += self.position - nAgent.position;
		l = np.sqrt(np.sum(v**2))
		if l<=0:
			return np.array([0.0,0.0])
		v *= Agent.maxspeed/l
		#Steering = Desired minus Velocity
		steer = v - self.velocity
		return steer;


	def checkTermination(self):
		if self.position[0]<=0 or self.position[0]>=width or self.position[1]<=0 or self.position[1]>=height:
			self.active = False
			return

def GetMaxAndStdForPS(u, edofMat):
	sList = []
	for i in range(nelx):
		for j in range(nely):
			ele=i*nely+j
			theta, ps1, ps2 = fea.PrincipalStress(u[edofMat[ele]], 0.5, 0.5)
			sList.append(abs(ps1))
			sList.append(abs(ps2))
	sSTD = np.std(sList)
	sMAX = np.max(sList);
	return sSTD, sMAX
	

if __name__ == "__main__":

	nelx, nely = 120, 80
	# BC's and support
	ndof = 2*(nelx+1)*(nely+1)
	fixed=[0,1,2*nely,2*nely+1]
	# Set load
	f=np.zeros((ndof,1))
	f[2*nelx*(nely+1)+1,0] = -1

	swarm(fixed,f,nelx,nely)
	input("Press enter to exit program...")
