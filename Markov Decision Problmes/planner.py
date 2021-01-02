import numpy as np
import pulp as p
import argparse
parser = argparse.ArgumentParser()
from pulp import *


class MDPs():
    """docstring for MDPs"""
    def __init__(self, nS,nA,Reward,tranProb,gamma,mdptype):

        self.nS = nS
        self.nA = nA
        self.Reward = Reward
        self.tranProb = tranProb
        self.gamma = gamma
        self.mdptype = mdptype
    def VI(self):
        Vi = np.zeros(self.nS)
        Vf = np.zeros(self.nS)

        pi = np.zeros(self.nS)

        count=0
        while True:
            Vf = np.zeros(self.nS)
            pi = np.zeros(self.nS)
            for s in range(self.nS):
                max_tmp = np.sum((self.tranProb[s,0,:])*(self.Reward[s,0,:]+(self.gamma)*Vi))
                for a in range(1,self.nA):
                    tmp = np.sum((self.tranProb[s,a,:])*(self.Reward[s,a,:]+(self.gamma*Vi)))
                    if tmp > max_tmp:
                        pi[s]=a
                        max_tmp = tmp
                Vf[s] = max_tmp
        

            if np.allclose(Vf, Vi, rtol=1e-13, atol=1e-15) or count > 20000 :
                    break
            Vi = Vf.copy()
            count+=1
        #print(count)
        return Vf, pi

        
    def Value_function(self, s, pi):
        V = np.zeros(self.nS)
        eps = 1e-10
        V1 = np.ones(self.nS)
        while (np.linalg.norm(V1-V, ord=np.inf))>eps:
            V = V1
            V = V.reshape(1,1,self.nS)
            V1 = np.sum(self.tranProb[s, pi,:]*(self.Reward[s, pi,:] + self.gamma*V), axis=-1)
            V = V.reshape(-1)
        return V

    def HPI(self):
        pi = np.zeros(self.nS,int)
        s = np.array([i for i in range(self.nS)])
        Vs = self.Value_function(s,pi)
        Qs = np.sum(self.tranProb*(self.Reward+self.gamma*Vs), axis=-1)
        Vs = Vs.reshape(self.nS,1)
        diff = Qs - Vs
        max_diff = np.argmax(diff,-1)
        while(not np.all(pi == max_diff)):
            pi = max_diff
            Vs = self.Value_function(s, pi)
            Qs = np.sum(self.tranProb*(self.Reward+self.gamma*Vs), axis=-1)
            Vs = Vs.reshape(self.nS,1)
            diff = Qs - Vs

            if diff.any() > 0:
                max_diff = np.argmax(diff,-1)

            Vs = self.Value_function(s,pi)
            Vs = Vs.reshape(self.nS)
        return Vs, pi 

    def bellman(self, state, action, V):
        tranProb = self.tranProb[state,action,:]
        Reward = self.Reward[state,action,:]
        tranProb = tranProb.reshape((self.nS, 1))
        Reward = Reward.reshape((self.nS, 1))
        V = V.reshape((self.nS,1))
        constant = Reward + self.gamma*V
        result  = np.transpose(tranProb).dot(constant)
        return result[0][0]


    def LP(self):
        prob = LpProblem("MDP_using_lp", LpMaximize)
        Vs = LpVariable.dict("Value", range(self.nS))
        prob += lpSum([-Vs[i] for i in range(self.nS)])
        Vf = np.zeros((self.nS,1,1),dtype = LpVariable)

        for i in range(self.nS):
            Vf[i][0] = Vs[i]

        for i in range(self.nS):
            for j in range(self.nA):
                const = self.bellman(i,j,Vf)
                prob += Vs[i] >= const

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        Vstar = np.zeros((self.nS,1))
        for i in range(self.nS):
            Vstar[i,0] = Vs[i].value()

        Vstar = Vstar.reshape((1,1,self.nS))
        pi = np.argmax(np.sum(self.tranProb*(self.Reward+self.gamma*Vstar),axis=-1),axis=-1)
        Vstar = Vstar.reshape(self.nS)
        return Vstar, pi







if __name__ == '__main__':
	parser.add_argument("--mdp", type=str,help='path to mdp file')
	parser.add_argument("--algorithm", type=str, help="algoritham is one of vi,hpi,lp")

	args = parser.parse_args()

	mdpfile = args.mdp
	algo = args.algorithm
	a = []
	with open(mdpfile) as fp:
	    lines = fp.readlines()
	    for line in lines:
	        a.append(line.split())

	nS = int(a[0][1])
	nA = int(a[1][1])
	dt = {}
	end = []
	for i in range(1,len(a[3])):
		end.append(a[3][i])
	#endS = np.array(end.copy())
	dt['mdptype'] = a[-2][1]
	dt['gamma'] = float(a[-1][1])
	dt['start'] = int(a[2][1])
	Reward = np.zeros((nS,nA,nS))
	tranProb = np.zeros((nS,nA,nS))
	m = len(a)
	for i in range(4,len(a)-2):
		Reward[int(a[i][1])][int(a[i][2])][int(a[i][3])] = float(a[i][4])
		tranProb[int(a[i][1])][int(a[i][2])][int(a[i][3])] = float(a[i][5])

	solve_mdp = MDPs(nS, nA, Reward, tranProb, dt['gamma'], dt['mdptype'])

	if algo == 'vi':
		Vstar, piStar = solve_mdp.VI()
		for i in range(nS):
			print('%10.7f'%Vstar[i]," ",piStar[i])

	if algo == 'hpi':
		Vstar, piStar = solve_mdp.HPI()
		for i in range(nS):
			print('%10.7f'%Vstar[i]," ",piStar[i])

            

	if algo == 'lp':
		Vstar, piStar = solve_mdp.LP()
		for i in range(nS):
			print('%10.7f'%Vstar[i]," ",piStar[i])








