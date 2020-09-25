import numpy as np
import random
import argparse

from policies import eps_greedy, UCB, KLUCB, thompson_sampling, thompson_sampling_with_hint

parser = argparse.ArgumentParser()
parser.add_argument('--instance', help='path to instance file')
parser.add_argument('--algorithm', help=' al is one of epsilon-greedy, ucb, kl-ucb, thompson-sampling, and thompson-sampling-with-hint.')
parser.add_argument('--randomSeed', help=' rs is a non-negative integer')
parser.add_argument('--epsilon', help='ep is a number in [0, 1]')
parser.add_argument('--horizon', help='hz is a non-negative integer')

args = parser.parse_args()

algo = (args.algorithm)
instance_file = args.instance
eps = float(args.epsilon)
T = int(args.horizon)
seed = int(args.randomSeed)

true_mean = []
a = []
with open(instance_file) as fp:
    lines = fp.readlines()
    for line in lines:
        n = (line.split())
        n = float(n[0])
        a.append(n)
true_mean = np.array(a)

sorted_true_mean = np.sort(true_mean)

if __name__ == "__main__":

    if algo == "epsilon-greedy":
        REG = eps_greedy(true_mean, T, eps,seed)
    
    if algo == "ucb":
        REG = UCB(true_mean,T,seed)

    if algo == "kl-ucb":
        REG  = KLUCB(true_mean, T,seed)

    if algo == "thompson-sampling": 
        REG = thompson_sampling(true_mean,T,seed)

    if algo == "thompson-sampling-with-hint":
        REG = thompson_sampling_with_hint(true_mean, T, seed, sorted_true_mean)

    print(instance_file+",",algo+",", str(seed)+",", str(eps)+",", str(T)+",", REG)
    #print(algo, end=" ")
    # print(seed, end=" ")
    # print(eps, end=" ")
    # print(T, end=" ")
    # print(REG)







    