import numpy as np
import sys, argparse
from MDP import MDP_solver

parser = argparse.ArgumentParser()

def main():
    parser.add_argument('--grid')
    parser.add_argument('--value_policy')
    args = parser.parse_args()
    
    filepath = args.grid
    f = open(filepath)
    lines = [[i for i in line.split()] for line in f.read().split('\n')]
    f.close()
    lines.pop()
    n = len(lines)
    
    act = {0:'N', 1:'E', 2:'W', 3:'S'}
    num_states = 0
    s = {}
    a_st = 0
    a_en = 0
    for p in range(1, n-1):
        for q in range(1, n-1):
            if lines[p][q]=='1':
                continue
            else:
                s[(p-1)*(n-1)+(q-1)] = num_states
                num_states += 1
                if lines[p][q]=='2':
                    start = s[(p-1)*(n-1)+(q-1)]
                    a_st = [p, q]
                elif lines[p][q]=='3':
                    end = s[(p-1)*(n-1)+(q-1)]
    filepath = args.value_policy
    f = open(filepath)
    lines = [[i for i in line.split()] for line in f.read().split('\n')]
    lines.pop()
    pi = np.zeros(len(lines))
    for i, j in enumerate(lines):
        pi[i] = int(j[1])
    q = start
    sx = ""
    while q!=end:
        if pi[q] == 0:
            sx += act[0]+" "
            a_st[0] = a_st[0]-1
        elif pi[q]==1:
            sx += act[1]+" "
            a_st[1] = a_st[1]+1
        elif pi[q]==2:
            sx+=act[2]+" "
            a_st[1] = a_st[1]-1
        else:
            sx += act[3]+" "
            a_st[0] = a_st[0]+1
        q = s[(a_st[0]-1)*(n-1)+(a_st[1]-1)]
    print(sx)
#     return sx
        
    
    
    
    
    
if __name__ =='__main__':
    main()