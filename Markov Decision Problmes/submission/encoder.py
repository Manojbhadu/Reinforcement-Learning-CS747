import numpy as np
import sys,argparse

parser = argparse.ArgumentParser()

def main():
    parser.add_argument('--grid')
    args = parser.parse_args()
    
    filepath = args.grid
    f = open(filepath)
    lines = [[i for i in line.split()] for line in f.read().split('\n')]
    f.close()
    lines.pop()
    n = len(lines)
    num_states = 0
    s = {}
    for p in range(1, n-1):
        for q in range(1, n-1):
            if lines[p][q]=='1':
                continue
            else:
                s[(p-1)*(n-1)+(q-1)] = num_states
                num_states += 1
                if lines[p][q]=='2':
                    start = s[(p-1)*(n-1)+(q-1)]
                elif lines[p][q]=='3':
                    end = s[(p-1)*(n-1)+(q-1)]
                    
    print("numStates", num_states)
    print("numActions", 4)
    print("start", start)
    print("end", end)
    for p in range(1, n-1):
        for q in range(1, n-1):
            if (p-1)*(n-1)+(q-1) in s and s[(p-1)*(n-1)+(q-1)]!=end:
                if lines[p-1][q]=='0':
                    print("transition", s[(p-1)*(n-1)+(q-1)], 0, s[(p-2)*(n-1)+(q-1)], -1, 1)
                elif lines[p-1][q]=='3':
                    print("transition", s[(p-1)*(n-1)+(q-1)], 0, s[(p-2)*(n-1)+(q-1)], num_states*2, 1)

                if lines[p+1][q]=='0':
                    print("transition", s[(p-1)*(n-1)+(q-1)], 3, s[p*(n-1)+(q-1)], -1, 1)
                elif lines[p+1][q]=='3':
                    print("transition", s[(p-1)*(n-1)+(q-1)], 3, s[p*(n-1)+(q-1)], num_states*2, 1)

                if lines[p][q-1]=='0':
                    print("transition", s[(p-1)*(n-1)+(q-1)], 2, s[(p-1)*(n-1)+(q-2)], -1, 1)
                elif lines[p][q-1]=='3':
                    print("transition", s[(p-1)*(n-1)+(q-1)], 2, s[(p-1)*(n-1)+(q-2)], num_states*2, 1)

                if lines[p][q+1]=='0':
                    print("transition", s[(p-1)*(n-1)+(q-1)], 1, s[(p-1)*(n-1)+q], -1, 1)
                elif lines[p][q+1]=='3':
                    print("transition", s[(p-1)*(n-1)+(q-1)], 1, s[(p-1)*(n-1)+q], num_states*2, 1)

    print("mdptype episodic")
    print("discount ", 0.95)
    
if __name__ == "__main__":
    main()