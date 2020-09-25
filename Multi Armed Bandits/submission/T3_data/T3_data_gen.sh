
#!/bin/bash

a=('epsilon-greedy')
file=("../instances/i-2.txt")
horizon=(102400)
eps_=(0.001)


for i in "${file[@]}"; do	
	for j in "${a[@]}"; do
  		for k in {0..49}; do
  			for t in "${horizon[@]}"; do
  				#for eps in `seq -f "%f" 0.01 0.01 .1`; do
  				for eps in "${eps_[@]}"; do
  					python3 bandit.py --instance $i --algorithm $j --randomSeed $k --epsilon $eps --horizon $t
  				done
  			done
  		done
  	done
done
