
#!/bin/bash

a=('thompson-sampling' 'thompson-sampling-with-hint')
file=("../instances/i-1.txt" "../instances/i-2.txt" "../instances/i-3.txt")
horizon=(100 400 1600 6400 25600 102400)


for i in "${file[@]}"; do	
	for j in "${a[@]}"; do
  		for k in {0..49}; do
  			for t in "${horizon[@]}"; do
  				python3 bandit.py --instance $i --algorithm $j --randomSeed $k --epsilon 0.02 --horizon $t
  			done
  		done
  	done
done

