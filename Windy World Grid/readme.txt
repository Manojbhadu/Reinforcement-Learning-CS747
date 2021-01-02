
1. for generating all plots and data.
	first delete everything inside figures and data folder(not folder) otherwise running program will update them

	run: bash data_plot.sh

	it will generate follwing things:
	data: sarsa.txt, expected_sarsa.txt, q_learning.txt (for 4 moves) and kings_move_with/without_stochastic.txt
	plots: sarsa.png, expected_sarsa.png, q_learning.png( for 4 moves), kings_move_with/without_stochastic.png, task4.png(combine plot for king moves), task5.png( comparison plot of three update algorithms for 4 moves), combined_all.png(4 moves (three alogos), kings_move(stochastic and non-stochastic))

2. for indivusal algorithms

	run: 

	1. python3 grid_world.py --update sarsa >> data/sarsa.txt
	   this will generate plot sarsa.png ans save data at data/sarsa.txt for 4 moves using Sarsa

	2. python3 grid_world.py --update expected_sarsa >> data/expected_sarsa.txt
		this will generate plot expected_sarsa.png ans save data at data/expected_sarsa.txt for 4 moves using Expected Sarsa

	3. python3 grid_world.py --update q_learning >> data/q_learning.txt
		this will generate plot q_learning.png ans save data at data/q_learning.txt for 4 moves using Q learning

	4. python3 Kings_move.py --stochastic 1 >> data/Kings_move_with_stochastic.txt
		this will generate plot Kings_move_with_stochastic.png ans save data at data/Kings_move_with_stochastic.txt for Kings move with stochastic

	5. python3 Kings_move.py --stochastic 0 >> data/Kings_move_without_stochastic.txt
		this will generate plot Kings_move_without_stochastic.png ans save data at data/Kings_move_without_stochastic.txt for Kings move without stochastic

	6. For using stochastic for 4 moves change line 261 in grid_world.py and repeat steps 1,2,3(not part of assignment)

	
