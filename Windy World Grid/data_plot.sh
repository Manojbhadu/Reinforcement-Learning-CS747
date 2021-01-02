
#!/bin/bash


python3 grid_world.py --update sarsa >> data/sarsa.txt
python3 grid_world.py --update expected_sarsa >> data/expected_sarsa.txt
python3 grid_world.py --update q_learning >> data/q_learning.txt
python3 Kings_move.py --stochastic 1 >> data/Kings_move_with_stochastic.txt
python3 Kings_move.py --stochastic 0 >> data/Kings_move_without_stochastic.txt

python3 plots.py