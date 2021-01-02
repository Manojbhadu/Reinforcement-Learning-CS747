import numpy as np
import matplotlib.pyplot as plt


avg_steps_sarsa=[]
with open('data/sarsa.txt') as fp:
    lines = fp.readlines()
    for line in lines:
        data = line.split()
        avg_steps_sarsa.append(int(data[0]))

avg_steps_exp_sarsa=[]
with open('data/expected_sarsa.txt') as fp:
    lines = fp.readlines()
    for line in lines:
        data = line.split()
        avg_steps_exp_sarsa.append(int(data[0]))

avg_steps_q_learning=[]
with open('data/q_learning.txt') as fp:
    lines = fp.readlines()
    for line in lines:
        data = line.split()
        avg_steps_q_learning.append(int(data[0]))


avg_steps_0=[]
with open('data/Kings_move_without_stochastic.txt') as fp:
    lines = fp.readlines()
    for line in lines:
        data = line.split()
        avg_steps_0.append(int(data[0]))

avg_steps_1=[]
with open('data/Kings_move_with_stochastic.txt') as fp:
    lines = fp.readlines()
    for line in lines:
        data = line.split()
        avg_steps_1.append(int(data[0]))




#for generating combined plots of sarsa, q learning, expected sarsa
plt.plot(avg_steps_sarsa, np.arange(1, len(avg_steps_sarsa) + 1),label = "sarsa")
plt.plot(avg_steps_exp_sarsa, np.arange(1, len(avg_steps_exp_sarsa) + 1),label = "expected Sarsa")
plt.plot(avg_steps_q_learning, np.arange(1, len(avg_steps_q_learning) + 1),label = "Q learning")
plt.xlabel('Time steps')
plt.ylabel('Episodes')
plt.title("Comined plot for 4 moves using different updates")
plt.grid()
plt.legend()
plt.savefig('figures/task5.png')
plt.close()



#for generating combined plot of 4moves and  kings move (with stochasticity and without stochasticity)
plt.plot(avg_steps_sarsa, np.arange(1, len(avg_steps_sarsa) + 1),label = "sarsa_4_moves")
plt.plot(avg_steps_exp_sarsa, np.arange(1, len(avg_steps_exp_sarsa) + 1),label = "expected sarsa_4_moves")
plt.plot(avg_steps_q_learning, np.arange(1, len(avg_steps_q_learning) + 1),label = "Q-learning_4_moves")
plt.plot(avg_steps_0, np.arange(1, len(avg_steps_0) + 1),label = "Non-stochastic-with-kings-moves")
plt.plot(avg_steps_1, np.arange(1, len(avg_steps_1) + 1),label = "stochastic-with-kings-moves")
plt.xlabel('Time steps')
plt.ylabel('Episodes')
plt.title("Comined plot for 4 moves and Kings moves with and without stochasticity")
plt.grid()
plt.legend()
plt.savefig('figures/combined_all.png')
plt.close()


#for generating combined plot kings move with stochasticity and without stochasticity
plt.plot(avg_steps_0, np.arange(1, len(avg_steps_0) + 1),label = "Non-stochastic")
plt.plot(avg_steps_1, np.arange(1, len(avg_steps_1) + 1),label = "stochastic")
plt.xlabel('Time steps')
plt.ylabel('Episodes')
plt.title("Comined plot for Kings moves with and without stochasticity")
plt.grid()
plt.legend()
plt.savefig('figures/task4.png')
plt.close()


