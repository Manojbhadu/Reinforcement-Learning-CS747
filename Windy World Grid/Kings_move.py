import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
# Grid world height
gH = 7

# world width
gW = 10

# wind strength for each column
Wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_NORTH_EAST = 4
ACTION_NORTH_WEST = 5
ACTION_SOUTH_EAST = 6
ACTION_SOUTH_WEST = 7


# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

# reward for each step
REWARD = -1.0

START = [3, 0]
GOAL = [3, 7]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,ACTION_NORTH_EAST, ACTION_NORTH_WEST,ACTION_SOUTH_EAST,ACTION_SOUTH_WEST]


#defining steps with effect of winds
def step(state, action,stochastic):
    i, j = state

    if stochastic == 0:
        wind_strength = Wind[j]
    else: 
        wind_strength =  Wind[j] + np.random.randint(-1, 2)

    if action == ACTION_UP:
        a = min(max(i - 1 - wind_strength, 0),gH-1)
        b = min(max(j, 0), gW-1)
        return [a,b]

        
    elif action == ACTION_DOWN:
        a = min(max(i + 1 - wind_strength, 0), gH-1)
        b = min(max(j, 0), gW-1)
       
        return [a,b] 

    elif action == ACTION_LEFT:
        a = min(max(i - wind_strength, 0),gH-1)
        b = min(max(j - 1, 0),gW-1)
        return [a,b] 

    elif action == ACTION_RIGHT:
        a = min(max(i - wind_strength, 0),gH-1)
        b = min(max(j + 1, 0), gW-1)
        return [a,b] 
    
    
    elif action == ACTION_NORTH_EAST:
        a = min(max(i - 1 - wind_strength, 0), gH-1)
        b = min(max(j + 1, 0), gW-1)
        return [a,b]

    elif action == ACTION_NORTH_WEST:
        a = min(max(i - 1 - wind_strength, 0), gH-1)
        b = min(max(j - 1, 0), gW-1)
        return [a,b]

    elif action == ACTION_SOUTH_EAST:
        a = min(max(i + 1 - wind_strength, 0), gH-1)
        b = min(max(j + 1, 0), gW-1)
        return [a,b]
    elif action == ACTION_SOUTH_WEST:
        a = min(max(i + 1 - wind_strength, 0), gH-1)
        b = min(max(j - 1, 0), gW-1)
        return [a,b]




#defining episode which gives time steps in episode and updated q 
def episode_sarsa(q_value,seed,stochastic):
    # track the total time steps in this episode
    time_steps = 0
    np.random.seed(seed)
    # initialize state
    state = START

    # choose an action based on epsilon-greedy algorithm
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # keep going until get to the goal state
    while state != GOAL:
        next_state = step(state, action,stochastic)
        if np.random.binomial(1, EPSILON) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # Sarsa update
        q_value[state[0], state[1], action] += \
            ALPHA * (REWARD + q_value[next_state[0], next_state[1], next_action] -
                     q_value[state[0], state[1], action])

        #Expected sarsa update

        state = next_state
        action = next_action
        time_steps += 1
        #print(time_steps)
    return time_steps




def plots(stochasticity):
    q_value = np.zeros((gH, gW, 8))

    episode_limit = 500
    total_seeds = 10
    avg_steps = np.zeros(500)
    for i in range(total_seeds):
        q_value = np.zeros((gH, gW, 8))
        steps = []
        ep = 0
        while ep < episode_limit:
            steps.append(episode_sarsa(q_value,i,stochasticity))
            ep += 1
        steps = np.add.accumulate(steps)
        avg_steps = avg_steps + np.array(steps)
    avg_steps = avg_steps/total_seeds
    avg_steps = avg_steps.astype(int)
    plt.plot(avg_steps, np.arange(1, len(avg_steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    
    plt.grid()
    if stochasticity == 1:
        plt.title("King Move with stochastic")
        plt.savefig('figures/Kings_move_'+'with_'+'stochastic.png')
    elif stochasticity == 0:
        plt.title("King Move without stochastic")
        plt.savefig('figures/Kings_move_'+'without_'+'stochastic.png')
    plt.close()

    return avg_steps

if __name__ == '__main__':
    parser.add_argument('--stochastic', help='either 0 or 1')
    args = parser.parse_args()
    stochastic = int((args.stochastic))
    avg_steps = plots(stochastic)
    for i in range(len(avg_steps)):
        print(avg_steps[i])
