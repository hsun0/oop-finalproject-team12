import gymnasium as gym
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse


def print_success_rate(rewards_per_episode):
    """Calculate and print the success rate of the agent."""
    total_episodes = len(rewards_per_episode)
    success_count = np.sum(rewards_per_episode)
    success_rate = (success_count / total_episodes) * 100
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

def run(episodes, is_training=True, render=False, epsilon_decay_rate=0.0001, min_exploration_rate=0.0, epsilon=1.0, discount_factor_g=0.9, start_learning_rate_a=0.5, min_learning_rate_a=0.1, learning_decay_rate=0.0001):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    # learning_rate_a = 0.9 # alpha or learning rate
    # discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    # epsilon = 1         # 1 = 100% random actions
    # epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    start_learning_rate_a = 0.5
    min_learning_rate_a = 0.1
    learning_rate_a = start_learning_rate_a
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset(seed=int(rng.integers(0, 10000)))[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200

        def bfs_shortest_distances(goal_state, grid_size):
            """Perform BFS to calculate shortest distances from all states to the goal state."""
            distances = np.full((grid_size, grid_size), np.inf)
            queue = [(goal_state, 0)]  # (state, distance)
            visited = set()

            while queue:
                current, dist = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                distances[current // grid_size, current % grid_size] = dist

                # Add neighbors (up, down, left, right) to the queue
                for action in [-1, 1, -grid_size, grid_size]:
                    neighbor = current + action
                    if 0 <= neighbor < grid_size * grid_size and neighbor not in visited:
                        # Ensure valid moves within the grid
                        if action == -1 and current % grid_size == 0:  # Left edge
                            continue
                        if action == 1 and current % grid_size == grid_size - 1:  # Right edge
                            continue
                        queue.append((neighbor, dist + 1))

            return distances

        goal_state = 63  # Bottom-right corner
        grid_size = 8
        shortest_distances = bfs_shortest_distances(goal_state, grid_size)

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = rng.integers(0, env.action_space.n) # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            if is_training:
                if terminated and reward == 0:
                    distance_to_goal = shortest_distances[new_state // grid_size, new_state % grid_size]
                    reward += (0.01 / (distance_to_goal + 1))  # Closer to goal gets higher reward
                    pass

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, min_exploration_rate)
        # print("learning rate:", learning_rate_a)
        learning_rate_a = max(learning_rate_a - learning_decay_rate, min_learning_rate_a)

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')
    
    if is_training == False:
        print(print_success_rate(rewards_per_episode))
        total_episodes = len(rewards_per_episode)
        success_count = np.sum(rewards_per_episode)
        success_rate = (success_count / total_episodes) * 100
        return success_rate

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    min_exploration_rate = 0.0005935994732258189
    epsilon_decay_rate = 0.00022073098440142171
    discount_factor_g = 0.9779776722671771
    start_learning_rate_a = 0.38674403968529697
    min_learning_rate_a = 0.0021001398462318953
    learning_decay_rate = 0.00027184353604355613

    mean = 0
    itnum = 10
    for _ in range(itnum):
        run(15000, is_training=True, render=False, epsilon_decay_rate=epsilon_decay_rate, min_exploration_rate=min_exploration_rate, epsilon=1.0, discount_factor_g=discount_factor_g, learning_decay_rate=learning_decay_rate)
        mean += run(750, is_training=False, render=False, epsilon_decay_rate=epsilon_decay_rate, min_exploration_rate=min_exploration_rate, epsilon=0.02, discount_factor_g=discount_factor_g, learning_decay_rate=learning_decay_rate)
    mean /= itnum
    print(f"Average success rate over {itnum} runs: {mean}%")