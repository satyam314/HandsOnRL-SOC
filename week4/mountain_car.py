# import gymnasium as gym
# import numpy as np
# import matplotlib.pyplot as plt

# '''
# The first task to work with a gym env is to initialise it using gym.make(name_of_env) and reset it using .reset() function. This resets the env to a starting position, with some noise in state. It returns a tuple of the initial state of environment and a dictionary containing info (Not important for the moment).

# Just as a environment in RL, you can take action based on current state. This is done using env.step(action), and it returns the following 5 values:

# 1. Next State: The state the environment transitioned to after taking the step.

# 2. Reward: Reward received for taking the action

# 3. Terminated: A boolean which is true if the environment terminated after taking the action. The condition for this termination is provided in the documentation of the environment.

# 4. Truncated: A boolean which is true if the environment was truncated after taking the action, usually because we cannot run a environment for infinite time, and hence every environment has a truncation period. More details in the documentation

# Note: You must call env.reset() after either termination or truncation.

# 5. Info: A dictionary containing information of env.

# The observation space of Mountation Car is an array of two variables: Position of car(x - coordinate) and velocity of cart. 

# Three actions are possible, as mentioned in documentation
# '''


import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import tqdm

class QLearningAgent:
    def __init__(self, environment_name: str) -> None:
        self.env_name = environment_name
        self.environment = gym.make(environment_name)
        self.current_state, _ = self.environment.reset()

        self.state_dim = len(self.current_state)
        self.num_actions = self.environment.action_space.n

        self.low_bound = self.environment.observation_space.low
        self.high_bound = self.environment.observation_space.high

        # Hyperparameters
        self.discrete_bins = [25, 25]
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.total_episodes = 25000
        self.exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.995

        self.q_values = np.random.uniform(low=-2, high=0, size=(*self.discrete_bins, self.num_actions))

    def discretize_state(self, state):
        scaled_state = (state - self.low_bound) / (self.high_bound - self.low_bound)
        indices = (scaled_state * np.array(self.discrete_bins)).astype(int)
        return tuple(np.clip(indices, 0, np.array(self.discrete_bins) - 1))

    def update_q_values(self, state, action, reward, next_state, terminal):
        current_index = self.discretize_state(state)
        next_index = self.discretize_state(next_state)

        future_q_value = 0 if terminal else np.max(self.q_values[next_index])
        target = reward + self.discount_factor * future_q_value
        error = target - self.q_values[current_index][action]

        self.q_values[current_index][action] += self.learning_rate * error

    def choose_action(self):
        state_index = self.discretize_state(self.current_state)
        if np.random.rand() < self.exploration_rate:
            return self.environment.action_space.sample()
        else:
            return np.argmax(self.q_values[state_index])

    def perform_step(self):
        action = self.choose_action()
        next_state, reward, done, _, _ = self.environment.step(action)

        self.update_q_values(self.current_state, action, reward, next_state, done)

        self.current_state = next_state
        return done

    def evaluate_agent(self):
        test_env = gym.make(self.env_name, render_mode="human")
        finished = False
        test_state, _ = test_env.reset()

        while not finished:
            action = np.argmax(self.q_values[self.discretize_state(test_state)])
            test_state, _, done, _, _ = test_env.step(action)
            test_env.render()
            finished = done

        test_env.close()

    def train_agent(self, evaluation_interval):
        episode_rewards = []

        for episode in tqdm.tqdm(range(1, self.total_episodes + 1)):
            finished = False
            total_reward = 0

            while not finished:
                finished = self.perform_step()
                total_reward += 1  # Customize reward tracking based on your environment

            episode_rewards.append(total_reward)
            self.current_state, _ = self.environment.reset()
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

            if episode % evaluation_interval == 0:
                self.evaluate_agent()

        self.plot_training_curve(episode_rewards)

    def plot_training_curve(self, rewards):
        plt.figure(figsize=(12, 6))
        plt.plot(rewards, color='royalblue', linewidth=2, label='Rewards per Episode')
        plt.title('Training Progress - Rewards by Episode', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Reward', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best', fontsize=12)
        plt.show()

if __name__ == "__main__":
    agent = QLearningAgent("MountainCar-v0")
    agent.train_agent(evaluation_interval=1000)
