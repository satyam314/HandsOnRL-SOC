import numpy as np
import matplotlib.pyplot as plt
from bandits import Bandit
from agents import *

def plot_results(agents, bandit, num_steps=1000):
    rewards = np.zeros((len(agents), num_steps))
    regrets = np.zeros((len(agents), num_steps))
    
    for i, agent in enumerate(agents):
        total_reward = 0
        for step in range(num_steps):
            reward = agent.act()
            total_reward += reward
            rewards[i, step] = reward
            regrets[i, step] = bandit.get_regret()
        bandit.reset_regret()  # Reset the bandit for the next agent

    # Plotting reward per step
    plt.figure(figsize=(12, 6))
    for i, agent in enumerate(agents):
        plt.plot(rewards[i], label=type(agent).__name__)
    plt.xlabel('Steps')
    plt.ylabel('Reward per Step')
    plt.title('Reward per Step by Different Agents')
    plt.legend()
    plt.savefig("reward.png")
    plt.show()

    # Plotting total regret
    plt.figure(figsize=(12, 6))
    for i, agent in enumerate(agents):
        plt.plot(regrets[i], label=type(agent).__name__)
    plt.xlabel('Steps')
    plt.ylabel('Total Regret')
    plt.title('Total Regret Accumulated by Different Agents')
    plt.legend()
    plt.savefig("regrets.png")
    plt.show()

# Test setup
n_bandits = 10
bandit = Bandit(n_bandits, "Bernoulli")

# Creating different agents
greedy_agent = GreedyAgent(bandit, initialQ=1.0)
eps_greedy_agent = epsGreedyAgent(bandit, epsilon=0.1)
ucb_agent = UCBAAgent(bandit, c=2)
gradient_bandit_agent = GradientBanditAgent(bandit, alpha=0.1)
thompson_agent = ThompsonSamplerAgent(bandit)

# List of agents
agents = [greedy_agent, eps_greedy_agent, ucb_agent, gradient_bandit_agent, thompson_agent]

# Plot the results
plot_results(agents, bandit, num_steps=1000)
