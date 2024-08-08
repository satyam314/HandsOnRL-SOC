from bandits import Bandit
# Import libraries if you need them
import random
import math
import numpy as np

class Agent:
    def __init__(self, bandit: Bandit) -> None:
        self.bandit = bandit
        self.banditN = bandit.getN()

        self.rewards = 0
        self.numiters = 0
    

    def action(self) -> int:
        '''This function returns which action is to be taken. It must be implemented in the subclasses.'''
        raise NotImplementedError()

    def update(self, choice : int, reward : int) -> None:
        '''This function updates all member variables you may require. It must be implemented in the subclasses.'''
        raise NotImplementedError()

    # dont edit this function
    def act(self) -> int:
        choice = self.action()
        reward = self.bandit.choose(choice)

        self.rewards += reward
        self.numiters += 1

        self.update(choice,reward)
        return reward

class GreedyAgent(Agent):
    def __init__(self, bandits: Bandit, initialQ : float) -> None:
        super().__init__(bandits)
        self.q_values = [initialQ] * self.banditN
        self.action_counts = [0] * self.banditN
        # add any member variables you may require
        
    # implement
    def action(self) -> int:
        return self.q_values.index(max(self.q_values))

    # implement
    def update(self, choice: int, reward: int) -> None:
        self.action_counts[choice] += 1
        self.q_values[choice] += (reward - self.q_values[choice]) / self.action_counts[choice]

class epsGreedyAgent(Agent):
    def __init__(self, bandits: Bandit, epsilon : float) -> None:
        super().__init__(bandits)
        self.epsilon = epsilon
        # add any member variables you may require
        self.q_values = [0.0] * self.banditN
        self.action_counts = [0] * self.banditN
    
    # implement
    def action(self) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.banditN - 1)
        else:
            return self.q_values.index(max(self.q_values))

    # implement
    def update(self, choice: int, reward: int) -> None:
        self.action_counts[choice] += 1
        self.q_values[choice] += (reward - self.q_values[choice]) / self.action_counts[choice]

class UCBAAgent(Agent):
    def __init__(self, bandits: Bandit, c: float) -> None:
        super().__init__(bandits)
        self.c = c
        # add any member variables you may require
        self.q_values = [0.0] * self.banditN
        self.action_counts = [0] * self.banditN

    # implement
    def action(self) -> int:
        for i in range(self.banditN):
            if self.action_counts[i] == 0:
                return i
        ucb_values = [self.q_values[i] + self.c * math.sqrt(math.log(self.numiters + 1) / self.action_counts[i]) for i in range(self.banditN)]
        return ucb_values.index(max(ucb_values))

    # implement
    def update(self, choice: int, reward: int) -> None:
        self.action_counts[choice] += 1
        self.q_values[choice] += (reward - self.q_values[choice]) / self.action_counts[choice]

class GradientBanditAgent(Agent):
    def __init__(self, bandits: Bandit, alpha : float) -> None:
        super().__init__(bandits)
        self.alpha = alpha
        # add any member variables you may require
        self.preferences = [0.0] * self.banditN
        self.avg_reward = 0.0

    # implement
    def action(self) -> int:
        exp_preferences = np.exp(self.preferences)
        probabilities = exp_preferences / np.sum(exp_preferences)
        return np.random.choice(range(self.banditN), p=probabilities)

    # implement
    def update(self, choice: int, reward: int) -> None:
        self.avg_reward += (reward - self.avg_reward) / (self.numiters + 1)
        exp_preferences = np.exp(self.preferences)
        probabilities = exp_preferences / np.sum(exp_preferences)
        for i in range(self.banditN):
            if i == choice:
                self.preferences[i] += self.alpha * (reward - self.avg_reward) * (1 - probabilities[i])
            else:
                self.preferences[i] -= self.alpha * (reward - self.avg_reward) * probabilities[i]

class ThompsonSamplerAgent(Agent):
    def __init__(self, bandits: Bandit) -> None:
        super().__init__(bandits)
        # add any member variables you may require
        self.successes = [1] * self.banditN
        self.failures = [1] * self.banditN

    # implement
    def action(self) -> int:
        sampled_values = [np.random.beta(self.successes[i], self.failures[i]) for i in range(self.banditN)]
        return sampled_values.index(max(sampled_values))

    # implement
    def update(self, choice: int, reward: int) -> None:
        if reward > 0:
            self.successes[choice] += 1
        else:
            self.failures[choice] += 1