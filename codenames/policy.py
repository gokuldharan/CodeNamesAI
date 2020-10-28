import numpy as np


class greedyPolicy:
    def evaluate(self, Q, state):
        return np.argmax(Q[state])

class epsilonGreedyExploration:
    def __init__(self, epsilon, alpha, seed=None):
        self.epsilon = epsilon
        self.alpha = alpha
        if seed is not None:
            random.seed(seed)

    def evaluate(self, Q, state):
        num_actions = Q.shape[1]
        if np.random.rand() < self.epsilon:
            self.epsilon *= self.alpha
            # I assume actions are indexed 0...n-1
            return np.random.randint(0,num_actions)
        return np.argmax(Q[state])

class softmaxExploration:
    def __init__(self, tau, alpha, seed=None):
        #As tau -> 0, this approaches greedy selection
        #As tau -> inf, this approaches uniform random selection
        self.tau = tau
        self.alpha = alpha
        if seed is not None:
            random.seed(seed)

    def evaluate(self, Q, state):
        weights = np.exp(Q[state] / self.tau)
        weights /= np.sum(weights)
        self.tau *= self.alpha

        num_actions = Q.shape[1]
        return np.random.choice(range(num_actions),p=weights)

