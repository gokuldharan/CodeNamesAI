import numpy as np


#Action_Mask needs to look like [0 1 0 0 0 1.... 1] (dim(action_space)x 1),
# where 1 indicates that the corresponding word is a valid guess
class greedyPolicy:
    def evaluate(self, Q, state, action_mask):
        masked_action_values = np.ma.masked_array(Q[state],1-action_mask)
        return np.argmax(masked_action_values)

class epsilonGreedyExploration:
    def __init__(self, epsilon, alpha, seed=None):
        self.epsilon = epsilon
        self.alpha = alpha
        if seed is not None:
            np.random.seed(seed)

    def evaluate(self, Q, state, action_mask):
        num_actions = Q.shape[1]
        if np.random.rand() < self.epsilon:
            self.epsilon *= self.alpha
            # I assume actions are indexed 0...n-1
            masked_actions = np.where(np.array(action_mask))[0]
            return np.random.choice(masked_actions)
        masked_action_values = np.ma.masked_array(Q[state],1-action_mask)
        return np.argmax(masked_action_values)

class softmaxExploration:
    def __init__(self, tau, alpha, seed=None):
        #As tau -> 0, this approaches greedy selection
        #As tau -> inf, this approaches uniform random selection
        #Start with tau roughly around 10?
        self.tau = tau
        self.alpha = alpha
        if seed is not None:
            np.random.seed(seed)

    def evaluate(self, Q, state, action_mask):
        masked_actions = np.where(np.array(action_mask))[0]
        weights = np.exp(Q[state] / self.tau)[masked_actions]
        weights /= np.sum(weights)
        self.tau *= self.alpha

        num_actions = Q.shape[1]

        return np.random.choice(masked_actions,p=weights)

