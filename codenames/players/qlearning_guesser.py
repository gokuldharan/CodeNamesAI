from policy import epsilonGreedyExploration
from policy import greedyPolicy
from players.guesser import Guesser
import numpy as np

class AIGuesser(Guesser):
    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None, Q_file=None):
        super().__init__()
        self.brown_ic = brown_ic
        self.glove_vecs = glove_vecs
        self.word_vectors = word_vectors
        self.num = 0

        self.explorer = epsilonGreedyExploration(0.5, 0.95)
        self.word_pool = None
        self.state = 0
        self.words_in_play = None
        self.action_mask = None
        if not Q_file == None:
            self.Q = np.load(Q_file)
            self.train = False
            self.policy =  greedyPolicy
        else:
            self.Q = None
            self.train = True
            self.policy = None
            

    def get_board_state(self, Q, action_mask, state):
        self.action_mask = action_mask
        if self.train:
            self.Q = Q
        self.state = state

    def set_board(self, words):
        self.words = words

    def set_clue(self, clue, num):
        self.clue = clue
        self.num = num
        print("The clue is:", clue, num)
        li = [clue, num]  
        self.state = clue
        return li

    def keep_guessing(self):
        return self.num > 0

    def get_answer(self):

        if not self.train:
            action_index = self.policy.evaluate(self.Q, self.state, self.action_mask)
            action_string = self.word_pool[action_index]
            return action_string
        action_index = self.explorer.evaluate(self.Q, self.state, self.action_mask)
        action_string = self.word_pool[action_index]
        return action_string

    def get_word_bank(self, word_pool):
        self.word_pool = word_pool

        