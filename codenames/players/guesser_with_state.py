import collections
import operator
import scipy.spatial.distance
import scipy.stats
from scipy.stats import beta

from players.guesser import Guesser

class AIGuesserWithState(Guesser):

    def __init__(self, word_vectors=None):
        super().__init__()
        self.word_vectors = word_vectors
        self.num = 0
        self.is_first_turn = True

    def set_board(self, words):
        self.words = words

        if self.is_first_turn:
            self.state = {}
            for word in words:
                self.state[word] = (7, 18)  # alpha, beta
            self.is_first_turn = False
        else:
            # TODO(Aditi): Update the existing probability distributions given the answer to our guess.
            guess_true_value = self.words[self.guess_index]
            if guess_true_value == "*Red*":  # We were correct!
                # Do something
                pass
            elif guess_true_value == "*Blue*":
                # Do something
                pass
            elif guess_true_value == "*Civilian*":
                # Do something
                pass
            else:
                # Assassin.. game's over.
                pass
        print("State: ", self.state)
                
    def set_clue(self, clue, num):
        self.clue = clue
        self.num = num
        print("The clue is:", clue, num)
        li = [clue, num]
        return li

    def keep_guessing(self):
        # TODO(aditij): Modify this logic to use thresholds.
        return self.num > 0

    def get_answer(self):
        embedding_distances = self.compute_distance(self.clue, self.words)
        sorted_embedding_distances = {k: v for k, v in sorted(embedding_distances.items(), key=lambda item: item[1])}
        print("Embedding distances: ", sorted_embedding_distances)
        self.updated_state = self.state
        for word in self.words:
            if word[0] == "*":  # Already guessed.
                continue

            embedding_distance = embedding_distances[word]
            (a, b) = self.state[word]
            beta_prior = (beta.stats(a, b)[0])  # Returns the mean of the distribution.
            print(beta_prior)

            # Assume we have 10 alpha (red) psuedocounts to give out.
            # TODO(aditi): Update psuedocounts.
        
        self.num -= 1
        self.guess = min(embedding_distances.items(), key=operator.itemgetter(1))[0]
        self.guess_index = self.words.index(self.guess)
        print("Guess is: ", self.guess, self.guess_index)
        return self.guess

    def compute_distance(self, clue, board):
        # Maps each word to embedding distance from clue.
        w2v = {}
        for word in board:
            try:
                if word[0] == '*':
                    continue
                w2v[word] = scipy.spatial.distance.cosine(self.word_vectors[clue],
                                                          self.word_vectors[word.lower()])
            except KeyError:
                continue
            
        return w2v
