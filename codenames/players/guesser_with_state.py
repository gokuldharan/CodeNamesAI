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
            del self.state[self.guess]
            del self.updated_state[self.guess]
            
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

    def beta_mean((alpha, beta)):
        return beta.mean(alpha, beta)

    def get_answer(self):
        embedding_distances = self.compute_distance(self.clue, self.words)
        sorted_words = [k for k, v in sorted(embedding_distances.items(), key=lambda item: item[1])]
        print("Words sorted by embedding distances: ", sorted_words)
        self.updated_state = self.state

        # First closest word.
        first = sorted_words[0]
        self.updated_state[first] = (self.state[first][0] + 5, self.state[first][1])

        # Second closest word. Guaranteed to exist.
        second = sorted_words[1]
        self.updated_state[second] = (self.state[second][0] + 3, self.state[second][1])

        # Third closest word. Usually exists. Let's leave this as is.
        third = sorted_words[2]
        self.updated_state[third] = (self.state[third][0] + 2, self.state[thrid][1])
        
        # Return the word with the highest Beta distribution mean.
        sorted_states = self.updated_state
        sorted_by_updated_beta = [k for k, v in sorted(self.updated_state.items(), key=lambda item: beta_mean(item[1]))]
        self.guess = sorted_by_updated_beta[0]
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
