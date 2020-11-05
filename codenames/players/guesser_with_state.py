# python3 run_game.py human players.guesser_with_state.AIGuesserWithState --w2v players/GoogleNews-vectors-negative300.bin
#--seed 3442

import collections
import operator
import scipy.spatial.distance
import scipy.stats
from scipy.stats import beta

from players.guesser import Guesser

class AIGuesserWithState(Guesser):

    def __init__(self, word_vectors=None, bert_vecs=None, glove_vecs=None):
        super().__init__()

        if bert_vecs is not None:
            self.chosen_embedding = bert_vecs
        elif glove_vecs is not None:
            self.chosen_embedding = glove_vecs
        else:
            self.chosen_embedding = word_vectors

        self.num = 0
        self.is_first_turn = True

    def set_board(self, words):
        self.words = words

        if self.is_first_turn:
            self.state = {}
            for word in words:
                self.state[word] = (8, 17)  # alpha, beta
            self.is_first_turn = False

        else:
            del self.state[self.guess]

        print("State: ", self.state)

    # Gets called in the beginning of each clue.
    # Do all the initialization here.
    def set_clue(self, clue, num):
        self.clue = clue
        self.num = num
        self.start_turn = True

    def keep_guessing(self):
        # TODO(aditij): Modify this logic to use thresholds.
        return self.num > 0

    def get_answer(self):

        # In the beginning of every turn, we calculate the first, second, and third top guesses
        # related to the clue.
        if self.start_turn:
            embedding_distances = self.compute_distance(self.clue, self.words)
            sorted_words = [k for k, v in sorted(embedding_distances.items(), key=lambda item: item[1])]
            print("Words sorted by embedding distances: ", sorted_words)

            # First closest word.
            self.first = sorted_words[0]
            self.state[self.first] = (self.state[self.first][0] + 5, self.state[self.first][1])

            # Second closest word. Guaranteed to exist.
            self.second = sorted_words[1]
            self.state[self.second] = (self.state[self.second][0] + 3, self.state[self.second][1])

            # Third closest word. Usually exists.
            self.third = sorted_words[2]
            self.state[self.third] = (self.state[self.third][0] + 2, self.state[self.third][1])

            self.start_turn = False

        print("state is: ", self.state)
        print("\n")

        sorted_by_beta = [k for k, v in sorted(self.state.items(), key=lambda item: -beta.mean(item[1][0], item[1][1]))]
        print("sorted_by_beta: ", sorted_by_beta)
        print("\n")
        # Note this guess may or may not be related to the current clue.
        self.guess = sorted_by_beta[0]
        self.guess_index = self.words.index(self.guess)
        print("Guess is: ", self.guess)
        self.num -= 1
        return self.guess

    def finish_turn(self, game_state):
        print("In finish_turn: ", game_state)
        if game_state == "GameCondition.CONTINUE":  # Hit white or blue.
            # If we hit white or blue, we discount existing psuedocounts,
            # so that we give higher weight to words that correlated with the present clue rather than past clues.
            discount = 0.5
            print("In finish_turn CONTINUE")

            if self.first in self.state:
                self.state[self.first] = (self.state[self.first][0] - int(5 * discount), self.state[self.first][1])

            if self.second in self.state:
                self.state[self.second] = (self.state[self.second][0] - int(3 * discount), self.state[self.second][1])

            if self.third in self.state:
                self.state[self.third] = (self.state[self.third][0] - int(2 * discount), self.state[self.third][1])


        if game_state == "GameCondition.HIT_RED":  # Hit red, and our turn is over.
            print("In finish_turn HIT_RED")
            # Move psuedocounts from alpha to beta.

            if self.guess != self.first and self.guess != self.second and self.guess != self.third:
                # We made a guess unrelated to the current clue, so leave current clue pseudocounts as-is.
                pass

            else:  # The guess we made was related to the current clue.
                if self.guess != self.first and self.first in self.state:
                    self.state[self.first] = (self.state[self.first][0] - 5, self.state[self.first][1] + 5)
                    print("Updated state for: ", self.first)

                if self.guess != self.second and self.second in self.state:
                    self.state[self.second] = (self.state[self.second][0] - 3, self.state[self.second][1] + 3)
                    print("Updated state for: ", self.second)

                if self.guess != self.third and self.third in self.state:
                    self.state[self.third] = (self.state[self.third][0] - 2, self.state[self.third][1] + 2)
                    print("Updated state for: ", self.third)

        print("After finish_turn: ", self.state)

    def compute_distance(self, clue, board):
        # Maps each word to embedding distance from clue.
        w2v = {}
        for word in board:
            try:
                if word[0] == '*':
                    continue
                w2v[word] = scipy.spatial.distance.cosine(self.chosen_embedding[clue],
                                                          self.chosen_embedding[word.lower()])
            except KeyError:
                continue

        return w2v
