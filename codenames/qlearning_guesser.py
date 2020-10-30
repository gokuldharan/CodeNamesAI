from policy import greedyEpsilon

class AIGuesser(Guesser):
	def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None, word_bank = None):
        super().__init__()
        self.brown_ic = brown_ic
        self.glove_vecs = glove_vecs
        self.word_vectors = word_vectors
        self.num = 0
        self.explorer = epsilonGreedyExploration()
        self.word_bank = word_bank
        self.state = 0

    def set_board(self, words):
        self.words = words

    def set_clue(self, clue, num):
        self.clue = clue
        self.num = num
        print("The clue is:", clue, num)
        li = [clue, num]
        self.state = clue   #Not sure if we're passing in the clue index or the clue string here
        					# May need to modify game.py to pass in the clue index to set_clue

        return li

    def keep_guessing(self):
        return self.num > 0

    def get_answer(Q, action_mask):
    	action_index = guesser.evaluate(Q, self.state, action_mask)
        action_string = self.word_bank[action_index]
        return action_string
        