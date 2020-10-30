from abc import ABC, abstractmethod
import os
import utils

class Codemaster(ABC):
    """codemaster abstract class that mimics the spymaster in the codenames game"""

    def __init__(self, wordlist_len):
        """Set up word list and handle pretrained vectors"""
        self.cm_wordlist = []

        wordlist_file = 'players/cm_wordlist.txt'
        if wordlist_len is not None:
            wordlist_file = utils.getSubsetName(wordlist_file, wordlist_len)
            if not os.path.isfile(wordlist_file):
                utils.saveRandomSubset('players/cm_wordlist.txt', wordlist_len)

        with open(wordlist_file) as infile:
            for line in infile:
                self.cm_wordlist.append(line.rstrip())

    @abstractmethod
    def set_game_state(self, words_on_board, key_grid):
        """A set function for wordOnBoard and keyGrid """
        pass

    @abstractmethod
    def get_clue(self):
        """Function that returns a clue word and number of estimated related words on the board"""
        pass


class HumanCodemaster(Codemaster):

    def __init__(self):
        super().__init__()
        pass

    def set_game_state(self, words_in_play, map_in_play):
        self.words = words_in_play
        self.maps = map_in_play

    def get_clue(self):
        clue_input = input("Input CM Clue:\nPlease enter a Word followed by a space and a Number >> ")
        clue_input = clue_input.strip()
        type(clue_input)
        clue = clue_input.split(" ")

        if len(clue) == 1:
            clue.append('1')
        return clue
