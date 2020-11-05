import time
import json

from game import Game
from players.codemaster_glove_03 import AICodemaster as cm_glove03
from players.codemaster_w2v_03 import AICodemaster as cm_w2v03
from players.guesser_with_state import AIGuesserWithState as g_w2v_with_state
from players.guesser_w2v import AIGuesser as g_w2v
#from players.vector_codemaster import VectorCodemaster
#from players.vector_guesser import VectorGuesser

class Simulation:
    """Example of how to share vectors, pass kwargs, and call Game directly instead of by terminal"""

    start_time = time.time()
    glove_50d = Game.load_glove_vecs("players/glove.6B/glove.6B.50d.txt")
    print(f"{time.time() - start_time:.2f}s to load glove50d")

    start_time = time.time()
    w2v = Game.load_w2v("players/GoogleNews-vectors-negative300.bin")
    print(f"{time.time() - start_time:.2f}s to load w2v")

    print("\nclearing results folder...\n")
    Game.clear_results()

    num_games = 100
    for seed in range(num_games):        
        game_name = "glv03-w2v-with-state-" + str(seed)
        print("starting " + game_name)
        cm_kwargs = {"glove_vecs": glove_50d}
        g_kwargs = {"word_vectors": w2v}
        Game(cm_glove03, g_w2v_with_state, seed=seed, do_print=False,  game_name=game_name, cm_kwargs=cm_kwargs, g_kwargs=g_kwargs).run()
        
    # display the results (AVG NUM TUNRS, MIN NUM TURNS, WIN PERCENTAGE).
    num_turns = 0
    min_num_turns = 25
    num_wins = 0
    with open("results/bot_results_new_style.txt") as f:
        for line in f.readlines():
            game_json = json.loads(line.rstrip())
            turns = game_json["total_turns"]
            num_turns += turns
            min_num_turns = min(turns, min_num_turns)

            red_cards_guessed = game_json["R"]
            if red_cards_guessed == 8:
                num_wins += 1
            

    avg_turns = 1.0 * num_turns / num_games
    win_pct = 1.0 * num_wins / num_games
    print(f"avg_turns={avg_turns}, min_turns={min_num_turns}, win_pct={win_pct}")


if __name__ == "__main__":
    Simulation()
