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

    num_games = 3
    for seed in range(num_games):
        game_name = "glv03-w2v-" + str(seed)
        print("starting " + game_name)
        cm_kwargs = {"glove_vecs": glove_50d}
        g_kwargs = {"word_vectors": w2v}
        Game(cm_glove03, g_w2v, seed=seed, do_print=False,  game_name=game_name, cm_kwargs=cm_kwargs, g_kwargs=g_kwargs).run()
        
        game_name = "glv03-w2v-with-state-" + str(seed)
        print("starting " + game_name)
        cm_kwargs = {"glove_vecs": glove_50d}
        g_kwargs = {"word_vectors": w2v}
        Game(cm_glove03, g_w2v_with_state, seed=seed, do_print=False,  game_name=game_name, cm_kwargs=cm_kwargs, g_kwargs=g_kwargs).run()
        

    # display the results
    with open("results/bot_results_new_style.txt") as f:
        for line in f.readlines():
            game_json = json.loads(line.rstrip())
            game_name = game_json["game_name"]
            game_time = game_json["time_s"]
            game_score = game_json["total_turns"]

            print(f"time={game_time:.2f}, turns={game_score}, name={game_name}")


if __name__ == "__main__":
    Simulation()
