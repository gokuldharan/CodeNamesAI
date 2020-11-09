import time
import json

from game import Game
from players.codemaster_w2vglovebert import AICodemaster as cm
from players.guesser_w2vglovebert import AIGuesser as g_unif
from players.guesser_with_state import AIGuesserWithState as g_with_state
from players.guesser_random import AIGuesser as g_rand



#Just change these for diff guesser types, embeddings
CM_EMB = "BERT"     #BERT, GLOVE, or W2V
G_EMB = "GLOVE"
G_TYPE = "BASELINE" #BASELINE, WITH_STATE, or RANDOM (ALSO Q?)




argToVec = {}
argToKey = {"BERT":"bert_vecs", "GLOVE":"glove_vecs", "W2V" : "word_vectors"}
typeToGuesser = {"BASELINE":g_unif, "WITH_STATE":g_with_state, "RANDOM":g_rand}

class Simulation:
    """Example of how to share vectors, pass kwargs, and call Game directly instead of by terminal"""

    start_time = time.time()
    argToVec["GLOVE"] = Game.load_glove_vecs("players/glove.6B.50d.txt")
    print(f"{time.time() - start_time:.2f}s to load glove50d")

    start_time = time.time()
    argToVec["BERT"] = Game.load_bert_vecs("players/bert_embeddings.txt")
    print(f"{time.time() - start_time:.2f}s to load BERT")

    start_time = time.time()
    argToVec["W2V"] = Game.load_w2v("players/GoogleNews-vectors-negative300.bin")
    print(f"{time.time() - start_time:.2f}s to load w2v")

    print("\nclearing results folder...\n")
    game_name = str(CM_EMB) +"-"+ str(G_EMB) + "-" + str(G_TYPE)
    Game.clear_results(name = game_name)

    cm_vec = argToVec[CM_EMB]
    cm_key = argToKey[CM_EMB]
    g_vec = argToVec[G_EMB]
    g_key = argToKey[G_EMB]

    g = typeToGuesser[G_TYPE]

    start_time = time.time()

    num_games = 100
    for seed in range(num_games):
        print("starting " + game_name + str(seed))
        cm_kwargs = {cm_key: cm_vec}
        g_kwargs = {g_key: g_vec}
        Game(cm, g, seed=seed, do_print=False,  game_name=game_name, cm_kwargs=cm_kwargs, g_kwargs=g_kwargs, uniquify_output=True).run()

    # display the results (AVG NUM TUNRS, MIN NUM TURNS, WIN PERCENTAGE).
    num_turns = 0
    min_num_turns = 25
    num_wins = 0
    num_turns_in_wins = 0
    with open("results/" + game_name + ".txt") as f:
        for line in f.readlines():
            game_json = json.loads(line.rstrip())
            turns = game_json["total_turns"]
            num_turns += turns
            min_num_turns = min(turns, min_num_turns)

            red_cards_guessed = game_json["R"]
            if red_cards_guessed == 8:
                num_wins += 1
                num_turns_in_wins += turns


    avg_turns = 1.0 * num_turns / num_games
    win_pct = 1.0 * num_wins / num_games
    avg_turns_wins = 1.0 * num_turns_in_wins / num_wins
    print(f"avg_turns={avg_turns}, min_turns={min_num_turns}, win_pct={win_pct}, avg_turns_in_wins={avg_turns_wins}")


if __name__ == "__main__":
    Simulation()
