import os
import random


def getSubsetName(file, n):
    return file.split(".txt")[0] + str(n) + ".txt"


def saveRandomSubset(file, n):
    print("Generating new subset of size " + str(n) + " of file " + file)
    f = open(file, "r")
    words = f.read().splitlines()
    random.shuffle(words)

    filename = getSubsetName(file, n)
    assert(not os.path.isfile(filename))

    f = open(filename, "x")
    for word in words[:n]:
        f.write(str(word) + '\n')

def genBERTembeddings(input_file, output_file):
    #must start server first with bert-serving-start -model_dir xxx -num_worker=xxx
    from bert_serving.client import BertClient
    bc = BertClient()

    f =  open(input_file, "r")
    word_pool = f.read().splitlines()

    bert_vecs = {}
    for word in word_pool:
        bert_vecs[word] = bc.encode([word.lower()])[0].tolist()

    f = open(output_file, "x")
    for word,embedding in bert_vecs.items():
        f.write(str(word.lower()))
        for w in embedding:
            f.write(" "+str(w))
        f.write("\n")
