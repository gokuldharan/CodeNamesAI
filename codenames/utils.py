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
