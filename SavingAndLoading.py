import os

import numpy as np


def SaveWeightsAndInserts(matrix, inserts, weightsFile, insertsFile):
    weights = np.array(matrix)
    inserts = np.array(inserts)
    np.save(weightsFile, weights)
    np.save(insertsFile, inserts)


def LoadWeightsAndInserts(weightsFile, numsFile):
    weightsNP = np.load(weightsFile)
    numsNP = np.load(numsFile)

    weights = weightsNP.tolist()
    nums = numsNP.tolist()
    return weights, nums


def DeleteWeights(file):
    os.remove(file)
