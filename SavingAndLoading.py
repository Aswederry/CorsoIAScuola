import os
import numpy as np



def SaveWeightsAndInserts(matrix, inserts, weightsFile, insertsFile):
    weights = np.array(matrix)
    inserts = np.array(inserts)
    np.save(weightsFile, weights)
    np.save(insertsFile, inserts)


def LoadWeightsAndInserts(weightsFile, numsFile):
    if os.path.exists(weightsFile):
        weightsNP = np.load(weightsFile)
        weights = weightsNP.tolist()
    else:
        print("Non esistono i pesi...")
        return

    if os.path.exists(numsFile):
        numsNP = np.load(numsFile)
        nums = numsNP.tolist()
    else:
        print("Non esistono i numeri...")
        return




    return weights, nums


def DeleteWeights(file):
    if os.path.exists(file):
        os.remove(file)
        print("Pesi cancellati")
    else:
        print("Non c'è niente da cancellare")
