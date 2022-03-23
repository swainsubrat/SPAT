import numpy as np
import pickle

from typing import List, Dict

def accuracy(Y: List, predY: List) -> float:
    """
    Get accuracy
    """
    Y = np.array(Y)
    predY = np.array(predY)
    accuracy = (Y == predY).sum()/ float(len(Y))
    accuracy = np.round(accuracy * 100, 2)

    return accuracy

def save(path: str, params: Dict) -> None:
    """
    Save model to path
    """
    outfile = open(path, 'wb')
    pickle.dump(params, outfile)
    outfile.close()


def load(path: str) -> Dict:
    """
    Load model from path
    """
    infile = open(path, 'rb')
    params = pickle.load(infile)
    infile.close()

    return params

if __name__ == "__main__":
    a = [1, 2, 3]
    b = [1, 2, 3]
    print(accuracy(a, b))