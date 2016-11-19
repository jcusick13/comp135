# analysis.py
from ann import *


def learn(w, d, traindata, testdata):
    """Builds neural network based off of user parameters
    and training dataset dimensionality. Trains network
    by updaing weights according to 200 iterations
    of backpropagation algorithm using traindata.

    w: int, number of nodes (width) of each hidden layer
    d: int, number of (depth) of hidden layer
    traindata: arff, training dataset of only
                numerical features
    testdata: arff, test dataset of only
                numerical features
    """

    # Construct network and initialize weights
    nn = NeuralNet(w, d, traindata, testdata)

    # Update weights with training dataset
    for example in traindata:
        # Create list from named tuple (ignore example classification)
        inputs = [val for val in example[:-1]]
