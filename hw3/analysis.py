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

    # Create Arff classes from input files
    train = Arff(traindata)
    test = Arff(testdata)

    for example in train.data[:5]:

        # Create list from named tuple (ignore example classification)
        inputs = [val for val in example[:-1]]

        # Convert output into one-hot endcoding
        classification = int(example[-1])
        onehot = ''  # string encoding
        outputs = []  # list integer encoding

        for i in range(10):
            if i == classification:
                onehot += '1'
                outputs.append(1)
            else:
                onehot += '0'
                outputs.append(0)

        # Update network weights
        nn.update_weights(inputs, outputs)


if __name__ == '__main__':
    train = r'optdigits_train.arff'
    test = r'optdigits_test.arff'

    learn(2, 2, train, test)
