# analysis.py
from ann import *
import sys


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

    # Train network on training dataset
    for x in range(200):
        for example_tn in train.data:

            # Create list from named tuple (ignore example classification)
            inputs_tn = [val for val in example_tn[:-1]]

            # Convert output into one-hot endcoding
            classification_tn = int(example_tn[-1])
            onehot_tn = ''  # string encoding
            outputs_tn = []  # list integer encoding

            for i in range(10):  # set to 3 for iris dataset
                if i == classification_tn:
                    onehot_tn += '1'
                    outputs_tn.append(1)
                else:
                    onehot_tn += '0'
                    outputs_tn.append(0)

            # Update network weights
            nn.update_weights(inputs_tn, outputs_tn)

    # Test network on testing dataset
    correct = 0.0
    total = 0.0
    for example_tt in test.data:

        # Create list from named tuple (ignore example classification)
        inputs_tt = [val for val in example_tt[:-1]]

        # Convert output into one-hot encoding
        classification_tt = int(example_tt[-1])
        onehot_tt = ''  # string encoding
        outputs_tt = []  # list integer encoding

        for i in range(10):  # set to 3 for iris dataset
            if i == classification_tt:
                onehot_tt += '1'
                outputs_tt.append(1)
            else:
                onehot_tt += '0'
                outputs_tt.append(0)

        # Make prediction
        predict = nn.assign_output(inputs_tt)

        # Determine correctness
        if predict == onehot_tt:
            correct += 1.0
        total += 1.0

    # Return accuracy of test set classification
    return correct / total


if __name__ == '__main__':

    # Expose function call to command line
    w = int(sys.argv[1])
    d = int(sys.argv[2])
    train = sys.argv[3]
    test = sys.argv[4]

    print 'Test set accuracy (w=%i, d=%i): ' % (w, d) + str(learn(w, d, train, test))


    # Below code used to generate results used in report
    # --------------------------------------------------
    # train = r'optdigits_train.arff'
    # test = r'optdigits_test.arff'

    # # Complete analysis runs
    # print 'TEST ACCURACY (d=0): ' + str(learn(0, 0, train, test))

    # depth = [1, 2, 3, 4]
    # width = [1, 2, 5, 10]

    # for d in depth:
    #     for w in width:
    #         print 'TEST ACCURACY (d=%i, w=%i): ' % (d, w) + str(learn(w, d, train, test))


