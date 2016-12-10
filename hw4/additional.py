# additional.py
from arff_reader import Arff
from perceptron import *

if __name__ == '__main__':

    # Create Arff objects for train/test sets
    traindata = Arff('additionalTraining.arff')
    testdata = Arff('additionalTesting.arff')

    results = []

    # Primal Perceptron
    pp = PrimalPerceptron(len(traindata.field_names))
    pp.train(traindata, 50)
    results.append(pp.test(testdata))

    # Polynomial Kernel
    res_poly = []
    for i in range(1, 6):
        poly_kp = KernelPerceptron(len(traindata.data), poly=True, d=i)
        poly_kp.train(traindata, 50)
        results.append(poly_kp.test(traindata, testdata))

    # RBF kernel
    res_rbf = []
    for i in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        rbf_kp = KernelPerceptron(len(traindata.data), RBF=True, s=i)
        rbf_kp.train(traindata, 50)
        results.append(rbf_kp.test(traindata, testdata))

    # Print results
    for i in results:
        print i,
