# analysis.py
from arff_reader import *
from perceptron import *

if __name__ == '__main__':

    ################################
    ##                            ##
    ##  Primal Perceptron Results ##
    ##                            ##
    ################################

    print 'Primal Perceptron'
    print '-----------------'

    # A
    a_traindata = Arff(r'ATrain.arff')
    a_testdata = Arff(r'ATest.arff')

    a_pp = PrimalPerceptron(len(a_traindata.field_names))
    a_pp.train(a_traindata, 50)
    print 'A: %f' % (a_pp.test(a_testdata))

    # B
    b_traindata = Arff(r'BTrain.arff')
    b_testdata = Arff(r'BTest.arff')

    b_pp = PrimalPerceptron(len(b_traindata.field_names))
    b_pp.train(b_traindata, 50)
    print 'B: %f' % (b_pp.test(b_testdata))

    # C
    c_traindata = Arff(r'CTrain.arff')
    c_testdata = Arff(r'CTest.arff')

    c_pp = PrimalPerceptron(len(c_traindata.field_names))
    c_pp.train(c_traindata, 50)
    print 'C: %f' % (c_pp.test(c_testdata))

    # back
    back_traindata = Arff(r'backTrain.arff')
    back_testdata = Arff(r'backTest.arff')

    back_pp = PrimalPerceptron(len(back_traindata.field_names))
    back_pp.train(back_traindata, 50)
    print 'Back: %f' % (back_pp.test(back_testdata))

    # breast
    breast_traindata = Arff(r'breastTrain.arff')
    breast_testdata = Arff(r'breastTest.arff')

    breast_pp = PrimalPerceptron(len(breast_traindata.field_names))
    breast_pp.train(breast_traindata, 50)
    print 'Breast: %f' % (breast_pp.test(breast_testdata))

    # sonar
    sonar_traindata = Arff(r'sonarTrain.arff')
    sonar_testdata = Arff(r'sonarTest.arff')

    sonar_pp = PrimalPerceptron(len(sonar_traindata.field_names))
    sonar_pp.train(sonar_traindata, 50)
    print 'Sonar: %f' % (sonar_pp.test(sonar_testdata))

    ###############################
    ##                           ##
    ## Polynomial Kernel Results ##
    ##                           ##
    ###############################

    print '\nPolynomial Kernel'
    print '-----------------'

    # A
    a_traindata = Arff(r'ATrain.arff')
    a_testdata = Arff(r'ATest.arff')

    for i in range(1, 6):
        a_kp = KernelPerceptron(len(a_traindata.data), poly=True, d=i)
        a_kp.train(a_traindata, 50)
        print 'A, d=%i: %f' % (i, a_kp.test(a_traindata, a_testdata))
    print '\n'

    # B
    b_traindata = Arff(r'BTrain.arff')
    b_testdata = Arff(r'BTest.arff')

    for i in range(1, 6):
        b_kp = KernelPerceptron(len(b_traindata.data), poly=True, d=i)
        b_kp.train(b_traindata, 50)
        print 'B, d=%i: %f' % (i, b_kp.test(b_traindata, b_testdata))
    print '\n'

    # C
    c_traindata = Arff(r'CTrain.arff')
    c_testdata = Arff(r'CTest.arff')

    for i in range(1, 6):
        c_kp = KernelPerceptron(len(c_traindata.data), poly=True, d=i)
        c_kp.train(c_traindata, 50)
        print 'C, d=%i: %f' % (i, c_kp.test(c_traindata, c_testdata))
    print '\n'

    # back
    back_traindata = Arff(r'backTrain.arff')
    back_testdata = Arff(r'backTest.arff')

    for i in range(1, 6):
        back_kp = KernelPerceptron(len(back_traindata.data), poly=True, d=i)
        back_kp.train(back_traindata, 50)
        print 'Back, d=%i: %f' % (i, back_kp.test(back_traindata, back_testdata))
    print '\n'

    # breast
    breast_traindata = Arff(r'breastTrain.arff')
    breast_testdata = Arff(r'breastTest.arff')

    for i in range(1, 6):
        breast_kp = KernelPerceptron(len(breast_traindata.data), poly=True, d=i)
        breast_kp.train(breast_traindata, 50)
        print 'Breast, d=%i: %f' % (i, breast_kp.test(breast_traindata, breast_testdata))
    print '\n'

    # sonar
    sonar_traindata = Arff(r'sonarTrain.arff')
    sonar_testdata = Arff(r'sonarTest.arff')

    for i in range(1, 6):
        sonar_kp = KernelPerceptron(len(sonar_traindata.data), poly=True, d=i)
        sonar_kp.train(sonar_traindata, 50)
        print 'Sonar, d=%i: %f' % (i, sonar_kp.test(sonar_traindata, sonar_testdata))

    ########################  
    ##                    ##
    ## RBF Kernel Results ##
    ##                    ##
    ########################

    print '\nRBF Kernel'
    print '----------'

    # A
    a_traindata = Arff(r'ATrain.arff')
    a_testdata = Arff(r'ATest.arff')

    for i in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        a_kp = KernelPerceptron(len(a_traindata.data), RBF=True, s=i)
        a_kp.train(a_traindata, 50)
        print 'A, s=%f: %f' % (i, a_kp.test(a_traindata, a_testdata))
    print '\n'

    # B
    b_traindata = Arff(r'BTrain.arff')
    b_testdata = Arff(r'BTest.arff')

    for i in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        b_kp = KernelPerceptron(len(b_traindata.data), RBF=True, s=i)
        b_kp.train(b_traindata, 50)
        print 'B, s=%f: %f' % (i, b_kp.test(b_traindata, b_testdata))
    print '\n'

    # C
    c_traindata = Arff(r'CTrain.arff')
    c_testdata = Arff(r'CTest.arff')

    for i in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        c_kp = KernelPerceptron(len(c_traindata.data), RBF=True, s=i)
        c_kp.train(c_traindata, 50)
        print 'C, s=%f: %f' % (i, c_kp.test(c_traindata, c_testdata))
    print '\n'

    # back
    back_traindata = Arff(r'backTrain.arff')
    back_testdata = Arff(r'backTest.arff')

    for i in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        back_kp = KernelPerceptron(len(back_traindata.data), RBF=True, s=i)
        back_kp.train(back_traindata, 50)
        print 'Back, s=%f: %f' % (i, back_kp.test(back_traindata, back_testdata))
    print '\n'

    # breast
    breast_traindata = Arff(r'breastTrain.arff')
    breast_testdata = Arff(r'breastTest.arff')

    for i in [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]:
        breast_kp = KernelPerceptron(len(breast_traindata.data), RBF=True, s=i)
        breast_kp.train(breast_traindata, 50)
        print 'Breast, s=%f: %f' % (i, breast_kp.test(breast_traindata, breast_testdata))
    print '\n'

    # sonar
    sonar_traindata = Arff(r'sonarTrain.arff')
    sonar_testdata = Arff(r'sonarTest.arff')

    for i in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        sonar_kp = KernelPerceptron(len(sonar_traindata.data), RBF=True, s=i)
        sonar_kp.train(sonar_traindata, 50)
        print 'Sonar, s=%f: %f' % (i, sonar_kp.test(sonar_traindata, sonar_testdata))
