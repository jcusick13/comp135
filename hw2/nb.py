# nb.py
import math


def learn_nb(index, m=0.0):
    """Takes in an index file with the syntax:
        1|yes|
    Where the first item defines the index number
    of an article and the second item its class.

    Returns:
        1. a dictionary of prior probability for each word in
        the vocab with a list of its corresponding word count
        for 'no' articles and 'yes' articles.
        i.e. dict['hockey'] = [0.07, 0.62]

        2. a list of class prior probability for each class
        i.e. class_prior = [0.47, 0.53] where class_prior[0]
        is the 'no' prior and class_prior[1] is the 'yes' prior

        3. alphabetized list of unique words from all documents

        4. a dictionary of document count for each word in the
        vocab with a list of 'no' and 'yes' docs that it appears in
        i.e. dict['hockey'] = [12, 347]

    The folder this file resides in must have the folder struct:
    --pp2data
    ----/ibmmac
    ----/sport

    index: str, filename of index file
    m: float, m-parameter for smoothing eqn. default=0.0
    """

    # Create dict of class count, word count, doc count
    clcount = {'no': 0.0, 'yes': 0.0}
    words = {}
    doc_count = {}

    # Create list of unique words for bag of words (unique vocab)
    bag = []

    # Ensure smoothing parameter is float
    if m != 0.0:
        m = float(m)

    with open(index, 'r') as idx:

        for line in idx:
            # Capture document title
            ln_splt = line.split('|')
            doc_num = ln_splt[0]
            cl = ln_splt[1].strip()

            # Tally class count
            if cl == 'yes':
                clcount['yes'] += 1.0
            else:
                clcount['no'] += 1.0

            # Read individual document
            with open(r'pp2data/ibmmac/%s.clean' % (doc_num), 'r') as doc:

                # 'no' articles
                if cl == 'no':
                    for l in doc:
                        for word in l.split(' '):
                            # Clean word
                            w = word.strip().lower()
                            # Skip blank spaces
                            if w == '':
                                continue
                            # Update word count
                            if w in words:
                                words[w][0] += 1.0
                            else:
                                words[w] = [1.0, 0.0]
                            # Update bag of words & doc count for new words
                            if w not in bag:
                                bag.append(w)
                                if w in doc_count:
                                    doc_count[w][0] += 1.0
                                else:
                                    doc_count[w] = [1.0, 0.0]

                # 'yes' articles
                else:
                    for l in doc:
                        for word in l.split(' '):
                            # Clean word
                            w = word.strip().lower()
                            # Skip blank spaces
                            if w == '':
                                continue
                            # Update word count
                            if w in words:
                                words[w][1] += 1.0
                            else:
                                words[w] = [0.0, 1.0]
                            # Update bag of words & doc count for new words
                            if w not in bag:
                                bag.append(w)
                                if w in doc_count:
                                    doc_count[w][1] += 1.0
                                else:
                                    doc_count[w] = [0.0, 1.0]

    # Add total word count to dict
    count0 = reduce(lambda a, b: a + b, [x[0] for x in words.itervalues()])
    count1 = reduce(lambda a, b: a + b, [x[1] for x in words.itervalues()])
    words['*tot_word_count*'] = [count0, count1]

    # Create dictionary of prior probability of each word (Type1)
    prior = {}

    # Create dictionary of prior probability of each word (Type2)
    prior_bag = {}

    for k in words.iterkeys():
        # Only add individual word probabilties to prior
        if k == '*tot_word_count*':
            continue

        # Priors with smoothing for Type1
        # -------------------------------

        # 'no' prior
        val0 = (words[k][0] + m) / (words['*tot_word_count*'][0] +
                                    (m * len(words) - 1))
        # 'yes' prior
        val1 = (words[k][1] + m) / (words['*tot_word_count*'][1] +
                                    (m * len(words) - 1))
        prior[k] = [val0, val1]

        # Priors with smoothing for Type2
        # -------------------------------

        # 'no' prior
        val0 = (doc_count[k][0] + m) / (clcount['no'] + (m * 2))
        # 'yes' prior
        val1 = (doc_count[k][1] + m) / (clcount['yes'] + (m * 2))

        prior_bag[k] = [val0, val1]

    # Get totals from clcount dict
    class_total = clcount['no'] + clcount['yes']

    # Create class prior list ['no', 'yes']
    class_prior = [clcount['no'] / class_total, clcount['yes'] / class_total]

    # Alphabetize bag to return
    bag.sort()

    return prior, class_prior, bag, prior_bag


def classify_nb_t1(document, prior, cp):
    """Classifies a document as 'no' or 'yes' using Naive
    Bayes with the prior probailites of all of the words in
    wordct.

    document: str, filename of document to classify
    prior: dict, key=word,
                value=['no' prior, 'yes' prior] (floats)
    cp: list, ['no' class prior, 'yes' class prior]
    """

    # Initiliaze 'no' and 'yes' scores with log of class score
    nb_no = math.log(cp[0])
    nb_yes = math.log(cp[1])

    # Read through document, update scores for 'no' and 'yes' for each word
    with open(document, 'r') as doc:
        for line in doc:
            for word in line.split(' '):
                # Clean and add each word to dict, avoid blank lines
                w = word.strip().lower()
                # Add log of class score for each outcome
                if w != '' and w in prior.keys():
                    # Skip words that haven't been seen in test set
                    if prior[w][0] > 0:
                        nb_no += math.log(prior[w][0])
                    if prior[w][1] > 0:
                        nb_yes += math.log(prior[w][1])

    # Return output classification
    if nb_no > nb_yes:
        return 0
    else:
        return 1


def classify_nb_t2(document, prior, cp, vocab):
    """Classifies a document as 'no' or 'yes' using Naive
    Bayes with the prior probabilities of all of the words in
    wordct. Represents document as a binary bag of words.

    document: str, filename of document to classify
    prior: dict, key=word,
                value=['no' prior, 'yes' prior] (floats)
    cp: list, ['no' class prior, 'yes' class prior]
    vocab: list, alphabeticaly ordered unique words in full vocabulary
        of training sets
    """

    # Create empty bag of words for test document
    bag = [0 for x in range(len(vocab))]

    # Initiliaze 'no' and 'yes' scores with log of class score
    nb_no = math.log(cp[0])
    nb_yes = math.log(cp[1])

    # Read through entire document, update bag as words are seen
    with open(document, 'r') as doc:
        for line in doc:
            for word in line.split(' '):
                # Clean each word, avoid blank lines
                w = word.strip().lower()
                # Add mark in bag for previously unseen words,
                # with respect to alphabetical ordering
                if w in vocab:
                    if bag[vocab.index(w)] == 0:
                        bag[vocab.index(w)] = 1

    # Update 'no'/'yes' scores for words seen in test document
    index = 0
    for mark in bag:
        if mark == 1:
            nb_no += math.log(prior[vocab[index]][0])
            nb_yes += math.log(prior[vocab[index]][1])
        index += 1

    # Return output classification
    if nb_no > nb_yes:
        return 0
    else:
        return 1


def naive_bayes(prior, cp, bag, test, variant):
    """Implements the Naive Bayes algorithm for text
    classification. Returns accuracy of classification
    algorithm.

    train: str, index filename of documents to train on
    test: str, index filename of documents to test
    variant: str, accepts 'Type1' or 'Type2' where
        'Type1' is a traditional implemention giving each word
            in the document its own representation
        'Type2' is a bag of words representation of incoming
            documents to test on
    """

    # Determine folder to test from (ibmmac or sport)
    fldr = test.split(r'/')[-2]

    # Count of classifications [incorrect, correct]
    result = [0.0, 0.0]

    print 'Beginning test set classification (%s)...' % (variant)
    with open(test, 'r') as idx:
        for line in idx:

            # Capture document title
            ln_splt = line.split('|')
            doc_num = ln_splt[0]
            cl = ln_splt[1].strip()

            # Classify document
            fname = r'pp2data/%s/%s.clean' % (fldr, doc_num)

            if variant == 'Type1':
                nb_cl = classify_nb_t1(fname, prior, cp)

            elif variant == 'Type2':
                nb_cl = classify_nb_t2(fname, prior, cp, bag)

            else:
                print 'Please choose a valid variant type.'
                return

            # Record classification results
            if nb_cl == 0:
                # 'no' nb classification
                if cl == 'no':
                    result[1] += 1.0
                else:
                    result[0] += 1.0
            else:
                # 'yes' nb classification
                if cl == 'no':
                    result[0] += 1.0
                else:
                    result[1] += 1.0

    # Calculate final classification results accuracy
    return result[1] / (result[0] + result[1])


if __name__ == '__main__':

    train = r'pp2data/ibmmac/index_train'
    test = r'pp2data/ibmmac/index_test'

    print 'Calculating training set priors...'
    prior1, cp, bag, prior2 = learn_nb(train, 0.0)
    print naive_bayes(prior1, cp, bag, test, 'Type1')
    print naive_bayes(prior2, cp, bag, test, 'Type2')
