# -*- mode: Python; coding: utf-8 -*-

from __future__ import division
from corpus import BlogsCorpus, NamesCorpus
from document import *
from naive_bayes import NaiveBayes
import sys
from random import shuffle, seed


def accuracy(classifier, test, verbose=sys.stderr):
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
    return sum(correct) / len(correct)


# names = NamesCorpus(document_class=Name)
# seed(hash("names"))
# shuffle(names)
# train, test = names[:6000], names[6000:]
# classifier = NaiveBayes()
# classifier.train(train)
# print accuracy(classifier, test)



classifier = NaiveBayes()
classifier.train([EvenOdd(0, True), EvenOdd(1, False)])
test = [EvenOdd(i, i % 2 == 0) for i in range(2, 1000)]
print accuracy(classifier, test)