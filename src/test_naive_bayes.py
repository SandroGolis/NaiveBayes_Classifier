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
# classifier = NaiveBayes(document_class=Name)
# classifier.train(train)
# accuracy(classifier, test)



# classifier = NaiveBayes(document_class = EvenOdd)
# classifier.train([EvenOdd(0, True), EvenOdd(1, False)])
# test = [EvenOdd(i, i % 2 == 0) for i in range(2, 1000)]
# accuracy(classifier, test)

def split_blogs_corpus(document_class):
    """Split the blog post corpus into training and test sets"""
    blogs = BlogsCorpus(document_class=document_class)
    seed(hash("blogs"))
    shuffle(blogs)
    return (blogs[:3000], blogs[3000:])

train, test = split_blogs_corpus(BagOfWords)
classifier = NaiveBayes(document_class = BagOfWords)
classifier.train(train)
accuracy(classifier, test)

















