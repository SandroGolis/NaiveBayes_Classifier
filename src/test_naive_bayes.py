# -*- mode: Python; coding: utf-8 -*-

from __future__ import division
from unittest import TestCase, main
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


class NaiveBayesTest(TestCase):
    u"""Tests for the naïve Bayes classifier."""

    def test_even_odd(self):
        """Classify numbers as even or odd"""
        classifier = NaiveBayes(document_class = EvenOdd)
        classifier.train([EvenOdd(0, True), EvenOdd(1, False)])
        test = [EvenOdd(i, i % 2 == 0) for i in range(2, 1000)]
        self.assertEqual(accuracy(classifier, test), 1.0)

    def split_names_corpus(self, document_class=Name):
        """Split the names corpus into training and test sets"""
        names = NamesCorpus(document_class=document_class)
        self.assertEqual(len(names), 5001 + 2943)  # see names/README
        seed(hash("names"))
        shuffle(names)
        return (names[:6000], names[6000:])

    def test_names_nltk(self):
        """Classify names using NLTK features"""
        train, test = self.split_names_corpus()
        classifier = NaiveBayes(document_class = Name)
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.70)

    def split_blogs_corpus(self, document_class):
        """Split the blog post corpus into training and test sets"""
        blogs = BlogsCorpus(document_class=document_class)
        self.assertEqual(len(blogs), 3232)
        seed(hash("blogs"))
        shuffle(blogs)
        return (blogs[:3000], blogs[3000:])

    def test_blogs_bag(self):
        """Classify blog authors using bag-of-words"""
        train, test = self.split_blogs_corpus(BagOfWords)
        classifier = NaiveBayes(document_class = BagOfWords)
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.55)

    def split_blogs_corpus_imba(self, document_class):
        blogs = BlogsCorpus(document_class=document_class)
        imba_blogs = blogs.split_imbalance()
        return (imba_blogs[:1600], imba_blogs[1600:])

    def test_blogs_imba(self):
        train, test = self.split_blogs_corpus_imba(BagOfWords)
        classifier = NaiveBayes(document_class = BagOfWords)
        classifier.train(train)
        # you don't need to pass this test
        self.assertGreater(accuracy(classifier, test), 0.1)


if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)

# names = NamesCorpus(document_class=Name)
# seed(hash("names"))
# shuffle(names)
# train, test = names[:6000], names[6000:]
# classifier = NaiveBayes(document_class=Name)
# classifier.train(train)
# accuracy(classifier, test)
#
#
#
# classifier = NaiveBayes(document_class = EvenOdd)
# classifier.train([EvenOdd(0, True), EvenOdd(1, False)])
# test = [EvenOdd(i, i % 2 == 0) for i in range(2, 1000)]
# accuracy(classifier, test)
#
# def split_blogs_corpus(document_class):
#     """Split the blog post corpus into training and test sets"""
#     blogs = BlogsCorpus(document_class=document_class)
#     seed(hash("blogs"))
#     shuffle(blogs)
#     return (blogs[:3000], blogs[3000:])
#
# train, test = split_blogs_corpus(BagOfWords)
# classifier = NaiveBayes(document_class = BagOfWords)
# classifier.train(train)
# accuracy(classifier, test)

















