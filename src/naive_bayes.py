from __future__ import division
from string import ascii_lowercase

import itertools

from classifier import Classifier
import numpy as np
from document import *


class NaiveBayes(Classifier):
    """A naive Bayes classifier."""

    def __init__(self, model=None, document_class=Document):
        super(NaiveBayes, self).__init__(model)
        self.col_to_label = dict()
        self.document_class = document_class
        # todo add f_vector_len?
        self.vocab = {}
        # todo need this exception? maybe some default f_vector hendling?
        if (document_class is not EvenOdd and
            document_class is not Name and
            document_class is not BagOfWords):
            print("NB classifier: unknown document type")
            raise Exception('NB classifier: unknown document type')

    def get_model(self):
        return self.seen

    def set_model(self, model):
        self.seen = model

    model = property(get_model, set_model)

    def extract_f_vector(self, features_list):
        if self.document_class is EvenOdd:
            return self.extract_from_even(features_list)
        elif self.document_class is Name:
            return self.extract_from_name(features_list)
        elif self.document_class is BagOfWords:
            return self.extract_from_bag(features_list)

    def extract_from_even(self, features_list):
        """ f1: is the number even
            f2: is the number odd """
        if not features_list:
            return np.array([0, 0])
        if features_list[0]:
            return np.array([1, 0])
        else:
            return np.array([0, 1])

    def extract_from_name(self, features_list):
        """ V1: vector of letters indicating the FIRST letter
            V2: vector of letters indicating the LAST letter
            todo V3: how many of each letter
            todo V4: which letters appear """
        f_vector_length = 2 * len(ascii_lowercase)
        if not features_list:
            return np.full(f_vector_length, 0, int)
        first_letter = features_list[0]
        last_letter = features_list[1]
        # create feature list of V1||V2
        f_list = [1 if c == first_letter else 0 for c in ascii_lowercase]
        f_list += [1 if c == last_letter else 0 for c in ascii_lowercase]
        return np.array(f_list)

    def extract_from_bag(self, word_list):
        """ V1: vocabulary binary vector """
        f_vector_length = len(self.vocab)
        if not word_list:
            return np.full(f_vector_length, 0, int)
        f_list = [0]*f_vector_length
        # todo no counting, only existence (set)
        # todo make normalize function to use here and in create_vocabulary
        world_list_set = set(word.lower() for word in word_list if word.isalpha())
        for word in world_list_set:
            if word in self.vocab:
                idx = self.vocab[word]
                f_list[idx] = 1
        return np.array(f_list)

    def train(self, documents):
        """documentation"""
        # find log of prior probability for each class
        col_count = 0
        label_to_col = {}
        seen_labels = {}
        for doc in documents:
            if doc.label in seen_labels:
                seen_labels[doc.label] += 1
            else:
                seen_labels[doc.label] = 1
                label_to_col[doc.label] = col_count
                self.col_to_label[col_count] = doc.label
                col_count += 1

        # todo: it is possible to combine these for loops over the documents
        self.make_vocabulary(documents)


        prior_log = []
        for col in self.col_to_label:
            label = self.col_to_label[col]
            prob = seen_labels[label] / len(documents)
            prior_log.append(np.log(prob))

        label_log_prob = np.array(prior_log)

        # find logLikelihood of features
        num_classes = len(label_to_col)
        # todo nicer way to know how many features there will be.
        # todo For ex. store this after defining vocabulary and increment every time another feature set is added
        num_features = len(self.extract_f_vector(documents[0].features()))
        features_freq = np.zeros((num_features, num_classes))
        for doc in documents:
            f_vector = self.extract_f_vector(doc.features())
            col_for_f_vector = label_to_col[doc.label]
            features_freq[:, col_for_f_vector] += f_vector

        # smoothing
        total_per_label = np.sum(features_freq, axis=0)
        features_freq += np.ones(total_per_label.shape, int)
        normalizer = total_per_label + np.full(total_per_label.shape, num_features, int)
        features_freq /= normalizer

        # stack all probabilities to one array
        # result: self.all_log_prob
        # log of each cell in the following matrix
        # |---------------------------|
        # | P(f1|C1) | ... | P(f1|Cn) |
        # | P(f2|C1) | ... | P(f2|Cn) |
        # |    .     |  .  |     .    |
        # |    .     |  .  |     .    |
        # |    .     |  .  |     .    |
        # | P(fm|C1) | ... | P(fm|Cn) |
        # | P(C1)    | ... | P(Cn)    |
        # |---------------------------|
        likelihood_log_prob = np.log(features_freq)
        # todo deal with saving/loading model. What should it be?
        self.all_log_prob = np.vstack((likelihood_log_prob, label_log_prob))

    def classify(self, document):
        """documentation"""
        # todo deal with words that don't appear in a vocabulary
        f_vector = self.extract_f_vector(document.features())
        f_vector = np.append(f_vector, 1)

        # assuming features that dont appear have 0 value
        sum_of_probabilities = f_vector.dot(self.all_log_prob)

        index = np.argmax(sum_of_probabilities)
        return self.col_to_label[index]

    def make_vocabulary(self, documents):
        if self.document_class is not BagOfWords:
            return
        # todo maybe tokenize better with nltk instead of regular split
        all_words = []
        for doc in documents:
            for word in doc.features():
                if (word.isalpha()):
                    all_words.append(word.lower())
        # todo normalization - sometimes have punctuations inside words. need to avoid
        normalized_set = sorted(set(all_words))
        values = range(0, len(normalized_set))
        self.vocab = dict(itertools.izip(normalized_set, values))
