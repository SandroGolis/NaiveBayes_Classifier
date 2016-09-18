from __future__ import division
from classifier import Classifier
import numpy as np


class NaiveBayes(Classifier):
    """A naive Bayes classifier."""

    def __init__(self, model={}):
        super(NaiveBayes, self).__init__(model)
        self.col_to_label = dict()

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)

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

        prior_log =[]
        for col in self.col_to_label:
            label = self.col_to_label[col]
            prob = seen_labels[label] / len(documents)
            prior_log.append(np.log(prob))

        label_log_prob = np.array(prior_log)

        # find logLikelihood of features
        num_classes = len(label_to_col)
        num_features = len(documents[0].features())
        features_freq = np.zeros((num_features, num_classes))
        for doc in documents:
            f_vector = np.array(doc.features())
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
        # | P(f1|C1) | ... | P(f1|Cn) |
        # | P(f2|C1) | ... | P(f2|Cn) |
        # |    .     |  .  |     .    |
        # |    .     |  .  |     .    |
        # |    .     |  .  |     .    |
        # | P(fm|C1) | ... | P(fm|Cn) |
        # | P(C1)    | ... | P(Cn)    |
        likelihood_log_prob = np.log(features_freq)
        self.all_log_prob = np.vstack((likelihood_log_prob, label_log_prob))

    def classify(self, document):
        """documentation"""
        # todo deal with words that don't appear in a vocabulary
        list_of_features = document.features()
        list_of_features.append(1)
        new_f_vector = np.array(list_of_features)


        # assuming features that dont appear have 0 value
        sum_of_probabilities = new_f_vector.dot(self.all_log_prob)

        index = np.argmax(sum_of_probabilities)
        return self.col_to_label[index]

