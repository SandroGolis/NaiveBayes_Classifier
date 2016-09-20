from __future__ import division
from classifier import Classifier
from feature_extractors import *


class NaiveBayes(Classifier):
    """A naive Bayes classifier."""

    def __init__(self, model=None):
        super(NaiveBayes, self).__init__(model)
        self.my_model = {'all_log_prob': None, 'col_to_label': None, 'vocabulary': None}

    def get_prior_log_probabilities(self, documents):
        col_count = 0
        label_to_col = {}
        seen_labels = {}
        for doc in documents:
            if doc.label in seen_labels:
                seen_labels[doc.label] += 1
            else:
                seen_labels[doc.label] = 1
                label_to_col[doc.label] = col_count
                col_count += 1
        col_to_label = {v: k for k, v in label_to_col.items()}
        self.my_model["col_to_label"] = col_to_label
        prior_log = []
        for col in col_to_label:
            label = col_to_label[col]
            prob = seen_labels[label] / len(documents)
            prior_log.append(np.log(prob))
        return np.array(prior_log), label_to_col

    def extract_f_vector(self, doc):
        if type(doc) is EvenOdd:
            return extract_from_even(doc.features())
        elif type(doc) is Name:
            return extract_from_name(doc.features())
        elif type(doc) is BagOfWords:
            return extract_from_bag(doc.features(), self.my_model["vocabulary"])
        # todo default extraction

    def classify(self, document):
        """ Classifies a given document to one of the
            classes, according to NB model and trained data"""
        f_vector = self.extract_f_vector(document)
        f_vector = np.append(f_vector, np.array([1]))       # adding last "feature" for prior log probability
        all_log_prob = self.my_model["all_log_prob"]
        sum_of_probabilities = f_vector.dot(all_log_prob)
        index = np.argmax(sum_of_probabilities)
        return self.my_model["col_to_label"][index]

    def train(self, documents):
        """ Instantiate a model according to Naive Bayes Classifier
            - Calculations are made in log space to avoid underflow
            - Laplace smoothing is made on the features vectors """
        prior_log_prob, label_to_col = self.get_prior_log_probabilities(documents)
        self.my_model["vocabulary"] = make_vocabulary(documents)

        # find frequencies of features
        num_classes = len(label_to_col)
        num_features = len(self.extract_f_vector(documents[0]))
        features_freq = np.zeros((num_features, num_classes))
        for doc in documents:
            f_vector = self.extract_f_vector(doc)
            col_for_f_vector = label_to_col[doc.label]
            features_freq[:, col_for_f_vector] += f_vector

        # laplace smoothing
        total_per_label = np.sum(features_freq, axis=0)
        features_freq += np.ones(total_per_label.shape, int)
        normalizer = total_per_label + np.full(total_per_label.shape, num_features, int)
        features_freq /= normalizer

        # stack all probabilities to one matrix and take log
        # result: self.all_log_prob
        # |-----------------------------------|
        # | log P(f1|C1) | ... | log P(f1|Cn) |
        # | log P(f2|C1) | ... | log P(f2|Cn) |
        # |        .     |  .  |         .    |
        # |        .     |  .  |         .    |
        # |        .     |  .  |         .    |
        # | log P(fm|C1) | ... | log P(fm|Cn) |
        # | log P(C1)    | ... | log P(Cn)    |
        # |-----------------------------------|
        likelihood_log_prob = np.log(features_freq)
        all_log_prob = np.vstack((likelihood_log_prob, prior_log_prob))
        self.my_model["all_log_prob"] = all_log_prob

    def get_model(self):
        return self.my_model

    def set_model(self, model):
        self.my_model = model

    model = property(get_model, set_model)










