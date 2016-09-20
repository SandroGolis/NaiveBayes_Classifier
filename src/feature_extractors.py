import itertools
from document import *
import numpy as np
from string import ascii_lowercase



def make_vocabulary(documents):
    doc_type = type(documents[0])
    if doc_type is not BagOfWords:
        return
    # todo NLTK tokenization?
    all_words = []
    for doc in documents:
        for word in doc.features():
            if (word.isalpha()):
                all_words.append(normalize(word))
    normalized_set = sorted(set(all_words))
    values = range(0, len(normalized_set))
    return dict(itertools.izip(normalized_set, values))

def normalize(word):
    return word.lower()


def extract_from_even(features_list):
    """ f1: is the number even
        f2: is the number odd """
    if not features_list:
        return np.array([0, 0])
    if features_list[0]:
        return np.array([1, 0])
    else:
        return np.array([0, 1])


def extract_from_name(features_list):
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


def extract_from_bag(word_list, vocab):
    """ V1: vocabulary binary vector """
    f_vector_length = len(vocab)
    if not word_list:
        return np.full(f_vector_length, 0, int)
    f_list = [0] * f_vector_length
    # todo no counting, only existence (set)

    world_list_set = set(normalize(word) for word in word_list if word.isalpha())
    for word in world_list_set:
        if word in vocab:
            idx = vocab[word]
            f_list[idx] = 1
    return np.array(f_list)