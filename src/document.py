class Document(object):
    """A document completely characterized by its features."""

    max_display_data = 30  # limit for data abbreviation

    def __init__(self, data, label=None, source=None):
        self.data = data
        self.label = label
        self.source = source

    # def __str__(self):
    #     return self.__repr__()

    # def __repr__(self):
    #     return (u"<%s: %s>" % (self.label, self.abbrev()) if self.label else
    #             u"%s" % self.abbrev())

    def abbrev(self):
        return (self.data if len(self.data) < self.max_display_data else
                self.data[0:self.max_display_data] + "...")

    def features(self):
        """A list of features that characterize this document."""
        return [self.data]


class EvenOdd(Document):
    def features(self):
        """Is the data even or odd?"""
        return [self.data % 2 == 0]

class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""
        return self.data.split()

class Name(Document):
    def features(self, letters="abcdefghijklmnopqrstuvwxyz"):
        """From NLTK's names_demo_features: first & last letters, how many
        of each letter, and which letters appear."""
        name = self.data
        return ([name[0].lower(), name[-1].lower()] +
                [name.lower().count(letter) for letter in letters] +
                [letter in name.lower() for letter in letters])