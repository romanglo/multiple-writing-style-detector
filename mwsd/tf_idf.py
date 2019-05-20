# tf_idf.py
"""
TFIDF - term frequency–inverse document frequency
"""

import numpy as np
from typing import List


def if_tdf(setcell: List[int], words):
    uniqueWords = np.unique(words)

    mat = numpy.zeros(shape=(len(setcell), len(uniqueWords)))

    for i in range(len(setcell)):
        word = np.loadtxt(setcell[i], dtype=str)
        uniqueId = ismember(words[i], word)
        np.sum(uniqueId, )


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(ietm, None) for item in a]
