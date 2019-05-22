"""
This script was created with the use of:
https://github.com/subramanyata/myprojects/tree/master/word2vec
"""

from future.utils import raise_with_traceback

import logging
import os
import time

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from nltk import word_tokenize


def preprocess_document(doc, stop_words):

    tokens = word_tokenize(doc.lower())

    logging.debug(f"The document contains {len(tokens)} tokens")

    if stop_words is None:
        stop_words = []

    # Remove stopwords, numbers and punctuation.
    tokens = [
        token for token in tokens
        if token not in stop_words and token.isalpha()
    ]

    logging.debug(
        f"Number of tokens after removing stop words, numbers and punctuation: {len(tokens)}"
    )

    return tokens


def train_word2vec(train_data,
                   stop_words,
                   iter=5,
                   worker_no=4,
                   vector_size=100):
    if not train_data:
        logging.error("no training data")
        return None

    w2v_corpus = [preprocess_document(data, stop_words) for data in train_data]

    model = Word2Vec(
        w2v_corpus, iter=iter, workers=worker_no, size=vector_size, sg=1)
    logging.info("Model Created Successfully")
    return model


def load_model(path, keyed_vectors=False, binary=False):
    if not os.path.isfile(path):
        err_msg = f"Received embedding model binary not found! Path: {path}"
        logging.error(err_msg)
        raise_with_traceback(FileNotFoundError(err_msg))

    start_time = time.time()

    if keyed_vectors:
        model = KeyedVectors.load_word2vec_format(path, binary=binary)
    else:
        model = Word2Vec.load(path)

    running_time = time.time() - start_time

    logging.debug(f"Model loading time: {running_time} seconds")

    return model


def text2mat(model, text, stop_words, skip_tokens=None):
    words = preprocess_document(text, stop_words)

    word_vectors = model.wv
    vector_size = model.vector_size
    mat = np.zeros((len(words), vector_size))

    if skip_tokens is None:
        skip_tokens = []

    for i, word in enumerate(words):
        if word in word_vectors.vocab and word not in skip_tokens:
            mat[i, :] = word_vectors.word_vec(word, use_norm=False)

    return mat, words


def text2ids(model, text, stop_words, skip_tokens=None, skipped_token_id=-1):
    words = preprocess_document(text, stop_words)

    word_vectors = model.wv

    ids = np.repeat(skipped_token_id, len(words))

    if skip_tokens is None:
        skip_tokens = []

    for i, word in enumerate(words):
        if word in word_vectors.vocab and word not in skip_tokens:
            ids[i] = word_vectors.vocab[word].index

    return ids, words
