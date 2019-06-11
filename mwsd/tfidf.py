"""
Term Frequency–Inverse Document Frequency
=====================================================================

**tfidf** module providing an implementation of an tf-idf (using sklearn) and provide an option to extract the top N
keywords in text.

This script was created with the use of:
https://github.com/kavgan/nlp-in-practice/tree/master/tf-idf
"""

import re
from builtins import str

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def preprocess_document(doc):
    text = doc.lower()  # lowercase
    text = re.sub("</?.*?>", " <> ", text)  # remove tags
    text = re.sub("(\\d|\\W)+", " ",
                  text)  # remove special characters and digits
    return text


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, top_n):
    # use only top n items from vector
    sorted_items = sorted_items[:top_n]

    scores = []
    features = []

    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        scores.append(round(score, 3))
        features.append(feature_names[idx])

    return list(zip(features, scores))


def create_transformers(docs, stop_words):

    docs = [docs] if isinstance(docs, str) else docs

    cleared_docs = []
    for doc in docs:
        cleared_docs.append(preprocess_document(doc))

    count_vectorizer = CountVectorizer(stop_words=stop_words)
    word_count_vector = count_vectorizer.fit_transform(cleared_docs)

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    return tfidf_transformer, count_vectorizer, cleared_docs


def tfidf(docs, stop_words):
    tfidf_transformer, count_vectorizer, cleared_docs = create_transformers(
        docs, stop_words)

    tf_idf_vector = tfidf_transformer.transform(
        count_vectorizer.transform(cleared_docs))

    return tf_idf_vector


def get_n_top_keywords(doc, stop_words, top_n):

    tfidf_transformer, count_vectorizer, cleared_docs = create_transformers(
        doc, stop_words)

    # generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(
        count_vectorizer.transform(cleared_docs))

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    feature_names = count_vectorizer.get_feature_names()

    # extract only the top n
    top_n_keywords = extract_topn_from_vector(feature_names, sorted_items,
                                              top_n)
    return top_n_keywords
