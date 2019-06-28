# -*- coding: utf-8 -*-
"""
Multi Writing Style Detection
=====================================================================

**mwsd** is a Python package providing an implementation of an algorithm to Detect
a Multi Writing Style in texts. This project implements a solution of detecting numerous
writing styles in a text. There are many different ways to measure how similar two documents
are, or how similar a document is to a query. The project implements the first algorithm of
the article with minor changes, which don't affect the outcomes. This algorithm is suggested
in the "Patterning of writing style evolution by means of dynamic similarity" by Konstantin
Amelina, Oleg Granichina, Natalia Kizhaevaa and Zeev Volkovich
(http://www.math.spbu.ru/user/gran/papers/Granichin_Pattern_Recognition.pdf).

Module main feature is determining whether 2 texts are written in the same writing style.

Project Home Page:
https://github.com/romanglo/multiple-writing-style-detector
"""
from __future__ import absolute_import

from .version import __version__

from .visualize import visualize
from .utils import download_nltk_dependencies
from .algorithm import execute_algorithm, execute_dzv, execute_zv, execute_dzv_clustering
from .algorithm import DEFAULT_CHUNK_SIZE, DEFAULT_N_TOP_KEYWORDS, DEFAULT_T, DEFAULT_CLUSTERING_K, DEFAULT_CLUSTERING_SPAWN


def initialize():
    """
    Initialize necessary the Multi Writing Style Detection algorithm.

    **Warning** The module may work without calling this method under certain circumstances,
                but it is recommended to call initialize() before calling the execute() function.
    """
    download_nltk_dependencies()


def execute(first_text,
            second_text,
            model=None,
            T=DEFAULT_T,
            chunk_size=DEFAULT_CHUNK_SIZE,
            n_top_keywords=DEFAULT_N_TOP_KEYWORDS):
    """
    Execute the Multi Writing Style Detection algorithm.

    Parameters
    ----------
    first_text : str
        The first text for the algorithm.
    second_text : str
        The first text for the algorithm.
    model : gensim.models.KeyedVectors or gensim.models.Word2Vec [optional]
        The embedding model
    T: int
        T look ahead when calculating the algorithm
    chunk_size:
        The text is divided into chunks, which is their size.
    n_top_keywords:
        The algorithm uses a certain amount of keywords, that is the quantity.

    Returns
    ----------
    (np.array, np.array, tuple): ZV distance (1 dimensional array), DZV distance (2 dimensional array), Clustering

    Clustering contains (according to indexes):
    0 - Labels array (1 dimensional array).
    1 - Distance of each element from its medoid (1 dimensional array).
    2 - Silhouette score (float).

    Algorithm
    ----------
    Full algorithm documentation is at the link:  https://github.com/romanglo/multiple-writing-style-detector#algorithm
    1. The algorithm receives two texts for input.
    2. Find the N top keywords using tf–idf.
    3. Remove from the texts the stopwords and words that not in the N (initially defined amount)  top keywords.
    4. Gather groups of 'L' (initially defined amount)  keywords out of the text.
    5. Use word2vec to represent each word as a vector for both documents.
    6. Calculate the correlation between all the words in each group (L) using the Kolmogorov-Smirnov statistic. Each group (L) became a vector of L(L-1)/2 dimensionality.
    7. Find an association between the vector and its 'T' (initially defined amount) predecessors, using ZV formula.
    8. Measure the distance between the documents using DZV formula.
    9. PAM (known also as k-medoids) clustering into two clusters.

    """
    return execute_algorithm(
        first_text=first_text,
        second_text=second_text,
        model=model,
        T=T,
        chunk_size=chunk_size,
        n_top_keywords=n_top_keywords)


def zv(text,
       model=None,
       T=DEFAULT_T,
       chunk_size=DEFAULT_CHUNK_SIZE,
       n_top_keywords=DEFAULT_N_TOP_KEYWORDS):
    """
    Calculate ZV distance of text.

    Parameters
    ----------
    text : str
        The text for the algorithm.
    model : gensim.models.KeyedVectors or gensim.models.Word2Vec [optional]
        The embedding model
    T: int
        T look ahead when calculating the algorithm
    chunk_size:
        The text is divided into chunks, which is their size.
    n_top_keywords:
        The algorithm uses a certain amount of keywords, that is the quantity.

    Returns
    ----------
    np.array: ZV distance (1 dimensional array)
    """
    return execute_zv(
        text=text,
        model=model,
        T=T,
        chunk_size=chunk_size,
        n_top_keywords=n_top_keywords)


def dzv(first_text,
        second_text,
        model=None,
        T=DEFAULT_T,
        chunk_size=DEFAULT_CHUNK_SIZE,
        n_top_keywords=DEFAULT_N_TOP_KEYWORDS):
    """
    Calculate DZV distance of two texts.

    Parameters
    ----------
    first_text : str
        The first text for the algorithm.
    second_text : str
        The first text for the algorithm.
    model : gensim.models.KeyedVectors or gensim.models.Word2Vec [optional]
        The embedding model
    T: int
        T look ahead when calculating the algorithm
    chunk_size:
        The text is divided into chunks, which is their size.
    n_top_keywords:
        The algorithm uses a certain amount of keywords, that is the quantity.

    Returns
    ----------
    np.array: DZV distance (2 dimensional array)
    """
    return execute_dzv(
        first_text=first_text,
        second_text=second_text,
        model=model,
        T=T,
        chunk_size=chunk_size,
        n_top_keywords=n_top_keywords)


def dzv_clustering(dzv, k=DEFAULT_CLUSTERING_K,
                   spawn=DEFAULT_CLUSTERING_SPAWN):
    """
    Cluster DZV result using k-mediods clustering .

    Parameters
    ----------
    dzv : np.array (2 dimensional array)
        DZV distance matrix to cluster.
    k : int
        Number of desired clusters (> 2)
    spawn : int
        The number of spawns in the clustering (> 1)
    T: int
        T look ahead when calculating the algorithm

    Returns
    ----------
    (np.array, np.array, float): labels array (1 dimensional array), distance of each element from its medoid (1 dimensional array), Silhouette score
    """
    return execute_dzv_clustering(dzv=dzv, k=k, spawn=spawn)


def visualize_algorithm_result(zv,
                               dzv,
                               clustering_result,
                               show_plot=True,
                               plot_saving_path=None):
    """
    Visualize the result of the algorithm

    Parameters
    ----------

    (np.array, np.array, tuple): ZV distance (1 dimensional array), DZV distance (2 dimensional array), Clustering

    zv : np.array (1 dimensional array)
        DZV distance array
    dzv : np.array (2 dimensional array)
        DZV distance matrix.
    clustering_result : tuple(np.array, np.array, float)
        DZV clustering result.
    show_plot : bool
        To call plt.show() or not.
    plot_saving_path: str
        Path to save the figure, None will do nothing.
    """
    visualize(
        zv=zv,
        dzv=dzv,
        clustering_result=clustering_result,
        show_plot=show_plot,
        plot_saving_path=plot_saving_path)


__all__ = ['algorithm', 'mediods', 'tfidf', 'utils', 'visualize', 'word2vec']
