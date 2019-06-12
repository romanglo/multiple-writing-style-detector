"""
Multi Writing Style Detection
=====================================================================

**mwsd** is a Python package providing an implementation of an algorithm to Detect
a Multi Writing Style in texts.

Module main feature is determining whether 2 texts are written in the same writing style.
"""

from .version import __version__

from .utils import download_nltk_dependencies
from .algorithm import execute_algorithm, execute_dzv, execute_zv
from .algorithm import DEFAULT_CHUNK_SIZE, DEFAULT_N_TOP_KEYWORDS, DEFAULT_T


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
    (np.array, np.array): ZV distance (1 dimensional array), DZV distance (2 dimensional array)
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


__all__ = ['tfidf', 'word2vec', 'utils', 'algorithm']
