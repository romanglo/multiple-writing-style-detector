from .version import __version__

from .utils import download_nltk_dependencies
from .algorithm import execute as algorithm_execute, DEFAULT_CHUNK_SIZE, DEFAULT_N_TOP_KEYWORDS, DEFAULT_T


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
        The embedding model\n


    Returns
    ----------
    (np.array, np.array): ZV result (1 dimensional array), (2 dimensional array)
    """
    return algorithm_execute(first_text, second_text, T, chunk_size,
                             n_top_keywords)


__all__ = ['tfidf', 'word2vec', 'utils', 'algorithm']

__doc__ = """
Multi Writing Style Detection
=====================================================================

**mwsd** is a Python package providing an implementation of an algorithm to detect
a Multi Writing Style Detection in texts.

Module main feature is determining whether 2 texts are written in the same writing style.
"""
