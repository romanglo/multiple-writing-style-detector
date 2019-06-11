from __future__ import division

import logging
import time
from builtins import int
from collections import Counter

import numpy as np
from future.utils import raise_with_traceback
from scipy.stats import ks_2samp

from .tfidf import get_n_top_keywords
from .utils import (UNSUPPORTED_LANGAUGE, ISO_639_1_codes_to_nltk_codes,
                    detect_language, get_stop_words, split_to_chunks)
from .word2vec import text2ids, train_word2vec

DEFAULT_T = 10
DEFAULT_CHUNK_SIZE = 50
DEFAULT_N_TOP_KEYWORDS = 1000


def _calculate_similarity_matrix(model, words, chunk_size=DEFAULT_CHUNK_SIZE):
    chunks = split_to_chunks(words, chunk_size)

    logging.debug(f"No. of words: {len(words)}, Chunk size: {chunk_size}")

    chunks_num = int(
        len(words) / chunk_size) + (1 if (len(words) % chunk_size != 0) else 0)

    logging.debug(f"No. of chunks {chunks_num}")

    similarites_vector_size = (chunk_size * (chunk_size - 1)) // 2
    similarites_matrix = np.zeros((chunks_num, similarites_vector_size))

    similarity_func = np.vectorize(
        lambda x, y: model.similarity(x, y), otypes=[float])

    chunk_index = 0
    for chunk in chunks:
        in_chunk_index = 0
        for i in range(len(chunk)):
            similarities = similarity_func(chunk[i], chunk[i + 1:len(chunk)])
            similarites_matrix[chunk_index, in_chunk_index:in_chunk_index +
                               len(similarities)] = similarities
            in_chunk_index += len(similarities)

        chunk_index += 1

    logging.debug(f"Similarity matrix shape {similarites_matrix.shape}")

    return similarites_matrix


def _calculate_zv_distances(similarites_matrix, T=DEFAULT_T):
    chunks_num = similarites_matrix.shape[0]
    ZVs = np.zeros(chunks_num - T + 1)

    for i in range(ZVs.shape[0]):
        ZVs[i] = np.sum(
            np.apply_along_axis(
                ks_2samp,
                1,
                similarites_matrix[i + 1:i + T + 1],
                similarites_matrix[i],
            )[:, 0],
            axis=0)

    return ZVs


def _calculate_dzv_distances(first_similarites_matrix,
                             second_similarites_matrix,
                             T=DEFAULT_T):

    DZV = np.zeros((first_similarites_matrix.shape[0] - T + 1,
                    second_similarites_matrix.shape[0] - T + 1))

    for i in range(DZV.shape[0]):

        zv_1 = np.sum(
            np.apply_along_axis(
                ks_2samp,
                1,
                first_similarites_matrix[i + 1:i + T + 1],
                first_similarites_matrix[i],
            )[:, 0],
            axis=0)

        for j in range(DZV.shape[1]):

            zv_2 = np.sum(
                np.apply_along_axis(
                    ks_2samp,
                    1,
                    second_similarites_matrix[j + 1:j + T + 1],
                    second_similarites_matrix[j],
                )[:, 0],
                axis=0)

            zv_3 = np.sum(
                np.apply_along_axis(
                    ks_2samp,
                    1,
                    second_similarites_matrix[j + 1:j + T + 1],
                    first_similarites_matrix[i],
                )[:, 0],
                axis=0)

            zv_4 = np.sum(
                np.apply_along_axis(
                    ks_2samp,
                    1,
                    first_similarites_matrix[i + 1:i + T + 1],
                    second_similarites_matrix[j],
                )[:, 0],
                axis=0)

            DZV[i, j] = abs(zv_1 + zv_2 - zv_3 - zv_4)

    return DZV


def zv_process(text,
               model,
               stop_words,
               keywords,
               T=DEFAULT_T,
               chunk_size=DEFAULT_CHUNK_SIZE):

    start_time = time.time()
    ids, words = text2ids(
        model=model,
        text=text,
        stop_words=stop_words,
        acceptable_tokens=keywords,
        remove_skipped_tokens=True)
    end_time = time.time()
    logging.debug(f"word2vec runs {end_time-start_time:.4f} seconds")

    start_time = time.time()
    sim_mat = _calculate_similarity_matrix(model, words, chunk_size)
    end_time = time.time()
    logging.debug(
        f"similarity matrix calculation took {end_time-start_time:.4f} seconds"
    )

    del ids, words

    start_time = time.time()
    ZVs = _calculate_zv_distances(sim_mat, T)
    end_time = time.time()
    logging.debug(
        f"ZV distances calculation took {end_time-start_time:.4f} seconds")

    del sim_mat

    return ZVs


def dzv_process(first_text,
                second_text,
                model,
                stop_words,
                keywords,
                T=DEFAULT_T,
                chunk_size=DEFAULT_CHUNK_SIZE):

    start_time = time.time()
    first_text_ids, first_text_words = text2ids(
        model=model,
        text=first_text,
        stop_words=stop_words,
        acceptable_tokens=keywords,
        remove_skipped_tokens=True)
    end_time = time.time()
    logging.debug(
        f"first text word2vec runs {end_time-start_time:.4f} seconds")

    start_time = time.time()
    first_sim_mat = _calculate_similarity_matrix(model, first_text_words,
                                                 chunk_size)
    end_time = time.time()
    logging.debug(
        f"first text similarity matrix calculation took {end_time-start_time:.4f} seconds"
    )

    del first_text_ids, first_text_words

    start_time = time.time()
    second_text_ids, second_text_words = text2ids(
        model=model,
        text=second_text,
        stop_words=stop_words,
        acceptable_tokens=keywords,
        remove_skipped_tokens=True)
    end_time = time.time()
    logging.debug(
        f"second text word2vec runs {end_time-start_time:.4f} seconds")

    start_time = time.time()
    second_sim_mat = _calculate_similarity_matrix(model, second_text_words,
                                                  chunk_size)
    end_time = time.time()
    logging.debug(
        f"second text similarity matrix calculation took {end_time-start_time:.4f} seconds"
    )

    del second_text_ids, second_text_words

    start_time = time.time()
    DZV = _calculate_dzv_distances(
        first_similarites_matrix=first_sim_mat,
        second_similarites_matrix=second_sim_mat,
        T=T)

    end_time = time.time()
    logging.debug(
        f"DZV matrix calculation took {end_time-start_time:.4f} seconds")

    del first_sim_mat, second_sim_mat

    return DZV


def _preprocess(texts, model=None, n_top_keywords=DEFAULT_N_TOP_KEYWORDS):
    langauge_ISO_639_1 = Counter(
        [detect_language(text) for text in texts]).most_common(1)[0][0]

    language_nltk = ISO_639_1_codes_to_nltk_codes(langauge_ISO_639_1)

    if (language_nltk == UNSUPPORTED_LANGAUGE):
        unsupported_language_msg = f"The texts are in unsupported language: {langauge_ISO_639_1} (ISO 639-1 code)"
        logging.error(unsupported_language_msg)
        raise_with_traceback(Exception(unsupported_language_msg))

    stop_words = get_stop_words(language_nltk)

    full_text = " ".join(texts)
    if model is None:
        model = train_word2vec(full_text, stop_words, iter=20)

    keywords = get_n_top_keywords(full_text, stop_words,
                                  int(n_top_keywords * 1.5))
    n_top_keyword = [
        keyword[0] for keyword in keywords if keyword[0] in model.wv.vocab
    ]
    n_top_keyword = n_top_keyword[:min(len(n_top_keyword), n_top_keywords)]

    return stop_words, model, n_top_keyword


def execute_algorithm(first_text,
                      second_text,
                      model=None,
                      T=DEFAULT_T,
                      chunk_size=DEFAULT_CHUNK_SIZE,
                      n_top_keywords=DEFAULT_N_TOP_KEYWORDS):

    del_model = model is None

    start_time = time.time()
    stop_words, model, n_top_keyword = _preprocess(
        texts=[first_text, second_text],
        model=model,
        n_top_keywords=n_top_keywords)
    end_time = time.time()
    logging.debug(f"Preprocessing took {end_time-start_time:.4f} seconds")

    start_time = time.time()
    ZV = zv_process(
        text=first_text + " " + second_text,
        model=model,
        stop_words=stop_words,
        keywords=n_top_keyword,
        T=T,
        chunk_size=chunk_size)
    end_time = time.time()
    logging.debug(f"ZV calculation took {end_time-start_time:.4f} seconds")

    start_time = time.time()
    DZV = dzv_process(
        first_text=first_text,
        second_text=second_text,
        model=model,
        stop_words=stop_words,
        keywords=n_top_keyword,
        T=T,
        chunk_size=chunk_size)
    end_time = time.time()
    logging.debug(f"DZV calculation took {end_time-start_time:.4f} seconds")

    if del_model:
        del model

    return ZV, DZV


def execute_zv(text,
               model=None,
               T=DEFAULT_T,
               chunk_size=DEFAULT_CHUNK_SIZE,
               n_top_keywords=DEFAULT_N_TOP_KEYWORDS):

    del_model = model is None

    start_time = time.time()
    stop_words, model, n_top_keyword = _preprocess(
        texts=[text], model=model, n_top_keywords=n_top_keywords)
    end_time = time.time()
    logging.debug(f"Preprocessing took {end_time-start_time:.4f} seconds")

    start_time = time.time()
    ZV = zv_process(
        text=text,
        model=model,
        stop_words=stop_words,
        keywords=n_top_keyword,
        T=T,
        chunk_size=chunk_size)
    end_time = time.time()
    logging.debug(f"ZV calculation took {end_time-start_time:.4f} seconds")

    if del_model:
        del model

    return ZV


def execute_dzv(first_text,
                second_text,
                model=None,
                T=DEFAULT_T,
                chunk_size=DEFAULT_CHUNK_SIZE,
                n_top_keywords=DEFAULT_N_TOP_KEYWORDS):

    del_model = model is None

    start_time = time.time()
    stop_words, model, n_top_keyword = _preprocess(
        texts=[first_text, second_text],
        model=model,
        n_top_keywords=n_top_keywords)
    end_time = time.time()
    logging.debug(f"Preprocessing took {end_time-start_time:.4f} seconds")

    start_time = time.time()
    DZV = dzv_process(
        first_text=first_text,
        second_text=second_text,
        model=model,
        stop_words=stop_words,
        keywords=n_top_keyword,
        T=T,
        chunk_size=chunk_size)
    end_time = time.time()
    logging.debug(f"DZV calculation took {end_time-start_time:.4f} seconds")

    if del_model:
        del model

    return DZV
