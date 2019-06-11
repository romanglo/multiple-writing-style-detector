from __future__ import division

import logging
import sys
import time
from builtins import int
from collections import Counter

import numpy as np
from scipy.stats import ks_2samp

import tfidf
import utils
import word2vec

T = 10
chunk_size = 50
n_top_keywords = 1000


def calculate_similarity_matrix(model, words, chunk_size=chunk_size):
    chunks = utils.chunks(words, chunk_size)

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


def calculate_zv_distances(similarites_matrix, T=T):
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


def calculate_dzv_distances(first_similarites_matrix,
                            second_similarites_matrix,
                            T=T):

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


def zv_process(first_text, second_text, model, stop_words, keywords):

    start_time = time.time()
    ids, words = word2vec.text2ids(
        model=model,
        text=first_text + " " + second_text,
        stop_words=stop_words,
        acceptable_tokens=keywords,
        remove_skipped_tokens=True)
    end_time = time.time()
    logging.debug(f"word2vec runs {end_time-start_time:.4f} seconds")

    start_time = time.time()
    sim_mat = calculate_similarity_matrix(model, words)
    end_time = time.time()
    logging.debug(
        f"similarity matrix calculation took {end_time-start_time:.4f} seconds"
    )

    del ids, words

    start_time = time.time()
    ZVs = calculate_zv_distances(sim_mat)
    end_time = time.time()
    logging.debug(
        f"ZV distances calculation took {end_time-start_time:.4f} seconds")

    del sim_mat

    return ZVs


def dzv_process(first_text, second_text, model, stop_words, keywords):

    start_time = time.time()
    first_text_ids, first_text_words = word2vec.text2ids(
        model=model,
        text=first_text,
        stop_words=stop_words,
        acceptable_tokens=keywords,
        remove_skipped_tokens=True)
    end_time = time.time()
    logging.debug(
        f"first text word2vec runs {end_time-start_time:.4f} seconds")

    start_time = time.time()
    first_sim_mat = calculate_similarity_matrix(model, first_text_words)
    end_time = time.time()
    logging.debug(
        f"first text similarity matrix calculation took {end_time-start_time:.4f} seconds"
    )

    del first_text_ids, first_text_words

    start_time = time.time()
    second_text_ids, second_text_words = word2vec.text2ids(
        model=model,
        text=second_text,
        stop_words=stop_words,
        acceptable_tokens=keywords,
        remove_skipped_tokens=True)
    end_time = time.time()
    logging.debug(
        f"second text word2vec runs {end_time-start_time:.4f} seconds")

    start_time = time.time()
    second_sim_mat = calculate_similarity_matrix(model, second_text_words)
    end_time = time.time()
    logging.debug(
        f"second text similarity matrix calculation took {end_time-start_time:.4f} seconds"
    )

    del second_text_ids, second_text_words

    start_time = time.time()
    DZV = calculate_dzv_distances(
        first_similarites_matrix=first_sim_mat,
        second_similarites_matrix=second_sim_mat)

    end_time = time.time()
    logging.debug(
        f"DZV matrix calculation took {end_time-start_time:.4f} seconds")
    del first_sim_mat, second_sim_mat

    return DZV


def process():

    # model_name = r"data\GoogleNews-vectors-negative300.bin"

    data_path = r"data\wiki_articles"

    files = utils.get_files_list_from_dir(data_path)[:5]
    texts = utils.read_text_from_files(files, encoding='utf-8')

    langauge_ISO_639_1 = Counter(
        [utils.detect_language(text) for text in texts]).most_common(1)[0][0]

    language_nltk = utils.ISO_639_1_codes_to_nltk_codes(langauge_ISO_639_1)

    if (language_nltk == utils.UNSUPPORTED_LANGAUGE):
        logging.error(
            f"The texts are in unsupported language: {langauge_ISO_639_1} (ISO 639-1 code)"
        )
        return

    stop_words = utils.get_stop_words(language_nltk)

    model = word2vec.train_word2vec(texts, stop_words, iter=20)

    # model = word2vec.load_model(model_name, keyed_vectors=True, binary=True)

    keywords = tfidf.get_n_top_keywords(texts, stop_words,
                                        int(n_top_keywords * 1.5))
    n_top_keyword = [
        keyword[0] for keyword in keywords if keyword[0] in model.wv.vocab
    ]
    n_top_keyword = n_top_keyword[:min(len(n_top_keyword), n_top_keywords)]

    full_text = " ".join(texts)
    first_text = full_text[:len(full_text) // 2]
    second_text = full_text[len(full_text) // 2:]

    ZV = zv_process(
        first_text=first_text,
        second_text=second_text,
        model=model,
        stop_words=stop_words,
        keywords=n_top_keyword)

    DZV = dzv_process(
        first_text=first_text,
        second_text=second_text,
        model=model,
        stop_words=stop_words,
        keywords=n_top_keyword)

    del model

    return ZV, DZV


def main(argv):
    try:
        utils.initialize_logging_config(logging.DEBUG)
        utils.download_nltk_dependencies()
        process()
    except KeyboardInterrupt:
        logging.info("Process aborted by the user!")
    except Exception:
        logging.exception(
            "Some error occurred during the running! Process aborted..")


# Run the program
if __name__ == "__main__":
    main(sys.argv[1:])
