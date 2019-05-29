from __future__ import division

import logging
import sys
from builtins import int

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

import tfidf
import utils
import word2vec

T = 10
chunk_size = 50
n_top_keywords = 1000


def calculate_similarity_matrix(model, ids, words, chunk_size=chunk_size):
    chunks = utils.chunks(list(zip(ids, words)), chunk_size)

    logging.debug(f"No. of words: {len(words)}, Chunk size: {chunk_size}")

    chunks_num = int(
        len(ids) / chunk_size) + (1 if (len(ids) % chunk_size != 0) else 0)

    logging.debug(f"No. of chunks {chunks_num}")

    similarites_vector_size = (chunk_size * (chunk_size - 1)) // 2
    similarites_matrix = np.zeros((chunks_num, similarites_vector_size))

    logging.debug(f"Similarity matrix shape {similarites_matrix.shape}")

    chunk_index = 0
    for chunk in chunks:
        in_chunk_index = 0
        for i in range(len(chunk)):
            for j in range(i + 1, len(chunk)):
                similarites_matrix[
                    chunk_index][in_chunk_index] = model.similarity(
                        chunk[i][1], chunk[j][1])
                in_chunk_index += 1
        chunk_index += 1

    return similarites_matrix


def calculate_zv_distances(similarites_matrix, show_plt=False):
    chunks_num = similarites_matrix.shape[0]
    ZVs = np.zeros(chunks_num - T + 1)
    ZV_statistic = np.zeros((T))
    for i in range(chunks_num - 1, T - 1, -1):
        for j in range(i - 1, i - 1 - T, -1):
            ZV_statistic[i - j - 1], _ = ks_2samp(similarites_matrix[i],
                                                  similarites_matrix[j])
        ZVs[i - T + 1] = np.sum(ZV_statistic)

    if show_plt:
        plt.clf()
        plt.plot(ZVs)
        plt.ylabel('ZV')
        plt.show()

    return ZVs


def zv_process(first_text, second_text, model, stop_words, keywords):
    ids, words = word2vec.text2ids(
        model=model,
        text=first_text + " " + second_text,
        stop_words=stop_words,
        acceptable_tokens=keywords,
        remove_skipped_tokens=True)

    sim_mat = calculate_similarity_matrix(model, ids, words)

    del ids, words

    ZVs = calculate_zv_distances(sim_mat, show_plt=True)

    del sim_mat

    return ZVs


def dzv_process(first_text, second_text, model, stop_words, keywords):

    first_text_ids, first_text_words = word2vec.text2ids(
        model=model,
        text=first_text,
        stop_words=stop_words,
        acceptable_tokens=keywords,
        remove_skipped_tokens=True)

    first_sim_mat = calculate_similarity_matrix(model, first_text_ids,
                                                first_text_words)
    del first_text_ids, first_text_words

    second_text_ids, second_text_words = word2vec.text2ids(
        model=model,
        text=second_text,
        stop_words=stop_words,
        acceptable_tokens=keywords,
        remove_skipped_tokens=True)

    second_sim_mat = calculate_similarity_matrix(model, second_text_ids,
                                                 second_text_words)
    del second_text_ids, second_text_words

    # TODO calculate DZV distances

    del first_sim_mat, second_sim_mat


def process():
    # model_name = r"data\GoogleNews-vectors-negative300.bin"

    data_path = r"data\wiki_articles"
    files = utils.get_files_list_from_dir(data_path)
    texts = utils.read_text_from_files(files, encoding='latin-1')
    stop_words = utils.get_stop_words('english')

    model = word2vec.train_word2vec(texts, stop_words, iter=20)

    # model = word2vec.load_model(model_name, keyed_vectors=True, binary=True)

    keywords = tfidf.get_n_top_keywords(texts, stop_words,
                                        int(n_top_keywords * 1.5))
    n_top_keyword = [
        keyword[0] for keyword in keywords if keyword[0] in model.wv.vocab
    ]
    n_top_keyword = n_top_keyword[:min(len(n_top_keyword), n_top_keywords)]

    first_text = " ".join(texts[:len(texts) // 2])
    second_text = " ".join(texts[len(texts) // 2:])

    dzv_process(
        first_text=first_text,
        second_text=second_text,
        model=model,
        stop_words=stop_words,
        keywords=n_top_keyword)

    del model


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
