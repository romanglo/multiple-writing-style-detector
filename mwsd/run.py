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


def process():
    # model_name = r"data\GoogleNews-vectors-negative300.bin"

    data_path = r"data\wiki_articles"
    files = utils.get_files_list_from_dir(data_path)
    texts = utils.read_text_from_files(files, encoding='latin-1')
    stop_words = utils.get_stop_words('english')

    T = 10
    chunk_size = 50
    n_top_keywords = 1000

    model = word2vec.train_word2vec(texts, stop_words, iter=20)

    # model = word2vec.load_model(model_name, keyed_vectors=True, binary=True)

    keywords = tfidf.get_n_top_keywords(texts, stop_words,
                                        int(n_top_keywords * 1.5))
    keyword_words = [
        keyword[0] for keyword in keywords if keyword[0] in model.wv.vocab
    ]
    keyword_words = keyword_words[:min(len(keyword_words), n_top_keywords)]

    ids, words = word2vec.text2ids(
        model=model,
        # texts[-1],
        text=" ".join(texts[0:2]),
        stop_words=stop_words,
        acceptable_tokens=keyword_words,
        remove_skipped_tokens=True)

    chunks = utils.chunks(list(zip(ids, words)), chunk_size)

    chunks_num = int(
        len(ids) / chunk_size) + (1 if (len(ids) % chunk_size != 0) else 0)

    similarites_vector_size = (chunk_size * (chunk_size - 1)) // 2
    similarites_matrix = np.zeros((chunks_num, similarites_vector_size))

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

    if model is not None:
        del model

    ZVs = np.zeros(chunks_num - T + 1)
    ZV_statistic = np.zeros((T))
    for i in range(chunks_num - 1, T - 1, -1):
        for j in range(i - 1, i - 1 - T, -1):
            ZV_statistic[i - j - 1], _ = ks_2samp(similarites_matrix[i],
                                                  similarites_matrix[j])
        ZVs[i - T + 1] = np.sum(ZV_statistic)

    plt.plot(ZVs)
    plt.ylabel('ZV')
    plt.show()


def main(argv):
    try:
        utils.initialize_logging_config(logging.DEBUG)
        utils.download_nltk_dependencies()
        process()
    except KeyboardInterrupt:
        logging.info("\n\nProcess aborted by the user!")
    except Exception:
        logging.error(
            "Some error occurred during the running! Process aborted..")


# Run the program
if __name__ == "__main__":
    main(sys.argv[1:])
