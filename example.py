from __future__ import division

import logging
import sys

import mwsd


def initialize_logging_config(logging_level):
    logging.basicConfig(
        format=
        u'%(asctime)s %(name)-7s [%(filename)10s:%(lineno)3s - %(funcName)20s()] %(levelname)-8s:: %(message)s.',
        level=logging_level)


def process():

    # model_name = r"data\GoogleNews-vectors-negative300.bin"

    data_path = r"data\wiki_articles"

    files = mwsd.utils.get_files_list_from_dir(data_path)
    texts = mwsd.utils.read_text_from_files(files, encoding='utf-8')

    full_text = " ".join(texts)
    first_text = full_text[:len(full_text) // 2]
    second_text = full_text[len(full_text) // 2:]

    ZV, DZV = mwsd.execute(first_text, second_text)

    # TODO visualize result


def main(argv):
    # TODO add arguments from user
    try:
        initialize_logging_config(logging.DEBUG)
        mwsd.initialize()
        process()
    except KeyboardInterrupt:
        logging.info("Process aborted by the user!")
    except Exception:
        logging.exception(
            "Some error occurred during the running! Process aborted..")


# Run the program
if __name__ == "__main__":
    main(sys.argv[1:])
