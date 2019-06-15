from __future__ import division, unicode_literals

import logging
import sys

import mwsd


def initialize_logging_config(logging_level):
    logging.basicConfig(format=u'%(asctime)s %(name)-7s [%(filename)10s:%(lineno)3s - %(funcName)20s()] %(levelname)-8s:: %(message)s.', level=logging_level)


def process():

    first_text = " ".join(
        mwsd.utils.read_text_from_files(
            mwsd.utils.get_files_list_from_dir(r"data\2"), encoding='utf-8'))

    second_text = " ".join(
        mwsd.utils.read_text_from_files(
            mwsd.utils.get_files_list_from_dir(r"data\3"), encoding='utf-8'))

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
