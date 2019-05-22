import logging
import os
import re

from nltk import download
from nltk.corpus import stopwords
from smart_open import open


def chunks(iterable, chunk_size):
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]


def removeMultipleSpaces(inStr: str) -> str:
    if not inStr:
        return None

    outStr = re.sub(' +', ' ', inStr)

    return outStr


def removeChars(inStr: str, charsToRemove: str) -> str:
    if not inStr or not charsToRemove:
        return None

    translation_table = dict.fromkeys(map(ord, charsToRemove), None)
    outStr = inStr.translate(translation_table)

    return outStr


def get_stop_words(language):
    return stopwords.words(language)


def download_nltk_dependencies():
    ids = ['stopwords', 'punkt']
    for id in ids:
        download(id)


def initialize_logging_config(logging_level):
    logging.basicConfig(
        format=
        u'%(asctime)s %(name)-7s [%(filename)10s:%(lineno)3s - %(funcName)20s()] %(levelname)-8s:: %(message)s.',
        level=logging_level)


def get_files_list_from_dir(path, extension=None):
    if not os.path.isdir(path):
        logging.warning(f"Received path: \"{path}\" is not a folder path!")
        return []
    files = os.listdir(path)
    if not files:
        logging.warning(f"Received folder: \"{path}\" is empty!")
        return []

    if not extension:
        return [os.path.join(path, file) for file in files]
    else:
        return [
            os.path.join(path, file) for file in files
            if file.endswith("extension")
        ]


def read_text_from_files(text_files, encoding=None):
    if not encoding:
        return [
            open(file, 'r').read() for file in text_files
            if file.endswith(".txt")
        ]
    else:
        return [
            open(file, 'r', encoding=encoding).read() for file in text_files
            if file.endswith(".txt")
        ]
