import logging
import os
import re

from langdetect import detect
from nltk import download
from nltk.corpus import stopwords
from smart_open import open

UNSUPPORTED_LANGAUGE = "unsupported"


def ISO_639_1_codes_to_nltk_codes(code):
    switcher = {
        "ar": "arabic",
        "az": "azerbaijani",
        "bn": "danish",
        "en": "english",
        "fi": "finnish",
        "fr": "french",
        "de": "german",  # "dutch"
        "el": "greek",
        "hu": "hungarian",
        "id": "indonesian",
        "kk": "kazakh",
        "ne": "nepali",
        "no": "norwegian",
        "pt": "portuguese",
        "ro": "romanian",
        "ru": "russian",
        "es": "spanish",
        "sv": "swedish",
        "tr": "turkish"
    }
    return switcher.get(code, UNSUPPORTED_LANGAUGE)


def detect_language(text):
    try:
        return detect(text)
    except Exception:
        logging.exception(
            f"Failed on try to detect the language of the text: {text}")


def split_to_chunks(iterable, chunk_size):
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
