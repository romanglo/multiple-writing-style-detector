# str_helper.py

import re
import string


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
