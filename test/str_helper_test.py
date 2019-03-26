# str_util_test.py

import unittest

import mwsd.str_helper as StrHelper


class StrUtils_Test(unittest.TestCase):
    def test_removeMultipleSpaces(self):
        inStr = "   Hello  world 1234  "
        outStr = StrHelper.removeMultipleSpaces(inStr)
        expectedStr = " Hello world 1234 "
        self.assertEqual(
            outStr, expectedStr,
            "The strings should be equal after calling removeMultipleSpaces() method"
        )

    def test_removeChars(self):
        inStr = "Hello world 1234"
        outStr = StrHelper.removeChars(inStr, "o24")
        expectedStr = "Hell wrld 13"
        self.assertEqual(
            outStr, expectedStr,
            "The strings should be equal after calling removeChars() method")


if __name__ == '__main__':
    unittest.main()
