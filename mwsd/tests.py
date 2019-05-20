import importlib
import os
import traceback
import unittest

from mwsd import *

TEST_FILE_EXTENSION = "_test.py"
TEST_FOLDER_NAME = "test"


def checkDependencies():
    filepath = "requirements.txt"

    print("\nChecking project dependencies from", filepath, ":")

    with open(filepath) as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            dep = line[:line.find("=")]
            print("importing", dep, "..")
            importlib.import_module(dep)
            print("imported!")
            line = fp.readline()

    print("All dependencies exists!")


def runUnittests():

    testmodules = []

    for path, subdirs, files in os.walk("""./%s/""" % (TEST_FOLDER_NAME)):
        for name in files:
            if (name.endswith(TEST_FILE_EXTENSION)):
                testmodules.append(
                    """%s.%s""" % (TEST_FOLDER_NAME, name.replace(".py", "")))

    suite = unittest.TestSuite()

    for t in testmodules:
        try:
            # If the module defines a suite() function, call it to get the suite.
            mod = __import__(t, globals(), locals(), ["suite"])
            suitefunction = getattr(mod, "suite")
            suite.addTest(suitefunction())
        except (ImportError, AttributeError):
            # else, just load all the test cases from the module.
            suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

    unittest.TextTestRunner().run(suite)


# Run the program
if __name__ == "__main__":
    try:
        checkDependencies()
        runUnittests()
    except SystemExit:
        pass
    except Exception as e:
        print("Some error occurred during the running! Process aborted..")
        print("\nError:", str(e))
        traceback.print_tb(e.__traceback__)
