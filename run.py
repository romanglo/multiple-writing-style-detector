import getopt
import sys
import traceback

DEFAULT_VERBOSE = True

HELP = """\Multiple Writing Style Detection :
              --help        : help description
              --verbose     : 0 for minimum prints and 1 for more prints [default='%d']
           """ % (DEFAULT_VERBOSE)

HELP_ON_ERROR = "\nIncorrect command!\n" + HELP


def readArguments(argv):

    verbose = DEFAULT_VERBOSE

    try:
        opts, args = getopt.getopt(argv, None, ["help", "verbose="])
        for opt, arg in opts:
            if opt == "--help":
                print(HELP)
                sys.exit()
            elif opt == "--verbose":
                verbose = int(arg)
    except getopt.GetoptError:
        print(HELP_ON_ERROR)
        sys.exit()

    print("\nArguments:")
    if verbose == 1:
        print("verbose = true")
    else:
        print("verbose = false")

    return verbose


def main(argv):
    verbose = readArguments(argv)
    try:
        pass
    except KeyboardInterrupt:
        print("\n\nProcess aborted by the user!")
    except Exception as e:
        print(
        )  # There is a chance that there was a exception in the middle of progress bar
        print("Some error occurred during the running! Process aborted..")
        if (verbose):
            print("\nError:", str(e))
            traceback.print_tb(e.__traceback__)
        else:
            print(
                "For more details, it is recommended to run with the verbose on option."
            )


# Run the program
if __name__ == "__main__":
    main(sys.argv[1:])
