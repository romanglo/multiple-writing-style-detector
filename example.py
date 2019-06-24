from __future__ import unicode_literals

import argparse
import logging
import os
import sys
import codecs
import json

import mwsd
from mwsd.visualize import visualize

DEFAULT_VERBOSE = True
DEFAULT_SAVE_OUTPUT = True


def read_arguments(argv):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "-fi",
        "--first_input",
        type=str,
        required=True,
        help="First text path")
    ap.add_argument(
        "-si",
        "--second_input",
        type=str,
        required=True,
        help="Second text path")
    ap.add_argument(
        "-so",
        "--save_output",
        type=bool,
        default=DEFAULT_SAVE_OUTPUT,
        required=False,
        help="Save the results to execution folder")
    ap.add_argument(
        "-v",
        "--verbose",
        type=bool,
        default=DEFAULT_VERBOSE,
        required=False,
        help="Verbose logging")

    args, _ = ap.parse_known_args()
    argsDict = vars(args)

    return argsDict


def initialize_logging_config(logging_level):
    logging.basicConfig(
        format=
        u'%(asctime)s %(name)-7s [%(filename)10s:%(lineno)3s - %(funcName)20s()] %(levelname)-8s:: %(message)s.',
        level=logging_level)


def process(first_input, second_input, save_output):
    mwsd.initialize()

    if not os.path.isfile(first_input):
        raise IOError("First input file does not exist: {}".format(first_input))
    if not os.path.isfile(second_input):
        raise IOError("Second input file does not exist: {}".format(second_input))

    first_text, second_text = mwsd.utils.read_text_from_files(
        [first_input, second_input], encoding='utf-8')

    ZV, DZV, medoids = mwsd.execute(first_text, second_text)

    plot_saving_path = 'mwsd_result.png' if save_output else None

    visualize(
        ZV, DZV, medoids, show_plot=True, plot_saving_path=plot_saving_path)

    if (save_output):
        result = {
            'zv': ZV.tolist(),
            'dzv': DZV.tolist(),
            'medoids': [{
                'elements': [element.tolist() for element in medoid.elements],
                'kernal': medoid.kernel.tolist()
            } for medoid in medoids]
        }

        json.dump(
            result,
            codecs.open('mwsd_result.json', 'w', encoding='utf-8'),
            sort_keys=True,
            indent=4)


def main(args):
    try:
        initialize_logging_config(logging.
                                  DEBUG if args['verbose'] else logging.INFO)
        process(args['first_input'], args['second_input'], args['save_output'])
    except KeyboardInterrupt:
        logging.info("Process aborted by the user!")
    except Exception:
        logging.exception(
            "Some error occurred during the running! Process aborted..")


if __name__ == "__main__":
    args = read_arguments(sys.argv[1:])
    main(args)
