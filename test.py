import argparse
import os

import configs
import datasets
from ibydmt import ConceptTester


def test_type(value):
    if value not in ["global", "global_cond", "local_cond"]:
        raise argparse.ArgumentTypeError(f"Invalid test type: {value}")
    return value


def concept_type(value):
    if value not in ["dataset", "class", "image"]:
        raise argparse.ArgumentTypeError(f"Invalid concept type: {value}")
    return value


def setup_logging(level):
    import logging

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.root.setLevel(level)
    loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if "ibydmt" in name
    ]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--test_type", type=test_type, default="global")
    parser.add_argument("--concept_type", type=concept_type, default="dataset")
    parser.add_argument(
        "--workdir", type=str, default=os.path.dirname(os.path.realpath(__file__))
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging")
    return parser.parse_args()


def main(args):
    config_name = args.config_name
    test_type = args.test_type
    concept_type = args.concept_type
    workdir = args.workdir
    log_level = args.log_level

    setup_logging(log_level)

    tester = ConceptTester(config_name)
    tester.test(test_type, concept_type, workdir=workdir)


if __name__ == "__main__":
    args = config()
    main(args)
