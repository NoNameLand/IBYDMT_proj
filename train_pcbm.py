import argparse
import logging
import os

import configs
import datasets
from ibydmt.tester import get_test_classes
from ibydmt.utils.config import ConceptType, Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.config import get_config
from ibydmt.utils.pcbm import PCBM

logger = logging.getLogger(__name__)


def setup_logging(level):
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
    parser.add_argument("--concept_type", type=str)
    parser.add_argument(
        "--workdir", type=str, default=os.path.dirname(os.path.realpath(__file__))
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging")
    return parser.parse_args()


def train_pcbm(config: Config, concept_type: ConceptType, workdir: str = c.WORKDIR):
    logger.info(
        f"Training PCBM for dataset {config.data.dataset.lower()} and backbone"
        f" {config.data.backbone}"
    )
    test_classes = get_test_classes(config)

    for class_name in test_classes:
        concept_class_name = None
        if concept_type == ConceptType.CLASS.value:
            concept_class_name = class_name

        pcbm = PCBM(config, concept_class_name=concept_class_name)
        pcbm.train()
        pcbm.eval()
        pcbm.save(workdir)


def main(args):
    config_name = args.config_name
    concept_type = args.concept_type
    workdir = args.workdir
    log_level = args.log_level

    setup_logging(log_level)

    config: Config = get_config(config_name)
    for backbone_config in config.sweep(["data.backbone"]):
        train_pcbm(backbone_config, concept_type, workdir=workdir)


if __name__ == "__main__":
    args = config()
    main(args)
