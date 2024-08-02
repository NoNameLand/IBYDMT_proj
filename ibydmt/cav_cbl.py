import logging
import os
import pickle

from ibydmt.utils.concepts import get_concepts
from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c

logger = logging.getLogger(__name__)


class CAVCBL:
    def __init__(
        self,
        config: Config,
        workdir=c.WORKDIR,
        concept_class_name=None,
        concept_image_idx=None,
    ):
        self.config = config
