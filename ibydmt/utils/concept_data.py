import logging
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ibydmt.bottlenecks import AttributeBottleneck, ZeroShotBottleneck
from ibydmt.utils.concepts import get_concepts
from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.data import get_dataset, get_embedded_dataset

logger = logging.getLogger(__name__)


def get_dataset_with_concepts(
    config: Config,
    train=True,
    concept_class_name=None,
    concept_image_idx=None,
    workdir=c.WORKDIR,
):
    return DatasetWithConcepts(
        config,
        train=train,
        concept_class_name=concept_class_name,
        concept_image_idx=concept_image_idx,
        workdir=workdir,
    )


class DatasetWithConcepts(Dataset):
    def __init__(
        self,
        config: Config,
        train=True,
        concept_class_name=None,
        concept_image_idx=None,
        workdir=c.WORKDIR,
    ):
        super().__init__()
        dataset = get_embedded_dataset(config, train=train, workdir=workdir)
        self.classes = dataset.classes
        self.concept_name, self.concepts = get_concepts(
            config,
            workdir=workdir,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
        )

        op = dataset.op
        root = os.path.join(workdir, "concept_data")
        data_dir = os.path.join(root, config.data.dataset.lower())
        data_path = os.path.join(
            data_dir, f"{op}_{config.backbone_name()}_{self.concept_name}.npy"
        )
        if not os.path.exists(data_path):
            os.makedirs(data_dir, exist_ok=True)

            semantics = project_dataset_with_concepts(
                config,
                train=train,
                concept_class_name=concept_class_name,
                concept_image_idx=concept_image_idx,
                workdir=workdir,
            )
            np.save(data_path, semantics)

        self.embedding = dataset.embedding
        self.semantics = np.load(data_path)
        self.label = dataset.label

    def __len__(self):
        return self.semantics.shape[0]


def project_dataset_with_concepts(
    config: Config,
    train=True,
    concept_class_name=None,
    concept_image_idx=None,
    workdir=c.WORKDIR,
):
    logger.info(
        f"Projecting dataset {config.data.dataset.lower()} (train = {train}) with"
        f" backbone = {config.data.backbone},"
        f" concept_class_name = {concept_class_name},"
        f" concept_image_idx = {concept_image_idx}"
    )

    if config.data.bottleneck_type == "zero_shot":
        dataset = get_embedded_dataset(config, train=train, workdir=workdir)

        concept_bottleneck = ZeroShotBottleneck.load_or_train(
            config,
            workdir=workdir,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
        )
        semantics = concept_bottleneck(dataset.embedding)
    # elif config.data.bottleneck_type == "cav":
    #     dataset = get_embedded_dataset(config, train=train, workdir=workdir)

    #     concept_bottleneck = CAVBottleneck(
    #         config,
    #         workdir=workdir,
    #         concept_class_name=concept_class_name,
    #         concept_image_idx=concept_image_idx,
    #     )
    #     semantics = concept_bottleneck(dataset.embedding)
    elif config.data.bottleneck_type == "attribute":
        dataset = get_dataset(config, train=train, workdir=workdir)

        concept_bottleneck = AttributeBottleneck(
            config,
            workdir=workdir,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
        )
        semantics = concept_bottleneck(dataset)
    return semantics
