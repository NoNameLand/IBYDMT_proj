import logging
import os
import pickle
from abc import abstractmethod
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import torch

from ibydmt.multimodal import get_text_encoder
from ibydmt.utils.concepts import get_concepts
from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.data import get_dataset, get_embedded_dataset

logger = logging.getLogger(__name__)


class ConceptBottleneck:
    def __init__(
        self,
        config: Config,
        concept_class_name: Optional[str] = None,
        concept_image_idx: Optional[None] = None,
        workdir=c.WORKDIR,
    ):
        self.config = config
        self.concept_name, self.concepts = get_concepts(
            config,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
            workdir=workdir,
        )

    @abstractmethod
    def encode_dataset(self, train: bool, workdir=c.WORKDIR, device=c.DEVICE):
        pass


bottlenecks: Mapping[str, ConceptBottleneck] = {}


def register_bottleneck(name):
    def register(cls: ConceptBottleneck):
        if name in bottlenecks:
            raise ValueError(f"Bottleneck {name} is already registered")
        bottlenecks[name] = cls

    return register


def get_bottleneck(
    config: Config,
    concept_class_name: Optional[str] = None,
    concept_image_idx: Optional[str] = None,
    workdir=c.WORKDIR,
) -> ConceptBottleneck:
    return bottlenecks[config.data.bottleneck](
        config,
        concept_class_name=concept_class_name,
        concept_image_idx=concept_image_idx,
        workdir=workdir,
    )


@register_bottleneck(name="zeroshot")
class ZeroShotBottleneck(ConceptBottleneck):
    def __init__(
        self,
        config: Config,
        concept_class_name: Optional[None] = None,
        concept_image_idx: Optional[None] = None,
        workdir=c.WORKDIR,
    ):
        super().__init__(
            config,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
            workdir=workdir,
        )

    def encode_dataset(self, train: bool, workdir=c.WORKDIR, device=c.DEVICE):
        dataset = get_embedded_dataset(self.config, train=train, workdir=workdir)

        state_path = self.state_path(workdir=workdir)
        cbm_exists = os.path.exists(state_path)
        if cbm_exists:
            with open(state_path, "rb") as f:
                concepts, _ = pickle.load(f)
            if concepts != self.concepts:
                cbm_exists = False

        if not cbm_exists:
            cbm = self.train(device=device)
            with open(self.state_path(workdir=workdir), "wb") as f:
                pickle.dump((self.concepts, cbm), f)

        with open(state_path, "rb") as f:
            _, cbm = pickle.load(f)

        return dataset.embedding @ cbm.T

    def state_path(self, workdir=c.WORKDIR):
        state_dir = os.path.join(workdir, "weights", self.config.name.lower())
        os.makedirs(state_dir, exist_ok=True)
        return os.path.join(
            state_dir, f"{self.config.backbone_name()}_cbm_{self.concept_name}.pkl"
        )

    @torch.no_grad()
    def train(self):
        logger.info(
            "Training zero-shot CBM for dataset"
            f" {self.config.data.dataset.lower()} with backbone ="
            f" {self.config.data.backbone} and concept_name = {self.concept_name}"
        )
        encode_text = get_text_encoder(self.config, device=self.device)

        cbm = encode_text(self.concepts).float()
        cbm /= torch.linalg.norm(cbm, dim=1, keepdim=True)
        return cbm.cpu().numpy()


@register_bottleneck(name="cub_attribute")
class CUBAttributeBottleneck(ConceptBottleneck):
    def __init__(
        self,
        config: Config,
        concept_class_name: Optional[str] = None,
        concept_image_idx: Optional[str] = None,
        workdir=c.WORKDIR,
    ):
        super().__init__(
            config,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
            workdir=workdir,
        )

        confident_image_attribute_label_path = os.path.join(
            workdir, "concept_data", "cub", "confident_image_attribute_labels.parquet"
        )
        attribute_df = pd.read_parquet(confident_image_attribute_label_path)
        attribute_df = attribute_df[attribute_df["attribute_name"].isin(self.concepts)]
        self.attribute_df = attribute_df

    def encode_dataset(self, train: bool, workdir=c.WORKDIR, device=c.DEVICE):
        dataset = get_dataset(self.config, train=train, workdir=workdir)

        semantics = -1 * np.ones(
            (len(dataset.image_idx), len(self.concepts)), dtype=int
        )
        image_df = self.attribute_df.groupby("image_idx")
        for group in image_df:
            image_idx = group[0]
            dataset_image_idx = dataset.image_idx.index(image_idx)

            for _, row in group[1].iterrows():
                concept_idx = self.concepts.index(row["attribute_name"])
                semantics[dataset_image_idx, concept_idx] = int(row["attribute_label"])

        return semantics
