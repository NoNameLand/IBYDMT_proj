import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch

from ibydmt.multimodal import get_text_encoder
from ibydmt.utils.concepts import get_concepts
from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c

logger = logging.getLogger(__name__)


class ZeroShotBottleneck:
    def __init__(
        self,
        config: Config,
        workdir=c.WORKDIR,
        concept_class_name=None,
        concept_image_idx=None,
    ):
        self.config = config

        self.concept_name, self.concepts = get_concepts(
            config,
            workdir=workdir,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
        )
        self.cbm = None

    def state_path(self, workdir=c.WORKDIR):
        state_dir = os.path.join(workdir, "weights", self.config.name.lower())
        os.makedirs(state_dir, exist_ok=True)
        return os.path.join(
            state_dir, f"{self.config.backbone_name()}_cbm_{self.concept_name}.pkl"
        )

    @staticmethod
    def load_or_train(
        config: Config,
        workdir=c.WORKDIR,
        concept_class_name=None,
        concept_image_idx=None,
        device=c.DEVICE,
    ):
        model = ZeroShotBottleneck(
            config,
            workdir=workdir,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
        )
        state_path = model.state_path(workdir=workdir)

        cbm_exists = os.path.exists(state_path)
        if cbm_exists:
            with open(state_path, "rb") as f:
                concepts, _ = pickle.load(f)

            if concepts != model.concepts:
                cbm_exists = False

        if cbm_exists:
            with open(state_path, "rb") as f:
                _, cbm = pickle.load(f)
            model.cbm = cbm
        else:
            model.train(device=device)
            model.save(workdir)
        return model

    def save(self, workdir=c.WORKDIR):
        with open(self.state_path(workdir=workdir), "wb") as f:
            pickle.dump((self.concepts, self.cbm), f)

    def __call__(self, h):
        return h @ self.cbm.T

    @torch.no_grad()
    def train(self, device=c.DEVICE):
        logger.info(
            "Training zero-shot CBM for dataset"
            f" {self.config.data.dataset.lower()} with backbone ="
            f" {self.config.data.backbone} and concept_name = {self.concept_name}"
        )
        encode_text = get_text_encoder(self.config, device=device)

        cbm = encode_text(self.concepts).float()
        cbm /= torch.linalg.norm(cbm, dim=1, keepdim=True)
        cbm = cbm.cpu().numpy()

        self.cbm = cbm


class AttributeBottleneck:
    def __init__(
        self,
        config: Config,
        workdir=c.WORKDIR,
        concept_class_name=None,
        concept_image_idx=None,
    ):
        self.config = config

        self.concept_name, self.concepts = get_concepts(
            config,
            workdir=workdir,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
        )

        confident_image_attribute_label_path = os.path.join(
            workdir,
            "concept_data",
            config.name.lower(),
            "confident_image_attribute_labels.parquet",
        )
        df = pd.read_parquet(confident_image_attribute_label_path)
        concept_df = df[df["attribute_name"].isin(self.concepts)]
        self.concept_df = concept_df

    def __call__(self, dataset):
        image_df = self.concept_df.groupby("image_idx")

        semantics = -1 * np.ones(
            (len(dataset.image_idx), len(self.concepts)), dtype=int
        )
        for group in image_df:
            image_idx = group[0]
            dataset_image_idx = dataset.image_idx.index(image_idx)

            for _, row in group[1].iterrows():
                concept_idx = self.concepts.index(row["attribute_name"])
                semantics[dataset_image_idx, concept_idx] = int(row["attribute_label"])

        return semantics
