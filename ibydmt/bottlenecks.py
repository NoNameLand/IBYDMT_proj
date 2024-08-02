import logging
import os
import pickle

import torch

from ibydmt.utils.concepts import get_concepts
from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.multimodal import get_text_encoder

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


class CAVBottleneck:
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

        with open(self.state_path(workdir=workdir), "rb") as f:
            attribute_idx, w, _ = pickle.load(f)
            attribute_idx = attribute_idx.tolist()

        good_attribute_path = os.path.join(
            workdir, "concepts", config.name.lower(), "good_attributes.txt"
        )
        with open(good_attribute_path, "r") as f:
            good_attribute = f.readlines()
        good_attribute_idx = [int(line.split()[0]) for line in good_attribute]

        self.w = w[[attribute_idx.index(_idx) for _idx in good_attribute_idx]]

    def state_path(self, workdir=c.WORKDIR):
        state_dir = os.path.join(workdir, "weights", self.config.name.lower())
        os.makedirs(state_dir, exist_ok=True)
        return os.path.join(
            workdir,
            "weights",
            self.config.name.lower(),
            f"{self.config.backbone_name()}_cav.pkl",
        )

    def __call__(self, h):
        return h @ self.w.T
