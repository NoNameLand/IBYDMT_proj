import logging
import os
import socket

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.multimodal import get_image_encoder

logger = logging.getLogger(__name__)

datasets = {}


def register_dataset(name):
    def register(cls):
        if name in datasets:
            raise ValueError(f"Dataset {name} is already registered")
        datasets[name] = cls

    return register


def get_dataset(config, train=True, transform=None, workdir=c.WORKDIR):
    name = config.data.dataset.lower()
    root = os.path.join(workdir, "data")
    hostname = socket.gethostname()
    if hostname == "io85":
        root = os.path.join(root, hostname)
    return datasets[name](root, train=train, transform=transform)


def get_embedded_dataset(config, train=True, workdir=c.WORKDIR):
    return EmbeddedDataset(config, train=train, workdir=workdir)


class EmbeddedDataset(Dataset):
    def __init__(self, config: Config, train=True, workdir=c.WORKDIR):
        super().__init__()
        dataset = get_dataset(config, train=train, workdir=workdir)
        self.op = dataset.op
        self.classes = dataset.classes

        root = os.path.join(workdir, "concept_data")
        data_dir = os.path.join(root, config.data.dataset.lower())

        data_path = os.path.join(
            data_dir, f"{self.op}_{config.backbone_name()}.parquet"
        )
        if not os.path.exists(data_path):
            os.makedirs(data_dir, exist_ok=True)

            embedding, label = embed_dataset(config, train=train, workdir=workdir)

            df = pd.DataFrame({"embedding": embedding, "label": label})
            df.to_parquet(data_path)

        self.data = pd.read_parquet(data_path)
        self.embedding = np.vstack(self.data["embedding"].values)
        self.label = np.stack(self.data["label"].values)


@torch.no_grad()
def embed_dataset(config: Config, train=True, workdir=c.WORKDIR, device=c.DEVICE):
    logger.info(
        f"Encoding dataset {config.data.dataset.lower()} (train = {train}) with"
        f" backbone = {config.data.backbone}"
    )

    dataset = get_dataset(config, train=train, workdir=workdir)
    encode_image = get_image_encoder(config, device=device)

    embedding, label = [], []
    for image, target in tqdm(dataset):
        h = encode_image(image).float()
        h /= torch.linalg.norm(h, dim=1, keepdim=True)
        h = h.cpu().numpy()

        embedding.extend(h)
        label.append(target)

    return embedding, label
