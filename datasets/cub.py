import logging
import os
from typing import List, Mapping

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import find_classes, make_dataset
from tqdm import tqdm

from ibydmt.utils.concepts import register_image_concept_trainer
from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.data import get_dataset, register_dataset

logger = logging.getLogger(__name__)

rng = np.random.default_rng()


@register_dataset(name="cub")
class CUB(VisionDataset):
    def __init__(self, root, train=None, transform=None):
        super().__init__(root, transform=transform)
        self.op = "all"

        image_root = os.path.join(root, "CUB", "images")
        wnids, wnid_to_idx = find_classes(image_root)
        self.classes = [wnid.split(".")[1].replace("_", " ") for wnid in wnids]

        with open(os.path.join(root, "CUB", "images.txt"), "r") as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
            filename_to_idx = {filename: int(idx) for idx, filename in lines}

        self.samples = make_dataset(image_root, wnid_to_idx, extensions=".jpg")
        self.image_idx = [
            filename_to_idx["/".join(filename.split("/")[-2:])]
            for filename, _ in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def validate_class_idx_for_testing(
        self, config: Config, class_idx: Mapping[str, List[int]], workdir=c.WORKDIR
    ):
        dataset_idx_to_image_idx = self.image_idx

        attribute_df = get_confident_image_attribute_labels(config, workdir=workdir)

        def validate_image_attribute_count(image_group):
            n_positive = n_negative = config.data.num_concepts // 2

            image_n_negative = sum(image_group == 0)
            image_n_positive = sum(image_group == 1)
            return image_n_negative >= n_negative and image_n_positive >= n_positive

        attribute_df["valid"] = attribute_df.groupby("image_idx")[
            "attribute_label"
        ].transform(validate_image_attribute_count)
        attribute_df = attribute_df[attribute_df["valid"]]
        valid_image_idx = attribute_df["image_idx"].unique()

        return {
            class_name: list(
                filter(
                    lambda idx: dataset_idx_to_image_idx[idx] in valid_image_idx,
                    _class_idx,
                ),
            )
            for class_name, _class_idx in class_idx.items()
        }


def get_confident_image_attribute_labels(config: Config, workdir=c.WORKDIR):
    confident_image_attribute_label_path = os.path.join(
        workdir, "concept_data", "cub", "confident_image_attribute_labels.parquet"
    )
    if not os.path.exists(confident_image_attribute_label_path):
        logging.info("Retrieving confident image attribute labels")
        attribute_dir = os.path.join(workdir, "data", "CUB", "attributes")
        attribute_path = os.path.join(attribute_dir, "attributes.txt")
        image_attribute_labels_path = os.path.join(
            attribute_dir, "image_attribute_labels.txt"
        )

        with open(attribute_path, "r") as f:
            attribute_idx_to_name = {
                int(line.split()[0]): line.split()[1] for line in f
            }

        dataset = get_dataset(config, workdir=workdir)
        classes = dataset.classes
        image_idx_to_dataset_idx = {
            image_idx: idx for idx, image_idx in enumerate(dataset.image_idx)
        }

        image_idx = []
        class_name = []
        attribute_idx = []
        attribute_name = []
        attribute_label = []
        with open(image_attribute_labels_path, "r") as f:
            for line in tqdm(f):
                chunks = line.strip().split()
                # malformed line
                if len(chunks) != 5:
                    continue

                _image_idx, _attribute_idx, _attribute_label, _confidence, _ = chunks
                if int(_confidence) >= 3:
                    _image_idx = int(_image_idx)

                    _dataset_idx = image_idx_to_dataset_idx[_image_idx]
                    _, _label = dataset.samples[_dataset_idx]
                    _class_name = classes[_label]

                    _attribute_idx = int(_attribute_idx)
                    _attribute_name = attribute_idx_to_name[_attribute_idx]
                    _attribute_label = int(_attribute_label)

                    image_idx.append(_image_idx)
                    class_name.append(_class_name)
                    attribute_idx.append(_attribute_idx)
                    attribute_name.append(_attribute_name)
                    attribute_label.append(_attribute_label)

        df = pd.DataFrame(
            {
                "image_idx": image_idx,
                "class_name": class_name,
                "attribute_idx": attribute_idx,
                "attribute_name": attribute_name,
                "attribute_label": attribute_label,
            }
        )
        df.to_parquet(confident_image_attribute_label_path)
    return pd.read_parquet(confident_image_attribute_label_path)


@register_image_concept_trainer(name="cub")
def train_image_concepts(
    config: Config, concept_image_idx: int, workdir=c.WORKDIR, device=c.DEVICE
):
    dataset = get_dataset(config, train=False)
    dataset_idx_to_image_idx = dataset.image_idx

    attribute_df = get_confident_image_attribute_labels(config, workdir=workdir)
    attribute_frequency = attribute_df.groupby("attribute_name")[
        "attribute_label"
    ].mean()
    attribute_class_frequency = attribute_df.groupby(["class_name", "attribute_name"])[
        "attribute_label"
    ].mean()

    image_idx = dataset_idx_to_image_idx[concept_image_idx]
    _, label = dataset.samples[concept_image_idx]
    class_name = dataset.classes[label]

    _attribute_class_frequency = attribute_class_frequency.loc[class_name]

    image_df = attribute_df[attribute_df["image_idx"] == image_idx]
    positive_attribute = image_df[image_df["attribute_label"] == 1][
        "attribute_name"
    ].values
    negative_attribute = image_df[image_df["attribute_label"] == 0][
        "attribute_name"
    ].values

    positive_attribute_score = [
        _attribute_class_frequency.loc[attribute] - attribute_frequency.loc[attribute]
        for attribute in positive_attribute
    ]
    sorted_idx = np.argsort(positive_attribute_score)[::-1]
    positive_attribute = positive_attribute[sorted_idx]

    n = config.data.num_concepts // 2
    positive_attribute = positive_attribute[:n].tolist()
    negative_attribute = rng.choice(negative_attribute, n, replace=False).tolist()
    return positive_attribute + negative_attribute
