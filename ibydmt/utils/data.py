import os

import torch
from torch.utils.data import Dataset

from ibydmt.utils.concepts import get_concepts
from ibydmt.utils.constants import device, workdir

datasets = {}


def register_dataset(name):
    def register(cls):
        if name in datasets:
            raise ValueError(f"Dataset {name} is already registered")
        datasets[name] = cls

    return register


def get_dataset(config, workdir=workdir, train=True, transform=None):
    name = config.data.dataset.lower()
    root = os.path.join(workdir, "data")
    return datasets[name](root, train=train, transform=transform)


def get_dataset_with_concepts(
    config,
    workdir=workdir,
    train=True,
    transform=None,
    concept_class_name=None,
    concept_image_idx=None,
    return_image=False,
):
    return DatasetWithConcepts(
        config,
        workdir=workdir,
        train=train,
        transform=transform,
        return_image=return_image,
        concept_class_name=concept_class_name,
        concept_image_idx=concept_image_idx,
    )


@torch.no_grad()
def project_dataset_with_concepts(
    config,
    workdir=workdir,
    concept_class_name=None,
    concept_image_idx=None,
    device=device,
):
    pass


class DatasetWithConcepts(Dataset):
    def __init__(
        self,
        config,
        workdir=workdir,
        train=True,
        transform=None,
        concept_class_name=None,
        concept_image_idx=None,
        return_image=False,
    ):
        super().__init__()
        self.dataset = dataset = get_dataset(
            config, workdir=workdir, train=train, transform=transform
        )
        self.root = dataset.root
        self.train = dataset.train
        self.classes = dataset.classes
        self.concept_name, self.concepts = get_concepts(
            config,
            workdir=workdir,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
        )
        self.return_image = return_image
