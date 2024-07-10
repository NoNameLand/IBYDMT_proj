import logging
import os

from ibydmt.utils.config import Config
from ibydmt.utils.constants import device, workdir
from ibydmt.utils.splice import (
    train_class_concepts,
    train_dataset_concepts,
    train_image_concepts,
)

logger = logging.getLogger(__name__)


def get_concept_name(concept_class_name=None, concept_image_idx=None):
    concept_name = "dataset"
    if concept_class_name is not None:
        concept_name = concept_class_name.lower().replace(" ", "_")
    elif concept_image_idx is not None:
        concept_name = f"image_{concept_image_idx}"
    return concept_name


def get_concept_path(
    config: Config,
    workdir: str = workdir,
    concept_class_name: str = None,
    concept_image_idx: str = None,
):
    concept_name = get_concept_name(
        concept_class_name=concept_class_name, concept_image_idx=concept_image_idx
    )
    concept_dir = os.path.join(workdir, "concepts", config.name.lower())
    return os.path.join(concept_dir, f"{concept_name}.txt")


def get_concepts(
    config: Config,
    workdir: str = workdir,
    concept_class_name: str = None,
    concept_image_idx: str = None,
):
    concept_name = get_concept_name(
        concept_class_name=concept_class_name, concept_image_idx=concept_image_idx
    )
    concept_path = get_concept_path(
        config,
        workdir=workdir,
        concept_class_name=concept_class_name,
        concept_image_idx=concept_image_idx,
    )
    if not os.path.exists(concept_path):
        train_concepts(
            config,
            workdir=workdir,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
        )

    with open(concept_path, "r") as f:
        concepts = f.read().splitlines()

    return concept_name, concepts


def train_concepts(
    config: Config,
    workdir: str = workdir,
    concept_class_name: str = None,
    concept_image_idx: str = None,
    device=device,
):
    logger.info(
        f"Training concepts for dataset {config.data.dataset.lower()},"
        f" concept_class_name = {concept_class_name},"
        f" concept_image_idx = {concept_image_idx}"
    )
    if concept_class_name is not None:
        concepts = train_class_concepts(
            config, concept_class_name, workdir=workdir, device=device
        )
    elif concept_image_idx is not None:
        concepts = train_image_concepts(
            config, concept_image_idx, workdir=workdir, device=device
        )
    else:
        concepts = train_dataset_concepts(config, workdir=workdir, device=device)

    concept_path = get_concept_path(
        config,
        workdir=workdir,
        concept_class_name=concept_class_name,
        concept_image_idx=concept_image_idx,
    )
    os.makedirs(os.path.dirname(concept_path), exist_ok=True)
    with open(concept_path, "w") as f:
        for concept in concepts:
            f.write(f"{concept}\n")
