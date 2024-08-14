import logging
import os
from typing import List, Mapping, Optional, Protocol

from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c

logger = logging.getLogger(__name__)


class DatasetConceptTrainer(Protocol):
    def __call__(self, config: Config, workdir=c.WORKDIR, device=c.DEVICE) -> List[str]:
        ...


class ClassConceptTrainer(Protocol):
    def __call__(
        self,
        config: Config,
        concept_class_name: str,
        workdir: str = c.WORKDIR,
        device: str = c.DEVICE,
    ) -> List[str]:
        ...


class ImageConceptTrainer(Protocol):
    def __call__(
        self,
        config: Config,
        concept_image_idx: int,
        workdir: str = c.WORKDIR,
        device: str = c.DEVICE,
    ) -> List[str]:
        ...


dataset_concept_trainers: Mapping[str, DatasetConceptTrainer] = {}
class_concept_trainers: Mapping[str, ClassConceptTrainer] = {}
image_concept_trainers: Mapping[str, ImageConceptTrainer] = {}


def register_dataset_concept_trainer(name):
    def register(f: DatasetConceptTrainer):
        dataset_concept_trainers[name] = f

    return register


def register_class_concept_trainer(name):
    def register(f: ClassConceptTrainer):
        class_concept_trainers[name] = f

    return register


def register_image_concept_trainer(name):
    def register(f: ImageConceptTrainer):
        image_concept_trainers[name] = f

    return register


def get_concept_trainer(
    config: Config,
    concept_class_name: Optional[str] = None,
    concept_image_idx: Optional[int] = None,
):
    registrar = dataset_concept_trainers
    if concept_class_name is not None:
        registrar = class_concept_trainers
    if concept_image_idx is not None:
        registrar = image_concept_trainers

    return registrar[config.data.dataset.lower()]


def get_concept_name(
    concept_class_name: Optional[str] = None, concept_image_idx: Optional[int] = None
):
    concept_name = "dataset"
    if concept_class_name is not None:
        concept_name = concept_class_name.lower().replace(" ", "_")
    elif concept_image_idx is not None:
        concept_name = f"image_{concept_image_idx}"
    return concept_name


def get_concept_path(
    config: Config,
    workdir: str = c.WORKDIR,
    concept_class_name: Optional[str] = None,
    concept_image_idx: Optional[int] = None,
):
    concept_name = get_concept_name(
        concept_class_name=concept_class_name, concept_image_idx=concept_image_idx
    )
    concept_dir = os.path.join(workdir, "concepts", config.name.lower())
    return os.path.join(concept_dir, f"{concept_name}.txt")


def get_concepts(
    config: Config,
    workdir: str = c.WORKDIR,
    concept_class_name: Optional[str] = None,
    concept_image_idx: Optional[int] = None,
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
    concept_class_name: Optional[str] = None,
    concept_image_idx: Optional[int] = None,
    workdir: str = c.WORKDIR,
    device=c.DEVICE,
):
    logger.info(
        f"Training concepts for dataset {config.data.dataset.lower()},"
        f" concept_class_name = {concept_class_name},"
        f" concept_image_idx = {concept_image_idx}"
    )
    concept_trainer = get_concept_trainer(
        config,
        concept_class_name=concept_class_name,
        concept_image_idx=concept_image_idx,
    )
    if isinstance(concept_trainer, DatasetConceptTrainer):
        concepts = concept_trainer(config, workdir=workdir, device=device)
    if isinstance(concept_trainer, ClassConceptTrainer):
        concepts = concept_trainer(
            config, concept_class_name, workdir=workdir, device=device
        )
    if isinstance(concept_trainer, ImageConceptTrainer):
        concepts = concept_trainer(
            config, concept_image_idx, workdir=workdir, device=device
        )

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
