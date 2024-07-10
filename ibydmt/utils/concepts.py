import os

from ibydmt.utils.constants import workdir
from ibydmt.utils.splice import (
    train_class_concepts,
    train_dataset_concepts,
    train_image_concepts,
)


def _one_or_neither(*args):
    return sum([1 if arg is not None else 0 for arg in args]) <= 1


def get_concept_name(concept_class_name=None, concept_image_idx=None):
    assert _one_or_neither(concept_class_name, concept_image_idx), ValueError(
        "Only one of `concept_class_name` or `concept_image_idx` should be provided."
    )

    concept_name = "dataset"
    if concept_class_name is not None:
        concept_name = concept_class_name.lower().replace(" ", "_")
    elif concept_image_idx is not None:
        concept_name = f"image_{concept_image_idx}"
    return concept_name


def get_concept_path(
    config, workdir=workdir, concept_class_name=None, concept_image_idx=None
):
    concept_name = get_concept_name(
        concept_class_name=concept_class_name, concept_image_idx=concept_image_idx
    )
    concept_dir = os.path.join(workdir, "concepts", config.name.lower())
    return os.path.join(concept_dir, f"{concept_name}.txt")


def get_concepts(
    config, workdir=workdir, concept_class_name=None, concept_image_idx=None
):
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


def train_concepts(
    config, workdir=workdir, concept_class_name=None, concept_image_idx=None
):
    if concept_class_name is not None:
        concepts = train_class_concepts(config, concept_class_name)
    elif concept_image_idx is not None:
        concepts = train_image_concepts(config, concept_image_idx)
    else:
        concepts = train_dataset_concepts(config)

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
