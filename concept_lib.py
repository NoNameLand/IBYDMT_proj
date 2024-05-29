import os

from splice_lib import (
    train_class_concepts,
    train_dataset_concepts,
    train_image_concepts,
)

workdir = os.path.dirname(os.path.realpath(__file__))


def _one_or_neither(*args):
    return sum([1 if arg is not None else 0 for arg in args]) <= 1


def get_concept_name(concept_class_name=None, concept_image_idx=None):
    assert _one_or_neither(concept_class_name, concept_image_idx), ValueError(
        "Only one of concept_class_name or concept_image_idx should be provided."
    )

    concept_name = "dataset"
    if concept_class_name is not None:
        concept_name = concept_class_name.lower().replace(" ", "_")
    elif concept_image_idx is not None:
        concept_name = f"image_{concept_image_idx}"
    return concept_name


def get_concepts(config, concept_class_name=None, concept_image_idx=None):
    concept_name = get_concept_name(
        concept_class_name=concept_class_name, concept_image_idx=concept_image_idx
    )

    concept_dir = os.path.join(workdir, "concepts", config.name.lower())

    concept_path = os.path.join(concept_dir, f"{concept_name}.txt")
    if not os.path.exists(concept_path):
        train_concepts(
            config,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
        )

    with open(concept_path, "r") as f:
        concepts = f.read().splitlines()

    return concept_name, concepts


def train_concepts(config, concept_class_name=None, concept_image_idx=None):
    concept_name = get_concept_name(
        concept_class_name=concept_class_name, concept_image_idx=concept_image_idx
    )

    concept_dir = os.path.join(workdir, "concepts", config.name.lower())
    os.makedirs(concept_dir, exist_ok=True)

    if concept_class_name is not None:
        if concept_class_name == "user":
            raise NotImplementedError("User-defined concepts are not yet supported.")
        else:
            concepts = train_class_concepts(config, concept_class_name)
    elif concept_image_idx is not None:
        concepts = train_image_concepts(config, concept_image_idx)
    else:
        concepts = train_dataset_concepts(config)

    concept_path = os.path.join(concept_dir, f"{concept_name}.txt")
    with open(concept_path, "w") as f:
        for concept in concepts:
            f.write(f"{concept}\n")
