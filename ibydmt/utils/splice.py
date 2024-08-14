import logging

import clip
import numpy as np
import splice
import torch
from nltk.stem import WordNetLemmatizer
from torch.utils.data import DataLoader

from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.data import get_dataset

logger = logging.getLogger(__name__)
rng = np.random.default_rng()

try:
    lem = WordNetLemmatizer()
except:
    print(
        'Please download the WordNet data by running `nltk.download("wordnet");'
        ' nltk.download("omw-1.4")`'
    )


def _lemmatize(concept):
    concept = concept.strip()
    subwords = concept.split(" ")
    return [lem.lemmatize(word) for word in subwords]


def _select_concepts(config: Config, weights, classes):
    num_concepts = config.data.num_concepts

    vocabulary = splice.get_vocabulary(config.splice.vocab, config.splice.vocab_size)
    sorted_indices = torch.argsort(weights, descending=True)

    class_lemmas = []
    for class_name in classes:
        class_lemmas.extend(_lemmatize(class_name))
    class_lemmas = set(class_lemmas)

    concepts = set()
    for idx in sorted_indices:
        concept = vocabulary[idx.item()]
        concept_lemmas = _lemmatize(concept)

        contained = any(
            [
                lemma in class_lemma
                for lemma in concept_lemmas
                for class_lemma in class_lemmas
            ]
        )
        contains = any(
            [
                class_lemma in lemma
                for lemma in concept_lemmas
                for class_lemma in class_lemmas
            ]
        )
        if contained or contains:
            logger.debug(f"Skipping concept: {concept}")
            continue

        concepts.add(" ".join(concept_lemmas))
        if len(concepts) == num_concepts:
            break

    return list(concepts)


def _preamble(config: Config, workdir, train, device):
    _, preprocess = clip.load(config.data.backbone, device=device)
    dataset = get_dataset(config, workdir=workdir, train=train, transform=preprocess)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    return (
        dataset,
        loader,
        splice.load(
            f"clip:{config.data.backbone}",
            config.splice.vocab,
            config.splice.vocab_size,
            l1_penalty=config.splice.l1_penalty,
            return_weights=True,
        ),
    )


def train_dataset_concepts(config: Config, workdir=c.WORKDIR, device=c.DEVICE):
    dataset, loader, splicemodel = _preamble(config, workdir, True, device)
    classes = dataset.classes
    weights, l0_norm, cosine = splice.decompose_dataset(
        loader, splicemodel=splicemodel, device=device
    )
    logger.info(f"Average SpLiCe Decomposition L0 Norm: {l0_norm:.0f}")
    logger.info(f"Average CLIP, SpLiCe Cosine Similarity: {cosine:.4f}")
    return _select_concepts(config, weights, classes)


def train_class_concepts(
    config: Config, concept_class_name: str, workdir=c.WORKDIR, device=c.DEVICE
):
    dataset, loader, splicemodel = _preamble(config, workdir, True, device)
    classes = dataset.classes
    assert concept_class_name in classes, ValueError(
        f"{concept_class_name} not in {classes}"
    )
    label = classes.index(concept_class_name)
    weights, l0_norm, cosine = splice.decompose_classes(
        loader, label, splicemodel=splicemodel, device=device
    )
    logger.info(f"Average SpLiCe Decomposition L0 Norm: {l0_norm:.0f}")
    logger.info(f"Average CLIP, SpLiCe Cosine Similarity: {cosine:.4f}")
    return _select_concepts(config, weights[label], classes)


def train_image_concepts(
    config: Config, concept_image_idx: int, workdir=c.WORKDIR, device=c.DEVICE
):
    dataset, _, splicemodel = _preamble(config, workdir, False, device)
    image, label = dataset[concept_image_idx]
    weights, l0_norm, cosine = splice.decompose_image(
        image.unsqueeze(0), splicemodel=splicemodel, device=device
    )
    logger.info(f"Average SpLiCe Decomposition L0 Norm: {l0_norm:.0f}")
    logger.info(f"Average CLIP, SpLiCe Cosine Similarity: {cosine:.4f}")
    return _select_concepts(config, weights.squeeze(), [dataset.classes[label]])
