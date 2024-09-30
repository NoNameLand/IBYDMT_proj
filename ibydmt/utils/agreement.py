from typing import Any, Mapping, Optional

import numpy as np
import scipy.stats as stats
from tqdm import tqdm

from ibydmt.tester import get_local_test_idx, get_test_classes
from ibydmt.utils.config import ConceptType, Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.config import TestType
from ibydmt.utils.data import get_dataset
from ibydmt.utils.pcbm import PCBM
from ibydmt.utils.result import TestingResults


def weightedtau(rank1, rank2):
    idx1 = np.arange(len(rank1))
    idx2 = np.array([rank1.index(r) for r in rank2])
    return stats.weightedtau(idx1, idx2, rank=False).statistic


def importance_agreement(
    config: Config,
    test_type: str,
    concept_type: str,
    workdir=c.WORKDIR,
    cardinality: Optional[int] = None,
    results_kw: Optional[Mapping[str, Any]] = None,
):
    n_elements, results = load_all_backbone_results(
        config,
        test_type,
        concept_type,
        workdir=workdir,
        cardinality=cardinality,
        results_kw=results_kw,
    )

    n_backbones = len(results)
    importance_agreement = np.zeros((n_elements, n_backbones, n_backbones))
    for i, results1 in enumerate(results.values()):
        for j, results2 in enumerate(results.values()):
            for k, (el, el_results1) in enumerate(results1.items()):
                el_results2 = results2[el]

                concepts1 = el_results1["sorted_concepts"]
                importance1 = el_results1["sorted_importance"]

                concepts2 = el_results2["sorted_concepts"]
                importance2 = el_results2["sorted_importance"]
                assert set(concepts1) == set(concepts2), ValueError(
                    "The concepts being compared across models are different."
                )
                importance_agreement[k, i, j] = np.mean(
                    [
                        i1 == importance2[concepts2.index(c1)]
                        for c1, i1 in zip(concepts1, importance1)
                    ]
                )
    return list(results.keys()), importance_agreement


def rank_agreement(
    config: Config,
    test_type: str,
    concept_type: str,
    pcbm: bool = False,
    workdir=c.WORKDIR,
    cardinality: Optional[int] = None,
    results_kw: Optional[Mapping[str, Any]] = None,
):
    n_elements, results = load_all_backbone_results(
        config,
        test_type,
        concept_type,
        pcbm=pcbm,
        workdir=workdir,
        cardinality=cardinality,
        results_kw=results_kw,
    )

    n_backbones = len(results)
    rank_agreement = np.zeros((n_elements, n_backbones, n_backbones))
    for i, results1 in enumerate(results.values()):
        for j, results2 in enumerate(results.values()):
            for k, (el, el_results1) in enumerate(results1.items()):
                el_results2 = results2[el]

                concepts1 = el_results1["sorted_concepts"]
                concepts2 = el_results2["sorted_concepts"]
                assert set(concepts1) == set(concepts2), ValueError(
                    "The concepts being compared across models are different."
                )
                rank_agreement[k, i, j] = weightedtau(concepts1, concepts2)
    return list(results.keys()), rank_agreement


def load_all_backbone_results(
    config: Config,
    test_type: str,
    concept_type: str,
    pcbm: bool = False,
    workdir=c.WORKDIR,
    cardinality: Optional[int] = None,
    results_kw: Optional[Mapping[str, Any]] = None,
):
    if pcbm:
        assert test_type == TestType.GLOBAL_COND.value, ValueError(
            "PCBM can only be used with global conditional tests."
        )
    if test_type == TestType.LOCAL_COND.value:
        assert cardinality is not None, ValueError(
            "Cardinality must be provided for local conditional tests."
        )

    dataset = get_dataset(config)
    classes = dataset.classes

    test_classes = get_test_classes(config)
    test_classes_idx = [classes.index(c) for c in test_classes]
    n_elements = len(test_classes)

    if test_type == TestType.LOCAL_COND.value:
        test_idx = get_local_test_idx(config)
        n_elements = sum([len(v) for v in test_idx.values()])

    results = {}
    for backbone_config in tqdm(config.sweep(["data.backbone"])):
        backbone = backbone_config.data.backbone

        results[backbone] = {}
        for class_idx, class_name in zip(test_classes_idx, test_classes):
            if pcbm:
                concept_class_name = None
                if concept_type == ConceptType.CLASS.value:
                    concept_class_name = class_name

                model = PCBM.load_or_train(
                    backbone_config,
                    concept_class_name=concept_class_name,
                    workdir=workdir,
                )
                concepts = model.concepts

                weights = model.weights()
                class_weights = weights[class_idx]
                class_weights = np.abs(class_weights)
                class_weights /= np.sum(class_weights)

                sorted_idx = np.argsort(class_weights)[::-1]
                sorted_concepts = [concepts[i] for i in sorted_idx]
                results[backbone][class_name] = {
                    "sorted_idx": sorted_idx,
                    "sorted_concepts": sorted_concepts,
                }
            else:
                backbone_results = TestingResults.load(
                    backbone_config,
                    test_type,
                    concept_type,
                    workdir=workdir,
                    results_kw=results_kw,
                )

                if test_type == TestType.LOCAL_COND.value:
                    for image_idx in test_idx[class_name]:
                        results[backbone][image_idx] = backbone_results.sort(
                            class_name, image_idx=image_idx, cardinality=cardinality
                        )
                else:
                    results[backbone][class_name] = backbone_results.sort(class_name)

    return n_elements, results
