from typing import Any, Mapping, Optional

import numpy as np
from scipy.stats import weightedtau
from tqdm import tqdm

from ibydmt.tester import get_test_classes
from ibydmt.utils.config import ConceptType, Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.config import TestType
from ibydmt.utils.data import get_dataset
from ibydmt.utils.pcbm import PCBM
from ibydmt.utils.result import TestingResults


def rank_agreement(
    config: Config,
    test_type: str,
    concept_type: str,
    pcbm: bool = False,
    workdir=c.WORKDIR,
    results_kw: Optional[Mapping[str, Any]] = None,
):
    if pcbm:
        assert test_type == TestType.GLOBAL_COND.value, ValueError(
            "PCBM can only be used with global conditional tests."
        )

    dataset = get_dataset(config)
    classes = dataset.classes

    test_classes = get_test_classes(config)
    test_classes_idx = [classes.index(c) for c in test_classes]

    backbone_configs = config.sweep(["data.backbone"])

    results = {}
    for _, backbone_config in enumerate(tqdm(backbone_configs)):
        backbone = backbone_config.data.backbone

        results[backbone] = {}
        if pcbm:
            for class_idx, class_name in zip(test_classes_idx, test_classes):
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
                results[backbone][class_name] = sorted_concepts
        else:
            backbone_results = TestingResults.load(
                backbone_config,
                test_type,
                concept_type,
                workdir=workdir,
                results_kw=results_kw,
            )

            for class_name in test_classes:
                (_, sorted_concepts, _, _, _) = backbone_results.sort(
                    class_name, fdr_control=True, with_importance=True
                )
                results[backbone][class_name] = sorted_concepts

    rank_agreement = np.zeros(
        (len(test_classes), len(backbone_configs), len(backbone_configs))
    )
    for i, (_, results1) in enumerate(results.items()):
        for j, (_, results2) in enumerate(results.items()):
            for k, class_name in enumerate(test_classes):
                concepts1 = results1[class_name]
                concepts2 = results2[class_name]
                assert set(concepts1) == set(concepts2), ValueError(
                    "The concepts being compared across models are different."
                )

                idx1 = np.arange(len(concepts1))
                idx2 = np.array([concepts1.index(c) for c in concepts2])
                rank_agreement[k, i, j] = weightedtau(idx1, idx2, rank=False).statistic
    return list(results.keys()), rank_agreement
