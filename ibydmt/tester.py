import json
import logging
import os
from itertools import product
from random import shuffle
from typing import Iterable

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ibydmt.classifiers import ZeroShotClassifier
from ibydmt.samplers import cKDE
from ibydmt.testing.fdr import FDRPostProcessor
from ibydmt.testing.procedure import SKIT, SequentialTester, cSKIT, xSKIT
from ibydmt.utils.concept_data import get_dataset_with_concepts
from ibydmt.utils.config import ConceptType, Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.config import TestType, get_config
from ibydmt.utils.data import get_dataset
from ibydmt.utils.result import TestingResults

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


def sweep(
    config: Config,
    sweep_keys=[
        "data.backbone",
        "testing.kernel",
        "testing.kernel_scale",
        "testing.tau_max",
    ],
    ckde_sweep_keys=["ckde.scale"],
    sweep_ckde=False,
):
    def _get(dict, key):
        keys = key.split(".")
        if len(keys) == 1:
            return dict[keys[0]]
        else:
            return _get(dict[keys[0]], ".".join(keys[1:]))

    def _set(dict, key, value):
        keys = key.split(".")
        if len(keys) == 1:
            dict[keys[0]] = value
        else:
            _set(dict[keys[0]], ".".join(keys[1:]), value)

    to_iterable = lambda v: v if isinstance(v, list) else [v]

    config_dict = config.to_dict()
    if sweep_ckde:
        sweep_keys += ckde_sweep_keys
    sweep_values = [_get(config_dict, key) for key in sweep_keys]
    sweep = list(product(*map(to_iterable, sweep_values)))

    configs: Iterable[Config] = []
    for _sweep in sweep:
        _config_dict = config_dict.copy()
        for key, value in zip(sweep_keys, _sweep):
            _set(_config_dict, key, value)

        configs.append(Config(_config_dict))
    return configs


def get_local_test_idx(config: Config, workdir: str = c.WORKDIR):
    results_dir = os.path.join(workdir, "results", config.name.lower(), "local_cond")
    os.makedirs(results_dir, exist_ok=True)

    test_idx_path = os.path.join(results_dir, "local_test_idx.json")
    if not os.path.exists(test_idx_path):
        dataset = get_dataset_with_concepts(config, train=False)
        class_idx = {
            class_name: np.nonzero(dataset.label == class_idx)[0].tolist()
            for class_idx, class_name in enumerate(dataset.classes)
        }

        samples_per_class = config.testing.get("samples_per_class", 2)
        test_idx = {
            class_name: rng.choice(
                _class_idx, samples_per_class, replace=False
            ).tolist()
            for class_name, _class_idx in class_idx.items()
        }

        with open(test_idx_path, "w") as f:
            json.dump(test_idx, f)

    with open(test_idx_path, "r") as f:
        test_idx = json.load(f)
    return test_idx


def sample_random_subset(concepts: Iterable[str], concept_idx: int, cardinality: int):
    sample_idx = list(set(range(len(concepts))) - {concept_idx})
    shuffle(sample_idx)
    return sample_idx[:cardinality]


def run_tests(config: Config, testers: Iterable[SequentialTester]):
    fdr_postprocessor = FDRPostProcessor(config)

    stop_value = len(testers) / config.testing.significance_level
    results = Parallel(n_jobs=-1)(
        delayed(tester.test)(stop_on="value", stop_value=stop_value, return_wealth=True)
        for tester in testers
    )
    rejected, tau, wealths = zip(*results)
    fdr_rejected, fdr_tau = fdr_postprocessor(wealths)
    return (rejected, tau), (fdr_rejected, fdr_tau)


def test_global(config: Config, concept_type: str, workdir: str = c.WORKDIR):
    logger.info(
        "Testing for global semantic importance of dataset"
        f" {config.data.dataset.lower()} with backbone = {config.data.backbone},"
        f" concept_type = {concept_type}, kernel = {config.testing.kernel},"
        f" kernel_scale = {config.testing.kernel_scale}, tau_max ="
        f" {config.testing.tau_max}"
    )

    dataset = get_dataset(config, workdir=workdir)
    predictions = ZeroShotClassifier.get_predictions(config, workdir=workdir)

    results = TestingResults(config, "global", concept_type)

    classes = dataset.classes
    for class_name in classes:
        logger.info(f"Testing class {class_name}")

        concept_class_name = None
        if concept_type == ConceptType.CLASS.value:
            concept_class_name = class_name

        concept_dataset = get_dataset_with_concepts(
            config, workdir=workdir, train=False, concept_class_name=concept_class_name
        )
        concepts = concept_dataset.concepts

        for _ in tqdm(range(config.testing.r)):
            testers = []
            for concept_idx, _ in enumerate(concepts):
                pi = rng.permutation(len(concept_dataset))
                Y = predictions[class_name].values[pi]
                Z = concept_dataset.semantics[:, concept_idx][pi]

                tester = SKIT(config, Y, Z)
                testers.append(tester)

            (rejected, tau), (fdr_rejected, fdr_tau) = run_tests(config, testers)
            results.add(class_name, concepts, rejected, tau)
            results.add(class_name, concepts, fdr_rejected, fdr_tau, fdr_control=True)

    results.save(workdir)


def test_global_cond(config: Config, concept_type: str, workdir: str = c.WORKDIR):
    logger.info(
        "Testing for global conditional semantic importance of dataset"
        f" {config.data.dataset.lower()} with concept_type = {concept_type}, kernel ="
        f" {config.testing.kernel}, kernel_scale = {config.testing.kernel_scale},"
        f" tau_max = {config.testing.tau_max}, ckde_scale = {config.ckde.scale}"
    )

    dataset = get_dataset(config, workdir=workdir)
    predictions = ZeroShotClassifier.get_predictions(config, workdir=workdir)
    return

    results = TestingResults(config, "global_cond", concept_type)

    classes = dataset.classes
    for class_name in classes:
        logger.info(f"Testing class {class_name}")

        concept_class_name = None
        if concept_type == ConceptType.CLASS.value:
            concept_class_name = class_name

        concept_dataset = get_dataset_with_concepts(
            config, workdir=workdir, train=False, concept_class_name=concept_class_name
        )
        concepts = concept_dataset.concepts

        ckde = cKDE(config, concept_class_name=concept_class_name)

        for _ in tqdm(range(config.testing.r)):
            testers = []
            for concept_idx, _ in enumerate(concepts):
                pi = rng.permutation(len(concept_dataset))
                Y = predictions[class_name].values[pi]
                Z = concept_dataset.semantics[pi]

                tester = cSKIT(config, Y, Z, concept_idx, ckde.sample_concept)
                testers.append(tester)

            (rejected, tau) = run_tests(config, testers)
            results.add(rejected, tau, class_name, concepts)

    results.save(workdir)


def test_local_cond(config: Config, concept_type: str, workdir: str = c.WORKDIR):
    logger.info(
        "Testing for local conditional semantic importance of dataset"
        f" {config.data.dataset.lower()} with concept_type = {concept_type}, kernel ="
        f" {config.testing.kernel}, kernel_scale = {config.testing.kernel_scale},"
        f" tau_max = {config.testing.tau_max}, ckde_scale = {config.ckde.scale}"
    )

    dataset = get_dataset(config, workdir=workdir)
    classifier = ZeroShotClassifier.load_or_train(config, workdir=workdir)

    test_idx = get_local_test_idx(config, workdir=workdir)
    cardinalities = config.testing.cardinalities
    results = TestingResults(config, "local_cond", concept_type)

    classes = dataset.classes
    for class_name, class_test_idx in test_idx.items():
        class_idx = classes.index(class_name)

        class_test = list(product(class_test_idx, cardinalities))
        for test_idx, cardinality in class_test:
            logger.info(
                f"Testing id = {test_idx} (class = {class_name}) with cardinality ="
                f" {cardinality}"
            )

            concept_class_name = None
            concept_image_idx = None
            if concept_type == ConceptType.CLASS.value:
                concept_class_name = class_name
            if concept_type == ConceptType.IMAGE.value:
                concept_image_idx = test_idx

            concept_dataset = get_dataset_with_concepts(
                config,
                workdir=workdir,
                train=False,
                concept_class_name=concept_class_name,
                concept_image_idx=concept_image_idx,
            )
            concepts = concept_dataset.concepts

            ckde = cKDE(
                config,
                concept_class_name=concept_class_name,
                concept_image_idx=concept_image_idx,
            )
            return

            z = concept_dataset.semantics[test_idx]

            for _ in tqdm(range(config.testing.r)):
                testers = []
                for concept_idx, _ in enumerate(concepts):
                    subset_idx = sample_random_subset(
                        concepts, concept_idx, cardinality
                    )

                    tester = xSKIT(
                        config,
                        z,
                        concept_idx,
                        subset_idx,
                        ckde.sample_embedding,
                        classifier,
                        class_idx=class_idx,
                        cond_p_kwargs=dict(m=config.testing.tau_max),
                    )

                    testers.append(tester)

                (rejected, tau), (fdr_rejected, fdr_tau) = run_tests(config, testers)
                results.add(
                    class_name,
                    concepts,
                    rejected,
                    tau,
                    idx=test_idx,
                    cardinality=cardinality,
                )
                results.add(
                    class_name,
                    concepts,
                    fdr_rejected,
                    fdr_tau,
                    fdr_control=True,
                    idx=test_idx,
                    cardinality=cardinality,
                )

    results.save(workdir)


class ConceptTester(object):
    def __init__(self, config_name: str):
        self.config: Config = get_config(config_name)

    def test(self, test_type: str, concept_type: str, workdir: str = c.WORKDIR):
        if test_type == TestType.GLOBAL.value:
            test_fn = test_global
            sweep_ckde = False
        if test_type == TestType.GLOBAL_COND.value:
            test_fn = test_global_cond
            sweep_ckde = True
        if test_type == TestType.LOCAL_COND.value:
            test_fn = test_local_cond
            sweep_ckde = True

        for config in sweep(self.config, sweep_ckde=sweep_ckde):
            test_fn(config, concept_type, workdir)
