import json
import os
from itertools import product

import ml_collections
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from concept_datasets import get_concept_dataset
from ibydmt import SKIT, cSKIT, xSKIT
from models.ckde import cKDE
from models.clip_classifier import CLIPClassifier
from test_utils import TestingResults, tqdm_joblib

n_jobs = 4
rng = np.random.default_rng()


def _test_sweep(config, test_fn, workdir, concept_type, sweep_ckde=False):
    ckde_config = config.ckde
    testing_config = config.testing

    to_iterable = lambda v: v if isinstance(v, list) else [v]

    ckde_keys, ckde_values = zip(*ckde_config.items())
    testing_keys, testing_values = zip(*testing_config.items())

    sweep_keys = testing_keys
    sweep_values = testing_values
    if sweep_ckde:
        sweep_keys += ckde_keys
        sweep_values += ckde_values

    sweep = list(product(*map(to_iterable, sweep_values)))

    for _sweep in sweep:
        _sweep_dict = {k: v for k, v in zip(sweep_keys, _sweep)}
        _config = ml_collections.ConfigDict(config)

        _testing_config = ml_collections.ConfigDict(
            {k: _sweep_dict[k] for k in testing_keys}
        )
        _config.testing = _testing_config

        if sweep_ckde:
            _ckde_config = ml_collections.ConfigDict(
                {k: _sweep_dict[k] for k in ckde_keys}
            )
            _config.ckde = _ckde_config

        test_fn(_config, workdir, concept_type)


def _test_global(config, workdir, concept_type):
    print(
        " ".join(
            [
                f"Testing for global semantic independence with",
                f"kernel = {config.testing.kernel},",
                f"kernel_scale = {config.testing.kernel_scale}",
                f"tau_max = {config.testing.tau_max}",
            ]
        )
    )

    predictions = CLIPClassifier.get_predictions(config, workdir)
    results = TestingResults(config, "global", concept_type=concept_type)

    def test(class_name, concept):
        concept_class_name = None
        if concept_type == "class":
            concept_class_name = class_name

        val_dataset = get_concept_dataset(
            config, train=False, concept_class_name=concept_class_name
        )
        val_dataset_df = val_dataset.to_df()

        Y = predictions[class_name].values
        Z_concept = val_dataset_df[concept].values

        rejected_hist, tau_hist = [], []
        for _ in range(config.testing.r):
            pi = np.random.permutation(len(Y))
            pi_Y, pi_Z_concept = Y[pi], Z_concept[pi]

            tester = SKIT(config.testing)
            rejected, tau = tester.test(pi_Y, pi_Z_concept)

            rejected_hist.append(rejected)
            tau_hist.append(tau)

        return {
            "class_name": class_name,
            "concept": concept,
            "rejected": rejected_hist,
            "tau": tau_hist,
        }

    tests = results.tests()
    for class_name, _ in tests:
        dataset_kw = {}
        if concept_type == "class":
            dataset_kw["concept_class_name"] = class_name

        get_concept_dataset(config, train=True, **dataset_kw)
        get_concept_dataset(config, train=False, **dataset_kw)

    with tqdm_joblib(tqdm(desc="Testing", total=len(tests))):
        for _results in Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(test)(class_name, concept) for class_name, concept in tests
        ):
            results.add(**_results)

    results.save(workdir)


def _test_global_cond(config, workdir, concept_type):
    print(
        " ".join(
            [
                "Testing for global conditional semantic independence with",
                f"kernel = {config.testing.kernel},",
                f"kernel_scale = {config.testing.kernel_scale}",
                f"tau_max = {config.testing.tau_max},",
                f"ckde_scale_method = {config.ckde.scale_method},",
                f"ckde_cale = {config.ckde.scale}",
            ]
        )
    )

    predictions = CLIPClassifier.get_predictions(config, workdir)
    results = TestingResults(config, "global_cond", concept_type=concept_type)

    def test(class_name, concept):
        concept_class_name = None
        if concept_type == "class":
            concept_class_name = class_name

        val_dataset = get_concept_dataset(
            config, train=False, concept_class_name=concept_class_name
        )

        model = cKDE(config, concept_class_name=concept_class_name)

        def cond_p(z, cond_idx):
            sample_z, _ = model.sample(z, cond_idx)
            return sample_z

        Z = val_dataset.Z
        Y = predictions[class_name].values

        concept_idx = val_dataset.concepts.index(concept)
        rejected_hist, tau_hist = [], []
        for _ in range(config.testing.r):
            pi = np.random.permutation(len(Y))
            pi_Y, pi_Z = Y[pi], Z[pi]

            tester = cSKIT(config.testing)
            rejected, tau = tester.test(pi_Y, pi_Z, concept_idx, cond_p)

            rejected_hist.append(rejected)
            tau_hist.append(tau)

        return {
            "class_name": class_name,
            "concept": concept,
            "rejected": rejected_hist,
            "tau": tau_hist,
        }

    tests = results.tests()
    for class_name, _ in tests:
        dataset_kw = {}
        if concept_type == "class":
            dataset_kw["concept_class_name"] = class_name

        get_concept_dataset(config, train=True, **dataset_kw)
        get_concept_dataset(config, train=False, **dataset_kw)

    with tqdm_joblib(tqdm(desc="Testing", total=len(tests))):
        for _results in Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(test)(class_name, concept) for class_name, concept in tests
        ):
            results.add(**_results)

    results.save(workdir)


def _get_test_idx(config, workdir):
    results_dir = os.path.join(workdir, "results", config.name.lower(), "local_cond")
    os.makedirs(results_dir, exist_ok=True)

    test_idx_path = os.path.join(results_dir, "test_idx.json")
    if not os.path.exists(test_idx_path):
        val_dataset = get_concept_dataset(config, train=False)
        class_idx = {class_name: [] for class_name in val_dataset.classes}
        for idx, (_, _, y) in enumerate(val_dataset):
            class_name = val_dataset.classes[y]
            class_idx[class_name].append(idx)

        test_samples_per_class = config.testing.get("samples_per_class", 2)
        test_idx = {
            class_name: rng.choice(
                _class_idx, test_samples_per_class, replace=False
            ).tolist()
            for class_name, _class_idx in class_idx.items()
        }
        with open(test_idx_path, "w") as f:
            json.dump(test_idx, f)

    with open(test_idx_path, "r") as f:
        test_idx = json.load(f)
    return test_idx


def _test_local_cond(config, workdir, concept_type):
    print(
        " ".join(
            [
                "Testing for local conditional semantic independence with",
                f"kernel = {config.testing.kernel},",
                f"kernel_scale = {config.testing.kernel_scale}",
                f"tau_max = {config.testing.tau_max}",
                f"scale_method = {config.ckde.scale_method}",
                f"scale = {config.ckde.scale}",
            ]
        )
    )

    test_idx = _get_test_idx(config, workdir)
    results = TestingResults(
        config, "local_cond", concept_type=concept_type, test_idx=test_idx
    )

    classifier = CLIPClassifier.load_or_train(config, workdir)

    def test(idx, class_name, concept, cardinality):
        concept_class_name = None
        concept_image_idx = None
        if concept_type == "class":
            concept_class_name = class_name
        elif concept_type == "image":
            concept_image_idx = idx

        val_dataset = get_concept_dataset(
            config,
            train=False,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
        )
        classes, concepts = val_dataset.classes, val_dataset.concepts

        class_idx = classes.index(class_name)
        concept_idx = concepts.index(concept)

        def sample_random_subset(concept_idx, cardinality):
            sample_idx = list(set(range(len(concepts))) - {concept_idx})
            return np.random.permutation(sample_idx)[:cardinality].tolist()

        model = cKDE(
            config,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
        )

        def cond_p(z, cond_idx, m):
            _, sample_h = model.sample(z, cond_idx, m=m)
            return sample_h

        def f(h):
            return classifier(h)[:, class_idx]

        _, z, _ = val_dataset[idx]

        rejected_hist, tau_hist, subset_hist = [], [], []
        for _ in range(config.testing.r):
            subset_idx = sample_random_subset(concept_idx, cardinality)
            subset = [concepts[idx] for idx in subset_idx]

            tester = xSKIT(config.testing)
            rejected, tau = tester.test(z, concept_idx, subset_idx, cond_p, f)

            rejected_hist.append(rejected)
            tau_hist.append(tau)
            subset_hist.append(subset)

        return {
            "class_name": class_name,
            "concept": concept,
            "idx": idx,
            "cardinality": cardinality,
            "rejected": rejected_hist,
            "tau": tau_hist,
            "subset": subset_hist,
        }

    tests = results.tests()
    for idx, class_name, _, _ in tests:
        dataset_kw = {}
        if concept_type == "class":
            dataset_kw["concept_class_name"] = class_name
        elif concept_type == "image":
            dataset_kw["concept_image_idx"] = idx

        get_concept_dataset(config, train=True, **dataset_kw)
        get_concept_dataset(config, train=False, **dataset_kw)

    with tqdm_joblib(tqdm(desc="Testing", total=len(tests))):
        for _results in Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(test)(idx, class_name, concept, cardinality)
            for idx, class_name, concept, cardinality in tests
        ):
            results.add(**_results)

    results.save(workdir)


def test(config, test_type, concept_type, workdir):
    if test_type in ["global", "all"]:
        _test_sweep(config, _test_global, workdir, concept_type)
    if test_type in ["global_cond", "all"]:
        _test_sweep(config, _test_global_cond, workdir, concept_type, sweep_ckde=True)
    if test_type in ["local_cond", "all"]:
        _test_sweep(config, _test_local_cond, workdir, concept_type, sweep_ckde=True)
