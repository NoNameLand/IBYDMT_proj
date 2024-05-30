import contextlib
import os
import pickle
from itertools import product

import joblib
import numpy as np

from concept_datasets import get_concept_dataset
from concept_lib import get_concepts
from datasets import get_dataset


class TestingResults:
    def __init__(
        self,
        config,
        test_type=None,
        concept_type="dataset",
        test_idx=None,
        kernel=None,
        kernel_scale=None,
        tau_max=None,
        ckde_scale_method=None,
        ckde_scale=None,
    ):
        self.config = config
        self.test_type = test_type
        self.concept_type = concept_type
        self.test_idx = test_idx

        testing_config = config.testing
        self.significance_level = testing_config.significance_level
        self.kernel = kernel or testing_config.kernel
        self.kernel_scale = kernel_scale or testing_config.kernel_scale
        self.tau_max = tau_max or testing_config.tau_max
        self.r = testing_config.r
        self.cardinalities = testing_config.get("cardinalities", [1, 2, 4, 8])

        ckde_config = config.ckde
        self.ckde_scale_method = ckde_scale_method or ckde_config.scale_method
        self.ckde_scale = ckde_scale or ckde_config.scale

        self._results = {}

    def state_path(self, workdir):
        state_dir = os.path.join(
            workdir, "results", self.config.name.lower(), self.test_type
        )
        os.makedirs(state_dir, exist_ok=True)

        state_name = (
            f"{self.kernel}_{self.kernel_scale}_{self.tau_max}_{self.concept_type}"
        )
        if "cond" in self.test_type:
            state_name = f"{state_name}_{self.ckde_scale_method}_{self.ckde_scale}"
        return os.path.join(state_dir, f"{state_name}.pkl")

    @staticmethod
    def load(
        config,
        workdir,
        test_type=None,
        concept_type=None,
        kernel=None,
        kernel_scale=None,
        tau_max=None,
        ckde_scale_method=None,
        ckde_scale=None,
    ):
        results = TestingResults(
            config,
            test_type=test_type,
            concept_type=concept_type,
            kernel=kernel,
            kernel_scale=kernel_scale,
            tau_max=tau_max,
            ckde_scale_method=ckde_scale_method,
            ckde_scale=ckde_scale,
        )
        with open(results.state_path(workdir), "rb") as f:
            _results = pickle.load(f)
        results._results = _results
        return results

    def save(self, workdir):
        with open(self.state_path(workdir), "wb") as f:
            pickle.dump(self._results, f)

    def tests(self):
        dataset = get_dataset(self.config)
        classes = dataset.classes

        if self.concept_type == "dataset":
            _, concepts = get_concepts(self.config)
            tests = list(product(classes, concepts))
        if self.concept_type == "class":
            tests = []
            for class_name in classes:
                _, concepts = get_concepts(self.config, class_name=class_name)
                tests.extend(list(product([class_name], concepts)))

            if self.test_idx is not None:
                local_tests = []
                for class_name, concept in tests:
                    _local_tests = list(
                        product(
                            self.test_idx[class_name],
                            [class_name],
                            [concept],
                            self.cardinalities,
                        )
                    )
                    local_tests.extend(_local_tests)
                tests = local_tests
        if self.concept_type == "image":
            assert self.test_idx is not None

            local_tests = []
            for class_name, idx in self.test_idx.items():
                for _idx in idx:
                    _, concepts = get_concepts(self.config, concept_image_idx=_idx)
                    _local_tests = list(
                        product([_idx], [class_name], concepts, self.cardinalities)
                    )
                    local_tests.extend(_local_tests)
            tests = local_tests

        return tests

    def add(
        self,
        class_name=None,
        concept=None,
        idx=None,
        cardinality=None,
        rejected=None,
        tau=None,
        subset=None,
    ):
        test_results = {"rejected": rejected, "tau": tau}
        if subset is not None:
            test_results["subset"] = subset

        results = self._results

        if idx is not None:
            if idx not in results:
                results[idx] = {}
            results = results[idx]

        if cardinality is not None:
            if cardinality not in results:
                results[cardinality] = {}
            results = results[cardinality]

        results[(class_name, concept)] = test_results

    def get(self, class_name, concepts, idx=None, cardinalities=None, reduction="mean"):
        def _get(results, rejected, tau):
            for i, concept in enumerate(concepts):
                test_results = results[(class_name, concept)]
                rejected[i] = test_results["rejected"]
                tau[i] = test_results["tau"]

        results = self._results
        if idx is not None:
            results = self._results[idx]

            if cardinalities is None:
                cardinalities = self.cardinalities

            rejected = np.empty((len(concepts), len(cardinalities), self.r))
            tau = np.empty((len(concepts), len(cardinalities), self.r))

            for j, cardinality in enumerate(cardinalities):
                _results = results[cardinality]
                _rejected = rejected[:, j]
                _tau = tau[:, j]

                _get(_results, _rejected, _tau)
        else:
            rejected = np.empty((len(concepts), self.r))
            tau = np.empty((len(concepts), self.r))

            _get(results, rejected, tau)

        tau /= self.tau_max

        if reduction == "mean":
            rejected = np.mean(rejected, axis=-1)
            tau = np.mean(tau, axis=-1)
        elif reduction == "none":
            pass
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented")

        return rejected, tau


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """From: https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
