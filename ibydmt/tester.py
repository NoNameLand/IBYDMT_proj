import json
import logging
import os
from itertools import product
from random import shuffle
from typing import List

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ibydmt.classifiers import CLIPClassifier
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


# class Tester(ABC):
#     def __init__(self):
#         pass

#     @abstractmethod
#     def test(self, *args, **kwargs) -> Tuple[bool, int]:
#         pass


# class SequentialTester(Tester):
#     def __init__(self, config):
#         super().__init__()
#         self.wealth = get_wealth(config.wealth)(config)

#         self.tau_max = config.tau_max


# class SKIT(SequentialTester):
#     """Global Independence Tester"""

#     def __init__(self, config):
#         super().__init__(config)
#         self.payoff = HSIC(config)

#     def test(self, Y: Float[Array, "N"], Z: Float[Array, "N"]) -> Tuple[bool, int]:
#         D = np.stack([Y, Z], axis=1)
#         for t in range(1, self.tau_max):
#             d = D[2 * t : 2 * (t + 1)]
#             prev_d = D[: 2 * t]

#             null_d = np.stack([d[:, 0], np.flip(d[:, 1])], axis=1)

#             payoff = self.payoff.compute(d, null_d, prev_d)
#             self.wealth.update(payoff)

#             if self.wealth.rejected:
#                 return (True, t)
#         return (False, t)


# class cSKIT(SequentialTester):
#     """Global Conditional Independence Tester"""

#     def __init__(self, config):
#         super().__init__(config)
#         self.payoff = cMMD(config)

#     def _sample(
#         self,
#         z: Float[Array, "N D"],
#         j: int = None,
#         cond_p: Callable[[Float[Array, "N D"], list[int]], Float[Array, "N D"]] = None,
#     ) -> Tuple[Float[Array, "N"], Float[Array, "N"], Float[Array, "N D-1"]]:
#         C = list(set(range(z.shape[1])) - {j})

#         zj, cond_z = z[:, [j]], z[:, C]
#         samples = cond_p(z, C)
#         null_zj = samples[:, [j]]
#         return zj, null_zj, cond_z

#     def test(
#         self,
#         Y: Float[Array, "N"],
#         Z: Float[Array, "N D"],
#         j: int,
#         cond_p: Callable[[Float[Array, "N D"], list[int]], Float[Array, "N D"]],
#     ) -> Tuple[bool, int]:
#         sample = functools.partial(self._sample, j=j, cond_p=cond_p)

#         prev_y, prev_z = Y[:1][:, None], Z[:1]
#         prev_zj, prev_null_zj, prev_cond_z = sample(prev_z)
#         prev_d = np.concatenate([prev_y, prev_zj, prev_null_zj, prev_cond_z], axis=-1)
#         for t in range(1, self.tau_max):
#             y, z = Y[[t]][:, None], Z[[t]]
#             zj, null_zj, cond_z = sample(z)

#             u = np.concatenate([y, zj, cond_z], axis=-1)
#             null_u = np.concatenate([y, null_zj, cond_z], axis=-1)

#             payoff = self.payoff.compute(u, null_u, prev_d)
#             self.wealth.update(payoff)

#             d = np.concatenate([y, zj, null_zj, cond_z], axis=-1)
#             prev_d = np.vstack([prev_d, d])

#             if self.wealth.rejected:
#                 return (True, t)
#         return (False, t)


# class xSKIT(SequentialTester):
#     """Local Conditional Independence Tester"""

#     def __init__(self, config):
#         super().__init__(config)
#         self.payoff = xMMD(config)

#         self._queue = deque()

#     def _sample(
#         self,
#         z: Float[Array, "D"],
#         j: int,
#         C: list[int],
#         cond_p: Callable[[Float[Array, "D"], list[int], int], Float[Array, "N D2"]],
#         model: Callable[[Float[Array, "N D2"]], Float[Array, "N"]],
#     ) -> Tuple[Float[Array, "1"], Float[Array, "1"]]:

#         if len(self._queue) == 0:
#             Cuj = C + [j]

#             h = cond_p(z, Cuj, self.tau_max)
#             null_h = cond_p(z, C, self.tau_max)

#             y = model(h)[:, None]
#             null_y = model(null_h)[:, None]

#             self._queue.extend(zip(y, null_y))

#         return self._queue.pop()

#     def test(
#         self,
#         z: Float[Array, "D"],
#         j: int,
#         C: list[int],
#         cond_p: Callable[[Float[Array, "D"], list[int], int], Float[Array, "N D2"]],
#         model: Callable[[Float[Array, "N D2"]], Float[Array, "N"]],
#     ) -> Tuple[bool, int]:
#         sample = functools.partial(self._sample, z, j, C, cond_p, model)

#         prev_d = np.stack(sample(), axis=1)
#         for t in range(1, self.tau_max):
#             y, null_y = sample()

#             payoff = self.payoff.compute(y, null_y, prev_d)
#             self.wealth.update(payoff)

#             d = np.stack([y, null_y], axis=1)
#             prev_d = np.vstack([prev_d, d])

#             if self.wealth.rejected:
#                 return (True, t)
#         return (False, t)


def sweep(
    config: Config,
    sweep_keys=[
        "testing.fdr_control",
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

    configs = []
    for _sweep in sweep:
        _config_dict = config_dict.copy()
        for key, value in zip(sweep_keys, _sweep):
            _set(_config_dict, key, value)
        configs.append(Config(_config_dict).freeze())
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


def sample_random_subset(concepts: List[str], concept_idx: int, cardinality: int):
    sample_idx = list(set(range(len(concepts))) - {concept_idx})
    shuffle(sample_idx)
    return sample_idx[:cardinality]


def run_tests(config: Config, testers: List[SequentialTester]):
    significance_level = config.testing.significance_level
    tau_max = config.testing.tau_max
    fdr_control = config.testing.fdr_control
    k = len(testers)

    if fdr_control:
        postprocessor = FDRPostProcessor()
        for tester in testers:
            tester.significance_level = significance_level / k

    results = Parallel(n_jobs=-1)(
        delayed(tester.test)(return_wealth=True) for tester in testers
    )
    rejected, tau, wealths = zip(*results)

    if fdr_control:
        rejected, tau = postprocessor(significance_level, wealths, tau_max=tau_max)

    return rejected, tau


def test_global(config: Config, concept_type: str, workdir: str = c.WORKDIR):
    logger.info(
        "Testing for global semantic importance of dataset"
        f" {config.data.dataset.lower()} with concept_type = {concept_type}, kernel ="
        f" {config.testing.kernel}, kernel_scale = {config.testing.kernel_scale},"
        f" tau_max = {config.testing.tau_max}, fdr_control ="
        f" {config.testing.fdr_control}"
    )

    dataset = get_dataset(config, workdir=workdir)
    predictions = CLIPClassifier.get_predictions(config, workdir=workdir)

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

            rejected, tau = run_tests(config, testers)
            results.add(rejected, tau, class_name, concepts)

    results.save(workdir)


def test_global_cond(config: Config, concept_type: str, workdir: str = c.WORKDIR):
    logger.info(
        "Testing for global conditional semantic importance of dataset"
        f" {config.data.dataset.lower()} with concept_type = {concept_type}, kernel ="
        f" {config.testing.kernel}, kernel_scale = {config.testing.kernel_scale},"
        f" tau_max = {config.testing.tau_max}, ckde_scale = {config.ckde.scale},"
        f" fdr_control = {config.testing.fdr_control}"
    )

    dataset = get_dataset(config, workdir=workdir)
    predictions = CLIPClassifier.get_predictions(config, workdir=workdir)

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

            rejected, tau = run_tests(config, testers)
            results.add(rejected, tau, class_name, concepts)

    results.save(workdir)


def test_local_cond(config: Config, concept_type: str, workdir: str = c.WORKDIR):
    logger.info(
        "Testing for local conditional semantic importance of dataset"
        f" {config.data.dataset.lower()} with concept_type = {concept_type}, kernel ="
        f" {config.testing.kernel}, kernel_scale = {config.testing.kernel_scale},"
        f" tau_max = {config.testing.tau_max}, ckde_scale = {config.ckde.scale},"
        f" fdr_control = {config.testing.fdr_control}"
    )

    dataset = get_dataset(config, workdir=workdir)
    classifier = CLIPClassifier.load_or_train(config, workdir=workdir)

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
                    )

                    testers.append(tester)

                rejected, tau = run_tests(config, testers)
                results.add(
                    rejected,
                    tau,
                    class_name,
                    concepts,
                    idx=test_idx,
                    cardinality=cardinality,
                )


class ConceptTester(object):
    def __init__(self, config_name: str):
        self.config = get_config(config_name)

    def test(self, test_type: str, concept_type: str, workdir: str = c.WORKDIR) -> None:
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
