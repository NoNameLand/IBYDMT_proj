import functools
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from itertools import product
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float
from joblib import Parallel, delayed

from ibydmt.utils.concept_data import get_dataset_with_concepts
from ibydmt.utils.config import Config, get_config
from ibydmt.utils.constants import workdir

# from ibydmt.payoff import HSIC, cMMD, xMMD
from ibydmt.utils.data import get_dataset
from ibydmt.utils.models.clip_classifier import CLIPClassifier
from ibydmt.wealth import Wealth, get_wealth

Array = Union[np.ndarray, torch.Tensor]


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


class SequentialTester(object):
    def __init__(self, config, *args):
        self.wealth: Wealth = get_wealth(config.wealth)(config)
        self.significance_level = config.significance_level
        self.tau_max = config.tau_max

        self.initialize(*args)

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def step(self):
        pass

    def test(self, stop_on="rejection", return_wealth=True):
        rejected = False
        tau = self.tau_max - 1
        for t in range(1, self.tau_max):
            payoff = self.step()

            self.wealth.update(payoff)
            if self.wealth.w >= 1 / self.significance_level:
                rejected = True
                tau = min(tau, t)
                if stop_on == "rejection":
                    break
        output = (rejected, tau)
        if return_wealth:
            output += (self.wealth,)
        return output


class SKIT(SequentialTester):
    def __init__(self, config):
        super().__init__(config)


def sweep(config: Config):
    to_iterable = lambda v: v if isinstance(v, list) else [v]

    sweep_keys, sweep_values = zip(*config.items())
    sweep = list(product(*map(to_iterable, sweep_values)))

    for _sweep in sweep:
        kwargs = {k: v for k, v in zip(sweep_keys, _sweep)}
        yield Config(**kwargs)


def run_tests(config: Config, testers: List[SequentialTester]):
    significance_level = config.testing.significance_level
    tau_max = config.testing.tau_max
    fdr_control = config.testing.fdr_control
    k = len(testers)


def test_global(config: Config, concept_type: str, workdir: str = workdir):
    dataset = get_dataset(config, workdir=workdir)
    classes = dataset.classes

    for class_name in classes:
        concept_class_name = None
        if concept_type == "class":
            concept_class_name = class_name

        concept_dataset = get_dataset_with_concepts(
            config, workdir=workdir, train=False, concept_class_name=concept_class_name
        )
        concepts = concept_dataset.concepts
        print(class_name, concepts)
        raise NotImplementedError


def test_global_cond(config: Config, concept_type: str, workdir: str = workdir):
    pass


def test_local_cond(config: Config, concept_type: str, workdir: str = workdir):
    pass


class ConceptTester(object):
    def __init__(self, config_name: str):
        self.config = get_config(config_name)

    def test(self, test_type: str, concept_type: str, workdir: str = workdir):
        if test_type == "global":
            test_fn = test_global
        if test_type == "global_cond":
            test_fn = test_global_cond
        if test_type == "local_cond":
            test_fn = test_local_cond

        for config in sweep(self.config):
            test_fn(config, concept_type, workdir)

    def _test(self, testers: List[SequentialTester]):
        testing_config = self.config.testing
        significance_level = testing_config.significance_level
        tau_max = testing_config.tau_max

        k = len(testers)
        _significance_level = significance_level / k
        for tester in testers:
            tester.significance_level = _significance_level

        wealths = Parallel(n_jobs=-1)(delayed(tester.test)()[-1] for tester in testers)
        wealths = np.array(
            [
                np.pad(
                    wealth,
                    (0, tau_max - len(wealth)),
                    mode="constant",
                    constant_values=-np.inf,
                )
                for wealth in wealths
            ]
        )

        s = []
        for n in range(1, k + 1):
            t = k / (significance_level * n)
            tester_idx, tau = np.nonzero(wealths >= t)
            raise NotImplementedError
