from abc import abstractmethod
from collections import deque
from enum import Enum
from typing import Optional, Protocol

import numpy as np
from jaxtyping import Float

from ibydmt.testing.payoff import HSIC, cMMD, xMMD
from ibydmt.testing.wealth import Wealth, get_wealth
from ibydmt.utils.config import Array, Config


class StoppingCriteria(Enum):
    REJECTION = "rejection"
    VALUE = "value"
    END = "end"


class ConditionalConceptSampler(Protocol):
    def __call__(
        self, z: Float[Array, "N M"], cond_idx: list[int], **kwargs: dict
    ) -> Float[Array, "N M"]:
        ...


class ConditionalEmbeddingSampler(Protocol):
    def __call__(
        z: Float[Array, "N M"], cond_idx: list[int], **kwargs: dict
    ) -> Float[Array, "N D"]:
        ...


class EmbeddingClassifier(Protocol):
    def __call__(self, h: Float[Array, "N D"], **kwargs: dict) -> Float[Array, "N K"]:
        ...


class SequentialTester:
    def __init__(self, config: Config):
        self.wealth: Wealth = get_wealth(config.testing.wealth)(config)
        self.significance_level = config.testing.significance_level
        self.tau_max = config.testing.tau_max

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def step(self):
        pass

    def test(
        self,
        stop_on: str = "rejection",
        return_wealth: bool = False,
        stop_value: Optional[float] = None,
    ):
        self.initialize()
        rejected, tau = False, self.tau_max - 1
        for t in range(1, self.tau_max):
            payoff = self.step()

            self.wealth.update(payoff)

            if stop_on == StoppingCriteria.VALUE.value:
                if self.wealth.w > stop_value:
                    break

            if not rejected and self.wealth.w > 1 / self.significance_level:
                rejected, tau = True, t

                if stop_on == StoppingCriteria.REJECTION.value:
                    break
        output = (rejected, tau)
        if return_wealth:
            output += (self.wealth,)
        return output


class SKIT(SequentialTester):
    def __init__(self, config: Config, Y: Float[Array, "N"], Zj: Float[Array, "N"]):
        super().__init__(config)
        self.payoff = HSIC(config)
        self.queue = deque(zip(Y, Zj))

    def sample_pair(self):
        return np.vstack([self.queue.pop(), self.queue.pop()])

    def initialize(self):
        self.prev_d = self.sample_pair()

    def step(self):
        d = self.sample_pair()
        null_d = np.stack([d[:, 0], np.flip(d[:, 1])], axis=1)

        payoff = self.payoff.compute(d, null_d, self.prev_d)

        self.prev_d = np.vstack([self.prev_d, d])
        return payoff


class cSKIT(SequentialTester):
    def __init__(
        self,
        config: Config,
        Y: Float[Array, "N"],
        Z: Float[Array, "N M"],
        j: int,
        cond_p: ConditionalConceptSampler,
        cond_p_kwargs: Optional[dict] = None,
    ):
        super().__init__(config)
        self.j = j
        self.C = list(set(range(Z.shape[1])) - {j})

        self.payoff = cMMD(config)
        self.queue = deque(zip(Y, Z))

        if cond_p_kwargs is None:
            cond_p_kwargs = {}

        self.cond_p = cond_p
        self.cond_p_kwargs = cond_p_kwargs

    def sample(self):
        y, z = self.queue.pop()
        y = y.reshape(1, 1)
        z = z.reshape(1, -1)

        zj, cond_z = z[:, [self.j]], z[:, self.C]
        samples = self.cond_p(z, self.C, **self.cond_p_kwargs)
        null_zj = samples[:, [self.j]]
        return y, zj, null_zj, cond_z

    def initialize(self):
        y, zj, null_zj, cond_z = self.sample()
        self.prev_d = np.concatenate([y, zj, null_zj, cond_z], axis=-1)

    def step(self):
        y, zj, null_zj, cond_z = self.sample()

        u = np.concatenate([y, zj, cond_z], axis=-1)
        null_u = np.concatenate([y, null_zj, cond_z], axis=-1)
        payoff = self.payoff.compute(u, null_u, self.prev_d)

        d = np.concatenate([y, zj, null_zj, cond_z], axis=-1)
        self.prev_d = np.vstack([self.prev_d, d])
        return payoff


class xSKIT(SequentialTester):
    def __init__(
        self,
        config: Config,
        z: Float[Array, "M"],
        j: int,
        C: list[int],
        cond_p: ConditionalEmbeddingSampler,
        classifier: EmbeddingClassifier,
        class_idx: Optional[int] = None,
        cond_p_kwargs: Optional[dict] = None,
        classifier_kwargs: Optional[dict] = None,
    ):
        super().__init__(config)
        self.z = z
        self.j = j
        self.C = C

        self.payoff = xMMD(config)
        self.queue = deque()

        if class_idx is None:
            class_idx = 0
        if cond_p_kwargs is None:
            cond_p_kwargs = {}
        if classifier_kwargs is None:
            classifier_kwargs = {}

        self.cond_p = cond_p
        self.classifier = classifier
        self.class_idx = class_idx
        self.cond_p_kwargs = cond_p_kwargs
        self.classifier_kwargs = classifier_kwargs

    def sample(self):
        return self.queue.pop()

    def initialize(self):
        Cuj = self.C + [self.j]

        h = self.cond_p(self.z, Cuj, **self.cond_p_kwargs)
        null_z = self.cond_p(self.z, self.C, **self.cond_p_kwargs)

        y = self.classifier(h)[:, [self.class_idx]]
        null_y = self.classifier(null_z)[:, [self.class_idx]]

        self.queue.extend(zip(y, null_y))
        self.prev_d = np.stack(self.sample(), axis=1)

    def step(self):
        y, null_y = self.sample()

        payoff = self.payoff.compute(y, null_y, self.prev_d)

        d = np.concatenate([y, null_y], axis=-1)
        self.prev_d = np.vstack([self.prev_d, d])
        return payoff
