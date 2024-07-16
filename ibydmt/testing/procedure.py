from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import torch
from jaxtyping import Float

from ibydmt.testing.payoff import HSIC
from ibydmt.testing.wealth import Wealth, get_wealth
from ibydmt.utils.config import Config

Array = np.ndarray | torch.Tensor


class SequentialTester:
    def __init__(self, config: Config, *args):
        self.wealth: Wealth = get_wealth(config.testing.wealth)(config)
        self.significance_level = config.testing.significance_level
        self.tau_max = config.testing.tau_max

        self.initialize(config, *args)

    @abstractmethod
    def initialize(self, config, *args):
        pass

    @abstractmethod
    def step(self):
        pass

    def test(self, stop_on="rejection", return_wealth=False):
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
            output += (self.wealth.wealth,)
        return output


class SKIT(SequentialTester):
    def __init__(self, config: Config, y: Float[Array, "N"], z: Float[Array, "N"]):
        super().__init__(config, y, z)

    def sample_pair(self):
        return np.vstack([self.queue.pop(), self.queue.pop()])

    def initialize(self, config, y, z):
        self.payoff = HSIC(config)
        self.queue = deque(zip(y, z))
        self.prev = self.sample_pair()

    def step(self):
        d = self.sample_pair()
        null_d = np.stack([d[:, 0], np.flip(d[:, 1])], axis=1)
        return self.payoff.compute(d, null_d, self.prev)


class cSKIT(SequentialTester):
    def __init__(self, config: Config, z: Float[Array, "N D"], j: int):
        pass
