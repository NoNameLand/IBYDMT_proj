import functools
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float

from ibydmt.payoff import HSIC, cMMD, xMMD
from ibydmt.wealth import get_wealth

Array = Union[np.ndarray, torch.Tensor]


class Tester(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def test(self, *args, **kwargs) -> Tuple[bool, int]:
        pass


class SequentialTester(Tester):
    def __init__(self, config):
        super().__init__()
        self.wealth = get_wealth(config.wealth)(config)

        self.tau_max = config.tau_max


class SKIT(SequentialTester):
    """Global Independence Tester"""

    def __init__(self, config):
        super().__init__(config)
        self.payoff = HSIC(config)

    def test(self, Y: Float[Array, "N"], Z: Float[Array, "N"]) -> Tuple[bool, int]:
        D = np.stack([Y, Z], axis=1)
        for t in range(1, self.tau_max):
            d = D[2 * t : 2 * (t + 1)]
            prev_d = D[: 2 * t]

            null_d = np.stack([d[:, 0], np.flip(d[:, 1])], axis=1)

            payoff = self.payoff.compute(d, null_d, prev_d)
            self.wealth.update(payoff)

            if self.wealth.rejected:
                return (True, t)
        return (False, t)


class cSKIT(SequentialTester):
    """Global Conditional Independence Tester"""

    def __init__(self, config):
        super().__init__(config)
        self.payoff = cMMD(config)

    def _sample(
        self,
        z: Float[Array, "N D"],
        j: int = None,
        cond_p: Callable[[Float[Array, "N D"], list[int]], Float[Array, "N D"]] = None,
    ) -> Tuple[Float[Array, "N"], Float[Array, "N"], Float[Array, "N D-1"]]:
        C = list(set(range(z.shape[1])) - {j})

        zj, cond_z = z[:, [j]], z[:, C]
        samples = cond_p(z, C)
        null_zj = samples[:, [j]]
        return zj, null_zj, cond_z

    def test(
        self,
        Y: Float[Array, "N"],
        Z: Float[Array, "N D"],
        j: int,
        cond_p: Callable[[Float[Array, "N D"], list[int]], Float[Array, "N D"]],
    ) -> Tuple[bool, int]:
        sample = functools.partial(self._sample, j=j, cond_p=cond_p)

        prev_y, prev_z = Y[:1][:, None], Z[:1]
        prev_zj, prev_null_zj, prev_cond_z = sample(prev_z)
        prev_d = np.concatenate([prev_y, prev_zj, prev_null_zj, prev_cond_z], axis=-1)
        for t in range(1, self.tau_max):
            y, z = Y[[t]][:, None], Z[[t]]
            zj, null_zj, cond_z = sample(z)

            u = np.concatenate([y, zj, cond_z], axis=-1)
            null_u = np.concatenate([y, null_zj, cond_z], axis=-1)

            payoff = self.payoff.compute(u, null_u, prev_d)
            self.wealth.update(payoff)

            d = np.concatenate([y, zj, null_zj, cond_z], axis=-1)
            prev_d = np.vstack([prev_d, d])

            if self.wealth.rejected:
                return (True, t)
        return (False, t)


class xSKIT(SequentialTester):
    """Local Conditional Independence Tester"""

    def __init__(self, config):
        super().__init__(config)
        self.payoff = xMMD(config)

        self._queue = deque()

    def _sample(
        self,
        z: Float[Array, "D"],
        j: int,
        C: list[int],
        cond_p: Callable[[Float[Array, "D"], list[int], int], Float[Array, "N D2"]],
        model: Callable[[Float[Array, "N D2"]], Float[Array, "N"]],
    ) -> Tuple[Float[Array, "1"], Float[Array, "1"]]:

        if len(self._queue) == 0:
            Cuj = C + [j]

            h = cond_p(z, Cuj, self.tau_max)
            null_h = cond_p(z, C, self.tau_max)

            y = model(h)[:, None]
            null_y = model(null_h)[:, None]

            self._queue.extend(zip(y, null_y))

        return self._queue.pop()

    def test(
        self,
        z: Float[Array, "D"],
        j: int,
        C: list[int],
        cond_p: Callable[[Float[Array, "D"], list[int], int], Float[Array, "N D2"]],
        model: Callable[[Float[Array, "N D2"]], Float[Array, "N"]],
    ) -> Tuple[bool, int]:
        sample = functools.partial(self._sample, z, j, C, cond_p, model)

        prev_d = np.stack(sample(), axis=1)
        for t in range(1, self.tau_max):
            y, null_y = sample()

            payoff = self.payoff.compute(y, null_y, prev_d)
            self.wealth.update(payoff)

            d = np.stack([y, null_y], axis=1)
            prev_d = np.vstack([prev_d, d])

            if self.wealth.rejected:
                return (True, t)
        return (False, t)
