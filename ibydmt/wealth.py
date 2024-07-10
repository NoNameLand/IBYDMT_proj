from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from ibydmt.utils import _get_cls, _register_cls

_WEALTH: Dict = {}


def register_wealth(name):
    return _register_cls(name, dict=_WEALTH)


def get_wealth(name):
    return _get_cls(name, dict=_WEALTH)


class Wealth(object):
    def __init__(self, config):
        self.w = 1.0
        self.wealth = [self.w]

    @abstractmethod
    def update(self, payoff):
        pass


@register_wealth("mixture")
class Mixture(Wealth):
    def __init__(self, config):
        super().__init__(config)

        self.grid_size = grid_size = config.grid_size
        self.wealth = np.ones((grid_size,))
        self.wealth_flag = np.ones(grid_size, dtype=bool)
        self.v = np.linspace(0.05, 0.95, grid_size)

    def update(self, payoff):
        raise NotImplementedError


@register_wealth("ons")
class ONS(Wealth):
    def __init__(self, config):
        super().__init__(config)

        self.v = 0
        self.a = 1

        self.min_v, self.max_v = config.get("min_v", 0), config.get("max_v", 1 / 2)
        self.wealth_flag = False

    def _update_v(self, payoff):
        z = payoff / (1 + self.v * payoff)
        self.a += z**2
        self.v = max(
            self.min_v, min(self.max_v, self.v + 2 / (2 - np.log(3)) * z / self.a)
        )

    def update(self, payoff):
        w = self.w * (1 + self.v * payoff)

        if w >= 0 and not self.wealth_flag:
            self.w = w
            self.wealth.append(self.w)
            self._update_v(payoff)
        else:
            self.wealth_flag = True
