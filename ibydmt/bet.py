from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from ibydmt.utils import _get_cls, _register_cls


class Bet(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass


_BETS: Dict[str, Bet] = {}


def register_bet(name):
    return _register_cls(name, dict=_BETS)


def get_bet(name):
    return _get_cls(name, dict=_BETS)


@register_bet("sign")
class Sign(Bet):
    def __init__(self, config):
        super().__init__()
        self.m = config.get("m", 0.5)
        self.prev_g = []

    def compute(self, g):
        return self.m * np.sign(g)


@register_bet("tanh")
class Tanh(Bet):
    def __init__(self, config):
        super().__init__()
        self.alpha = config.get("alpha", 0.20)
        self.prev_g = []

    def compute(self, g):
        if len(self.prev_g) < 2:
            scale = 1
        else:
            l, u = np.quantile(self.prev_g, [self.alpha / 2, 1 - self.alpha / 2])
            scale = u - l

        self.prev_g.append(g)

        return np.tanh(g / np.clip(scale, 1e-04, None))
