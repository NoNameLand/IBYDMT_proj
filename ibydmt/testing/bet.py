from abc import abstractmethod

import numpy as np

bets = {}


def register_bet(name):
    def register(cls):
        if name in bets:
            raise ValueError(f"Bet {name} is already registered")
        bets[name] = cls

    return register


def get_bet(name):
    return bets[name]


class Bet:
    def __init__(self):
        pass

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass


@register_bet("sign")
class Sign(Bet):
    def __init__(self):
        super().__init__()
        self.m = 0.5
        self.prev_g = []

    def compute(self, g):
        return self.m * np.sign(g)


@register_bet("tanh")
class Tanh(Bet):
    def __init__(self):
        super().__init__()
        self.alpha = 0.20
        self.prev_g = []

    def compute(self, g):
        if len(self.prev_g) < 2:
            scale = 1
        else:
            l, u = np.quantile(self.prev_g, [self.alpha / 2, 1 - self.alpha / 2])
            scale = u - l

        self.prev_g.append(g)

        return np.tanh(g / np.clip(scale, 1e-04, None))
