from typing import Iterable

import numpy as np

from ibydmt.testing.wealth import Wealth
from ibydmt.utils.config import Config


class FDRPostProcessor:
    def __init__(self, config: Config):
        self.significance_level = config.testing.significance_level
        self.tau_max = config.testing.tau_max

    def __call__(self, wealths: Iterable[Wealth]):
        k = len(wealths)

        rejected = np.zeros(k, dtype=bool)
        tau = (self.tau_max - 1) * np.ones(k, dtype=int)

        _wealths = np.stack(
            [
                np.pad(
                    w.wealth,
                    (0, self.tau_max - len(w.wealth)),
                    mode="constant",
                    constant_values=-np.inf,
                )
                for w in wealths
            ]
        )

        for n in range(1, k + 1):
            threshold = k / (self.significance_level * n)
            _idx, _tau = np.nonzero(_wealths > threshold)
            if len(_idx) == 0:
                break

            first_idx = np.argmin(_tau)
            rejected_idx = _idx[first_idx]
            rejected_tau = _tau[first_idx]

            rejected[rejected_idx] = True
            tau[rejected_idx] = rejected_tau

            _wealths[rejected_idx] = -np.inf

        rejected = tuple(rejected)
        tau = tuple(tau)
        return rejected, tau
