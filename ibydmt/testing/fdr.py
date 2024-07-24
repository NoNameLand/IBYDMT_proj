from typing import List

import numpy as np

from ibydmt.testing.wealth import Wealth


class FDRPostProcessor:
    def __call__(
        self, significance_level: float, wealths: List[Wealth], tau_max: int = None
    ):
        k = len(wealths)
        tau_max = tau_max or max([len(w.wealth) for w in wealths])

        rejected = np.zeros(k, dtype=bool)
        tau = (tau_max - 1) * np.ones(k, dtype=int)

        _wealths = np.stack(
            [
                np.pad(
                    w.wealth,
                    (0, tau_max - len(w.wealth)),
                    mode="constant",
                    constant_values=-np.inf,
                )
                for w in wealths
            ]
        )

        for n in range(1, k + 1):
            threshold = k / (significance_level * n)
            _idx, _tau = np.nonzero(_wealths >= threshold)
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
