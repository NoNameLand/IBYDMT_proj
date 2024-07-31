from abc import abstractmethod
from functools import reduce

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel

from ibydmt.testing.bet import Bet, get_bet
from ibydmt.utils.config import Config


class Payoff:
    def __init__(self, config: Config):
        self.bet: Bet = get_bet(config.testing.bet)()

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass


class Kernel:
    def __init__(self, kernel: str, scale_method: str, scale: float):
        if kernel == "linear":
            self.base_kernel = linear_kernel
        elif kernel == "rbf":
            self.base_kernel = rbf_kernel

            self.scale_method = scale_method
            self.scale = scale

            self.gamma = None
            self.recompute_gamma = True
            self.prev = None
        else:
            raise NotImplementedError(f"{kernel} is not implemented")

    def __call__(self, x, y):
        if self.base_kernel == linear_kernel:
            return self.base_kernel(x, y)
        if self.base_kernel == rbf_kernel:
            if self.scale_method == "constant":
                self.gamma = self.scale
            elif self.scale_method == "quantile":
                if self.prev is None:
                    self.prev = y

                if self.recompute_gamma:
                    dist = pairwise_distances(
                        self.prev.reshape(-1, self.prev.shape[-1])
                    )
                    scale = np.quantile(dist, self.scale)
                    gamma = 1 / (2 * scale**2) if scale > 0 else None
                    self.gamma = gamma

                    if len(self.prev) > 100:
                        self.recompute_gamma = False
                    self.prev = np.vstack([self.prev, x])
            else:
                raise NotImplementedError(
                    f"{self.scale} is not implemented for rbf_kernel"
                )
            return self.base_kernel(x, y, gamma=self.gamma)


class KernelPayoff(Payoff):
    def __init__(self, config: Config):
        super().__init__(config)

        self.kernel = config.testing.kernel
        self.scale_method = config.testing.kernel_scale_method
        self.scale = config.testing.kernel_scale

    @abstractmethod
    def witness_function(self, d, prev):
        pass

    def compute(self, d, null_d, prev):
        g = reduce(
            lambda acc, u: acc
            + self.witness_function(u[0], prev)
            - self.witness_function(u[1], prev),
            zip(d, null_d),
            0,
        )

        return self.bet.compute(g)


class HSIC(KernelPayoff):
    def __init__(self, config):
        super().__init__(config)

        kernel = self.kernel
        scale_method = self.scale_method
        scale = self.scale

        self.kernel_y = Kernel(kernel, scale_method, scale)
        self.kernel_z = Kernel(kernel, scale_method, scale)

    def witness_function(self, d, prev_d):
        y, z = d
        prev_y, prev_z = prev_d[:, 0], prev_d[:, 1]

        y_mat = self.kernel_y(y.reshape(-1, 1), prev_y.reshape(-1, 1))
        z_mat = self.kernel_z(z.reshape(-1, 1), prev_z.reshape(-1, 1))

        mu_joint = np.mean(y_mat * z_mat)
        mu_prod = np.mean(y_mat, axis=1) @ np.mean(z_mat, axis=1)
        return mu_joint - mu_prod


class cMMD(KernelPayoff):
    def __init__(self, config):
        super().__init__(config)

        kernel = self.kernel
        scale_method = self.scale_method
        scale = self.scale

        self.kernel_y = Kernel(kernel, scale_method, scale)
        self.kernel_zj = Kernel(kernel, scale_method, scale)
        self.kernel_cond_z = Kernel(kernel, scale_method, scale)

    def witness_function(self, d, prev_d):
        y, zj, cond_z = d[0], d[1], d[2:]

        prev_y, prev_zj, prev_null_zj, prev_cond_z = (
            prev_d[:, 0],
            prev_d[:, 1],
            prev_d[:, 2],
            prev_d[:, 3:],
        )

        y_mat = self.kernel_y(y.reshape(-1, 1), prev_y.reshape(-1, 1))
        zj_mat = self.kernel_zj(zj.reshape(-1, 1), prev_zj.reshape(-1, 1))
        cond_z_mat = self.kernel_cond_z(
            cond_z.reshape(-1, prev_cond_z.shape[1]),
            prev_cond_z.reshape(-1, prev_cond_z.shape[1]),
        )
        null_zj_mat = self.kernel_zj(zj.reshape(-1, 1), prev_null_zj.reshape(-1, 1))

        mu = np.mean(y_mat * zj_mat * cond_z_mat)
        mu_null = np.mean(y_mat * null_zj_mat * cond_z_mat)
        return mu - mu_null


class xMMD(KernelPayoff):
    def __init__(self, config):
        super().__init__(config)

        self.kernel = Kernel(self.kernel, self.scale_method, self.scale)

    def witness_function(self, d, prev_d):
        prev_y, prev_y_null = prev_d[:, 0], prev_d[:, 1]

        mu_y = np.mean(self.kernel(d.reshape(-1, 1), prev_y.reshape(-1, 1)))
        mu_y_null = np.mean(self.kernel(d.reshape(-1, 1), prev_y_null.reshape(-1, 1)))
        return mu_y - mu_y_null
