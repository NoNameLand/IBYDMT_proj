from typing import Tuple

import numpy as np
from jaxtyping import Float
from scipy.special import expit as sigmoid

rng = np.random.default_rng()

mu1, sigma1 = 1, 1
mu2, sigma2 = -1, 1
sigma3 = 1


def model(
    z: Float[np.ndarray, "N 3"], beta: Float[np.ndarray, "3"] = None
) -> Float[np.ndarray, "N"]:
    _beta = np.tile(beta[None, :], (z.shape[0], 1))

    gated_idx = 1
    _beta[:, gated_idx] = z[:, 2] * beta[gated_idx]
    return sigmoid(np.sum(_beta * z, axis=1))


def sample_z(n: int) -> Float[np.ndarray, "N 3"]:
    z1 = rng.normal(mu1, sigma1, size=n)
    z2 = rng.normal(mu2, sigma2, size=n)
    z3 = z1 + rng.normal(0, sigma3, size=n)
    return np.stack([z1, z2, z3], axis=1)


def sample(
    n: int, beta: Float[np.ndarray, "D"]
) -> Tuple[Float[np.ndarray, "N 3"], Float[np.ndarray, "N"]]:
    z = sample_z(n)
    mu = model(z, beta=beta)
    y = mu + rng.normal(0, 0.01, size=n)
    return z, y


def cond_p(
    z: Float[np.ndarray, "N 3"], cond_idx: list[int], m: int = 1
) -> Float[np.ndarray, "N 3"]:
    if z.ndim == 1:
        z = z[None, :]
        z = np.tile(z, (m, 1))

    n, d = z.shape

    if len(cond_idx) == 0:
        cond_z = sample_z(m)
    if len(cond_idx) == d:
        cond_z = z
    else:
        cond_z = np.empty((n, d))

        cond_z[:, 1] = z[:, 1] if 1 in cond_idx else rng.normal(mu2, sigma2, size=n)

        if 2 in cond_idx:
            cond_z[:, 2] = z3 = z[:, 2]

            tot_var = sigma3**1 + sigma1**2
            gamma1, gamma3 = sigma1**2 / tot_var, sigma3**2 / tot_var
            cond_mu = gamma1 * z3 + gamma3 * mu1
            cond_sigma = np.sqrt((1 / sigma1**2 + 1 / sigma3**2) ** -1)

            cond_z[:, 0] = cond_mu + rng.normal(0, cond_sigma, size=n)
        else:
            cond_z[:, 0] = z1 = (
                z[:, 0] if 0 in cond_idx else rng.normal(mu1, sigma1, size=n)
            )
            cond_z[:, 2] = z1 + rng.normal(0, sigma3, size=n)

    return cond_z
