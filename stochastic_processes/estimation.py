"""Provides numba-optimized maximum likelihood estimators of stochastic processes"""

import numpy as np
from numba import jit, types


@jit(nopython=True)
def maximum_likelihood_ou_mean_reversion(series, mu_static=0.0):

    """
    Function that applies the maximum likelihood method for O-U process parameters.
    Jit used to speed up calculation.

    :param series:          (np.array) Required. 1-d array of of numerics of dim (1,1) or (1,).
                                        The data for which paramters will be estimated.

    :param mu_static:       (float). Optional. Value for mu to be fixed to if required. Will halt calculation of mu from data.

    Returns: Tuple containing estimates for mu, alpha, sigma^2. All tuple members are of type float.

    Example -
        maximum_likelihood_ou_mean_reversion(X)
        Returns: (0.5, 1.0, 0.01)
    """

    x_i = series[1:].sum()
    x_i_sq = (series[1:] **2).sum()
    x_i_1 = series[:-1].sum()
    x_i_1_sq = (series[:-1] **2).sum()
    x_i_i_1 = (series[1:] * series[:-1]).sum()
    n = series.shape[0]

    if mu_static != 0.0:
        mu = mu_static
    else:
        mu = ((x_i * x_i_1_sq) - (x_i_1 * x_i_i_1)) / \
                ((n*(x_i_1_sq - x_i_i_1)) - ((x_i_1 **2) - (x_i * x_i_1)))

    alpha = -np.log(\
                    (x_i_i_1 + (mu * (-x_i - x_i_1 + (n * mu)))) / \
                    (x_i_1_sq + (mu * (-(2 * x_i_1) + (n * mu)))) \
                  )

    k = np.exp(-alpha)
    sigma = x_i_sq - k * ((2 * x_i_i_1) - (k * x_i_1_sq)) + \
            mu * ((2 * k * (1 - k) * x_i_1) - (2 * (1 - k) * x_i) + (n * mu * (1-k)**2))
    sigma *= (1/n)
    sigma *= ((2 * alpha) / (1 - (k**2)))
    return (mu, alpha, np.sqrt(sigma))


@jit(nopython=True)
def maximum_likelihood_general_langevin(series):

    """
    Function that applies the maximum likelihood method for general langevin process parameters.
    Jit used to speed up calculation.

    :param series:          (np.array) Required. 1-d array of of numerics of dim (1,1) or (1,).
                                        The data for which paramters will be estimated.

    Returns: Tuple containing estimates for mu and sigma^2. All tuple members are of type float.

    Example -
        maximum_likelihood_general_langevin(X)
        Returns: (0.5, 0.01)
    """

    x_i = series[1:].sum()
    x_i_sq = (series[1:] **2).sum()
    x_i_1 = series[:-1].sum()
    x_i_1_sq = (series[:-1] **2).sum()
    x_i_i_1 = (series[1:] * series[:-1]).sum()
    n = series.shape[0]

    mu = np.log(x_i_i_1/ x_i_1_sq)

    sigma = x_i_sq - (x_i_i_1 **2) / x_i_1_sq
    sigma *= (1/n)
    sigma *= ((2 * mu) / (np.exp(2 * mu) - 1))
    return (mu, np.sqrt(sigma))