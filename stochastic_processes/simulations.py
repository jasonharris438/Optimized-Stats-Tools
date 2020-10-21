"""Functions that implement O-U and langevin process simulations using Jit for speed improvements"""

from numpy import zeros, exp, int64, float64
import numba as nb


@nb.jit(nb.types.float64[:,:](nb.types.float64[:,:], nb.types.float64[:,:], nb.types.float64,
                              nb.types.float64, nb.types.int32, nb.types.float64), nopython=True)
def _langevin_simulations_inner_loop(x, dW, mu=2.0, sigma=0.4, m=1, delta=1.0):

    """
    Function that performs n many simultaneous simulation of general langevin process given a set of parameters.

    :param x:           (numpy.array) Required. 2-d array containing zeros. To be populated with simulations.

    :param dW:          (numpy.array) Required. 2-d array containing random normal distribution draws.
                                                Used as the noise component in O-U simulations.

    :param mu:          (float) Optional. Parameter mu from O-U process.

    :param sigma:       (float) Optional. Parameter sigma from O-U process.

    :param m:           (int) Optional. Length of simulation (number of obs).

    :param delta:       (float) Optional. Parameter for time increment size.

    :returns x          (numpy.ndarray). 2-d Array of floats containing langevin process simulations.
    """

    for i in range(m - 1):
            x[i + 1, :] = (x[i,:] * exp(mu * delta)) + (sigma * dW[i, :])

    return x


@nb.jit(nb.types.float64[:,:](nb.types.float64[:,:], nb.types.float64[:,:], nb.types.float64,
                              nb.types.float64, nb.types.float64, nb.types.int32, nb.types.float64),
                              nopython=True)
def _mean_reverting_simulations_inner_loop(x, dW, mu=2.0, alpha=0.01, sigma=0.4, m=1, delta=1.0):

    """
    Function that performs n many simultaneous simulation of O-U process given a set of parameters.

    :param x:           (numpy.array) Required. 2-d array containing zeros. To be populated with simulations.

    :param dW:          (numpy.array) Required. 2-d array containing random normal distribution draws.
                                                Used as the noise component in O-U simulations.

    :param mu:          (float) Optional. Parameter mu from O-U process.

    :param alpha:       (float) Optional. Parameter alpha from O-U process.

    :param sigma:       (float) Optional. Parameter sigma from O-U process.

    :param m:           (int) Optional. Length of simulation (number of obs).

    :param delta:       (float) Optional. Parameter for time increment size.

    :returns x          (numpy.ndarray). 2-d Array of floats containing O-U process simulations.
    """

    for i in range(m - 1):
            x[i + 1, :] = (x[i,:] * exp(-alpha * delta)) + \
                        (mu * (1 - exp(-alpha * delta))) + \
                        (sigma * dW[i, :])

    return x


@nb.jit(nb.types.float64[:,:](nb.types.int64, nb.types.float64[:,:], nb.types.float64, nb.types.float64,
                              nb.types.float64, nb.types.float64, nb.types.int64,
                              nb.types.int64, nb.types.float64), nopython=True)
def simulations(type, dW, start=0.0, mu=2.0, alpha=0.01, sigma=0.4, m=1, n=10000, delta=1.0):

    """
    Function that performs n many simultaneous simulation of O-U process given a set of parameters.

    :param dW:          (numpy.array) Required. 2-d array containing random normal distribution draws.
                                                Used as the noise component in O-U simulations.

    :param start:       (float) Optional. The starting x_0 value for the simulation.

    :param mu:          (float) Optional. Parameter mu from O-U process.

    :param alpha:       (float) Optional. Parameter alpha from O-U process.

    :param sigma:       (float) Optional. Parameter sigma from O-U process.

    :param m:           (int) Optional. Length of simulation (number of obs).

    :param n:           (int) Optional. Number of simulations.

    :param delta:       (float) Optional. Parameter for time increment size.

    :returns sims       (numpy.ndarray). 2-d Array of floats containing O-U process simulations.
    """

    x = zeros((m, n)).astype(float64)
    x[0, :] = start

    if type == 0:
        sims = _mean_reverting_simulations_inner_loop(x=x, dW=dW, mu=mu, alpha=alpha, sigma=sigma, m=m, delta=delta)
    elif type == 1:
        sims = _langevin_simulations_inner_loop(x=x, dW=dW, mu=mu, sigma=sigma, m=m, delta=delta)
    return sims