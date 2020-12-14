from numpy import abs, exp
from scipy.stats import poisson
from numba import jit, prange, types


@jit(types.int64(types.int64), nopython=True)
def factorial(x):

    """
    Numba-styled implementation of factorial calculation.

    :param x:   (int) Required.

    :returns n: (int).
    """

    n = 1
    for i in range(2, x+1):
        n *= i
    return n


@jit(types.float64(types.int64, types.float64), nopython=True)
def pois_survival_partial(x, mu):

    """
    Implements the partial derivative of the poisson CDF with respect to lambda.

    :param x:        (int) Required. The value in which the CDF is calculated from.
                                     Density is integers to the right of this number.

    :param mu:       (float) Required. The mean at which to calculate the density with.

    :returns val:    (float) Required. The derivative of the cumulative density.
    """

    ret_val = 0
    for i in range(x+1):
        summation = ((i * (mu**(i-1))) - mu**i) / factorial(i)
        ret_val += summation

    return exp(-mu) * ret_val


@jit(types.float64(types.int64, types.float64), nopython=True)
def poisson_cdf(x, mu):

    """
    Generates Numba-styled Poisson CDF calculation.

    :param x:   (int) Required. Point x (inclusive).

    :param mu:  (float) Required. Mean of distribution.

    :returns density: (float)
    """

    density = 0
    for i in range(x+1):
        point = exp(-mu) * (mu**i) / factorial(i)
        density += point

    return density


@jit(types.float64(types.int64, types.float64), nopython=True)
def newton_poisson_solver(thres, target):

    """
    Newton-Raphson method of solving the Poisson CDF for lambda (mean).
    This might break in some instances, but is a much more efficient way
    than brute force calculations. This uses the loss function of target cumulative
    density less the density at each step.

    Currently not able to deal with small target probability values. The mu variable (mean) will become negative
    with target values < ~0.06. The function defaults to returning a value of 0.05.
    Open to suggestions on how to overcome this!

    :param thres:   (int) Required. Number of trials inputted into survival function (i.e. one less than
                                    your desired cdf density x value).

    :param target:  (float) Required. The probability ( P(X=x) ) that is tested for.

    :returns mu:    (float). The value of lambda with the smallest deviation from the target.
    """

    mu = thres + 1
    fx = 0
    while abs(target - fx) > 0.0001:
        fx = 1 - poisson_cdf(thres, mu) #Create a function to generate the survival function.
        mu -= ((target - fx) / pois_survival_partial(thres, mu))

        if mu < 0:
            return 0.05
    return mu