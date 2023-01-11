# The goal of this program is to find the best x that best satisfies the equation ax = b given certain constraints on x (e.g., that x-values can only be in certain ranges).

from numpy import array, dot, reshape, shape, sum
from numpy.random import random
from scipy.optimize import minimize

from pyds import (
    flatten,
    isATensor,
    containsOnlyNumbers,
    isAPandasDataFrame,
    isANumber,
    isANumpyArray,
    isIterable,
    leastSquares,
)


def random_in_range(rmin, rmax):
    assert isANumber(rmin), "`rmin` must be a number!"
    assert isANumber(rmax), "`rmax` must be a number!"
    return random() * (rmax - rmin) + rmin


def flatten_to_tuples(x):
    assert isATensor(x), "`x` must be a tensor!"
    x_flat = flatten(x)
    return reshape(x_flat, [len(x_flat) // 2, 2])


def least_squares_with_bounds(a, b, bounds=None):
    aErrorMessage = "`a` must be a tensor that contains only numbers!"
    bErrorMessage = "`b` must be a tensor that contains only numbers!"

    assert isATensor(a), aErrorMessage
    assert containsOnlyNumbers(a), aErrorMessage
    assert isATensor(b), bErrorMessage
    assert containsOnlyNumbers(b), bErrorMessage

    if isAPandasDataFrame(a):
        a = a.values

    if not isANumpyArray(a):
        a = array(a)

    if isAPandasDataFrame(b):
        b = b.values

    if not isANumpyArray(b):
        b = array(b)

    if bounds is not None:
        bounds_error_message = "`bounds` must be an array of tuples where each tuple is a pair of minimum and maximum values!"

        assert isIterable(bounds), bounds_error_message
        bounds = flatten_to_tuples(bounds)

        for item in bounds:
            assert len(item) == 2, bounds_error_message
            assert containsOnlyNumbers(item), bounds_error_message

            assert (
                item[0] <= item[1]
            ), "Each `bounds` tuple must be in the form `(min, max)`!"

    x_shape = [a.shape[1], b.shape[1]]

    # ax = b
    # objective = sum((ax - b)^2)
    def objective(x_flat):
        return sum((dot(a, reshape(x_flat, x_shape)) - b) ** 2)

    # gradient = d/d_x of objective
    # = d/d_x of sum((ax - b)^2)
    # = 2 * a * (ax - b)
    def gradient(x_flat):
        return flatten(2 * dot(a.T, (dot(a, reshape(x_flat, x_shape)) - b)))

    # if there are no bounds, then just return OLS(a, b)
    if bounds is None:
        return leastSquares(a, b)

    # otherwise, set up x_init to fall within bounds (and to be flattened!)
    else:
        x_init = []

        for bound in bounds:
            x_init.append(random_in_range(bound[0], bound[1]))

    results = minimize(
        objective,
        x_init,
        args=(),
        method="TNC",
        jac=gradient,
        bounds=bounds,
        tol=None,
        callback=None,
        options={
            "accuracy": 0,
            "eps": 1e-08,
            "eta": -1,
            "finite_diff_rel_step": None,
            "ftol": -1,
            "gtol": -1,
            "maxCGit": -1,
            "maxfun": None,
            "mesg_num": None,
            "minfev": 0,
            "offset": None,
            "rescale": -1,
            "scale": None,
            "stepmx": 0,
            "xtol": -1,
        },
    )

    return reshape(results.x, x_shape)
