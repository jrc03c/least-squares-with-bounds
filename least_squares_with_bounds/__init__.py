# The goal of this program is to find the best x that best satisfies the equation ax = b given certain constraints on x (e.g., that x-values can only be in certain ranges).

from numpy import dot, reshape, sum, array
from numpy.random import random, normal
from scipy.optimize import minimize

from pyds import (
    flatten,
    isATensor,
    containsOnlyNumbers,
    isAPandasDataFrame,
    isAPandasSeries,
    isANumpyArray,
    isIterable,
    leastSquares,
)


def randomInRange(rmin, rmax):
    return random() * (rmax - rmin) + rmin


def leastSquaresWithBounds(a, b, bounds=None):
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
        tupleType = type((2, 3, 4))
        boundsErrorMessage = "`bounds` must be a one-dimensional array of tuples where each tuple is a pair of minimum and maximum values!"

        assert isIterable(bounds), boundsErrorMessage

        for item in bounds:
            assert type(item) == tupleType, boundsErrorMessage
            assert len(item) == 2, boundsErrorMessage
            assert containsOnlyNumbers(item), boundsErrorMessage

            assert (
                item[0] <= item[1]
            ), "Each `bounds` tuple must be in the form `(min, max)`!"

    xShape = [a.shape[1], b.shape[1]]

    # ax = b
    # objective = sum((ax - b)^2)
    def objective(xFlat):
        return sum((dot(a, reshape(xFlat, xShape)) - b) ** 2)

    # gradient = d/d_x of objective
    # = d/d_x of sum((ax - b)^2)
    # = 2 * a * (ax - b)
    def gradient(xFlat):
        return flatten(2 * dot(a.T, (dot(a, reshape(xFlat, xShape)) - b)))

    if bounds is None:
        # reallistically speaking, if there are no bounds, then we should just return
        # OLS(a, b); i.e.:
        # x = leastSquares(a, b)
        # return x

        # if there are no bounds, xInit is random (?)
        xInit = normal(size=xShape)

    # otherwise, set up xInit to fall within bounds
    else:
        xInit = []

        for i in range(0, xShape[0]):
            row = []

            for j in range(0, xShape[1]):
                pair = bounds[i * xShape[1] + j]
                pmin = pair[0]
                pmax = pair[1]
                row.append(randomInRange(pmin, pmax))

            xInit.append(row)

        xInit = array(xInit)

    results = minimize(
        objective,
        xInit,
        args=(),
        method="TNC",
        jac=gradient,
        bounds=bounds,
        tol=None,
        callback=None,
        options={
            "eps": 1e-08,
            "scale": None,
            "offset": None,
            "mesg_num": None,
            "maxCGit": -1,
            "maxiter": None,
            "eta": -1,
            "stepmx": 0,
            "accuracy": 0,
            "minfev": 0,
            "ftol": -1,
            "xtol": -1,
            "gtol": -1,
            "rescale": -1,
            "disp": False,
            "finite_diff_rel_step": None,
            "maxfun": None,
        },
    )

    return reshape(results.x, xShape)

