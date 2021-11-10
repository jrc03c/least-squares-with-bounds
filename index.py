# The goal of this program is to find the best x that best satisfies the equation ax = b given certain constraints on x (e.g., that x-values can only be in certain ranges).

from numpy import dot, reshape, sum, min, max, array
from numpy.random import normal, seed, random
from scipy.optimize import minimize
from pyds import flatten, rScore, apply


def randomInRange(rmin, rmax):
    return random() * (rmax - rmin) + rmin


def leastSquaresWithBounds(a, b, bounds):
    xShape = [a.shape[1], b.shape[1]]
    objective = lambda xFlat: sum((b - dot(a, reshape(xFlat, xShape))) ** 2)
    gradient = lambda xFlat: flatten(-2 * dot(a.T, b - dot(a, reshape(xFlat, xShape))))

    # set up xInit to fall within bounds
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


if __name__ == "__main__":
    seed(12345)
    a = normal(size=[100, 50])
    x = normal(size=[50, 25])
    b = dot(a, x)
    b += 0.01 * normal(size=b.shape)
    bounds = [(-1, 1) for i in range(0, len(flatten(x)))]

    xPred = leastSquaresWithBounds(a, b, bounds)

    print("r-score:", rScore(x, xPred))
    print("min:", min(xPred))
    print("max:", max(xPred))
