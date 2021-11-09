# The goal of this program is to find the best x that best satisfies the equation ax = b given certain constraints on x (e.g., that x-values can only be in certain ranges).

from numpy import dot, reshape, sum, min, max
from numpy.random import normal
from scipy.optimize import minimize
from autograd import grad
from pyds import flatten, rScore, apply


def getObjective(a, x, b):
    def objective(xFlat):
        xTemp = reshape(xFlat, x.shape)
        return sum((b - dot(a, xTemp)) ** 2)

    return objective


a = normal(size=[10, 7])
x = normal(size=[7, 5])
b = dot(a, x)
b += 0.01 * normal(size=b.shape)

objective = getObjective(a, x, b)
gradient = grad(objective)
xInit = normal(size=x.shape)
bounds = [(-1, 1) for i in range(0, len(flatten(x)))]

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

print("r-score:", rScore(x, reshape(results.x, x.shape)))
print("min:", min(results.x))
print("max:", max(results.x))
