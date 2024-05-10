# The goal of this program is to find the best x that best satisfies the equation ax = b given certain constraints on x (e.g., that x-values can only be in certain ranges).

from numpy import array, dot, reshape, sum
from pyds import (
    contains_only_numbers,
    flatten,
    is_a_numpy_array,
    is_a_pandas_dataframe,
    is_a_tensor,
    is_iterable,
    least_squares,
)
from scipy.optimize import minimize

from .flatten_to_tuples import flatten_to_tuples
from .random_in_range import random_in_range


def least_squares_with_bounds(a, b, bounds=None):
    a_error_message = "`a` must be a tensor that contains only numbers!"
    b_error_message = "`b` must be a tensor that contains only numbers!"

    assert is_a_tensor(a), a_error_message
    assert contains_only_numbers(a), a_error_message
    assert is_a_tensor(b), b_error_message
    assert contains_only_numbers(b), b_error_message

    if is_a_pandas_dataframe(a):
        a = a.values

    if not is_a_numpy_array(a):
        a = array(a)

    if is_a_pandas_dataframe(b):
        b = b.values

    if not is_a_numpy_array(b):
        b = array(b)

    if bounds is not None:
        bounds_error_message = "`bounds` must be an array of tuples where each tuple is a pair of minimum and maximum values!"

        assert is_iterable(bounds), bounds_error_message
        bounds = flatten_to_tuples(bounds)

        for item in bounds:
            assert len(item) == 2, bounds_error_message
            assert contains_only_numbers(item), bounds_error_message

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
        return least_squares(a, b)

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
