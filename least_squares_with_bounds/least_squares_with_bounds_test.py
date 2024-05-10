from numpy import dot, max, min, shape
from numpy.random import normal, random
from pyds import apply, least_squares, r_score

from .least_squares_with_bounds import least_squares_with_bounds


# test that running the `least_squares_with_bounds` function with no bounds values yields the same results as simply running `least_squares`
def test_without_bounds():
    a = normal(size=[100, 75])
    x = normal(size=[75, 50])
    b = dot(a, x)
    b += 1e-5 * normal(size=shape(b))
    x_pred_1 = least_squares(a, b)
    x_pred_2 = least_squares_with_bounds(a, b)
    assert r_score(x_pred_1, x_pred_2) > 0.99


# test that bounds are obeyed when given
def test_with_bounds():
    a = normal(size=[100, 50])
    x = normal(size=[50, 25])
    b = dot(a, x)
    b += 0.01 * normal(size=shape(b))
    bounds = apply(lambda: [0, 1], x)
    x_pred = least_squares_with_bounds(a, b, bounds=bounds)
    assert r_score(x, x_pred) > 0.5
    assert min(x_pred) >= 0
    assert max(x_pred) <= 1


# test that errors are thrown when using wrong kinds of data
def test_errors():
    a = normal(size=[10, 10])
    b = normal(size=[10, 10])
    a_missing = apply(lambda x: None if random() < 0.1 else x, a)
    b_missing = apply(lambda x: None if random() < 0.1 else x, b)
    bounds = [(0, 1) for i in range(0, 100)]

    bounds_wrong_1 = list(
        map(lambda pair: (pair[1], pair[0]) if random() < 0.1 else pair, bounds)
    )

    bounds_wrong_2 = list(map(lambda pair: "foo" if random() < 0.1 else pair, bounds))

    wrongs = [
        [a, b, bounds_wrong_1],
        [a, b, bounds_wrong_2],
        [a_missing, b, bounds],
        [a, b_missing, bounds],
        [True, False, True],
        [None, None, None],
        ["foo", "bar", "baz"],
        [{}, {}, {}],
        [2, 3, 4],
        [lambda x: x, lambda x: x, lambda x: x],
    ]

    for trio in wrongs:
        raised = False

        try:
            least_squares_with_bounds(trio[0], trio[1], trio[2])

        except Exception:
            raised = True

        assert raised
