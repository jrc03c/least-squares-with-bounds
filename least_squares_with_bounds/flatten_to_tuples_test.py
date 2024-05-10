from numpy import shape
from numpy.random import normal, random
from pyds import flatten, is_equal

from .flatten_to_tuples import flatten_to_tuples


def fake_flatten_to_tuples(x):
    out = []
    xflat = flatten(x)

    for i in range(0, len(xflat) // 2):
        out.append(xflat[i : i + 2])

    return out


def test_rights():
    for _ in range(0, 100):
        xshape = [2 * int(random() * 5 + 1) for _ in range(0, int(random() * 5 + 1))]
        x = normal(size=xshape)
        y_true = fake_flatten_to_tuples(x)
        y_pred = flatten_to_tuples(x)
        assert is_equal(shape(y_true), shape(y_pred))


def test_wrongs():
    wrongs = [
        0,
        -1,
        1,
        234,
        "Hello, world!",
        True,
        False,
        None,
        {"hello": "world"},
        lambda x: x,
        flatten_to_tuples,
    ]

    for wrong in wrongs:
        raised = False

        try:
            flatten_to_tuples(wrong)

        except Exception:
            raised = True

        assert raised
