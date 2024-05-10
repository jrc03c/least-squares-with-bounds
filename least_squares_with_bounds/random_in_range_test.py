from numpy.random import normal

from .random_in_range import random_in_range


def test_rights():
    for _ in range(0, 1000):
        rmin = -abs(normal() * 100)
        rmax = abs(normal() * 100)
        r = random_in_range(rmin, rmax)
        assert r > rmin
        assert r < rmax


def test_wrongs():
    wrongs = [
        "Hello, world!",
        True,
        False,
        None,
        [2, 3, 4],
        [[2, 3, 4], [5, 6, 7]],
        normal(size=[2, 3, 4, 5, 6]),
        {"hello": "world"},
        lambda x: x,
        random_in_range,
    ]

    for i in range(0, len(wrongs) - 1):
        for j in range(i + 1, len(wrongs)):
            a = wrongs[i]
            b = wrongs[j]
            raised = False

            try:
                random_in_range(a, b)

            except Exception:
                raised = True

            assert raised
