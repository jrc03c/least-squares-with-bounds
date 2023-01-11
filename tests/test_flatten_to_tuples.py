import unittest
from least_squares_with_bounds import flatten_to_tuples
from numpy import shape
from numpy.random import normal, random
from pyds import flatten, isEqual


def fake_flatten_to_tuples(x):
    out = []
    xflat = flatten(x)

    for i in range(0, len(xflat) // 2):
        out.append(xflat[i : i + 2])

    return out


class FlattenToTuplesTestCase(unittest.TestCase):
    def test_rights(self):
        for _ in range(0, 100):
            xshape = [
                2 * int(random() * 5 + 1) for _ in range(0, int(random() * 5 + 1))
            ]

            x = normal(size=xshape)
            yTrue = fake_flatten_to_tuples(x)
            yPred = flatten_to_tuples(x)
            self.assertTrue(isEqual(shape(yTrue), shape(yPred)))

    def test_wrongs(self):
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
            self.assertRaises(Exception, flatten_to_tuples, wrong)
