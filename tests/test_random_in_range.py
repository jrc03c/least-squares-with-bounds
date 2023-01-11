from least_squares_with_bounds import random_in_range
from numpy.random import normal
import unittest


class RandomInRangeTestCase(unittest.TestCase):
    def testRights(self):
        for _ in range(0, 1000):
            rmin = -abs(normal() * 100)
            rmax = abs(normal() * 100)
            r = random_in_range(rmin, rmax)
            self.assertGreater(r, rmin)
            self.assertLess(r, rmax)

    def testWrongs(self):
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
                self.assertRaises(Exception, random_in_range, a, b)
