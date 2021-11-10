import unittest
from least_squares_with_bounds import leastSquaresWithBounds
from numpy import dot, min, max
from numpy.random import seed, normal
from pyds import flatten, rScore


class LeastSquaresWithBoundsTestCase(unittest.TestCase):
    def testWithoutBounds(self):
        a = normal(size=[100, 50])
        x = normal(size=[50, 25])
        b = dot(a, x)
        b += 0.01 * normal(size=b.shape)

        xPred = leastSquaresWithBounds(a, b, bounds=None)
        self.assertGreater(rScore(x, xPred), 0.95)

    def testWithBounds(self):
        a = normal(size=[100, 50])
        x = normal(size=[50, 25])
        b = dot(a, x)
        b += 0.01 * normal(size=b.shape)
        bounds = [(0, 1) for value in flatten(x)]

        xPred = leastSquaresWithBounds(a, b, bounds=bounds)
        self.assertGreater(rScore(x, xPred), 0.5)
        self.assertGreaterEqual(min(xPred), 0)
        self.assertLessEqual(max(xPred), 1)

    def testErrors(self):
        pass

