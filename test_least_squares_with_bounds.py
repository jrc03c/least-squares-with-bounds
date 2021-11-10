import unittest
from least_squares_with_bounds import leastSquaresWithBounds
from numpy import dot
from numpy.random import seed, normal
from pyds import flatten, rScore


class LeastSquaresWithBoundsTestCase(unittest.TestCase):
    def test(self):
        seed(12345)
        a = normal(size=[100, 50])
        x = normal(size=[50, 25])
        b = dot(a, x)
        b += 0.01 * normal(size=b.shape)

        xPred = leastSquaresWithBounds(a, b, bounds=None)
        self.assertGreater(rScore(x, xPred), 0.95)

    def testErrors(self):
        pass

