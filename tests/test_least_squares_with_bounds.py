import unittest
from least_squares_with_bounds import least_squares_with_bounds
from numpy import dot, max, min, shape
from numpy.random import normal, random
from pyds import apply, leastSquares, rScore


class LeastSquaresWithBoundsTestCase(unittest.TestCase):
    # test that running the `leastSquaresWithoutBounds` function yields the same
    # results as simply running `leastSquares`
    def testWithoutBounds(self):
        a = normal(size=[100, 75])
        x = normal(size=[75, 50])
        b = dot(a, x)
        b += 1e-5 * normal(size=shape(b))

        xPred1 = leastSquares(a, b)
        xPred2 = least_squares_with_bounds(a, b)
        self.assertGreater(rScore(xPred1, xPred2), 0.99)

    # test that bounds are obeyed when given
    def testWithBounds(self):
        a = normal(size=[100, 50])
        x = normal(size=[50, 25])
        b = dot(a, x)
        b += 0.01 * normal(size=shape(b))
        bounds = apply(lambda: [0, 1], x)

        xPred = least_squares_with_bounds(a, b, bounds=bounds)
        self.assertGreater(rScore(x, xPred), 0.5)
        self.assertGreaterEqual(min(xPred), 0)
        self.assertLessEqual(max(xPred), 1)

    # test that errors are thrown when using wrong kinds of data
    def testErrors(self):
        a = normal(size=[10, 10])
        b = normal(size=[10, 10])
        aMissing = apply(lambda x: None if random() < 0.1 else x, a)
        bMissing = apply(lambda x: None if random() < 0.1 else x, b)
        bounds = [(0, 1) for i in range(0, 100)]

        boundsWrong1 = list(
            map(lambda pair: (pair[1], pair[0]) if random() < 0.1 else pair, bounds)
        )

        boundsWrong2 = list(map(lambda pair: "foo" if random() < 0.1 else pair, bounds))

        wrongs = [
            [a, b, boundsWrong1],
            [a, b, boundsWrong2],
            [aMissing, b, bounds],
            [a, bMissing, bounds],
            [True, False, True],
            [None, None, None],
            ["foo", "bar", "baz"],
            [{}, {}, {}],
            [2, 3, 4],
            [lambda x: x, lambda x: x, lambda x: x],
        ]

        for trio in wrongs:
            self.assertRaises(
                Exception, least_squares_with_bounds, trio[0], trio[1], trio[2]
            )
