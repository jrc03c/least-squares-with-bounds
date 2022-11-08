from numpy.random import normal
from numpy import dot
from pyds import leastSquares, rScore
from least_squares_with_bounds import leastSquaresWithBounds

a = normal(size=[100, 75])
x = normal(size=[75, 50])
b = dot(a, x)
b += 1e-5 * normal(size=b.shape)

xPred1 = leastSquares(a, b)
xPred2 = leastSquaresWithBounds(a, b)
print(rScore(xPred1, xPred2))
