# Intro

This package makes it a little easier to do linear regression when bounds need to be applied to the solution. It's basically just a wrapper around [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html), but it provides defaults for my current specific use case (e.g., the solver is set to `"TNC"`, the bounds are specified as an array of tuples, etc.).

# Installation

Install with `pip`:

```bash
pip install -U git+https://github.com/jrc03c/least-squares-with-bounds
```

Or instal with `conda`:

```bash
conda install pip git
pip install -U git+https://github.com/jrc03c/least-squares-with-bounds
```

# Usage

```py
from least_squares_with_bounds import least_squares_with_bounds

a = ...
b = ...
bounds = ...

x = least_squares_with_bounds(a, b, bounds)
```

# API

## `least_squares_with_bounds(a, b, bounds=None)`

Assuming an equation of the form $ax = b$, this function will solve for $x$ given $a$, $b$, and an array of bounds on $x$. The `bounds` parameter can either be in the shape that $x$ should be in (but where each value is a tuple rather than a single number), or it can be a 1-dimensional array of tuples.

# Example

Here's a fuller example that forces the values of $x$ to fall in the range $[0, 1]$ and then confirms that that is in fact the case. (I'm sure there are more Pythonic ways of doing things than what I've done below, so please pardon my ignorance!)

```py
from least_squares_with_bounds import least_squares_with_bounds
from numpy import dot, max, min
from numpy.random import normal

# generate `a` and `x`
a = normal(size=[100, 75])
x_true = normal(size=[75, 50])

# compute `b`, the dot product of `a` and `x` (plus a little noise)
b = dot(a, x_true)
b += 1e-5 * normal(size=b.shape)

# define bounds for `x`
bounds = []

for i in range(0, x_true.shape[0]):
    row = []

    for j in range(0, x_true.shape[1]):
        bound = [0, 1]
        row.append(bound)

    bounds.append(row)

# solve for `x`
x_pred = least_squares_with_bounds(a, b, bounds)

# confirm that `x` falls within the specified bounds
x_pred_min = min(x_pred)
x_pred_max = max(x_pred)

assert x_pred_min >= 0, "Uh-oh! The lowest value in x_pred is %.4f!" % (x_pred_min)

assert x_pred_max <= 1, "Uh-oh! The highest value in x_pred is %.4f!" % (x_pred_max)

print("Yay! All of the values in x_pred were correctly bounded to the range [0, 1]!")
```
