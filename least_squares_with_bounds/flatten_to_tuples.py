from numpy import reshape
from pyds import flatten, is_a_tensor


def flatten_to_tuples(x):
    assert is_a_tensor(x), "`x` must be a tensor!"
    x_flat = flatten(x)
    return reshape(x_flat, [len(x_flat) // 2, 2])
