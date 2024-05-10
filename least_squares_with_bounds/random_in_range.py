from numpy.random import random
from pyds import is_a_number


def random_in_range(rmin, rmax):
    assert is_a_number(rmin), "`rmin` must be a number!"
    assert is_a_number(rmax), "`rmax` must be a number!"
    return random() * (rmax - rmin) + rmin
