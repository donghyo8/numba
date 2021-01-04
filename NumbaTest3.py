from math import cos, log
from numba import jit

@jit(nopython=True)
def add(a, b):
    return a + b

add(5,10)