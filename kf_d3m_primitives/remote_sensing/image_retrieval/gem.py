import numpy as np
from numba import njit


@njit()
def gem(x, p=3):
    nobs, ndim = x.shape

    y = np.zeros(ndim)
    for r in range(nobs):
        for c in range(ndim):
            y[c] += x[r, c] ** p

    y /= nobs

    for c in range(ndim):
        y[c] = y[c] ** (1 / p)

    return y