import numpy as np
from numba import njit


@njit()
def linear_gem(x, p=3):
    nobs, ndim = x.shape

    y = np.zeros(ndim)
    for r in range(nobs):
        for c in range(ndim):
            y[c] += x[r, c] ** p

    y /= nobs

    for c in range(ndim):
        y[c] = y[c] ** (1 / p)

    return y

# segments the columns to run in parallel
@njit
def segmented_gem(x, start: int, end: int, p=3):
    nobs, _ = x.shape

    y = np.zeros(end-start)
    for r in range(nobs):
        for c in range(end-start):
            y[c] += x[r, start+c] ** p

    y /= nobs

    for c in range(end-start):
        y[c] = y[c] ** (1 / p)

    return y

# thread_helper_gem can be used as the target for a Thread is requires all the segmented_gem params and an additional list to add the output to
def thread_helper_gem(result: list, thread_idx: int, scores: list, start: int, end: int, p: int) -> None:
    result[thread_idx] = segmented_gem(scores, start, end, p)
    return