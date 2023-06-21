import copy

import numpy as np
import copy
from SVM import loss_function
from datetime import datetime
from tqdm import tqdm
from scipy.sparse import csc_matrix
from numba import njit

def loss_function(w, x, y):
    rows = 1 - y * (w @ x.T)
    rows_sq = np.maximum(rows, 0) ** 2
    return np.sum(rows_sq)

@njit
def fast_H(x_data, x_indptr, C):
    p = len(x_indptr-1)
    H = np.zeros(p)
    for i in range(p):
        total = 0
        start = x_indptr[i]
        end = x_indptr[i+1]
        col_sliced = x_data[start:end]
        for data in col_sliced:
            total += data
        H[i] = 1 + 2 * C * total
    return H

@njit
def very_fast(x_data, x_indptr, x_indices, i, idx):
    total = 0
    start = x_indptr[i]
    end = x_indptr[i+1]
    col_sliced = x_data[start:end]
    col_slicei = x_indices[start:end]
    for data, el in zip(col_sliced, col_slicei):
        if idx[el]:
            total += data
    return total

@njit
def very_fast_multiply_inside(x_data, x_indptr, x_indices, i, idx, bjs):
    total = 0
    start = x_indptr[i]
    end = x_indptr[i+1]
    col_sliced = x_data[start:end]
    col_slicei = x_indices[start:end]
    for data, el in zip(col_sliced, col_slicei):
        if idx[el]:
            total += data * bjs[el]
    return total

@njit
def fast_D(x_data, x_indptr, x_indices, i, idx, bjs, z):
    total = 0
    start = x_indptr[i]
    end = x_indptr[i+1]
    col_sliced = x_data[start:end]
    col_slicei = x_indices[start:end]
    for x, current_index in zip(col_sliced, col_slicei):
        if idx[current_index]:
            total += (bjs[current_index] - z * x)**2 
    return total

@njit
def fast_update_b(x_data, x_indptr, x_indices, i, z, b, b_idx):
    start = x_indptr[i]
    end = x_indptr[i+1]
    col_sliced = x_data[start:end]
    col_slicei = x_indices[start:end]
    for x, current_index in zip(col_sliced, col_slicei):
        b[current_index] -= z * x
        b_idx[current_index] = b[current_index] > 0


from numba.experimental import jitclass

from numba import int32, float64, boolean
spec = [
    ('C', float64),
    ('x_data', float64[:]),
    ('x2_data', float64[:]),
    ('xy_data', float64[:]),
    ('x_indptr', int32[:]),
    ('x_indices', int32[:]),
    ('y', float64[:]),
    ('H', float64[:]),
    ('w', float64[:]),
    ('beta', float64),
    ('ro', float64),
    ('eps', float64),
    ('max_iter', int32),
    ('D', float64[:]),
    ('b', float64[:]),
    ('b_idx', boolean[:])
]


@jitclass(spec)
class SubProblemSolver:
    def __init__(
            self, 
            C,
            x_data,
            x2_data,
            xy_data,
            x_indptr,
            x_indices,
            y,
            H,
            w,
            D,
            b,
            b_idx,
            beta=0.5, ro=0.01, eps=1e-9, max_iter=500
            ):
        self.C = C
        self.beta = beta  # in (0, 1)
        self.ro = ro  # in (0, 1/2)
        self.eps = eps  # solution accuracy (for stopping condition)
        self.max_iter = max_iter
        self.x_data = x_data
        self.x2_data = x2_data
        self.xy_data = xy_data
        self.x_indptr = x_indptr
        self.x_indices = x_indices
        self.y = y
        self.H = H
        self.w = w
        self.D = D
        self.b = b
        self.b_idx = b_idx
    
    def iteration(self):
        idx = np.random.permutation(len(self.w))
        for i in idx:
            z = self._sub_problem(i)
            self.w[i] += z
            fast_update_b(self.xy_data, self.x_indptr, self.x_indices, i, z, self.b, self.b_idx)
        stop = np.sum(self.D ** 2)
        return stop

    def _sub_problem(self, i) -> float64:
        D_hat_hat = self._D_hat_hat(i)
        D_hat = self._D_hat(i)
        self.D[i] = D_hat
        d = -D_hat / D_hat_hat
        lam = 1
        z = lam * d
        iter = 0
        while iter < self.max_iter:
            if lam <= D_hat_hat / ((self.H[i] / 2) + self.ro):
                break
            if self._D(z, i) - self._D(0, i) <= self.ro * z ** 2:
                break
            lam *= self.beta
            z *= self.beta
            iter += 1
        return z

    def _D(self, z, i) -> float64:
        res = np.sum(np.square(self.w))
        res -= self.w[i] ** 2 # subtract what shouldn't have been added
        res += (self.w[i] + z)**2 # add what should have been addeed
        res = .5 * res
        res += self.C * fast_D(self.xy_data, self.x_indptr, self.x_indices, i, self.b_idx, self.b, z)
        return res

    def _D_hat(self, i) -> float64:
        res = self.w[i]
        res -= 2 * self.C * very_fast_multiply_inside(self.xy_data, self.x_indptr, self.x_indices, i, self.b_idx, self.b)
        return res

    def _D_hat_hat(self, i) -> float64:
        res = 1
        res += 2 * self.C * very_fast(self.x2_data, self.x_indptr, self.x_indices, i, self.b_idx)

        return res

# Algorithm is from
# https://www.csie.ntu.edu.tw/~cjlin/papers/cdl2.pdf
class CoordinateDescent:
    def __init__(self, C, beta=0.5, ro=0.01, eps=1e-9, max_iter=500):
        # C - regularization parameter
        # beta, ro - algoritm parameters
        self.C = C
        self.beta = beta  # in (0, 1)
        self.ro = ro  # in (0, 1/2)
        self.eps = eps  # solution accuracy (for stopping condition)
        self.max_iter = max_iter
        self.w_history = []
        self.subiter = None

    def fit(self, x, y):
        # x - data
        # y - responses
        # to add bias just include a column of ones
        x = csc_matrix(x)
        x2 = x.multiply(x)
        xy = x.multiply(y[None].T)
        xy = csc_matrix(xy)
        self.x = x
        self.y = y

        w = np.zeros(x.shape[1])
        self.w = w
        b = np.ones(len(y)) - y * (w @ x.T)
        b_idx = b > 0
        H = fast_H(x2.data, x2.indptr, self.C) 

        self.subiter = SubProblemSolver(
            self.C,
            x.data,
            x2.data,
            xy.data,
            x.indptr,
            x.indices,
            y,
            H,
            w = w,
            D = np.zeros(x.shape[1]),
            b=b,
            b_idx=b_idx,
            beta=0.5, ro=0.01, eps=1e-9, max_iter=500
        )

    def process(self):
        if self.subiter is None:
            raise Exception("Algorithm is not fitted yet")
        # weight initialization
        self.w = self.subiter.w
        # stop condition- max iteration or max(D'_i(0)) small enough or sum(D'_i(0)^2) small enough or validation error
        iter = 0
        stop = self.eps + 1
        while iter < self.max_iter and stop >= self.eps:
            self.log(iter)
            stop = self.subiter.iteration()
            self.w = self.subiter.w
            iter += 1
        return self.w

    def log(self, iter):
        print(datetime.isoformat(datetime.now()), iter, loss_function(self.w, self.x, self.y))

    def fit_process(self, x, y):
        self.fit(x, y)
        w = self.process()
        return w
