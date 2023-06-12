import copy

import numpy as np
import copy
from SVM import loss_function

# Algorithm is from
# https://www.csie.ntu.edu.tw/~cjlin/papers/cdl2.pdf
class CoordinateDescent:
    def __init__(self, C, beta=0.5, ro=0.01, eps=1e-9, max_iter=500):
        # C - regularization parameter
        # beta, ro - algoritm parameters
        self.C = C
        self.beta = beta # in (0, 1)
        self.ro = ro # in (0, 1/2)
        self.eps = eps # solution accuracy (for stopping condition)
        self.max_iter = max_iter
        self.x = None
        self.y = None
        self.H = None
        self.D = np.array([])


    def fit(self, x, y):
        # x - data
        # y - responses
        # to add bias just include a column of ones
        self.x = x
        self.y = y
        Hs = [self._H(i) for i in range(x.shape[1])]
        self.H = Hs

    def process(self, clear = False):
        if self.x is None or self.y is None or self.H is None:
            raise Exception("Algorithm is not fitted yet")
        # weight initialization
        w = np.zeros(self.x.shape[1])
        # stop condition- max iteration or max(D'_i(0)) small enough or sum(D'_i(0)^2) small enough or validation error
        iter = 0
        stop = self.eps + 1
        while iter < self.max_iter and stop >= self.eps:
            self.D = np.array([])
            idx = np.random.permutation(len(w))
            for i in idx:
                z = self._sub_problem(w, i)
                w[i] += z
            iter += 1
            stop = sum(self.D**2)
        if clear:
            self.clear()
        return w

    def fit_process(self, x, y, clear = False):
        self.fit(x, y)
        w = self.process()
        if clear:
            self.clear()
        return w

    def clear(self):
        self.H = None
        self.x = None
        self.y = None


    def _sub_problem(self, w, i):
        bjs, idx = self._indi_b(w)
        D_hat_hat = self._D_hat_hat(w, 0, i, idx, bjs)
        D_hat = self._D_hat(w, 0, i, idx, bjs)
        self.D = np.append(self.D, D_hat)
        d = -D_hat/D_hat_hat
        lam = 1
        z = lam * d
        iter = 0
        while iter < self.max_iter:
            if lam <= D_hat_hat/((self._H(i)/2) + self.ro):
                break
            if self._D(w, z, i, idx, bjs) - self._D(w, 0, i, idx, bjs) <= self.ro * z**2:
                break
            lam *= self.beta
            z *= self.beta
            iter += 1
        return z


    def _b(self, w):
        bj = np.ones(len(self.y)) - self.y * (w @ self.x.T)
        return bj

    def _D(self, w, z, i, idx, bjs):
        wz = copy.deepcopy(w)
        wz[i] += z
        res = 1/2 * wz.T @ wz
        mm = self.multiply_elementwise(self.x[idx,i].T, self.y[idx])
        res += self.C * np.sum(np.square(bjs[idx]- z*mm))
        return res


    def _D_hat(self, w, z, i, idx, bjs):
        wz = copy.deepcopy(w)
        wz[i] += z
        res = wz[i]
        mm = self.multiply_elementwise(self.x[idx,i].T, self.y[idx])
        res -= 2 * self.C * np.sum(self.multiply_elementwise(mm, bjs[idx]-z*mm))
        return res

    def _D_hat_hat(self, w, z, i, idx, bjs):
        wz = copy.deepcopy(w)
        wz[i] += z
        res = 1
        res += 2 * self.C * np.sum(
            self.multiply_elementwise(self.x[idx, i], self.x[idx, i])
        )

        return res

    def _indi_b(self, w):
        bjs = self._b(w)
        idx = bjs > 0
        return bjs, idx

    def _H(self, i):
        if self.H is not None:
            return self.H[i]
        return 1 + 2 * self.C * np.sum(
            self.multiply_elementwise(self.x[:,i], self.x[:,i])
        )

    def multiply_elementwise(self, x1, x2):
        import scipy
        #if type(x1) == scipy.sparse._csr.csr_matrix or type(x2) == scipy.sparse._csc.csc_matrix :
        return x1.multiply(x2)
        #else:
         #   return x1 * x2
