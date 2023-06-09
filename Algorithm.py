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

    def process(self):
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
        self.clear()
        return w

    def clear(self):
        self.H = None
        self.x = None
        self.y = None


    def _sub_problem(self, w, i):
        D_hat_hat = self._D_hat_hat(w, 0, i)
        D_hat = self._D_hat(w, 0, i)
        self.D = np.append(self.D, D_hat)
        d = -D_hat/D_hat_hat
        lam = 1
        z = lam * d
        while True:
            if lam <= D_hat_hat/((self._H(i)/2) + self.ro):
                break
            if self._D(w, z, i) - self._D(w, 0, i) <= self.ro * z**2:
                break
            lam *= self.beta
            z *= self.beta
        return z


    def _b(self, w, j):
        bj = 1 - self.y[j] * w.T @ self.x[j,:]
        return bj

    def _D(self, w, z, i):
        wz = copy.deepcopy(w)
        wz[i] += z
        res = 1/2 * wz.T @ wz
        sm = 0
        for j in range(len(self.y)):
            if self._indi_b(wz, j):
                sm += self._b(wz, j)**2
        res += self.C * sm
        return res


    def _D_hat(self, w, z, i):
        wz = copy.deepcopy(w)
        wz[i] += z
        res = wz[i]
        sm = 0
        for j in range(len(self.y)):
            if self._indi_b(wz, j):
                sm += self.y[j] * self.x[j,i] * self._b(wz, j)
        res -= 2 * self.C * sm
        return res

    def _D_hat_hat(self, w, z, i):
        wz = copy.deepcopy(w)
        wz[i] += z
        res = 1
        sm = 0
        for j in range(len(self.y)):
            if self._indi_b(wz, j):
                sm += self.x[j, i]**2
        res += 2 * self.C * sm
        return res

    def _indi_b(self, w, j):
        return self._b(w, j) > 0

    def _H(self, i):
        if self.H is not None:
            return self.H[i]
        return 1 + 2 * self.C * np.sum(self.x[:,i]**2)


