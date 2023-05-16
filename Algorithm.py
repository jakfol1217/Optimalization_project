import copy

import pandas as pd
import numpy as np
import copy
from SVM import loss_function


class coordinate_descend:
    def __init__(self, C, beta=0.5, ro=0.01):
        # C - regularization parameter
        # beta, ro - algoritm parameters
        self.C = C
        self.beta = beta
        self.ro = ro
        self.x = None
        self.y = None
        self.H = None

    def fit(self, x, y):
        # x - data
        # y - responses

        # weight initialization
        self.x = x
        self.y = y
        w = np.random.normal(size=x.shape[1])
        Hs = [self._H(i) for i in range(x.shape[1])]
        self.H = Hs
        # initialize all the H_i, permutation of w, stop condition
        stop_condition = True
        while not stop_condition:
            idx = np.random.permutation(len(w))
            for i in idx:
                z = self._sub_problem(w, i)
                w[i] += z
        # TODO - find stop condition
        self.H = None
        self.x = None
        self.y = None


    def _sub_problem(self, w, i):
        D_hat_hat = self._D_hat_hat(w, 0, i)
        d = -self._D_hat(w, 0, i)/D_hat_hat
        lam = 1
        z = lam * d
        while True: #:(
            if lam <= D_hat_hat/((self._H(i)/2) + self.ro):
                break
            if self._D(w, z, i) - self._D(w, 0, i) <= self.ro * z**2:
                break
            lam *= self.beta
            z *= self.beta
        return z

        # TODO - testing

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
                sm += _b(wz, j)**2
        res += C * sm
        return res


    def _D_hat(self, w, z, i):
        wz = copy.deepcopy(w)
        wz[i] += z
        res = wz[i]
        sm = 0
        for j in range(len(self.y)):
            if self._indi_b(wz, j):
                sm += y[j] * x[j,i] * _b(wz, j)
        res -= 2 * C * sm
        return res

    def _D_hat_hat(self, w, z, i):
        wz = copy.deepcopy(w)
        wz[i] += z
        res = 1
        sm = 0
        for j in range(len(self.y)):
            if self._indi_b(wz, j):
                sm += x[j, i]**2
        res += 2 * C * sm

    def _indi_b(self, w, j):
        return self.b(w, j) > 0

    def _H(self, i):
        if self.H is not None:
            return self.H[i]
        return 1 + 2 * self.C * np.sum(self.x[:,i]**2)


