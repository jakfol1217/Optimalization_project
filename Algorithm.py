import copy

import numpy as np
import copy
from SVM import loss_function
from datetime import datetime
from tqdm import tqdm


def loss_function(w, x, y):
    rows = 1 - y * (w @ x.T)
    rows_sq = np.maximum(rows, 0) ** 2
    return np.sum(rows_sq)


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
        self.x = None
        self.x2 = None
        self.xy = None
        self.y = None
        self.H = None
        self.D = np.array([])
        self.w_history = []

    def fit(self, x, y):
        # x - data
        # y - responses
        # to add bias just include a column of ones
        self.x = x
        self.x2 = self.multiply_elementwise(x, x)
        self.y = y
        self.xy = self.multiply_elementwise(x, y[None].T)
        if "sparse" in str(type(self.xy)):
            # this multiplication results in COO matrix as opposed to CSC
            from scipy.sparse import csc_matrix
            self.xy = csc_matrix(self.xy)

        self.w = np.zeros(self.x.shape[1])
        self.b = np.ones(len(self.y)) - self.y * (self.w @ self.x.T)
        Hs = [self._H(i) for i in range(x.shape[1])]
        self.H = Hs

    def process(self, clear=False):
        if self.x is None or self.y is None or self.H is None:
            raise Exception("Algorithm is not fitted yet")
        # weight initialization
        w = np.zeros(self.x.shape[1])
        self.w = w
        # stop condition- max iteration or max(D'_i(0)) small enough or sum(D'_i(0)^2) small enough or validation error
        iter = 0
        stop = self.eps + 1
        while iter < self.max_iter and stop >= self.eps:
            print(datetime.isoformat(datetime.now()), iter, loss_function(w, self.x, self.y))
            self.D = np.array([])
            idx = np.random.permutation(len(w))
            # idx = np.random.choice(len(w))
            for i in idx:
                z = self._sub_problem(w, i)
                w[i] += z
                self.b -= z * self.xy[:, i].T
                self.b = np.asarray(self.b).reshape(-1)
            self.w_history.append(w)
            iter += 1
            stop = sum(self.D ** 2)
        if clear:
            self.clear()
        return w

    def fit_process(self, x, y, clear=False):
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
        D_hat_hat = self._D_hat_hat(w, i, idx, bjs)
        D_hat = self._D_hat(w, i, idx, bjs)
        self.D = np.append(self.D, D_hat)
        d = -D_hat / D_hat_hat
        lam = 1
        z = lam * d
        iter = 0
        while iter < self.max_iter:
            if lam <= D_hat_hat / ((self._H(i) / 2) + self.ro):
                break
            if self._D(w, z, i, idx, bjs) - self._D(w, 0, i, idx, bjs) <= self.ro * z ** 2:
                break
            lam *= self.beta
            z *= self.beta
            iter += 1
        return z

    def _b(self, w):
        # bj = np.ones(len(self.y)) - self.y * (w @ self.x.T)
        return self.b

    def _D(self, w, z, i, idx, bjs):
        wz = copy.deepcopy(w)
        wz[i] += z
        res = 1 / 2 * wz.T @ wz
        # Fixed _D:
        # mm = bjs[None].T - z * self.xy[:, i]
        # if z != 0:
        #   idx = mm > 0
        #res += self.C * np.sum(np.square(mm[idx]))
        mm = self.xy[idx, i]
        res += self.C * np.sum(np.square(bjs[idx][None].T - z * mm))
        return res

    def _D_hat(self, w, i, idx, bjs):
        #wz = copy.deepcopy(w)
        #wz[i] += z
        res = w[i]#wz[i]
        #mm = self.xy[:, i]
        #zmm = z * mm
        #bsj_zmm = bjs[None].T# - zmm
        res -= 2 * self.C * self.sum_vector_coo(self.multiply_elementwise(self.xy[:, i], bjs[None].T), idx)
        return res

    def _D_hat_hat(self, w, i, idx, bjs):
        #wz = copy.deepcopy(w)
        #wz[i] += z
        res = 1
        res += 2 * self.C * self.sum_vector_csr(self.x2[:, i], idx)

        return res

    def _indi_b(self, w):
        bjs = self._b(w)
        idx = bjs > 0
        return bjs, idx

    def _H(self, i):
        if self.H is not None:
            return self.H[i]
        return 1 + 2 * self.C * np.sum(self.x2[:, i])

    def multiply_elementwise(self, x1, x2):
        if "sparse" in str(type(x1)):
            return x1.multiply(x2)
        elif "sparse" in str(type(x2)):
            return x2.multiply(x1)
        else:
            return x1 * x2

    def sum_vector_coo(self, x, idx):
        total = 0
        for data, el in zip(x.data, x.col):
            if idx[el]:
                total += data
        return total

    def sum_vector_csr(self, x, idx):
        total = 0
        for data, el in zip(x.data, x.indices):
            if idx[el]:
                total += data
        return total