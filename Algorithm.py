import pandas as pd
import numpy as np

def loss_function(x, y, z):
    # x -- parameter vector, in literature usually w
    # y -- feature vector, y is in R^n
    # z -- response vector, with values {-1,1}
    # https://www.csie.ntu.edu.tw/~cjlin/papers/cdl2.pdf <- can't we use one of those methods?
    sm = 1 - x.T @ y
    return max(0, sm)

def regularization(x):
    return np.sum(x**2)

def SVM_algorithm(x, y, C):
    return C * loss_function(x, y) + regularization(x)
