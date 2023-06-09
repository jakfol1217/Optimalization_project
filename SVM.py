import numpy as np

def loss_function_row(w, x, y):
    # w -- parameter vector
    # x -- feature vector, x is in R^n
    # y -- response vector, with values {-1,1}
    sm = 1 - y * w.T @ x
    return max(0, sm)**2

def loss_function(w, x, y):
    sm = 0
    for i in range(len(y)):
        sm += loss_function_row(w, x[i,:], y[i])
    return sm

def regularization(w):
    return np.sum(w**2)

def SVM_algorithm(w, x, y, C):
    return C * loss_function(w, x, y) + regularization(w)
