import pandas as pd
import numpy as np

def loss_function(x, y):
    sm = 1 - x.T @ y
    return max(0, sm)

def regularization(x):
    return np.sum(x**2)

def SVM_algorithm(x, y, C):
    return C * loss_function(x, y) + regularization(x)
