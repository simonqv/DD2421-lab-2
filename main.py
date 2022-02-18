import random, math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


classA = np.concatenate((
    np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
    np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]
))

classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs = np.concatenate((classA, classB))
targets = np.concatenate((
    np.ones(classA.shape[0]),
    -np.ones(classB.shape[0])
))

N = inputs.shape[0]  # Number of rows (samples)
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]


def linear_kernel(x, y):
    return np.dot(x, y)

T = ...
K = ...
TK_MATRIX = np.array([[targets[i]*targets[j]*linear_kernel(inputs[i], inputs[j]) for j in range(N)] for i in range(N)])


def objective(a):
    amat = np.dot(a.T, a)
    tmp = amat * TK_MATRIX
    return 0.5*np.sum(tmp) - np.sum(a)



if __name__ == '__main__':
    print('Hello :^D')
    print(targets)
    print(TK_MATRIX)
