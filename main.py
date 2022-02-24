import random, math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

classA = np.concatenate((
    np.random.randn(10, 2) * 0.2 + [2.5, 0.5],
    np.random.randn(10, 2) * 0.2 + [-2.5, 0.5]
))

classB = np.random.randn(20, 2) * 0.2 + [0.0, 0.5]

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

C = 1e15
bounds = [(0, C) for _ in range(N)]
start = np.zeros(N)


def linear_kernel(x, y):
    return np.dot(x, y)


def poly_kernel(x, y):
    p = 2
    return (np.dot(x, y) + 1)**p


def radial_kernel(x, y):
    sig = 10
    return np.exp(-(np.linalg.norm(x - y)**2) / (2 * sig**2))


KERNEL_FUNC = poly_kernel


TK_MATRIX = np.array(
    [[targets[i] * targets[j] * KERNEL_FUNC(inputs[i], inputs[j]) for j in range(N)] for i in range(N)])


def objective(a):
    a.shape = (a.shape[0], 1)
    amat = np.dot(a, a.T)
    tmp = amat * TK_MATRIX
    return (0.5 * np.sum(tmp)) - np.sum(a)


def zerofun(a):
    return np.dot(a.T, targets)


ret = minimize(objective, start, bounds=bounds, constraints={'type': 'eq', 'fun': zerofun})
alpha = ret['x']

b_alpha = np.abs(alpha) > 1e-5

nonzero_alpha = alpha[b_alpha]
support_vector = inputs[b_alpha, :][0]
print(support_vector)


def calcB():
    s = 0
    for i in range(N):
        s += alpha[i] * targets[i] * KERNEL_FUNC(support_vector, inputs[i])

    s -= targets[b_alpha][0]
    return s


b = calcB()


def indicator(x, y):
    s = 0
    for i in range(N):
        s += alpha[i] * targets[i] * KERNEL_FUNC(np.array([x, y]), inputs[i])

    s -= b
    return s


plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b.')

plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.')

plt.axis('equal')


xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
grid = np.array([[indicator(x, y)
                  for x in xgrid]
                 for y in ygrid])
plt.contour(xgrid, ygrid, grid,
            (-1.0, 0.0, 1.0),
            colors=('red', 'black', 'blue'),
            linewidths=(1, 3, 1))

#plt.plot([support_vector[0]], [support_vector[1]], "x")

plt.savefig('svmplot.pdf')
plt.show()

