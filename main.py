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

# classA = np.array([[1, 1]])
# classB = np.array([[3, 3]])
# inputs = np.concatenate((classA, classB))
# targets = np.concatenate((
#     np.ones(classA.shape[0]),
#     -np.ones(classB.shape[0])
# ))

N = inputs.shape[0]  # Number of rows (samples)
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

print(inputs)
print(targets)

C = 1e-4
bounds = [(0, C) for _ in range(N)]
XC = ...
start = np.zeros(N)
# start = np.random.randn((N))


def linear_kernel(x, y):
    return np.dot(x, y)


TK_MATRIX = np.array([[targets[i]*targets[j]*linear_kernel(inputs[i], inputs[j]) for j in range(N)] for i in range(N)])


def objective(a):
    print('-'*10)
    amat = np.dot(a, a.T)
    a.shape = (a.shape[0], 1)
    print('s:', a.shape)
    print('1 amat:', np.dot(a.T, a))
    print('2 amat:', np.dot(a, a.T))
    tmp = amat * TK_MATRIX
    print('amat:',  amat.shape, 'TK:', TK_MATRIX.shape, 'tmp:', tmp.shape)
    print('-'*10)
    return 0.5*np.sum(tmp) - np.sum(a)


def zerofun(a):
    # print(np.dot(a.T, targets))
    return np.dot(a.T, targets)


ret = minimize(objective, start, bounds=bounds, constraints={'type':'eq', 'fun':zerofun})
alpha = ret['x']


if __name__ == '__main__':
    print('Hello :^D')
    # print(targets)
    print('TK_MATRIX')
    print(TK_MATRIX)
    # print(alpha)
    print('ret')
    print(ret)


b_alpha = np.abs(alpha) > 1e-5
print(b_alpha)

nonzero_alpha = alpha[b_alpha]
suport_vector = inputs[b_alpha, :]

b = ...


plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b.')

plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.')

plt.axis('equal')
# plt.savefig('svmplot.pdf')
plt.show()
