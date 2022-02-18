import random, math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def linear_kernel(x, y):
    return np.dot(x, y)
