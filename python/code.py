import numpy as np
import numexpr

sig = numexpr.NumExpr("1.0/(1.0+exp(-z))")
sizes = [4, 3, 3]
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
