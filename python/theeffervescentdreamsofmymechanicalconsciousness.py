import json
import numpy as np
import numexpr

data = json.load(open("data.json"))

sig = numexpr.NumExpr("1.0/(1.0+exp(-z))")
sizes = [4, 3, 3]
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

# The first flower

hidden = np.asarray([sig(np.dot(w0, data['training_data'][0][:-1]) +
    biases[0][i, 0]) for i, w0  in enumerate(weights[0])])

result = np.asarray([sig(np.dot(w1, hidden) +
    biases[1][i, 0]) for i, w1 in enumerate(weights[1])])

#print(result)
Cost = 0.5 * np.linalg.norm(result - data['training_data'][0][4]) ** 2
#print(Cost)
