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

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)

def backprop(x, y):
    """Return a tuple "(nabla_b, nabla_w)" representing the
    gradient for the cost function C_x.  "nabla_b" and
    "nabla_w" are layer-by-layer lists of numpy arrays, similar
    to "biases" and "weights"."""
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(biases, weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid_vec(z)
        activations.append(activation)
    # backward pass
    delta = cost_derivative(activations[-1], y) * \
        sigmoid_prime_vec(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.
    for l in xrange(2, len(sizes)):
        z = zs[-l]
        spv = sigmoid_prime_vec(z)
        delta = np.dot(weights[-l+1].transpose(), delta) * spv
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)

def cost_derivative(output_activations, y):
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."""
    return (output_activations-y) 
