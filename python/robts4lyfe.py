import json
import numexpr as ne


"""
This code rocks. Kilian is the best! Down with Coop! Down with Haskell!
"""


flower = json.load(open("data.json"))
tr_dat = flower["trainint_data"]

sigmoid = ne.NumExpr("1.0 / (1 + exp(-z))")


def compute_out(layer1, layer2):

    layer2.outputs = sigmoid(np.dot(layer2.weights,
                                    layer1.outputs))


def cost(y, h):

    return 0.5 * np.dot(y, h)


"""
gonna start off with one hidden layer of 5 sigmoid neurons for the craic
"""


class layer():

    def __init__(self, Nneurons, Ninputs=None):

        if Ninputs:
            # initialise with random numbers if a hidden layer
            self.biases = np.random.randn(Nneurons)
            self.weights = np.random.randn((Ninputs, Nneurons))
        else:
            pass

input_data = layer()
mid_layer = layer(5, 4)
result = layer(3, 5)


def fwd_prop():

    # get a random batch from the training data
    stoc_i = np.random.randint(0, len(tr_dat), 10)
    stoc = tr_dat[stoc]


