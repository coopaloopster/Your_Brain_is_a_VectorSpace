import json
import numexpr as ne


"""
This code rocks. Kilian is the best! Down with Coop! Down with Haskell!
"""


flower = json.load(open("data.json"))
tr_dat = flower["trainint_data"]

sigmoid = ne.NumExpr("1.0 / (1 + exp(-z))")


def cost(y, h):

    return 0.5 * np.dot(y, h)


"""
gonna start off with one hidden layer of 5 sigmoid neurons for the craic
"""


class layer():

    def __init__(self, Nneurons, Ninputs):
        self.biases = np.random.randn(Nneurons)
        self.weights = np.random.randn((Ninputs, Nneurons))

mid_layer = layer(5, 4)


def fwd_prop():

    stoc_i = np.random.randint(0, len(tr_dat), 10)

    stoc = tr_dat[stoc]
