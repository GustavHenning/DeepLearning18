
import numpy as np

from layers import Linear, Softmax, ReLU

class Net():
    # layers, weight regularization term, learning rate
    def __init__(self, layers=[], lam=0.1, l_rate=0.001):
        self.layers = layers
        self.l_rate = l_rate

    def forward(self, inp):
        out = inp
        for l in self.layers:
            out = l.forward(out)
        return out

    def backward(self, truth):
        for l in self.layers[::-1]:
            grad = l.backward(truth) if type(l) is Softmax else l.backward(grad)
        return grad

    def cost(self, inp=None, out=None):
        out = self.forward(inp) if out is None else out
        c = [(l.cost(truth, out) if type(layer) is Softmax else layer.cost()) for l in self.layers]
        return c.sum()

    def accuracy(self, truth, inp=None, out=None):
        pred = self.forward(inp) if out is None else out
        N = truth.shape[0]
        pred = np.argmax(pred, axis=0)
        truth = np.argmax(truth, axis=0)
        return np.sum(pred == truth) / N

    def cost_acc(self, truth, inp=None, out=None):
        return cost(inp, out), accuracy(truth, inp, out)

    def train(self, train, ind):
        inp = train['images'][:,ind]
        truth = train['one_hot'][:,ind]

        self.forward(inp)
        self.backward(truth)

    def update(self):
        for l in self.layers:
            l.update(self.l_rate)
