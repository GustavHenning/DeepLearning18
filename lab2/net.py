
import numpy as np

from tqdm import tqdm
from layers import Linear, Softmax, ReLU

class Net():
    # layers, weight regularization term, learning rate
    def __init__(self, layers=[], lam=0.1, l_rate=0.001, decay=1.0, mom=None):
        self.layers = layers
        self.l_rate = l_rate
        self.decay = decay
        self.mom = mom

    def forward(self, inp):
        out = inp
        for l in self.layers:
            out = l.forward(out)
        return out

    def backward(self, truth):
        for l in self.layers[::-1]:
            grad = l.backward(truth) if type(l) is Softmax else l.backward(grad)
        return grad

    def cost(self, truth, inp=None, out=None):
        out = self.forward(inp) if out is None else out
        c = [(l.cost(truth, out) if type(l) is Softmax else l.cost()) for l in self.layers]
        return np.sum(c)

    def accuracy(self, truth, inp=None, out=None):
        pred = self.forward(inp) if out is None else out
        N = truth.shape[1]
        pred = np.argmax(pred, axis=0)
        truth = np.argmax(truth, axis=0)
        return np.sum(pred == truth) / N

    def cost_acc(self, truth, inp=None, out=None):
        return self.cost(truth, inp, out), self.accuracy(truth, inp, out)

    def trainMiniBatch(self, train, val, epochs=10, batch_size=1000, shuffle=False):
        N = train['images'].shape[0]
        ind = np.arange(N)

        a_train, c_train, a_val, c_val = [], [], [], []
        trainImgT = train['images'].T
        trainTruthT = train['one_hot'].T
        valImgT = val['images'].T
        valTruthT = val['one_hot'].T

        for e in tqdm(range(epochs), ncols=50):
            if shuffle:
                np.random.shuffle(ind)

            for i in range(0, N, batch_size):
                batch_ind = ind[i: i + batch_size]
                x = train['images'][batch_ind].T
                truth = train['one_hot'][batch_ind].T

                # do the learning
                self.forward(x)
                self.backward(truth)
                self.update() if self.mom is None else self.updateMom()

            # Measure each epoch
            cost, acc = self.cost_acc(trainTruthT, trainImgT)
            c_train.append(cost)
            a_train.append(acc)

            cost, acc = self.cost_acc(valTruthT, valImgT)
            c_val.append(cost)
            a_val.append(acc)

        return np.array(a_train), np.array(c_train), np.array(a_val), np.array(c_val)

    def train(self, train, ind):
        inp = train['images'][:,ind]
        truth = train['one_hot'][:,ind]

    def update(self):
        for l in self.layers:
            l.update(self.l_rate)

    def updateMom(self):
        for l in self.layers:
            l.updateMom(self.l_rate, self.mom)
