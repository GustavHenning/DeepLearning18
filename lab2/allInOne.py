import itertools
import pickle
import scipy
import sys

import pandas as pd
from pandas import DataFrame as df
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import preprocessing
from laplotter import LossAccPlotter


ds_loc="../dataset/cifar-10-batches-py/"
plot_loc="./plots/"
show_not_save=False

def show_save(name):
    if show_not_save:
        plt.show()
    else:
        plt.savefig(plot_loc + name.replace(' ', '') + ".png")
        plt.close()

class CIFAR:
    def __init__(self):
        self.out_size = 10
        self.in_size = 32 * 32 * 3
        self.loaded_batches = {}
        self.load_labels()
        self.label_encoder = preprocessing.LabelBinarizer()
        self.label_encoder.fit([x for x in range(self.out_size)])

    def load_labels(self):
        with open(ds_loc + 'batches.meta', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            self.labels = [x.decode('ascii') for x in data[b'label_names']]

    def get_batch(self, batch_name):
        if not batch_name in self.loaded_batches:
            with open(ds_loc + batch_name, 'rb') as f:
                data = pickle.load(f, encoding='bytes')

            data[b'labels'] = np.array(data[b'labels'])

            self.loaded_batches[batch_name] = {
                'batch_name': data[b'batch_label'],
                'images': np.divide(data[b'data'], 255),
                'one_hot': self.label_encoder.transform(data[b'labels']),
                'labels': data[b'labels']
            }
        return self.loaded_batches[batch_name]

    def get_batches(self, *args):
        batches = [self.get_batch(name) for name in args]
        return {
            'batch_name': ", ".join(args),
            'images': np.vstack([b['images'] for b in batches]),
            'one_hot': np.vstack([b['one_hot'] for b in batches]),
            'labels': np.hstack([b['labels'] for b in batches])
        }

def barPlotLabels(dataset, labels, name):
    n = dataset['labels'].size
    y = [(dataset['labels'] == l).sum() / dataset['labels'].size for l in np.unique(dataset['labels'])]

    index = np.arange(len(labels))
    plt.bar(index, y)
    plt.xlabel('Label', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.xticks(index, labels, fontsize=9, rotation=30)
    plt.title('Label Distribution, n=' + str(n))
    show_save(name)

def show_image(img, label='', interpolation='gaussian'):
    squared_image = np.rot90(np.reshape(img, (32, 32, 3), order='F'), k=3)
    plt.imshow(squared_image, interpolation=interpolation)
    plt.axis('off')
    plt.title(label)

def plotImages(dataset, name):
    for plot_i, img_i  in enumerate(np.random.choice(dataset['images'].shape[0], 10, replace=False)):
        plt.subplot(2, 5, plot_i+1)
        show_image(dataset['images'][img_i], cifar.labels[dataset['labels'][img_i]])
    plt.suptitle('Sample images from ' + name, fontsize=20)
    show_save(name)

# Time to make a network

def num_gradient(x, truth, net, h=1e-6):
    Ws = net.W.shape
    Bs = net.b.shape

    gW = np.zeros(net.W.shape)
    gB = np.zeros(net.b.shape)

    p, _ = net.evaluate(x)
    c = net.cost(p, truth)

    # we dont make a temps, we subtract h later
    for i in tqdm(range(Bs[0]), ncols=50):
        net.b[i] += h
        p, _ = net.evaluate(x)
        next_c = net.cost(p, truth)
        gB[i] = (next_c - c) / h
        net.b[i] -= h

    for i in tqdm(range(Ws[0]), ncols=50):
        for j in range(Ws[1]):
            net.W[i, j] += h
            p, _ = net.evaluate(x)
            next_c = net.cost(p, truth)
            gW[i, j] = (next_c - c) / h
            net.W[i, j] -= h

    return gW, gB

def compareGradients(actualW, actualB, numericW, numericB):
    """
    computes the relative error between the 'actual' gradient and a controlled
    numerical one, where the goal is to achieve a very small positive difference.
    """
    rel_err_W = np.abs(np.subtract(actualW, numericW)).sum()
    rel_err_B = np.abs(np.subtract(actualB, numericB)).sum()
    print('Gradient relative error check:')
    print('Relative error W: {:.6e}'.format(rel_err_W))
    print('Relative error B: {:.6e}'.format(rel_err_B))

def plotResults(title, a_train, c_train, a_val, c_val):
    plotter = LossAccPlotter(title=title,
        show_averages=False,
        save_to_filepath= plot_loc + "lossAcc_{}.png".format(title),
        show_plot_window=show_not_save)

    for e in range(len(a_train)):
        plotter.add_values(e, loss_train=c_train[e], acc_train=a_train[e], loss_val=c_val[e], acc_val=a_val[e], redraw=False)

    plotter.redraw()
    plotter.block()

def weights_plot(net, dest_file, labels):
    for i, row in enumerate(net.W):
        img = (row - row.min()) / (row.max() - row.min())
        plt.subplot(2, 5, i+1)
        show_image(img, label=labels[i])
    if not show_not_save:
        plt.savefig(dest_file)
    plt.clf()
    plt.close()
    return dest_file

def trainMean():
    return train['images'].T.mean(axis=1)


def tryParameters(test_name, N_hidden, lam, l_rate, decay, mom, epochs=50, batch_size=250):
    net = Net([
    BatchNorm(cifar.in_size, trainMean()),
    Linear(cifar.in_size, N_hidden, lam=lam),
    ReLU(N_hidden),
    Linear(N_hidden, cifar.out_size, lam=lam),
    Softmax(cifar.out_size)],
                lam, l_rate, decay, mom)
    results = net.trainMiniBatch(train, val, epochs, batch_size, shuffle=True)
    print('{} Test Accuracy: {:.2f}'.format(test_name, net.accuracy(test['one_hot'].T, test['images'].T)))
    print('Final train a/c, val a/c: {:.2f}/{:.2f}, {:.2f}/{:.2f}'.format(results['last_a_train'], results['last_c_train'], results['last_a_val'], results['last_c_val']))
    plotResults(test_name, results['a_train'], results['c_train'], results['a_val'], results['c_val'])
    #weights_plot(net, "plots/weights_vizualisation_{}.png".format(test_name), labels)
    return results

cifar=CIFAR()
labels=cifar.labels

train = cifar.get_batches('data_batch_1')
val = cifar.get_batches('data_batch_2')
test = cifar.get_batches("test_batch")

# Create the network and test the accuracy before training
#tryParameters("initTest", N_hidden=50, lam=0.1, l_rate=0.001, decay=0.99, mom=0.99, epochs=1)
#tryParameters("initTest", N_hidden=50, lam=0.1, l_rate=0.001, decay=0.99, mom=0.99, epochs=25)

#
#
# TODO Compare gradients
#
#

def print_grad_diff(actW, numW, actB, numB):
    rel_err_W = np.abs(np.subtract(actW, numW)).sum()
    rel_err_B = np.abs(np.subtract(actB, numB)).sum()
    print('Gradient relative error check:')
    print('Relative error W: {:.6e}'.format(rel_err_W))
    print('Relative error B: {:.6e}'.format(rel_err_B))

def num_gradient(grad_x, grad_truth, g_net, lin, init_cost, h=1e-6):
    Ws = lin.W.shape
    Bs = lin.b.shape

    gW = np.zeros(Ws)
    gB = np.zeros(Bs)

    c = init_cost

    # we dont make a temps, we subtract h later
    for i in tqdm(range(Bs[0]), ncols=50):
        lin.b[i] += h
        next_c = g_net.cost(grad_truth, grad_x)
        gB[i] = (next_c - c) / h
        lin.b[i] -= h

    for i in tqdm(range(Ws[0]), ncols=50):
        for j in range(Ws[1]):
            lin.W[i, j] += h
            next_c = g_net.cost(grad_truth, grad_x)
            gW[i, j] = (next_c - c) / h
            lin.W[i, j] -= h

    print_grad_diff(lin.gW, gW, lin.gB, gB)
    return gW, gB

def gradient_check():
    # prepare a subset of the train data
    subset=50
    grad_train_img = train['images'][:subset, :].T
    grad_train_truth = train['one_hot'][:subset, :].T

    # init the network
    N_hidden=50
    lin = [Linear(cifar.in_size, N_hidden, lam=0.1), Linear(N_hidden, cifar.out_size, lam=0.1)]
    g_net = Net([
    lin[0],
    ReLU(N_hidden),
    lin[1],
    Softmax(cifar.out_size)],
                lam=0.1, l_rate=0.001, decay=0.99, mom=0.99)

    # do the pass
    grad_out = g_net.forward(grad_train_img)
    g_net.backward(grad_train_truth)
    cost = g_net.cost(grad_train_truth, out=grad_out)

    # calc the numeric grad for each linear layer
    for linear in lin:
        num_gradient(grad_train_img, grad_train_truth, g_net, linear, cost)


#gradient_check()

#
#
# TODO test overfitting
#
#

def test_overfit(momentum):
    subset=100
    img_temp = train['images']
    truth_temp = train['one_hot']
    train['images'] = train['images'][:subset, :]
    train['one_hot'] = train['one_hot'][:subset, :]

    tryParameters("overfitTest_mom_{}".format(momentum), N_hidden=50, lam=0, l_rate=0.005, decay=0.99, mom=momentum, epochs=200)

    train['images'] = img_temp
    train['one_hot'] = truth_temp

#test_overfit(momentum=0.5)
#test_overfit(momentum=0.75)
#test_overfit(momentum=0.8)
#test_overfit(momentum=0.95)

#
#
# TODO coarse and fine search
#
#

def search(reg_range, l_rate_range, dest_file, epochs=10):

    for i, params in enumerate(tqdm(list(itertools.product(reg_range, l_rate_range)), desc=dest_file, ncols=50)):
        run_name="{}_".format(i) + dest_file
        results = tryParameters(run_name,
                                N_hidden=50,
                                lam=params[0],
                                l_rate=params[1],
                                decay=0.995,
                                mom=0.8,
                                epochs=epochs)
        pd.DataFrame([results]).to_csv("results/" + dest_file, mode='a', header=(True if i == 0 else False))

def doSearch():
    # coarse search
    reg_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
    l_rate_range = list(reversed(reg_range))
    search(reg_range, l_rate_range, "coarse.csv", epochs=5)
    # fine search
    reg_range = np.linspace(0.005, 0.0005, num=7)
    l_rate_range = np.linspace(0.35, 0.1, num=4)
    search(reg_range, l_rate_range, "fine.csv", epochs=10)
#doSearch()

#
#
# TODO show best nets
#
#

def showBest():
    coarse_df = pd.read_csv("results/coarse.csv").sort_values(by=['last_a_val'], ascending=False)
    print(coarse_df.head(3))

    fine_df = pd.read_csv("results/fine.csv").sort_values(by=['last_a_val'], ascending=False)
    #print(fine_df.head(3))
showBest()

#
#
# Using best performance net, find results for extended set
#
#
def test_best():
    # we add val set
    train = cifar.get_batches('data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5')
    # remove last 1000
    train['images'] = train['images'][:49000, :]
    train['one_hot'] = train['one_hot'][:49000, :]

    val = cifar.get_batches('data_batch_5')
    # keep only last 1000
    val['images'] = val['images'][-1000:, :]
    val['one_hot'] = val['one_hot'][-1000:, :]

    tryParameters("bestTest", N_hidden=50, lam=0.00050, l_rate=0.095111, decay=0.995, mom=0.8, epochs=30)
    #truth = train['one_hot'].T

#test_best()




#
#
#
#
#
#


class Layer():
    def __init__(self, in_size, out_size, lam, name):
        self.in_size = in_size
        self.out_size = out_size
        self.name = name
        self.isActivation = (False if name is "linear" else True)

        if not self.isActivation:
            # Weights
            self.W = np.random.normal(loc=0.0, scale=0.01, size=(out_size, in_size))
            # Bias
            self.b = np.random.normal(loc=0.0, scale=0.01, size=(out_size, 1))
            # Weight regularization
            self.lam = lam
            self.mom = {
                'W' : np.zeros(self.W.shape),
                'b' : np.zeros(self.b.shape)
            }

            self.resetGrad()
        # this is a memory variable between forward/backward
        self.x = np.empty(shape=(self.in_size, 1))


    def forward(self, x):
        assert x is not None
        # sometimes we need to store input for backward
        self.x = x
        #print(self.name + " forward")
        #print(self.x.shape)

    def backward(self):
        assert self.x is not None
        #print(self.name + " back")
        #print(self.x.shape)

    def cost(self):
        return 0

    def resetGrad(self):
        self.gW = np.zeros(self.W.shape)
        self.gB = np.zeros(self.b.shape)

    # for non-activation layers to implement
    def update(self, l_rate=0.001):
        pass

    def updateMom(self, l_rate=0.001, momentum=0.0):
        pass

class Linear(Layer):
    def __init__(self, in_size, out_size, lam=0, name="linear"):
        super().__init__(in_size, out_size, lam, name)

    def forward(self, x):
        Layer.forward(self, x)
        # Wx + b
        return np.dot(self.W, x) + self.b

    def backward(self, grad):
        Layer.backward(self)
        N = self.x.shape[1]

        self.resetGrad()

        for i in range(N):
            p = self.x[:, i]
            g = grad[i, :]

            self.gW += np.outer(g, p)
            self.gB += np.reshape(g, self.gB.shape)

        # here's the difference in (10) and (11)
        self.gW = (1.0/N) * self.gW + 2 * self.lam * self.W
        self.gB /= N

        return np.dot(grad, self.W)

    def cost(self):
        return self.lam * np.power(self.W, 2).sum()

    def update(self, l_rate=0.001):
        self.W -= l_rate * self.gW
        self.b -= l_rate * self.gB

    def updateMom(self, l_rate=0.001, momentum=0.0):
        self.mom['W'] = momentum * self.mom['W'] + l_rate * self.gW
        self.mom['b'] = momentum * self.mom['b'] + l_rate * self.gB

        self.W -= self.mom['W']
        self.b -= self.mom['b']

class ReLU(Layer):
    def __init__(self, in_size, name="relu"):
        super().__init__(in_size, in_size, -1, name)

    def forward(self, x):
        Layer.forward(self, x)
        # max(0, x)
        return self.x * (self.x > 0)

    def backward(self, grad):
        Layer.backward(self)
        return np.multiply(grad, self.x.T > 0)

class Softmax(Layer):
    def __init__(self, in_size, name="softmax"):
        super().__init__(in_size, in_size, -1, name)

    def forward(self, x):
        assert x is not None
        try:
            # this should prevent error tried for
            e = np.exp(x - x.max())
            res = e / np.sum(e, axis=0)
        except FloatingPointError:
            # Gradient explosion scenario
            print("jesus take the wheel")
            res = np.ones(x)
        Layer.forward(self, res)
        return res

    def backward(self, truth):
        Layer.backward(self)
        assert self.x.shape[1] == truth.shape[1]
        N = truth.shape[1]

        cols = ((truth[:,i], self.x[:,i]) for i in range(N))
        grad = [self.softGrad(t, p) for (t, p) in cols]

        return np.vstack(grad)

    @staticmethod
    def softGrad(t, p):
        # Jacobian according for formulas in Ass1
        a = np.outer(p,p)
        b = np.dot(t, (np.diag(p) - a))
        c = np.dot(t, p)
        return -b/c

    def cost(self, truth, prob=None):
        x = self.x if prob is None else prob
        assert x.shape[1] == truth.shape[1]
        N = x.shape[1]

        Py = np.multiply(truth, x).sum(axis=0)
        Py[Py == 0] = np.finfo(float).eps # fix floats

        return - np.log(Py).sum() / N

class BatchNorm(Layer):
    # https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
    def __init__(self, in_size, mu=None, s=None, name="batch_norm"):
        super().__init__(in_size, in_size, -1, name)

        self.mu = mu if mu is not None else np.zeros(shape=(in_size, 1), dtype=float)
        self.s = s if s is not None else np.eye(in_size, dtype=float)

    def forward(self, x, train=False):
        Layer.forward(self, x)
        # if mu, s is passed: then it's eval time not training
        self.mu = x.mean(axis=1) if train else self.mu
        self.s = x.var(axis=1) if train else self.s
        return np.dot(self.s, (x.T - self.mu.T).T)

    def backward(self, grad):
        Layer.backward(self)
        # Not implemented yet
        return grad

#
#
#
#
#
#
#
#
#

class Net():
    # layers, weight regularization term, learning rate
    def __init__(self, layers=[], lam=0.1, l_rate=0.001, decay=None, mom=None):
        self.layers = layers
        self.lam = lam
        self.l_rate = l_rate
        self.decay = decay
        self.mom = mom

    def forward(self, inp, train=False):
        out = inp
        for l in self.layers:
            out = l.forward(out) if type(l) is not BatchNorm else l.forward(out, train)
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

    def hidden_size(self):
        for l in self.layers:
            if not l.isActivation: return l.out_size
        return -1

    def trainMiniBatch(self, train, val, epochs=10, batch_size=200, shuffle=False):
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

            if self.decay is not None:
                self.l_rate *= self.decay


            # Measure each epoch
            cost, acc = self.cost_acc(trainTruthT, trainImgT)
            c_train.append(cost)
            a_train.append(acc)

            cost, acc = self.cost_acc(valTruthT, valImgT)
            c_val.append(cost)
            a_val.append(acc)

            if c_train[0]*3 < c_train[-1]:
                print("Cost is rising, early stopping...")
                break
        return {
            'epochs': epochs,
            'N_hidden' : self.hidden_size(),
            'lam' : self.lam,
            'learning rate' : self.l_rate,
            'decay rate' : self.decay,
            'momentum' : self.mom,
            'last_a_train' : a_train[-1],
            'last_c_train' : c_train[-1],
            'last_a_val' : a_val[-1],
            'last_c_val' : c_val[-1],
            'a_train' : a_train,
            'c_train' : c_train,
            'a_val' : a_val,
            'c_val' : c_val,
            }

    def train(self, train, ind):
        inp = train['images'][:,ind]
        truth = train['one_hot'][:,ind]

    def update(self):
        for l in self.layers:
            l.update(self.l_rate)

    def updateMom(self):
        for l in self.layers:
            l.updateMom(self.l_rate, self.mom)
