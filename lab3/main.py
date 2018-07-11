import matplotlib
matplotlib.use('Agg')
import itertools
import pickle
import scipy
import sys

import pandas as pd
from pandas import DataFrame as df
import numpy as np
import matplotlib.pyplot as plt
from net import Net

from tqdm import tqdm
from sklearn import preprocessing
from laplotter import LossAccPlotter

from layers import Softmax, ReLU, Linear, BatchNorm

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


def tryParameters(test_name, lin_neurons, with_BN, lam, l_rate, decay, mom, epochs=50, batch_size=250):

    count = 0
    layers = []

    for N in lin_neurons:
        not_last_layer = count < (len(lin_neurons) - 1)
        layers.append(Linear(cifar.in_size if count == 0 else lin_neurons[count-1],
        N if not_last_layer else cifar.out_size,
        lam=lam))
        if not_last_layer:
            if with_BN:
                layers.append(BatchNorm(N))
            layers.append(ReLU(N))
        count += 1
    if len(lin_neurons) == 1 and with_BN:
        layers.append(BatchNorm(cifar.out_size))
    layers.append(Softmax(cifar.out_size))
    # init the network
    print(["{}:{},{}".format(l.name, l.in_size, l.out_size) for l in layers])
    net = Net(layers, lam=lam, l_rate=l_rate, decay=0.99, mom=0.99)
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

def gradient_check(lam, lin_neurons, with_BN):
    # prepare a subset of the train data
    subset=50
    grad_train_img = train['images'][:subset, :].T
    grad_train_truth = train['one_hot'][:subset, :].T

    count = 0
    layers = []

    for N in lin_neurons:
        not_last_layer = count < (len(lin_neurons) - 1)
        layers.append(Linear(cifar.in_size if count == 0 else lin_neurons[count-1],
        N if not_last_layer else cifar.out_size,
        lam=lam))
        if not_last_layer:
            if with_BN:
                layers.append(BatchNorm(N))
            layers.append(ReLU(N))
        count += 1
    if len(lin_neurons) == 1 and with_BN:
        layers.append(BatchNorm(cifar.out_size))
    layers.append(Softmax(cifar.out_size))
    # init the network
    print(["{}:{},{}".format(l.name, l.in_size, l.out_size) for l in layers])
    g_net = Net(layers, lam=lam, l_rate=0.001, decay=0.99, mom=0.99)

    # do the pass
    grad_out = g_net.forward(grad_train_img, train=True)
    g_net.backward(grad_train_truth)
    cost = g_net.cost(grad_train_truth, out=grad_out)


    # calc the numeric grad for each linear layer
    for linear in [l for l in layers if l.isActivation == False]:
        num_gradient(grad_train_img, grad_train_truth, g_net, linear, cost)


#
#
# TODO test comparison with/(out) batch norm
#
#
def compareBatchNorm():
    gradient_check(0.0, [50], False) # one layer without reg
    gradient_check(0.0, [50], True) # one layer without BN
    gradient_check(0.2, [50], False) # one layer with reg
    gradient_check(0.0, [50, 30], True) # one layer with BN
    gradient_check(0.0, [50, 30], False) # two layer
    gradient_check(0.0, [50, 30, 15], False)
    gradient_check(0.0, [50, 30, 15], True)
    gradient_check(0.0, [50, 50, 30, 15], True)

#compareBatchNorm()


def tryOne():
    tryParameters("oneLayerWithoutBN", lin_neurons=[50,30],
                                with_BN=False,
                                lam=0.0,
                                l_rate=0.001,
                                decay=0.99,
                                mom=0.8,
                                epochs=100)

    tryParameters("oneLayerWithBN", lin_neurons=[50],
                                with_BN=True,
                                lam=0.0,
                                l_rate=0.001,
                                decay=0.99,
                                mom=0.8,
                                epochs=100)

#tryOne()

#
#
#
# TODO Can we train a 3-layer network?
#
# Note: We use the best performing network parameters from the last exercise.
#

def tryThreeLayers():
    l_rates = [0.095111, 0.095111/4, 0.095111/10]
    for l in l_rates:
        tryParameters("3layerTest_lrate_{}".format(l), lin_neurons=[50, 30],
        with_BN=False, lam=0.00050,
        l_rate=l, decay=0.995,
        mom=0.8, epochs=200)

#tryThreeLayers()

#
#
# TODO coarse and fine search
#
#

def search(reg_range, l_rate_range, dest_file, epochs=10):

    for i, params in enumerate(tqdm(list(itertools.product(reg_range, l_rate_range)), desc=dest_file, ncols=50)):
        run_name="{}_".format(i) + dest_file
        results = tryParameters(run_name,
                                lin_neurons = [50, 30],
                                with_BN=True,
                                lam=params[0],
                                l_rate=params[1],
                                decay=0.995,
                                mom=0.8,
                                epochs=epochs)
        pd.DataFrame([results]).to_csv("results/" + dest_file, mode='a', header=(True if i == 0 else False))

def doSearch():
    # coarse search
    #reg_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
    #l_rate_range = list(reversed(reg_range))
    #l_rate_range[:] = [x/10.0 for x in l_rate_range]
    #print(l_rate_range)
    #search(reg_range, l_rate_range, "coarse.csv", epochs=100)
    # fine search
    reg_range = np.linspace(0.005, 0.0005, num=7)
    l_rate_range = np.linspace(0.001, 0.00001, num=7) # we lower this from feedback of ex2
    search(reg_range, l_rate_range, "fine.csv", epochs=300)
doSearch()


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
#showBest()

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
