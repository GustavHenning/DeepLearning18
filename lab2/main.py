import pickle
import scipy

import numpy as np
import matplotlib.pyplot as plt
from net import Net

from tqdm import tqdm
from sklearn import preprocessing
from laplotter import LossAccPlotter

from layers import Softmax, ReLU, Linear

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

def trainMiniBatch(train, val, net, eta=0.01, epochs=20, batch_size=100, shuffle=False):
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
            prob, pred = net.evaluate(x)
            gW, gB = net.slides_gradient(x, prob, truth)
            net.W -= eta * gW
            net.b -= eta * gB

        # Measure each epoch
        prob, pred = net.evaluate(trainImgT)
        c_train.append(net.cost(prob, trainTruthT))
        a_train.append(net.accuracy(pred, trainTruthT))

        prob, pred = net.evaluate(valImgT)
        c_val.append(net.cost(prob, valTruthT))
        a_val.append(net.accuracy(pred, valTruthT))

    return np.array(a_train), np.array(c_train), np.array(a_val), np.array(c_val)


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

def tryParameters(test_name, lam, epochs, batch_size, eta):
    net = Net(cifar.in_size, cifar.out_size, lam)
    a_train, c_train, a_val, c_val = trainMiniBatch(train, val, net, eta, epochs, batch_size)
    prob, pred = net.evaluate(test['images'].T)
    print('{} Test Accuracy: {:.2f}'.format(test_name, net.accuracy(pred, test['one_hot'].T)))
    plotResults(test_name, a_train, c_train, a_val, c_val)
    weights_plot(net, "plots/weights_vizualisation_{}.png".format(test_name), labels)


# Exercise 1
cifar=CIFAR()
labels=cifar.labels

train = cifar.get_batches('data_batch_1')
val = cifar.get_batches('data_batch_2')
test = cifar.get_batches("test_batch")

# Plot label distribution
#barPlotLabels(train, labels, "barLabels")
#plotImages(train, "Training set")
#plotImages(val, "Validation set")
#plotImages(test, "Test set")

# Create the network and test the accuracy before training
net = Net([Linear(cifar.in_size, 50), ReLU(50), Linear(50, cifar.out_size), Softmax(cifar.out_size)])
num_examples = 100
in_data = train['images'][0:num_examples].T
truth = train['one_hot'][0:num_examples].T
pred = net.forward(in_data)
net.backward(truth)
print('Initial Accuracy: {:.2f}'.format(net.accuracy(truth, None, pred)))

# Compare gradients
#numericW, numericB = num_gradient(in_data, truth, net)
#slidesW, slidesB = net.slides_gradient(in_data, prob, truth)
#compareGradients(slidesW, slidesB, numericW, numericB)

# Train the network
#a_train, c_train, a_val, c_val = trainMiniBatch(train, val, net)

#plotResults("Initial", a_train, c_train, a_val, c_val)

#weights_plot(net, plot_loc + "weights_vizualisation_Initial.png", labels)

# Parameter testing, using the greater train set
train = cifar.get_batches('data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4') #
val = cifar.get_batches('data_batch_5')
truth = train['one_hot'].T

#tryParameters("ParamTest1", 0, 40, 100, .1)
#tryParameters("ParamTest2", 0, 40, 100, .01)
#tryParameters("ParamTest3", .1, 40, 100, .01)
#tryParameters("ParamTest4", 1, 40, 100, .01)
