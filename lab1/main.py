import pickle
import scipy

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing

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

# Exercise 1
cifar=CIFAR()
labels=cifar.labels

train=cifar.get_batches("data_batch_1")
val=cifar.get_batches("data_batch_1")
test=cifar.get_batches("test_batch")

# Plot label distribution
barPlotLabels(train, labels, "barLabels")
plotImages(train, "Training set")
plotImages(val, "Validation set")
plotImages(test, "Test set")

# Time to make a network
class Net():
    def __init__(self, input_size, output_size, lam=0.1):
        self.in_size = input_size
        self.out_size = output_size
        # Guassian normal dist as sensible random initialization
        # Weights
        self.W = np.random.normal(loc=0.5, scale=0.1, size=(output_size, input_size))
        # Bias
        self.b = np.random.normal(loc=0.5, scale=0.1, size=(output_size, 1))
        # Lambda term
        self.lam = lam

    def softmax(self, x):
        try:
            e = np.exp(x)
            return e / np.sum(e, axis=0)
        except FloatingPointError:
            # Gradient explosion scenario TODO
            return np.ones(x)

    def evaluate(self, X):
        """
        X: input of size [in_size, N]
            returns:
        prob: probabilities of each class [out_size, N]
        pred: integer value of the most probable class: [1, N]
        """
        X = np.reshape(X, (self.in_size, -1))
        prob = self.softmax(np.dot(self.W, X) + self.b)
        pred = np.argmax(prob, axis=0)
        return prob, pred

    def cost(self, prob, truth):
        """
        prob: probablities of each class [out_size, N]
        (ground) truth: one hot encodings, one per image [out_size, N]
            returns:
        cost: cross entropy plus L2 regularization term to minimize
        equivalent to (5), (6) in Assignment 1
        """
        N = prob.shape[1]
        prob = np.reshape(prob, (self.out_size, -1))
        truth = np.reshape(truth, (self.out_size, -1))

        Py = np.multiply(truth, prob).sum(axis=0)
        Py[Py == 0] = np.finfo(float).eps # fix floats

        return - np.log(Py).sum() / N * self.lam * np.power(self.W, 2).sum()


net = Net(cifar.in_size, cifar.out_size)
num_examples = 3
in_data = train['images'][0:num_examples]
ground_truth = train['one_hot'][0:num_examples]
probabilities, predictions = net.evaluate(in_data)

print(probabilities)
print(predictions)
