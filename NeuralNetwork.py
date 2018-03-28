"""
Implementation of a simple feed forward neural network module in almost plain
python (pyplot, matplotlib).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles


class NeuralNetwork:
    def __init__(self, sizes):
        """
        Initialization of the random weights and biases.
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.Weights = [np.random.randn(m, n)
                        for m, n in zip(sizes[1:], sizes[:-1])]
        self.bias = [np.random.randn(m, 1) for m in sizes[1:]]

    def forwardpass(self, input):
        """
        Does a forwardpass on ``input`` and returns all the activations.
        """
        activations = []
        zs = []
        if np.array(input).shape == (len(input), ):
            z = np.array([input]).T
        else:
            z = input
        for W, b in zip(self.Weights, self.bias):
            z = np.dot(W, z) + b
            a = self.activation(z)
            activations.append(a)
            zs.append(z)
        return z, activations, zs

    def feedbatch(self, batch):
        """
        Feeds a batch into the neural network and returns all its forwardpasses
        of each element in the batch.
        """
        predictions = np.zeros((len(batch), 1))
        for i in range(len(batch)):
            prediction, _, _ = self.forwardpass(batch[i, :])
            predictions[i] = prediction
        return predictions

    def backprop(self, x, y):
        '''
        Returns the gradients of the lossfunction with respect to the weights
        and biases via backpropagation.
        '''
        if np.array(x).shape == (len(x), ):
            x = np.array([x]).T
        a0, activations, zs = self.forwardpass(x)
        activations.insert(0, x)
        grads_W = [np.zeros(w.shape) for w in self.Weights]
        grads_b = [np.zeros(b.shape) for b in self.bias]
        delta = 2*(a0-y)
        grads_W[-1] = np.dot(delta, activations[-2].T)
        grads_b[-1] = delta
        for l in range(2, self.num_layers):
            delta = np.dot(self.Weights[-l+1].T, delta)  \
                    * self.activation_prime(zs[-l])
            grads_b[-l] = delta
            grads_W[-l] = np.dot(delta, activations[-l-1].T)
        return grads_W, grads_b

    def update(self, data, eta=0.01):
        """
        Updates the weights and biases with learningrate=``eta`` using gradient
        descent and backprop.
        """
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.Weights]
        for x, y in data:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.Weights = [w - (eta/len(data))*nw
                        for w, nw in zip(self.Weights, nabla_w)]
        self.bias = [b - (eta/len(data)) * nb
                     for b, nb in zip(self.bias, nabla_b)]

    def activation(self, x):
        """
        Returns the activation of input x, where ``self.type`` denotes which
        activaion function we use (e.g. sigmoid, relu, etc..).
        """
        if self.type == 'sigmoid':
            return 1/(1+np.exp(-x))
        elif self.type == 'relu':
            op = lambda t: t if t >= 0 else 0
            vfunc = np.vectorize(op)
            return vfunc(x)
        elif self.type == 'heavyside':
            op = lambda t: 1 if t >= 0 else 0
            vfunc = np.vectorize(op)
            return vfunc(x)
        elif self.type == 'tanh':
            return 2/(1+np.exp(-2*x))-1

    def activation_prime(self, x):
        """
        Derivative of the activation function.
        """
        if self.type == 'sigmoid':
            return self.activation(x) * (1 - self.activation(x))
        elif self.type == 'relu':
            op = lambda t: 1 if t > 0 else 0
            vfunc = np.vectorize(op)
            return vfunc(x)
        elif self.type == 'heavyside':
            return 0
        elif self.type == 'tanh':
            return 1-self.activation(x)**2

    def lossfunction(self, data, type='LS'):
        '''
        Returns the loss on a given data set ``data``.
        '''
        loss = 0
        for x, y in data:
            x_predict, _, _ = self.forwardpass(x)
            loss = loss + np.dot(x_predict-y, x_predict-y)
        return float(1/len(data) * loss)

    def SGD(self, data_train, epochs=100, mini_batch_size=1, eta=0.01):
        '''
        Stochastic gradient descent (i.e mini_batch_size = 1) or minibatch SGD
        '''
        m = len(data_train)
        for i in range(epochs):
            np.random.shuffle(data_train)
            batches = [data_train[k:k+mini_batch_size]
                       for k in range(0, m, mini_batch_size)]
            # batches = [batches[0]]
            for batch in batches:
                self.update(batch, eta)

    def train(self, data_train, data_test, epochs=100, eta=0.01,
              mini_batch_size=0, type='sigmoid'):
        '''
        Trains the NeuralNetwork
        '''
        self.type = type
        if mini_batch_size == 0:
            mini_batch_size = len(data_train)
        x_test = np.array([x for x, y in data_test])
        x_train = np.array([x for x, y in data_train])
        y_train = np.array([y for x, y in data_train])
        fig = plt.figure(figsize=(8, 6))
        plt.ion()
        for i in range(epochs):
            #self.update(data_train, eta)
            self.SGD(data_train, epochs=1, mini_batch_size=mini_batch_size, eta=eta)
            if (i+1) % 100 == 0:
                print("Iteration: {iter:<5} | Training error: {losstrain:<6} | Test error: {losstest:<6}".format(iter=i+1, losstrain=round(self.lossfunction(data_train), 4), losstest=round(self.lossfunction(data_test), 4)))
                pred = self.feedbatch(x_test)
                t = 0.5
                b = pred < t
                pos = b[:, 0]
                neg = (pos == False)
                plt.clf()
                plt.title("After iteration "+str(i+1))
                plt.xlabel("x_1")
                plt.ylabel("x_2")
                plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
                plt.scatter(x_test[pos, 0], x_test[pos, 1])
                plt.scatter(x_test[neg, 0], x_test[neg, 1])
                plt.pause(0.005)
                plt.show()

    def draw(self):
        """
        Visualization of the neural network.
        """
        Ws = self.Weights
        layers = self.sizes
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'}, figsize=(8, 6))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.axis('off')

        nlayers = self.num_layers
        maxdots = max(layers)
        width = 1/(maxdots+2) * 0.2

        xs = np.linspace(0, 1, num=nlayers + 2)
        ys = [np.linspace(0, 1, num=layers[i] + 2) for i in range(nlayers)]

        for w, W in enumerate(Ws):
            [n, m] = W.shape
            for y in range(m):
                dot = Circle(xy=[xs[w+1], ys[w][y+1]],
                             radius=width,
                             alpha=1,
                             facecolor="grey")
                for x in range(n):
                    if abs(W[x][y]) > 0.1:
                        ccolor = "blue"
                        if W[x][y] < 0:
                            ccolor = "orange"
                        ligne = ConnectionPatch([xs[w+1], ys[w][y+1]], [xs[w+1+1], ys[w+1][x+1]],
                                                "data", "data",
                                                arrowstyle="simple, tail_width=" + str(abs(W[x][y])*0.05),
                                                color=ccolor,
                                                alpha=0.5)
                        ax.add_artist(ligne)
                dot.set_clip_box(ax.bbox)
                ax.add_artist(dot)
        dot = Circle(xy=[xs[-2], ys[-3][2]],
                     radius=width,
                     alpha=1,
                     facecolor="grey")
        dot.set_clip_box(ax.bbox)
        ax.add_artist(dot)
        plt.show()


def example1():
    np.random.seed(15)
    NN = NeuralNetwork([2, 4, 2, 1])
    X, y = make_blobs(n_samples=500, centers=2)
    x_test = X[:100, :]
    y_test = y[:100]
    x_train = X[100:, :]
    y_train = y[100:]
    data_train = list(zip(x_train, y_train))
    data_test = list(zip(x_test, y_test))
    NN.train(data_train, data_test, epochs=600, eta=0.03,
             mini_batch_size=100, type='sigmoid')


def example2():
    np.random.seed(123)
    NN = NeuralNetwork([2, 8, 4, 1])
    X, y = make_circles(n_samples=1000, factor=0.5, noise=.05)
    x_test = X[:200, :]
    y_test = y[:200]
    x_train = X[200:, :]
    y_train = y[200:]
    data_train = list(zip(x_train, y_train))
    data_test = list(zip(x_test, y_test))
    NN.train(data_train, data_test, epochs=600, eta=0.01,
             mini_batch_size=256, type='tanh')



example1()
