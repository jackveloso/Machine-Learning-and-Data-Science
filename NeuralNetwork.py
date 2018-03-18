import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch


class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.Weights = [np.random.randn(m, n)
                        for m, n in zip(sizes[1:], sizes[:-1])]
        self.bias = [np.random.randn(m, 1) for m in sizes[1:]]

    def activation(self, x, type='sigmoid'):
        if type == 'sigmoid':
            return [1/(1+np.exp(-item)) for item in x]
        elif type == 'relu':
            return [item if item >= 0 else 0 for item in x]
        elif type == 'heavyside':
            return [1 if item >= 0 else 0 for item in x]

    def evaluate(self, input, type='sigmoid'):
        if np.array(input).shape == (len(input), ):
            print('help')
            a = np.array([input]).T
        else:
            a = input
        for W, b in zip(self.Weights, self.bias):
            a0 = np.dot(W, a) + b
            a = self.activation(a0, type)
        return a0

    def backprop(self, x, y):
        pass

    def SGD(self, X_train, epochs, mini_batch_size, eta):
        pass

    def update_batch(self, batch, eta):
        pass

    def draw(self):
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


np.random.seed(123)
NN = NeuralNetwork([8, 4, 6, 3, 2, 1])
NN.draw()
print(NN.evaluate(input, 'sigmoid'))
