import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class Perceptron:
    def __init__(self):
        pass

    def heavyside(self, x):
        return np.array([1 if elem >= 0 else 0 for elem in x])[:, np.newaxis]

    def errorfunction(self, X, y):
        return 100

    def train_perceptron(self, X, y, learningrate=0.1, n_iterations=100):
        costs = []
        n_samples, n_features = X.shape
        self.W = np.zeros((n_features, 1))
        self.b = 0
        for iter in range(n_iterations):
            prediction = self.heavyside(np.dot(X, self.W)+self.b)
            grad_W = np.dot(X.T, prediction - y)
            grad_b = np.sum(prediction - y)
            self.W = self.W - learningrate * grad_W
            self.b = self.b - learningrate * grad_b
        return self.W, self.b


X, y = make_blobs(n_samples=1000, centers=2)
P = Perceptron()
fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Dataset")
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.show()
