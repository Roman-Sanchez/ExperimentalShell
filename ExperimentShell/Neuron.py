import numpy as np


class Neuron(object):

    def __init__(self, row, weights):
        self.weights = weights # holds all of the weights that are coming in
        self.inputData = []
        self.activation = 0
        self.target = 0
        self.error = 0

    def update_activation(self):
        h = 0

        for data_item, weight in zip(self.inputData, self.weights):
            h += data_item * weight

        self.activation = 1.0 / (1.0 + np.exp(-1.0 * h))
