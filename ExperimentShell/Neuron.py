
class Neuron(object):

    def __init__(self, row, weights):
        self.weights = weights # holds all of the weights that are coming in
        self.inputData = []
        self.activation = 0
        self.target = 0

