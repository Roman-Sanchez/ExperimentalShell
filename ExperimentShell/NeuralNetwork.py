from Neuron import Neuron
import random


class NeuralNetwork(object):

    # Constructor
    def __init__(self, num_hidden_layers, num_attributes, learning_rate, num_classes):
        self.num_hidden_layers = num_hidden_layers
        self.num_attributes = num_attributes
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.layers = []

    # returns a neuron given the layer and neuron number
    def get_neuron(self, layer_num, neuron_num):
        return self.layers[layer_num][neuron_num]

    def build_output_layer(self):

        lastIndex = len(self.layers) - 1
        numWeights = len(self.layers[lastIndex])
        layer = []

        for node in range (self.num_classes):
            layer.append(Neuron(lastIndex + 1, self.create_weights(numWeights)))

        self.layers.append(layer)

    # builds a layer and appends it to the layers list
    def build_layers(self):

        for numLayer in range(len(self.num_hidden_layers)):
            layer = []

            # if the layer number is 0 then the number of weights is dependent on the number of attributes
            # if it is not then it is dependent on the number of nodes in the previous layer
            if numLayer == 0:
                numWeights = self.num_attributes
            else:
                numWeights = len(self.layers[numLayer - 1])

            # build the individual neurons
            for numNodes in range(self.num_hidden_layers[numLayer]):
                layer.append(Neuron(numLayer, self.create_weights(numWeights)))

            self.layers.append(layer)

        self.build_output_layer()

    def create_weights(self, numWeights):

        weights = []

        for weight in range(numWeights):
            weights.append(random.uniform(-1, 1))

        return weights

    def predict(self):
        pass

    def fit(self):
        pass
