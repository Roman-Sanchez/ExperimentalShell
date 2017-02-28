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

    def fit(self, training_data, training_target):

        for data_row, target in zip(training_data, training_target):

            for neuron in self.layers[0]:
                neuron.inputData = data_row

            for num in range(10000):
                self.feedforward()
                self.back_propagate_output(target)
                self.back_propagate()
                self.update_weights()

    def back_propagate(self):
        index = len(self.layers) - 2

        if index > 0:
            for num in range(index + 1):
                cur_layer = self.layers[index]
                k_layer = self.layers[index + 1]
                num_neuron = 0
                for neuron in cur_layer:
                    summation = 0
                    cur_act = neuron.activation

                    for k_neuron in k_layer:
                        summation += k_neuron.error * k_neuron.weights[num_neuron]

                    neuron.error = cur_act * (1 - cur_act) * summation
                    num_neuron += 1
                index -= 1

    def update_weights(self):

        for layer in self.layers:
            for neuron in layer:
                tmp_weights = []
                for weight in neuron.weights:
                    tmp_weights.append(weight - self.learning_rate * neuron.error)
                neuron.weights = tmp_weights

    def back_propagate_output(self, target):
        last_index = len(self.layers) - 1

        for neuron in self.layers[last_index]:
            activation = neuron.activation
            neuron.error = (activation - target) * (activation*(1 - activation))

    def feedforward(self):
        cur_layer = 0
        total_layers = len(self.layers)

        for layer in self.layers:
            for neuron in layer:
                neuron.update_activation()

            if cur_layer < total_layers - 1:
                acts = self.build_activation_array(cur_layer)
                for neuron in self.layers[cur_layer + 1]:
                    neuron.inputData = acts
            cur_layer += 1

    # grab all of the activations from a specified layer and return it as an array
    def build_activation_array(self, num_layer):
        activations = []
        active_layer = self.layers[num_layer]

        for neuron in active_layer:
            activations.append(neuron.activation)

        return activations
