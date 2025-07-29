import numpy as np


class DenseLayer:
    def __init__(self, units, activation='relu', input_dim=None, weights_initializer='heUniform'):
        self.units = units
        self.activation_name = activation
        self.weights_initializer = weights_initializer
        self.input_dim = input_dim  # optional, might be set by the network
        self.weights = None  # to be initialized when input_dim is known
        self.bias = None


    def initialize_weights(self):
        if self.input_dim is None:
            raise ValueError("Input dimension must be set before initializing weights.")

        stddev = np.sqrt(2 / self.input_dim)
        self.weights = np.random.normal(0, stddev, (self.input_dim, self.units))


    def forward(self, inputs):
        pass


    def backward(self, grad_output):
        pass


    def activate(self, inputs):
        pass

    