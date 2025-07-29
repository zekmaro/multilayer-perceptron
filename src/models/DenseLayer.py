import numpy as np


class DenseLayer:
    def __init__(
        self, 
        units: int, 
        activation_name: str = 'relu', 
        input_dim: int = None, 
        weights_initializer: str = 'heUniform'
    ) -> None:
        """
        Initialize a Dense Layer.
        
        Args:
            units (int): Number of neurons in the layer.
            activation_name (str): Activation function name ('relu', 'sigmoid', 'softmax').
            input_dim (int, optional): Input dimension for the layer. Required for weight initialization.
            weights_initializer (str): Method to initialize weights ('heUniform' or others).
        """
        self.units = units
        self.activation_name = activation_name
        self.weights_initializer = weights_initializer
        self.input_dim = input_dim  # optional, might be set by the network
        self.weights = None  # to be initialized when input_dim is known
        self.bias = None
        self.z = None
        self.a = None


    def initialize_weights(self):
        if self.input_dim is None:
            raise ValueError("Input dimension must be set before initializing weights.")

        stddev = np.sqrt(2 / self.input_dim)
        self.weights = np.random.normal(0, stddev, (self.input_dim, self.units))


    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.a = self.activate(self.z)
        return self.a


    def backward(self, grad_output):
        pass


    def activate(self, inputs):
        if self.activation_name == 'relu':
            return np.maximum(0, inputs)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-inputs))
        elif self.activation_name == 'softmax':
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            return exp_values / np.sum(exp_values, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_name}")
