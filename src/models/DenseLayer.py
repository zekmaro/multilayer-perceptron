class DenseLayer:
    def __init__(self, units, activation='relu', input_dim=None, weights_initializer='heUniform'):
        self.units = units
        self.activation_name = activation
        self.weights_initializer = weights_initializer
        self.input_dim = input_dim  # optional, might be set by the network
        self.weights = None  # to be initialized when input_dim is known
        self.bias = None


    def initialize_weights(self):
        pass


    def forward(self, inputs):
        pass


    def backward(self, grad_output):
        pass


    def activate(self, inputs):
        pass

    