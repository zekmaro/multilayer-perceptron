import numpy as np


class Optimizer:
    def update(self, layer, grad_w, grad_b):
        raise NotImplementedError("Each optimizer must implement the update method.")


class GD(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate


    def update(self, layer, grad_w, grad_b):
        """
        Update the weights and biases of the layer using SGD.
        
        Args:
            layer (DenseLayer): The layer to update.
            grad_w (np.ndarray): Gradient of the loss with respect to the weights.
            grad_b (np.ndarray): Gradient of the loss with respect to the biases.
        """
        layer.weights -= self.learning_rate * grad_w
        layer.bias -= self.learning_rate * grad_b


class Momentum(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = {}  # key: layer, value: array
        self.velocity_b = {}

    def update(self, layer, grad_w, grad_b):
        if layer not in self.velocity_w:
            self.velocity_w[layer] = np.zeros_like(grad_w)
        if layer not in self.velocity_b:
            self.velocity_b[layer] = np.zeros_like(grad_b)

        self.velocity_w[layer] = (
            self.momentum * self.velocity_w[layer] + (1 - self.momentum) * grad_w
        )
        self.velocity_b[layer] = (
            self.momentum * self.velocity_b[layer] + (1 - self.momentum) * grad_b
        )

        layer.weights -= self.learning_rate * self.velocity_w[layer]
        layer.bias -= self.learning_rate * self.velocity_b[layer]



class NesterovMomentum(Momentum):
    pass


class RMSProp(Optimizer):
    def __init__(self, learning_rate: float = 0.001, decay_rate: float = 0.9, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache_w = {}  # Per-layer cache
        self.cache_b = {}

    def update(self, layer, grad_w, grad_b):
        if layer not in self.cache_w:
            self.cache_w[layer] = np.zeros_like(grad_w)
        if layer not in self.cache_b:
            self.cache_b[layer] = np.zeros_like(grad_b)

        self.cache_w[layer] = (
            self.decay_rate * self.cache_w[layer] + (1 - self.decay_rate) * np.square(grad_w)
        )
        self.cache_b[layer] = (
            self.decay_rate * self.cache_b[layer] + (1 - self.decay_rate) * np.square(grad_b)
        )

        layer.weights -= self.learning_rate * grad_w / (np.sqrt(self.cache_w[layer]) + self.epsilon)
        layer.bias -= self.learning_rate * grad_b / (np.sqrt(self.cache_b[layer]) + self.epsilon)



class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0


    def update(self, layer, grad_w, grad_b):
        """
        Update the weights and biases of the layer using Adam optimizer.
        
        Args:
            layer (DenseLayer): The layer to update.
            grad_w (np.ndarray): Gradient of the loss with respect to the weights.
            grad_b (np.ndarray): Gradient of the loss with respect to the biases.
        """
        pass


OPTIMIZER_CLASSES = {
    "gradient_descent": GD,
    "momentum": Momentum,
    "rmsprop": RMSProp,
    "adam": Adam,
}
