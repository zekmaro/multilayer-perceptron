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
    def __init__(self, learning_rate: float = 0.01, decay_rate: float = 0.9, epsilon: float = 1e-8):
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
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = {}
        self.v_w = {}
        self.m_b = {}
        self.v_b = {}
        self.t = 0


    def update(self, layer, grad_w, grad_b):
        """
        Update the weights and biases of the layer using Adam optimizer.
        
        Args:
            layer (DenseLayer): The layer to update.
            grad_w (np.ndarray): Gradient of the loss with respect to the weights.
            grad_b (np.ndarray): Gradient of the loss with respect to the biases.
        """
        if layer not in self.m_w:
            self.m_w[layer] = np.zeros_like(grad_w)
        if layer not in self.v_w:
            self.v_w[layer] = np.zeros_like(grad_w)
        if layer not in self.m_b:
            self.m_b[layer] = np.zeros_like(grad_b)
        if layer not in self.v_b:
            self.v_b[layer] = np.zeros_like(grad_b)

        self.t += 1

        self.m_w[layer] = self.beta1 * self.m_w[layer] + (1 - self.beta1) * grad_w
        self.m_b[layer] = self.beta1 * self.m_b[layer] + (1 - self.beta1) * grad_b

        self.v_w[layer] = self.beta2 * self.v_w[layer] + (1 - self.beta2) * np.square(grad_w)
        self.v_b[layer] = self.beta2 * self.v_b[layer] + (1 - self.beta2) * np.square(grad_b)

        m_w_hat = self.m_w[layer] / (1 - np.power(self.beta1, self.t))
        m_b_hat = self.m_b[layer] / (1 - np.power(self.beta1, self.t))

        v_w_hat = self.v_w[layer] / (1 - np.power(self.beta2, self.t))
        v_b_hat = self.v_b[layer] / (1 - np.power(self.beta2, self.t))

        layer.weights -= (self.learning_rate * m_w_hat) / (np.sqrt(v_w_hat) + self.epsilon)
        layer.bias -= (self.learning_rate * m_b_hat) / (np.sqrt(v_b_hat) + self.epsilon)


OPTIMIZER_CLASSES = {
    "gradient_descent": GD,
    "momentum": Momentum,
    "rmsprop": RMSProp,
    "adam": Adam,
}
