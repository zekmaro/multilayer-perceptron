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
        layer.biases -= self.learning_rate * grad_b


class Momentum(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None


    def update(self, layer, grad_w, grad_b):
        """
        Update the weights and biases of the layer using Momentum optimizer.
        
        Args:
            layer (DenseLayer): The layer to update.
            grad_w (np.ndarray): Gradient of the loss with respect to the weights.
            grad_b (np.ndarray): Gradient of the loss with respect to the biases.
        """
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(layer.weights)
            self.velocity_b = np.zeros_like(layer.biases)

        self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * grad_w
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * grad_b

        layer.weights += self.velocity_w
        layer.biases += self.velocity_b


class NesterovMomentum(Momentum):
    pass


class RMSProp(Optimizer):
    def __init__(self, learning_rate: float = 0.001, decay_rate: float = 0.9, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache_w = None
        self.cache_b = None


    def update(self, layer, grad_w, grad_b):
        """
        Update the weights and biases of the layer using RMSProp optimizer.
        
        Args:
            layer (DenseLayer): The layer to update.
            grad_w (np.ndarray): Gradient of the loss with respect to the weights.
            grad_b (np.ndarray): Gradient of the loss with respect to the biases.
        """
        


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
