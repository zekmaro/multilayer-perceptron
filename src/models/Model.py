from src.models.Network import Network
import numpy as np


class Model:
    def __init__(self):
        """
        Initialize the Model class.
        
        Attributes:
            layers (list): List of layers in the neural network.
        """
        self.loss_history = []


    def create_network(self, layers):
        """
        Create a neural network with the specified architecture.
        """
        for i in range(1, len(layers)):
            if layers[i].input_dim is None:
                layers[i].input_dim = layers[i - 1].units
            layers[i].initialize()
        return Network(layers)


    def fit(self, network, X, y, epochs=100, batch_size=32, learning_rate=0.01):
        """
        Train the model on the provided data.
        """

        # implement early stopping
        for _ in range(epochs):
            for layer in network.layers:
                layer.inputs = X
                X = layer.forward(X)
            
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            grad_output = self.compute_loss_gradient(X, y)
            
            for layer in reversed(network.layers):
                grad_output = layer.backward(grad_output)
                layer.weights -= learning_rate * layer.dL_dW
                layer.bias -= learning_rate * layer.dL_db


    def compute_loss(self, X, y, loss_function='cross_entropy'):
        """
        Compute the loss for the current predictions.
        
        Args:
            X (np.ndarray): Predictions from the network.
            y (np.ndarray): True labels.
            loss_function (str): Type of loss function to use.
        
        Returns:
            float: Computed loss value.
        """
        if loss_function == 'cross_entropy':
            return -np.mean(y * np.log(X + 1e-15))


    def compute_loss_gradient(self, X, y, loss_function='cross_entropy'):
        """
        Compute the gradient of the loss with respect to the predictions.
        
        Args:
            X (np.ndarray): Predictions from the network.
            y (np.ndarray): True labels.
            loss_function (str): Type of loss function to use.
        
        Returns:
            np.ndarray: Gradient of the loss with respect to predictions.
        """
        if loss_function == 'cross_entropy':
            return -y / (X + 1e-15)
        return None