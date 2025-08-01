from src.models.DenseLayer import DenseLayer
from typing import List
import numpy as np


class Network:
    """A class representing a neural network."""
    def __init__(self, layers: List[DenseLayer]):
        """
        Initialize the Network with a list of layers.
        
        Attributes:
            layers (List[DenseLayer]): List of layers in the network.
        """
        self.layers = layers


    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            inputs (np.ndarray): Input data for the network.
        
        Returns:
            np.ndarray: Output of the network after the forward pass.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs


    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the network.

        Args:
            grad_output (np.ndarray): Gradient of the loss
            with respect to the output of layer.
        
        Returns:
            np.ndarray: Gradient of the loss with respect to the input of layer.
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Make predictions using the network.
        
        Args:
            inputs (np.ndarray): Input data for prediction.
        
        Returns:
            np.ndarray: Predicted outputs.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
