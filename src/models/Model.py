from src.models.DenseLayer import DenseLayer
from src.models.Network import Network
from typing import List
import numpy as np


class Model:
    def __init__(self, name=None):
        """
        Initialize the Model class.
        
        Attributes:
            layers (list): List of layers in the neural network.
        """
        self.loss_history = []
        self.accuracy_history = []
        self.accurancy = 0.0
        self.name = name


    def create_network(self, layers: List[Network]) -> Network:
        """
        Create a neural network with the specified architecture.

        Args:
            layers (list): List of layer configurations.
        
        Returns:
            Network: An instance of the Network class
            initialized with the provided layers.
        """
        for i in range(0, len(layers)):
            if layers[i].input_dim is None:
                layers[i].input_dim = layers[i - 1].units
            layers[i].initialize()
        return Network(layers)


    def fit(
        self,
        network: Network,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 32,
        learning_rate: int = 0.01,
        epsilon: int = 1e-6
    ) -> None:
        """
        Train the model on the provided data.

        Args:
            network (Network): The neural network to train.
            X (np.ndarray): Input features for training.
            y (np.ndarray): Target labels for training.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
            learning_rate (float): Learning rate for weight updates.
            epsilon (float): Threshold for early stopping based on loss change.
        """
        last_loss = float('inf')
        for _ in range(epochs):
            inputs = X
            for layer in network.layers:
                layer.inputs = inputs
                inputs = layer.forward(inputs)

            y_pred = inputs
            pred_classes = np.argmax(y_pred, axis=1)
            accuracy = np.mean(pred_classes == y)
            loss = self.compute_loss(y_pred, y)

            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)

            grad_output = self.compute_loss_gradient(y_pred, y)
            print(f"Epoch {_ + 1}/{epochs} - loss: {loss:.4f} accuracy: {accuracy:.4f}")

            if abs(last_loss - loss) < epsilon:
                print(f"Early stopping at epoch {_ + 1}")
                break
            last_loss = loss

            for layer in reversed(network.layers):
                grad_output = layer.backward(grad_output)
                layer.weights -= learning_rate * layer.dL_dW
                layer.bias -= learning_rate * layer.dL_db


    def compute_loss(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            loss_function: str = 'cross_entropy'
        ) -> float:
        """
        Compute the loss for the current predictions.
        
        Args:
            y_pred (np.ndarray): Predictions from the network.
            y_true (np.ndarray): True labels.
            loss_function (str): Type of loss function to use.
        
        Returns:
            float: Computed loss value.
        """
        if loss_function == 'cross_entropy':
            return -np.mean(y_true * np.log(y_pred[:, 1] + 1e-15) + (1 - y_true) * np.log(y_pred[:, 0] + 1e-15))


    def compute_loss_gradient(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            loss_function: str = 'cross_entropy'
        ) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the predictions.
        
        Args:
            y_pred (np.ndarray): Predictions from the network.
            y_true (np.ndarray): True labels.
            loss_function (str): Type of loss function to use.
        
        Returns:
            np.ndarray: Gradient of the loss with respect to predictions.
        """
        if loss_function == 'cross_entropy':
            one_hot = np.zeros_like(y_pred)
            one_hot[np.arange(len(y_true)), y_true] = 1
            return y_pred - one_hot
        return None


    def predict(self, network: Network, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            network (Network): The trained neural network.
            X (np.ndarray): Input data for prediction.
        
        Returns:
            np.ndarray: Predicted outputs.
        """
        return network.predict(X)


    def get_model_accuracy(
            self,
            network: Network,
            X_test: np.ndarray,
            y_test: np.ndarray
        ) -> float:
        """
        Calculate the accuracy of the model on the test set.
        
        Args:
            network (Network): The trained neural network.
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.
        
        Returns:
            float: Accuracy of the model on the test set.
        """
        y_pred = self.predict(network, X_test)
        pred_classes = np.argmax(y_pred, axis=1)
        self.accuracy = np.mean(pred_classes == y_test)
        return self.accuracy
