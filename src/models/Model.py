from src.models.DenseLayer import DenseLayer
from src.models.Network import Network
from src.models.Optimizers import Optimizer
from typing import List
import numpy as np
import os
import pickle
import json


class Model:
    def __init__(self, name : str = None, optimizer: Optimizer = None):
        """
        Initialize the Model class.
        
        Attributes:
            layers (list): List of layers in the neural network.
        """
        self.loss_history = []
        self.accuracy_history = []
        self.name = name
        self.optimizer = optimizer


    def create_network(self, layers: List[DenseLayer], input_dim: int) -> Network:
        """
        Create a neural network with the specified architecture.

        Args:
            layers (list): List of layer configurations.
        
        Returns:
            Network: An instance of the Network class
            initialized with the provided layers.
        """
        if not layers:
            raise ValueError("The network must have at least one layer.")
        layers[0].input_dim = input_dim  # Set input dimension for the first layer
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
        learning_rate: float = 0.01,
        epsilon: float = 1e-6,
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
                self.optimizer.update(layer, layer.dL_dW, layer.dL_db)


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


    def get_accuracy(
            self,
            y_pred: np.ndarray,
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
        pred_classes = np.argmax(y_pred, axis=1)
        return np.mean(pred_classes == y_test)


    def get_precision(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate the precision of the model predictions.
        
        Args:
            y_pred (np.ndarray): Predicted labels.
            y_true (np.ndarray): True labels.
        
        Returns:
            float: Precision score.
        """
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        false_positive = np.sum((y_pred == 1) & (y_true == 0))
        return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0


    def get_recall(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate the recall of the model predictions.
        
        Args:
            y_pred (np.ndarray): Predicted labels.
            y_true (np.ndarray): True labels.
        
        Returns:
            float: Recall score.
        """
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        false_negative = np.sum((y_pred == 0) & (y_true == 1))
        return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    

    def get_f1_score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate the F1 score of the model predictions.

        Args:
            y_pred (np.ndarray): Predicted labels.
            y_true (np.ndarray): True labels.

        Returns:
            float: F1 score.
        """
        precision = self.get_precision(y_pred, y_true)
        recall = self.get_recall(y_pred, y_true)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    

    def get_confusion_matrix(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculate the confusion matrix for the model predictions.

        Args:
            y_pred (np.ndarray): Predicted labels.
            y_true (np.ndarray): True labels.
        
        Returns:
            np.ndarray: Confusion matrix.
        """
        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        return np.array([[TN, FP], [FN, TP]])


    def save(self, network, path: str, config: dict = None):
        """
        Save the model and network to the given directory.

        Args:
            network: The trained network (list of DenseLayer objects).
            path (str): Directory path to save the model.
            config (dict): Optional config dict to save alongside.
        """
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(self, f)

        with open(os.path.join(path, "network.pkl"), "wb") as f:
            pickle.dump(network, f)

        if config:
            with open(os.path.join(path, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
