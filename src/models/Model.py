from src.models.Network import Network
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


    def create_network(self, layers):
        """
        Create a neural network with the specified architecture.
        """
        for i in range(0, len(layers)):
            if layers[i].input_dim is None:
                layers[i].input_dim = layers[i - 1].units
            layers[i].initialize()
        return Network(layers)


    def fit(self, network, X, y, epochs=1000, batch_size=32, learning_rate=0.01, epsilon=1e-6):
        """
        Train the model on the provided data.
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


    def compute_loss(self, y_pred, y_true, loss_function='cross_entropy'):
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
            return -np.mean(y_true * np.log(y_pred[:, 1] + 1e-15) + (1 - y_true) * np.log(y_pred[:, 0] + 1e-15))


    def compute_loss_gradient(self, y_pred, y_true, loss_function='cross_entropy'):
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


    def predict(self, network, X):
        """
        Make predictions using the trained model.
        
        Args:
            network (Network): The trained neural network.
            X (np.ndarray): Input data for prediction.
        
        Returns:
            np.ndarray: Predicted outputs.
        """
        return network.predict(X)


    def get_model_accuracy(self, network, X_test, y_test):
        y_pred = self.predict(network, X_test)
        pred_classes = np.argmax(y_pred, axis=1)
        self.accuracy = np.mean(pred_classes == y_test)
        return self.accuracy
