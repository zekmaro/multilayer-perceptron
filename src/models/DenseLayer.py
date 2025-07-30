import numpy as np


class DenseLayer:
    """Class for a dense layer in a neural network."""
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
        
        Attributes:
            units (int): Number of neurons in the layer.
            activation_name (str): Name of the activation function.
            weights_initializer (str): Method to initialize weights.
            input_dim (int): Input dimension for the layer.
            inputs (np.ndarray): Input data to the layer.
            weights (np.ndarray): Weights of the layer.
            bias (np.ndarray): Biases of the layer.
            z (np.ndarray): Linear combination of inputs and weights plus bias.
            a (np.ndarray): Activated output of the layer.
            dL_dW (np.ndarray): Gradient of the loss with respect to weights.
            dL_db (np.ndarray): Gradient of the loss with respect to biases.
        """
        self.units = units
        self.activation_name = activation_name
        self.weights_initializer = weights_initializer
        self.input_dim = input_dim  # optional, might be set by the network
        self.inputs = None
        self.weights = None  # to be initialized when input_dim is known
        self.bias = None
        self.z = None
        self.a = None
        self.dL_dW = None
        self.dL_db = None


    def initialize(self) -> None:
        """Initialize weights and biases for the layer."""
        if self.input_dim is None:
            raise ValueError("Input dimension must be set before initializing weights.")

        stddev = np.sqrt(2 / self.input_dim)
        self.weights = np.random.normal(0, stddev, (self.input_dim, self.units))
        self.bias = np.zeros((1, self.units))


    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform forward pass through the layer.

        Args:
            inputs (np.ndarray): Input data to the layer.
        
        Returns:
            np.ndarray: Output after applying the layer's weights, bias, and activation function.
        """
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.a = self.activate(self.z)
        return self.a


    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Perform backward pass through the layer.
        
        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the layer's output.
        
        Returns:
            np.ndarray: Gradient of the loss with respect to the layer's input.
        """
        if self.activation_name == 'softmax':
            dL_dz = grad_output
        else:
            dL_dz = grad_output * self.activation_derivative(self.z)
        dL_dW = np.dot(self.inputs.T, dL_dz)
        dL_db = np.sum(dL_dz, axis=0)
        dL_dX = np.dot(dL_dz, self.weights.T)

        self.dL_dW = dL_dW
        self.dL_db = dL_db
        return dL_dX


    def activate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Apply the activation function to the inputs.
        
        Args:
            inputs (np.ndarray): Inputs to the activation function.
        
        Returns:
            np.ndarray: Activated outputs.
        """
        if self.activation_name == 'relu':
            return np.maximum(0, inputs)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-inputs))
        elif self.activation_name == 'softmax':
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            return exp_values / np.sum(exp_values, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_name}")


    def activation_derivative(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the activation function.
        
        Args:
            inputs (np.ndarray): Inputs to the activation function.
        
        Returns:
            np.ndarray: Derivative of the activated outputs.
        """
        if self.activation_name == 'relu':
            return np.where(inputs > 0, 1, 0)
        elif self.activation_name == 'sigmoid':
            sig = self.activate(inputs)
            return sig * (1 - sig)
        elif self.activation_name == 'softmax':
            # Softmax derivative is more complex and typically handled in the loss function
            raise NotImplementedError("Softmax derivative is not implemented.")
        else:
            raise ValueError(f"Unknown activation function: {self.activation_name}")
