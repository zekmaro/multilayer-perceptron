class Network:
    """A class representing a neural network."""
    def __init__(self, layers):
        self.layers = layers


    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs


    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    

    def predict(self, inputs):
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
