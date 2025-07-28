class Network:
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

	def activate(self, inputs):
		for layer in self.layers:
			inputs = layer.activate(inputs)
		return inputs
	
	