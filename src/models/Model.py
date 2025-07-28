from src.models.Network import Network


class Model:
	def create_network(self, layers):
		"""
		Create a neural network with the specified architecture.
		"""
		return Network(layers)


	def fit(self, X, y, epochs=100, batch_size=32, learning_rate=0.01):
		"""
		Train the model on the provided data.
		"""
		pass
