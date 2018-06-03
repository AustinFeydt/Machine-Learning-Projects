class network_structure:
	def __init__(self, input_units, hidden_units, weight_decay, training_iterations, debug):
		self.input_units = input_units + 1
		self.hidden_units = hidden_units + 1
		self.weight_decay = weight_decay
		if training_iterations > 0:
			self.training_iterations = training_iterations
		else:
			self.training_iterations = float('inf')
		self.debug = debug