import numpy as np
from network_structure import network_structure
from gradient_descent import gradient_descent

# This short script shows that our network IS able to converge for 
# simpler classification problems, like XOR:
examples = np.array([[0,0,0], [1,1,0], [1,0,1], [0,1,1]])
input_units = 2
hidden_units = 6
weight_decay = 0
training_iterations = 2500
network = network_structure(input_units, hidden_units, weight_decay, training_iterations, True)
gradient_descent(network, examples)
