import numpy as np
import numpy.random as nr
from network_structure import network_structure
from activation import run_network
from activation import eval_err

#Weight Initialization
def init_weight(network):
	#Get weight matrices of random values in the required range
	inputs = network.input_units
	hidden = network.hidden_units

	if network.debug:
		in_weights = nr.uniform(-1, 1, inputs * hidden)
		input_weights = np.reshape(in_weights, (hidden, inputs))
		
		hid_weights = nr.uniform(-1, 1, hidden)
		hidden_weights = np.reshape(hid_weights, (1, hidden))
	else:
		in_weights = nr.uniform(-0.1, 0.1, inputs * hidden)
		input_weights = np.reshape(in_weights, (hidden, inputs))
	
		hid_weights = nr.uniform(-0.1, 0.1, hidden)
		hidden_weights = np.reshape(hid_weights, (1, hidden))

	return input_weights, hidden_weights

# adjust weight matrices til error rate converges
def gradient_descent(network, examples):
	if network.debug:
		eta = 0.75
	else:
		eta = 0.01
		
	counter = 0
	input_weights, hidden_weights = init_weight(network)
	
	while counter < network.training_iterations:
		# # feed each input vector through the network, get the weight 
		# # sensitivities (dL/dW for every weight)
		dL_dW_out_tot, dL_dW_hid_tot = run_network(network, np.copy(examples), input_weights, hidden_weights)

		#now we'll adjust all the weights and try again
		input_weights -= eta*dL_dW_hid_tot
		hidden_weights -= eta*dL_dW_out_tot

		# evaluate the error
		rms, cc = eval_err(network, input_weights, hidden_weights, np.copy(examples))

		if counter%100 == 50:
			print 'RMS:'
			print rms
			print 'Correct Classifications:'
			print cc
		counter += 1
	return input_weights, hidden_weights