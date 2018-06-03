import numpy as np
from network_structure import network_structure

# returns the dL_dW matrices for each layer
def run_network(network, examples, input_weights, hidden_weights):
	weight_decay = network.weight_decay
	dL_dW_hid_tot = 0 * input_weights
	dL_dW_out_tot = 0 * hidden_weights

	for ex in examples:
		# get the class of the example
		ex_out = ex[-1]
		stim_vec = np.array(ex)
		stim_vec = np.reshape(stim_vec, (len(stim_vec), 1))
		#add a 1 for the bias
		stim_vec[-1] = 1

		#evaluate network on the example
		h_n_hid, network_out = feed_forward(stim_vec, input_weights, hidden_weights)

		#compute dL/dW for hidden-to-output weights
		dL_dW_out, delta = dW_out(weight_decay, network_out, ex_out, h_n_hid, hidden_weights)

		#comput dL/dW for input-to-hidden weights
		dL_dW_hid = dW_hidden(delta, weight_decay, h_n_hid, stim_vec, hidden_weights, input_weights)

		dL_dW_out_tot += dL_dW_out
		dL_dW_hid_tot += dL_dW_hid

	return dL_dW_out_tot, dL_dW_hid_tot

#sigmoid function that stops overflow errors!!!! I think this is why
# my neural network wouldn't work
def sigmoid(x):
	if x >= 0:
		z = np.exp(-x)
		return 1 / (1 + z)
	else:
		# if x is less than zero then z will be small, denom can't be
		# zero because it's 1+z.
		z = np.exp(x)
		return z / (1 + z)

#Now for the derivative of the sigmoid function
def sigmoid_deriv(x):
	return sigmoid(x) * (1 - sigmoid(x))

# Feed the input into the network and get the confidence
def feed_forward(ex_input, input_weights, hidden_weights):
	vec_sigm = np.vectorize(sigmoid)

	h_n_hid = vec_sigm(np.dot(input_weights, ex_input))
	h_n_hid[(len(h_n_hid) - 1),0] = 1
	network_out = vec_sigm(np.dot(hidden_weights, h_n_hid))

	return h_n_hid, network_out

# Compute the contribution of each output weight to the error
# we'll also add the contribution from the weight decay
def dW_out(decay, network_out, ex_out, h_n_hid, hidden_weights):
	delta = (network_out - ex_out) * network_out * (1 - network_out)
	dL =  np.dot(h_n_hid, delta)
	weight_decay = 2 * decay * hidden_weights
	output = np.transpose(dL) + weight_decay
	return output, delta

# Compute the contribution of each hidden weight to the error
# we also include the weight decay term
def dW_hidden(delta, decay, h_n_hid, ex_input, hidden_weights, input_weights):
	vec_sigmoid_deriv = np.vectorize(sigmoid_deriv)
	h_prime = vec_sigmoid_deriv(np.dot(input_weights, ex_input))
	delta_new = (np.dot(np.transpose(hidden_weights), delta))*h_prime
	dL = np.dot(delta_new, np.transpose(ex_input))
	weight_decay = 2 * decay * input_weights
	output = dL + weight_decay
	return output

def eval_err(network, input_weights, hidden_weights, examples):
	esqd = 0
	correct_count = 0
	for p in range(len(examples)):
		# trim the class variable
		ex_out = examples[p][-1]

		stim_vec = examples[p]
		stim_vec = np.reshape(stim_vec, (len(stim_vec), 1))
		#add a 1 for the bias
		stim_vec[-1] = 1

		output_hid, output_net = feed_forward(stim_vec, input_weights, hidden_weights)

		if np.abs(ex_out- output_net) < 0.5:
			correct_count+=1

		errvec= output_net - ex_out;
		esqd += errvec*errvec

	return np.sqrt(esqd/len(examples)), correct_count