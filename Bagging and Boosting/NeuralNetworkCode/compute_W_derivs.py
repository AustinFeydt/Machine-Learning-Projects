import numpy as np
from network_structure import network_structure
from copy import deepcopy



def compute_W_derivs(network, input_weights, hidden_weights, examples):
	dL_dW_hid = 0 * input_weights
	dL_dW_out  = 0 * hidden_weights

	for p in range(len(examples)):
		# trim the class variable
		ex_out = examples[p][-1]

		stim_vec = examples[p]
		stim_vec = np.reshape(stim_vec, (len(stim_vec), 1))
		#add a 1 for the bias
		stim_vec[-1] = 1

		output_hid, output_net = feed_forward(input_weights, hidden_weights,stim_vec)
		for k in range(network.hidden_units):
			err_k = output_net - ex_out
			h_prime_k = output_net * (1-output_net)
			delta_2 = h_prime_k * err_k
			dL_dW_out[(0,k)] = dL_dW_out[(0,k)] + delta_2 * output_hid[k]
			for j in range(network.input_units):
				h_prime_j = output_hid[k] * (1-output_hid[k])
				delta_1 = hidden_weights[0,k]*delta_2*h_prime_j
				dL_dW_hid[(k,j)] += stim_vec[j]*delta_1
	return dL_dW_hid, dL_dW_out


#do output first
#calculate loss, assuming actual was one
# adjust weights by h, compare the loss to the actual loss
# take step based on subtraction above
# then subtract h again
#start with highest confidence

def feed_forward(input_weights, hidden_weights, stim_vec):
	vec_sigmoid = np.vectorize(sigmoid)
	n_hid = np.dot(input_weights, stim_vec)
	output_hid = vec_sigmoid(n_hid)
	# overwrite the output of bias node
	output_hid[(len(output_hid) - 1),0] = 1

	n_out = np.dot(hidden_weights, output_hid)
	output_net = vec_sigmoid(n_out)
	return output_hid, output_net


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

