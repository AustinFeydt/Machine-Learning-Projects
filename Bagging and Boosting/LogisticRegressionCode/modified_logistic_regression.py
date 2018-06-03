import numpy as np
def modified_logistic_regression(examples, alt_examples, lda, ada_w):
	w = np.zeros(np.size(examples,1))
	return weight_gradient(examples, alt_examples, w, lda, ada_w)

# stochastic gradient descent to find optimal weights
def weight_gradient(examples, alt_examples, w, lda, ada_w):
	eta = 0.0001
	ll = 0
	for i in xrange(2500):
		for j in xrange(np.size(examples,0)):
			
			# predict output using sigmoid
			est_out = predict(alt_examples[j], w)
			
			# calculate difference of expected and actual output
			out_err = examples[j][-1] - est_out

			# size of weight's contribution to error
			weight_contr = np.sum(w)*float(lda)

			#calculate the gradient from log likelihood
			gradient = ada_w[j]*alt_examples[j]*out_err + weight_contr

			#adjust weight vector and continue
			w += eta*gradient
		ll_prev = ll
		ll = log_likelihood(examples, alt_examples, w, lda)
		if np.abs(np.abs(ll_prev) - np.abs(ll)) < 0.001:
			print 'Log-Likelihood: ' + str(ll)
			break
	return w

# use sigmoid function with linear combination of weights and features
# to predict the output
global predict
def predict(ex, w):
	x = np.dot(ex, w)
	return sigmoid(x)

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

# calculates the log likelihood to see if we're on the right track
def log_likelihood(examples, alt_examples, w, lda):
	ll = 0
	for i in xrange(np.size(examples,0)):
		score = np.dot(alt_examples[i], w)
		ll += examples[i][-1]*score - np.log(1 + np.exp(score))
		ll += 0.5*lda*np.sum(w)*np.sum(w)
	return ll