import numpy as np
import LogisticRegressionCode.modified_logistic_regression as l_r
from Attribute import Attribute

def logreg_booster(training_set, iterations, attributes):
	alt_examples = np.copy(training_set)
	alt_examples[:,-1] = 1
	
	num_exs = np.size(training_set,0)
	ada_w = (1.0/float(num_exs))*np.ones((num_exs))
	classifiers = {}
	epsilons = np.zeros(iterations)
	alphas = np.zeros(iterations)

	for i in xrange(iterations):
		classifiers[i] = get_classifier(training_set, alt_examples, ada_w)
		epsilons[i] = calc_epsilon(training_set, alt_examples, ada_w, classifiers[i])
		alphas[i] = calc_alpha(epsilons[i])
		ada_w = update_ada_weights(training_set, alt_examples, ada_w, classifiers[i], alphas[i])
	return classifiers, alphas

def get_classifier(training_set, alt_examples, ada_w):
	return l_r.modified_logistic_regression(training_set, alt_examples, 0, ada_w)

def calc_epsilon(training_set, alt_examples, ada_w, classifier):
	epsilon = 0
	num_examples = np.size(training_set,0)
	for i in xrange(num_examples):
		epsilon += update_epsilon(training_set[i], alt_examples[i], ada_w[i], classifier)
	return epsilon

def update_epsilon(example, alt_example, ada_w, classifier):
	out_pred = lr_classify(example, alt_example, classifier)
	if out_pred == example[-1]:
		return 0
	else:
		return ada_w
		
global classify
def lr_classify(example, alt_example, classifier):
	confidence = l_r.predict(alt_example,classifier)
	out_pred = np.round(confidence)
	return out_pred

def calc_alpha(epsilon):
	return 0.5*np.log(float(1-float(epsilon))/epsilon)

def update_ada_weights(training_set, alt_examples, ada_w, classifier, alpha):
	num_examples = np.size(training_set,0)
	z = 0
	ada_w_new = np.zeros(num_examples)
	for i in xrange(0, num_examples):
		example = training_set[i]
		alt_example = alt_examples[i]
		h = lr_classify(example, alt_example, classifier)
		y = example[-1]
		if y == h:
			hy = -1
		else:
			hy = 1
		ada_w_new[i] = ada_w[i] * np.exp(alpha * hy)
		z += ada_w_new[i]

	ada_w_new = ada_w_new / z
	# print ada_w_neada_w
	# print '~~~~~~~~~~~~~'
	return ada_w_new
