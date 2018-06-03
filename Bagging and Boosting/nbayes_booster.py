import numpy as np
import NBayesCode.modified_MAP as MAP
import NBayesCode.modified_nbayes as nbayes
import NBayesCode.bucket_cont as buc
from Attribute import Attribute

def nbayes_booster(training_set,iterations, attributes):
	buc.bucket_cont(10, training_set, attributes)
	num_exs = np.size(training_set,0)
	w = (1.0/float(num_exs))*np.ones((num_exs))
	classifiers = {}
	epsilons = np.zeros(iterations)
	alphas = np.zeros(iterations)

	for i in xrange(iterations):
		classifiers[i] = get_classifier(training_set, w)
		epsilons[i] = calc_epsilon(training_set, w, classifiers[i])
		alphas[i] = calc_alpha(epsilons[i])
		w = update_weights(training_set, w, classifiers[i], alphas[i])
	return classifiers, alphas

def get_classifier(training_set, w):
	probs = {}
	pos_probs, neg_probs = MAP.MAP(training_set, 0, w)
	probs[1] = pos_probs
	probs[0] = neg_probs
	return probs

def calc_epsilon(training_set, w, classifier):
	epsilon = 0
	num_examples = np.size(training_set,0)
	for i in xrange(num_examples):
		epsilon += update_epsilon(training_set[i], w[i], classifier, num_examples)
	return epsilon

def update_epsilon(example, w, classifier, num_examples):
	h = nb_classify(example, classifier, num_examples)
	if h == example[-1]:
		return 0
	else:
		return w
		
global classify
def nb_classify(example, classifier, num_examples):
	pos, neg = nbayes.predict(example, num_examples, classifier[1], classifier[0], 0)
	if pos >= neg:
		h = 1
	else:
		h = 0
	return h

def calc_alpha(epsilon):
	return 0.5*np.log(float(1-float(epsilon))/epsilon)

def update_weights(training_set, w, classifier, alpha):
	num_examples = np.size(training_set,0)
	z = 0
	w_new = np.zeros(num_examples)
	for i in xrange(0, num_examples):
		example = training_set[i]
		h = nb_classify(example, classifier, num_examples)
		y = example[-1]
		if y == h:
			hy = -1
		else:
			hy = 1
		w_new[i] = w[i] * np.exp(alpha * hy)
		z += w_new[i]

	w_new = w_new / z
	# print w_new
	# print '~~~~~~~~~~~~~'
	return w_new
