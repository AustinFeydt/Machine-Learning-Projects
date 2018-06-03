import numpy as np
import DecisionTreeCode.modified_ID3 as ID3
import DecisionTreeCode.modified_dtree as dtree
from Attribute import Attribute

def dtree_booster(training_set, iterations,attributes):
	num_exs = np.size(training_set,0)
	w = (1.0/float(num_exs))*np.ones((1,num_exs))
	classifiers = {}
	epsilons = {}
	alphas = {}

	# We treat our decision tree as a special case, because we end up
	# sorting our training set so many times (anytime we are finding a contin-
	# uous split). Thus, we tack on the weight as an extra attribute, and 
	# add it to the list of attributes
	weight_index = len(attributes)- 1
	# insert column of weights
	training_set = np.insert(training_set, weight_index, w,1)
	
	# update attributes dictionary
	attributes['WEIGHTS'] = Attribute('weights', 'WEIGHTS', '0', weight_index)
	attributes['CLASS'].index = weight_index + 1

	for i in xrange(iterations):
		classifiers[i] = get_classifier(training_set, attributes)
		epsilons[i] = calc_epsilon(training_set, classifiers[i])
		alphas[i] = calc_alpha(epsilons[i])
		update_weights(training_set, classifiers[i], alphas[i])
	return classifiers, alphas

def get_classifier(training_set, attributes):
	tree_builder = ID3.ID3(1, 0)
	return tree_builder.buildTree(training_set, attributes, {}, 1)

def calc_epsilon(training_set, classifier):
	epsilon = 0
	for i in xrange(np.size(training_set, 0)):
		epsilon += classify(training_set[i], classifier)
	if epsilon == 0:
		epsilon += 0.1
	return epsilon

def classify(example, classifier):
	weight = example[len(example) - 2]
	h = dtree.predict(classifier, example)
	if h == example[-1]:
		return 0
	else:
		return weight

def calc_alpha(epsilon):
	return 0.5*np.log(float(1-float(epsilon))/epsilon)

def update_weights(training_set, classifier, alpha):
	num_examples = np.size(training_set,0)
	z = 0
	w_new = np.zeros(num_examples)

	weight_index = len(training_set[0]) - 2
	for i in xrange(0, num_examples):
		example = training_set[i]
		h = dtree.predict(classifier, example)
		y = example[-1]
		
		if y == h:
			hy = -1
		else:
			hy = 1
		w_new[i] = example[weight_index] * np.exp(alpha * hy)
		z += w_new[i]

	w_new = w_new / z
	training_set[:, weight_index] = w_new
	return w_new
