import numpy as np
import standardizeData as sd
from copy import deepcopy
from Attribute import Attribute
from network_structure import network_structure
from gradient_descent import gradient_descent
from activation import feed_forward
from mldata import *
def main():
	# while True:
	# 	data_path = raw_input("Please enter path to the data: ")
	# 	try:
	# 		orig_array = parse_c45(data_path)
	# 		break
	# 	except Exception as error:
	# 		print 'Not a valid path!'

	# print 'Converting Example array to np array...'
	# examples = np.array(orig_array.to_float())
	
	# print 'Conversion done. \n Extracting attributes...'
	# attributes = getAttributes(orig_array)
	# print'Attributes successfully parsed'

	# print 'Standardizing the data...'
	# examples = sd.standardizeData(examples, attributes)
	# print 'All nominal and continuous attributes were standardized.'

	# # subtract 1 because we don't want class label to be an input
	# input_units = len(attributes) - 1

	# while True:
	# 	data_option = input("Enter '0' for cross validation, or '1' to run on full sample: " )
	# 	if data_option == 0:
	# 		cross_validate = True
	# 		break
	# 	elif data_option == 1:
	# 		cross_validate = False
	# 		break
	# 	else:
	# 		print 'Not a valid input!'

	# while True:
	# 	hidden_unit_option = input("Please enter a nonnegative integer to set the number of hidden units in the network: ")
	# 	try:
	# 		hidden_units = int(hidden_unit_option)
	# 	except ValueError:
	# 		print("That's not an int!")
	# 	if hidden_units <= 0:
	# 		print 'Not a valid input!'
	# 	else:
	# 		break
	
	# while True:
	# 	weight_decay_option = input("Enter a decimal to represent the weight decay coefficient (gamma): " )
	# 	try:
 #   			weight_decay = int(weight_decay_option)
 #   			break
	# 	except ValueError:
 #   			print("That's not an int!")

	# while True:
	# 	training_iteration_option = input("Please enter a positive integer to set the number of training iterations, or zero/ a nonnegative integer to train until convergence: ")
	# 	try:
	# 		training_iterations = int(training_iteration_option)
 #   			break
	# 	except ValueError:
 #   			print("That's not an int!")
	
	# subtract 1 because we don't want class label to be an input

	orig_array = parse_c45('voting')
	print 'Converting Example array to np array...'
	examples = np.array(orig_array.to_float())
	print 'Conversion done. \n Extracting attributes...'
	attributes = getAttributes(orig_array)
	print'Attributes successfully parsed'

	print 'Standardizing the data...'
	examples = sd.standardizeData(examples, attributes)
	print 'All nominal and continuous attributes were standardized.'

	input_units = len(attributes) - 1

   	network = network_structure(input_units, 20, 0, 5000, False)

	if False:
		partitions = partitionExamples(examples)
		accuracy, precision, recall = runOnFolds(network, partitions)
	else:
		accuracy, precision, recall = runOnFull(network, examples)

	display_results(accuracy, precision, recall, 0)


# Build and test an ANN for each fold
global runOnFolds
def runOnFolds(network, partitions):
	accuracies = np.zeros((5,1))
	precisions = np.zeros((5,1))
	recalls = np.zeros((5,1))

	# we iteratively concatenate folds together, leaving out a different
	# fold each time
	for x in xrange(0,5):
		first_part = True
		test_set = partitions[x]
		for y in xrange(0, x):
			if first_part:
				training_set = partitions[y]
				first_part = False
			else:
				training_set = np.concatenate([training_set, partitions[y]])
		for z in xrange(x + 1, 5):
			if first_part:
				training_set = partitions[z]
				first_part = False
			else:
				training_set = np.concatenate([training_set, partitions[z]])

		#build the tree, get key values, find the average accuracy
		input_weights, hidden_weights = gradient_descent(network, training_set)
		accuracy, precision, recall = get_stats(test_set, input_weights, hidden_weights)
		accuracies[x] = accuracy
		precisions[x] = precision
		recalls[x] = recall

	return accuracies, precisions, recalls

global runOnFull
def runOnFull(network, examples):
	accuracies = np.zeros((1,1))
	precisions = np.zeros((1,1))
	recalls = np.zeros((1,1))

	input_weights, hidden_weights = gradient_descent(network, examples)

	accuracy, precision, recall = get_stats(examples, input_weights, hidden_weights)

	accuracies[0] = accuracy
	precisions[0] = precision
	recalls[0] = recall

	return accuracies, precisions, recalls


global get_stats
def get_stats(examples, input_weights, hidden_weights):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	for ex in np.copy(examples):
		# get the class of the example
		ex_out = ex[-1]
		stim_vec = np.array(ex)
		stim_vec = np.reshape(stim_vec, (len(stim_vec), 1))
		#add a 1 for the bias
		stim_vec[-1] = 1

		z, confidence = feed_forward(stim_vec, input_weights, hidden_weights)
		if confidence > 0.5:
			if ex_out == 1:
				TP += 1
			else:
				FP += 1
		else:
			if ex_out == 0:
				TN += 1
			else:
				FN += 1

		TOT = float(len(examples))
		accuracy = float(TP + TN)/(TOT)
		if TP == 0 and FP == 0:
			print 'no positive classifications, cannot compute precision'
			precision =  0
		else:
			precision = float(TP)/float(TP + FP)
		if TP == 0 and FN == 0:
			print 'could not compute recall'
			recall = 0
		else:
			recall = float(TP)/float(TP + FN)

		return accuracy, precision, recall

global display_results
def display_results(accuracy, precision, recall, AROC):
	avg_acc = np.mean(accuracy, dtype=np.float64)
	avg_stnd_dev = np.var(accuracy, dtype=np.float64)**(-0.5)
	
	prec_acc = np.mean(precision, dtype=np.float64)
	prec_stnd_dev = np.var(precision, dtype=np.float64)**(-0.5)
	
	rec_acc = np.mean(recall, dtype=np.float64)
	rec_stnd_dev = np.var(recall, dtype=np.float64)**(-0.5)

	#Display results
	print('Accuracy: ' + str(avg_acc) + ' ' + str(avg_stnd_dev))
	print('Precision: ' + str(prec_acc) + ' ' + str(prec_stnd_dev))
	print('Recall: ' + str(rec_acc) + ' ' + str(rec_stnd_dev))
	print('Area under ROC: ' + str(0))

# Find all attributes and their types
global getAttributes
def getAttributes(orig_array):
	attributes = {}
	for feature in orig_array.schema:
		attributes[feature.name] = Attribute(feature.name, feature.type, feature.values, len(attributes))
	return attributes
	

# Partition the example set into 5 folds for Cross Validation
# This is a stratified partition, and keeps the ratio of class labes
# in each partition close to the ratio in the example set
global partitionExamples
def partitionExamples(examples):
	# Set the seed and shuffle the np array
	partitions = {}
	np.random.seed(12345)
	np.random.shuffle(examples)

	#Split by class label
	positives = examples[examples[:,-1] == 0]
	pos_size = np.size(positives, 0)
	negatives = examples[examples[:,-1] == 1]
	neg_size = np.size(negatives, 0)
	width = np.size(examples, 1)

	pos_chunk = pos_size / 5
	neg_chunk = neg_size / 5
	for x in xrange(0,4):
		partitions[x] = np.zeros(shape = (pos_chunk + neg_chunk, width))
		for y in xrange(0, pos_chunk):
			partitions[x][y] = positives[x*pos_chunk + y]
		for z in xrange(0, neg_chunk):
			partitions[x][pos_chunk + z] = negatives[x*neg_chunk + z]

	#handle last chunk separately, because division might not have been perfect
	x = 4
	last_pos_chunk = pos_size - 4 * pos_chunk
	last_neg_chunk = neg_size - 4 * neg_chunk
	partitions[4] = np.zeros(shape = (last_pos_chunk + last_neg_chunk, width))
	for y in xrange(0, last_pos_chunk):
		partitions[x][y] = positives[x*pos_chunk + y]
	for z in xrange(0, last_neg_chunk):
		partitions[x][last_pos_chunk + z] = negatives[x*neg_chunk + z]

	return partitions
if __name__ == '__main__':
	main()