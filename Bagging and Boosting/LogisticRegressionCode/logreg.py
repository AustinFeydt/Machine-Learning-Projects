import numpy as np
from mldata import *
from Attribute import Attribute
from logistic_regression import logistic_regression
from logistic_regression import predict

def main():
	while True:
		data_path = raw_input("Please enter path to the data: ")
		try:
			orig_array = parse_c45(data_path)
			break
		except Exception as error:
			print 'Not a valid path!'

	print 'Converting Example array to np array...'
	examples = np.array(orig_array.to_float())
	
	print 'Conversion done. \n Extracting attributes...'
	attributes = getAttributes(orig_array)
	print'Attributes successfully parsed'

	while True:
		data_option = input("Enter '0' for cross validation, or '1' to run on full sample: " )
		if data_option == 0:
			cross_validate = True
			break
		elif data_option == 1:
			cross_validate = False
			break
		else:
			print 'Not a valid input!'

	while True:
		lambda_option = input("Enter a nonnegative real number to set the value of lambda: ")
		if lambda_option < 0:
			print 'Not a valid input!'
		else:
			lda = lambda_option
			break

	if cross_validate:
		partitions = partitionExamples(examples)
		accuracy, precision,recall,confidence_table = runOnFolds(partitions, lda)
	else:
		accuracy, precision,recall,confidence_table = runOnFull(examples, lda)

	aROC = poolROC(confidence_table)
	display_results(accuracy, precision, recall, aROC)

global runOnFolds
def runOnFolds(partitions, lda):
	accuracies = np.zeros((5,1))
	precisions = np.zeros((5,1))
	recalls = np.zeros((5,1))
	confidences = {}
	
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

		print 'running on Fold ' + str(x + 1)
		alt_training_set = np.copy(training_set)
		alt_test_set = np.copy(test_set)
		# we'll set the last value to be 1 to incorporate bias into weight 
		# vector
		alt_training_set[:,-1] = 1
		alt_test_set[:, -1] = 1
		
		w = logistic_regression(training_set, alt_training_set, lda)
		
		accuracy, precision, recall, confidence = get_stats(test_set, alt_test_set, w)

		accuracies[x] = accuracy
		precisions[x] = precision
		recalls[x] = recall
		confidences[x] = confidence

	return accuracies, precisions, recalls, confidences

global runOnFull
def runOnFull(examples, lda):
	accuracies = np.zeros((1,1))
	precisions = np.zeros((1,1))
	recalls = np.zeros((1,1))
	confidences = {}

	alt_examples = np.copy(examples)
	#we'll set the last value to be 1 to incorporate bias into weight vector
	alt_examples[:,-1] = 1
	w = logistic_regression(examples, alt_examples, lda)

	accuracy, precision, recall, confidence = get_stats(examples, alt_examples, w)

	accuracies[0] = accuracy
	precisions[0] = precision
	recalls[0] = recall
	confidences[0] = confidence

	return accuracies, precisions, recalls, confidences

global get_stats
def get_stats(examples, alt_examples, w):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	confidences = np.zeros((np.size(examples,0), 2))

	for i in xrange(np.size(examples,0)):
		# get the class of the example
		ex_out = examples[i][-1]
		
		# get predicted output
		confidence = predict(alt_examples[i],w)
		out_pred = np.round(confidence)
		confidences[i][0] = ex_out
		confidences[i][1] = confidence

		if out_pred == 1:
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
	return accuracy, precision, recall, confidences


global poolROC
def poolROC(confidence_table):
	table = np.zeros((1,2))
	for i in xrange(len(confidence_table)):
		table = np.concatenate((table, confidence_table[i]), axis=0)

	# easy way to sort in descending order
	table = -1 * table
	table = table[np.argsort(table[:, 1])]
	table = -1 * table

	unique, counts = np.unique(table[:,0], return_counts = True)
	u = dict(zip(unique, counts))

	TP = float(0)
	FP = float(0)
	TN = float(u[0])
	FN = float(u[1])

	prev_TPR = 0
	prev_FPR = 0
	aROC = 0
	for i in xrange(np.size(table,0)):
		if table[i][0] == 1:
			TP += 1
			FN -= 1
		else:
			FP += 1
			TN -= 1
		
		TPR = float(TP / (TP + FN))
		FPR = float(FP/(FP + TN))

		aROC += 0.5* (FPR - prev_FPR) * (TPR + prev_TPR)
		prev_TPR = TPR
		prev_FPR = FPR
	
	return aROC

global display_results
def display_results(accuracy, precision, recall, aROC):
	avg_acc = np.mean(accuracy, dtype=np.float64)
	prec_acc = np.mean(precision, dtype=np.float64)
	rec_acc = np.mean(recall, dtype=np.float64)
	
	if np.size(accuracy) == 1:
		avg_stnd_dev = 0
		prec_stnd_dev = 0
		rec_stnd_dev = 0
	else:
		avg_stnd_dev = np.var(accuracy, dtype=np.float64)**(-0.5)
		prec_stnd_dev = np.var(precision, dtype=np.float64)**(-0.5)
		rec_stnd_dev = np.var(recall, dtype=np.float64)**(-0.5)

	#Display results
	print('\nResults:')
	print('Accuracy: ' + str(avg_acc) + ' ' + str(avg_stnd_dev))
	print('Precision: ' + str(prec_acc) + ' ' + str(prec_stnd_dev))
	print('Recall: ' + str(rec_acc) + ' ' + str(rec_stnd_dev))
	print('Area under ROC: ' + str(aROC))

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