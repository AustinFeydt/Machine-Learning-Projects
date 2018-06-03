import numpy as np
from dtree_booster import dtree_booster
from nbayes_booster import nbayes_booster, nb_classify
from LogisticRegressionCode.modified_logistic_regression import predict as logreg_predict
from logreg_booster import logreg_booster, lr_classify
from mldata import *
from Attribute import Attribute
import DecisionTreeCode.modified_dtree as dtree
import NBayesCode.modified_nbayes as nbayes
from  NBayesCode.modified_nbayes import predict as nbayes_predict

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
		learning_option = raw_input("Enter an option for a learning algorithm (dtree, ann, nbayes,  or logreg): ")
		if learning_option not in {'dtree', 'ann', 'nbayes', 'logreg'}:
			print 'Not a valid input!'
		elif learning_option == 'ann':
			print 'ann not supported, unfortunately'
		else:
			learner = learning_option
			break

	while True:
		iteration_option = input("Enter a nonnegative number for maximum iterations: ")
		if iteration_option <= 0:
			print 'Not a valid input!'
		else:
			iterations = iteration_option
			break

	if cross_validate:
		partitions = partitionExamples(examples)
		accuracy, precision,recall,confidence_table = runOnFolds(partitions, attributes, iterations, learner)
	else:
		accuracy, precision,recall,confidence_table = runOnFull(examples, examples, attributes, iterations, learner)

	aROC = poolROC(confidence_table)
	display_results(accuracy, precision, recall, aROC)

global runOnFolds
def runOnFolds(partitions, attributes, iterations, learner):
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

		accuracy, precision, recall, confidence = runOnFull(training_set, test_set, attributes, iterations, learner)

		accuracies[x] = accuracy[0]
		precisions[x] = precision[0]
		recalls[x] = recall[0]
		confidences[x] = confidence[0]

	return accuracies, precisions, recalls, confidences

global runOnFull
def runOnFull(examples, test_set, attributes, iterations, learner):
	accuracies = np.zeros((1,1))
	precisions = np.zeros((1,1))
	recalls = np.zeros((1,1))
	confidences = {}
	classifiers = {}
	alphas = {}

	if learner == 'dtree':
		classifiers, alphas = dtree_booster(examples, iterations, attributes)
	elif learner == 'logreg':
		classifiers, alphas = logreg_booster(examples, iterations, attributes)
	elif learner == 'nbayes':
		classifiers, alphas = nbayes_booster(examples ,iterations, attributes)
	#ann
	else:
		print ''
	accuracy, precision, recall, confidence = get_stats(test_set, learner, classifiers, alphas, iterations)

	accuracies[0] = accuracy
	precisions[0] = precision
	recalls[0] = recall
	confidences[0] = confidence
	return accuracies, precisions, recalls, confidences

global get_stats
def get_stats(test_set, learner, classifiers, alphas, iterations):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	num_examples = np.size(test_set, 0)
	confidences = np.zeros((num_examples, 2))
	alt_examples = np.copy(test_set)
	alt_examples[:,-1] = 1

	for i in xrange(num_examples):
		# get the class of the example
		ex = test_set[i]
		alt = alt_examples[i]
		ex_out = ex[-1]
		out_pred = 0

		if learner == 'dtree':
			for j in xrange(iterations):
				hj = dtree.predict(classifiers[j], ex)
				if hj == 0:
					hj = -1
				out_pred += alphas[j]*hj
			if out_pred > 0:
				out_pred = 1
				confidences[i][0] = out_pred
				confidences[i][1] = 1
			else:
				out_pred = 0
				confidences[i][0] = out_pred
				confidences[i][1] = 1
		
		elif learner == 'logreg':
			con = 0
			for j in xrange(iterations):
				hj = logreg_predict(alt, classifiers[j])
				con += hj
				hj = np.round(hj)
				if hj == 0:
					hj = -1
				out_pred += alphas[j]*hj
			con = con / float(iterations)
			if out_pred > 0:
				out_pred = 1
				confidences[i][0] = out_pred
				confidences[i][1] = con
			else:
				out_pred = 0
				confidences[i][0] = out_pred
				confidences[i][1] = con

		elif learner == 'nbayes':
			for j in xrange(iterations):
				pos_val, neg_val = nbayes_predict(ex, num_examples, classifiers[j][1], classifiers[j][0], 0)
				if pos_val > neg_val:
					hj = 1
				else:
					hj = -1
				out_pred += alphas[j]*hj
			if out_pred > 0:
				out_pred = 1
				confidences[i][0] = out_pred
				confidences[i][1] = np.exp(pos_val)
			else:
				out_pred = 0
				confidences[i][0] = out_pred
				confidences[i][1] = np.exp(neg_val)
		#ann
		else:
			print ''

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

	TOT = float(len(test_set))
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