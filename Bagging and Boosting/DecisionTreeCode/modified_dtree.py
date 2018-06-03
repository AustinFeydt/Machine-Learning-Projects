import numpy
from copy import deepcopy
from ID3 import ID3
from AttributeTest import AttributeTest
from mldata import *

def main():
	while True:
		data_path = raw_input("Please enter path to the data: ")
		try:
			orig_array = parse_c45(data_path)
			break
		except Exception as error:
			print 'Not a valid path!'
	
	print 'Converting Example array to numpy array...'
	examples = numpy.array(orig_array.to_float())
	
	print 'Conversion done. \n Extracting attributes...'
	attributes = getAttributes(orig_array)
	print'Attributes successfully parsed'

	while True:
		data_option = input("Enter '0' for cross validation, or '1' to run on full sample:" )
		if data_option == 0:
			cross_validate = True
			break
		elif data_option == 1:
			cross_validate = False
			break
		else:
			print 'Not a valid input!'

	while True:
		depth_option = input("Please enter a nonnegative integer to set the maximum depth of the tree, or enter 0 to grow the full tree:")
		if depth_option < 0:
			print 'Not a valid input!'
		elif depth_option == 0:
			depth = float("inf")
			break
		else:
			depth = depth_option
			break
	
	while True:
		split_option = input("Enter '0' for information gain, or '1' for gain ratio:" )
		if split_option != 0 and split_option != 1:
			print 'Not a valid input!'
		else:
			break

	tree_builder = ID3(depth, split_option)

	if cross_validate:
		partitions = partitionExamples(examples)
		trees = runOnFolds(tree_builder, partitions, attributes)

	else:
		# build tree on the entire data set
		single_tree = tree_builder.buildTree(examples, attributes, {}, 1)
		accuracy = process_tree(single_tree, examples)
		size = tree.subtree_size
		max_depth = tree_builder.max_depth
		first_feature = tree.attribute.name
		print('Accuracy: ' + str(accuracy))
		print('Sizes: ' + str(size))
		print('Maximum Depth: ' + str(max_depth))
		print('First Feature: ' + str(first_feature))


# Find all attributes and their types
global getAttributes
def getAttributes(orig_array):
	attributes = {}
	for feature in orig_array.schema:
		attributes[feature.name] = AttributeTest(feature.name, feature.type, feature.values, len(attributes))
	return attributes
	

# Partition the example set into 5 folds for Cross Validation
# This is a stratified partition, and keeps the ratio of class labes
# in each partition close to the ratio in the example set
global partitionExamples
def partitionExamples(examples):
	# Set the seed and shuffle the numpy array
	partitions = {}
	numpy.random.seed(12345)
	numpy.random.shuffle(examples)

	#Split by class label
	positives = examples[examples[:,-1] == 0]
	pos_size = numpy.size(positives, 0)
	negatives = examples[examples[:,-1] == 1]
	neg_size = numpy.size(negatives, 0)
	width = numpy.size(examples, 1)

	pos_chunk = pos_size / 5
	neg_chunk = neg_size / 5
	for x in xrange(0,4):
		partitions[x] = numpy.zeros(shape = (pos_chunk + neg_chunk, width))
		for y in xrange(0, pos_chunk):
			partitions[x][y] = positives[x*pos_chunk + y]
		for z in xrange(0, neg_chunk):
			partitions[x][pos_chunk + z] = negatives[x*neg_chunk + z]

	#handle last chunk separately, because division might not have been perfect
	x = 4
	last_pos_chunk = pos_size - 4 * pos_chunk
	last_neg_chunk = neg_size - 4 * neg_chunk
	partitions[4] = numpy.zeros(shape = (last_pos_chunk + last_neg_chunk, width))
	for y in xrange(0, last_pos_chunk):
		partitions[x][y] = positives[x*pos_chunk + y]
	for z in xrange(0, last_neg_chunk):
		partitions[x][last_pos_chunk + z] = negatives[x*neg_chunk + z]

	return partitions

# Build and test a tree for each fold
global runOnFolds
def runOnFolds(tree_builder, partitions, attributes):
	accuracies = {}
	sizes = {}
	max_depths = {}
	first_features = {}

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
				training_set = numpy.concatenate([training_set, partitions[y]])
		for z in xrange(x + 1, 5):
			if first_part:
				training_set = partitions[z]
				first_part = False
			else:
				training_set = numpy.concatenate([training_set, partitions[z]])

		#build the tree, get key values, find the average accuracy
		tree = tree_builder.buildTree(training_set, attributes, {}, 1)
		sizes[x] = tree.subtree_size
		max_depths[x] = tree_builder.max_depth
		first_features[x] = tree.attribute.name
		accuracies[x] = process_tree(tree, test_set)
		tree_builder.max_depth = 0

	#Display results
	print('Average Accuracy: ' + str(float(sum(accuracies.values())) / 5.0))
	print('Sizes: ' + str(sizes.values()))
	print('Maximum Depth: ' + str(max_depths.values()))
	print('First Feature: ' + str(first_features.values()))

# Evaluates predicted label for each element in the test set
global process_tree
def process_tree(tree, test_set):
	correct_classify_count = 0
	total_count = float(numpy.size(test_set, 0))
	for test in test_set:
		result = recurse_tree(tree, test)
		if result == test[-1]:
			correct_classify_count += 1
	return float(correct_classify_count) / total_count

global predict
def predict(tree, test):
	return recurse_tree(tree,test)

# This is the recursive method that travels from root to leaf based on 
# decision nodes 
global recurse_tree
def recurse_tree(tree, test):
	index = tree.attribute.index
	if tree.attribute.type == 'LEAF':
		return tree.attribute.values
	else:
		if tree.attribute.type == Feature.Type.CONTINUOUS:
			split = tree.attribute.split
			if (test[index] <= split):
				return recurse_tree(tree.children[0], test)
			else:
				return recurse_tree(tree.children[1], test)
		else:
			return recurse_tree(tree.children[test[index]], test)
if __name__ == '__main__':
	main()