import numpy
from copy import deepcopy
from ID3 import ID3
from AttributeTest import AttributeTest
import dtree
from mldata import *

def main():
	print'PARTS A & B'

	orig_spam = parse_c45('spam')
	orig_volcanoes = parse_c45('volcanoes')
	orig_voting = parse_c45('voting')

	print 'making numpy'
	spam_examples = numpy.array(orig_spam.to_float())
	volcanoes_examples = numpy.array(orig_volcanoes.to_float())
	voting_examples = numpy.array(orig_voting.to_float())

	spam_attributes = dtree.getAttributes(orig_spam)
	volcanoes_attributes = dtree.getAttributes(orig_volcanoes)
	voting_attributes = dtree.getAttributes(orig_voting)

	spam_partitions = dtree.partitionExamples(spam_examples)
	volcanoes_partitions = dtree.partitionExamples(volcanoes_examples)
	voting_partitions = dtree.partitionExamples(voting_examples)

	part_a_builder = ID3(1, 0)

	print 'Spam CV Accuracies for Depth = 1'
	spam_trees = dtree.runOnFolds(part_a_builder, spam_partitions, spam_attributes)
	print ''

	print 'Volcanoes CV Accuracies for Depth = 1'
	volcanoes_trees = dtree.runOnFolds(part_a_builder, volcanoes_partitions, volcanoes_attributes)
	print''

	print 'Voting CV Accuracies for Depth = 1'
	voting_trees = dtree.runOnFolds(part_a_builder, voting_partitions, voting_attributes)
	print''


	print'PART C'
	depth_1_builder = ID3(1, 0)
	depth_3_builder = ID3(3, 0)
	depth_5_builder = ID3(5, 0)
	depth_7_builder = ID3(7, 0)
	depth_9_builder = ID3(9, 0)

	print 'Spam Depth 1:'
	dtree.runOnFolds(depth_1_builder, spam_partitions, spam_attributes)


	print 'Spam Depth 3:'
	dtree.runOnFolds(depth_3_builder, spam_partitions, spam_attributes)


	print 'Spam Depth 5:'
	dtree.runOnFolds(depth_5_builder, spam_partitions, spam_attributes)


	print 'Spam Depth 7:'
	dtree.runOnFolds(depth_7_builder, spam_partitions, spam_attributes)


	print 'Spam Depth 9:'
	dtree.runOnFolds(depth_9_builder, spam_partitions, spam_attributes)

	print 'Volcanoes Depth 1:'
	dtree.runOnFolds(depth_1_builder, volcanoes_partitions, volcanoes_attributes)


	print 'Volcanoes Depth 3:'
	dtree.runOnFolds(depth_3_builder, volcanoes_partitions, volcanoes_attributes)


	print 'Volcanoes Depth 5:'
	dtree.runOnFolds(depth_5_builder, volcanoes_partitions, volcanoes_attributes)


	print 'Volcanoes Depth 7:'
	dtree.runOnFolds(depth_7_builder, volcanoes_partitions, volcanoes_attributes)


	print 'Volcanoes Depth 9:'
	dtree.runOnFolds(depth_9_builder, volcanoes_partitions, volcanoes_attributes)


	print 'Part D'
	depth_1_GR_builder = ID3(1,1)
	depth_3_GR_builder = ID3(3,1)
	depth_5_GR_builder = ID3(5,1)

	print 'Spam Depth 1 IG'
	dtree.runOnFolds(depth_1_builder, spam_partitions, spam_attributes)
	print ''
	print 'Spam Depth 3 IG'
	dtree.runOnFolds(depth_3_builder, spam_partitions, spam_attributes)
	print ''
	print 'Spam Depth 5 IG'
	dtree.runOnFolds(depth_5_builder, spam_partitions, spam_attributes)

	print 'Spam Depth 1 GR'
	dtree.runOnFolds(depth_1_GR_builder, spam_partitions, spam_attributes)
	print ''
	print 'Spam Depth 3 GR'
	dtree.runOnFolds(depth_3_GR_builder, spam_partitions, spam_attributes)
	print ''
	print 'Spam Depth 5 GR'
	dtree.runOnFolds(depth_5_GR_builder, spam_partitions, spam_attributes)
	print ''

	print 'Voting Depth 1 IG'
	dtree.runOnFolds(depth_1_builder, voting_partitions, voting_attributes)
	print ''
	print 'Voting Depth 3 IG'
	dtree.runOnFolds(depth_3_builder, voting_partitions, voting_attributes)
	print ''
	print 'Voting Depth 5 IG'
	dtree.runOnFolds(depth_5_builder, voting_partitions, voting_attributes)
	print ''

	print 'Voting Depth 1 GR'
	dtree.runOnFolds(depth_1_GR_builder, voting_partitions, voting_attributes)
	print ''
	print 'Voting Depth 3 GR'
	dtree.runOnFolds(depth_3_GR_builder, voting_partitions, voting_attributes)
	print ''
	print 'Voting Depth 5 GR'
	dtree.runOnFolds(depth_5_GR_builder, voting_partitions, voting_attributes)
	print ''

	print 'Volcanoes Depth 1 IG'
	dtree.runOnFolds(depth_1_builder, volcanoes_partitions, volcanoes_attributes)
	print ''
	print 'Volcanoes Depth 3 IG'
	dtree.runOnFolds(depth_3_builder, volcanoes_partitions, volcanoes_attributes)
	print ''
	print 'Volcanoes Depth 5 IG'
	dtree.runOnFolds(depth_5_builder, volcanoes_partitions, volcanoes_attributes)
	print ''

	print 'Volcanoes Depth 1 GR'
	dtree.runOnFolds(depth_1_GR_builder, volcanoes_partitions, volcanoes_attributes)
	print ''
	print 'Volcanoes Depth 3 GR'
	dtree.runOnFolds(depth_3_GR_builder, volcanoes_partitions, volcanoes_attributes)
	print ''
	print 'Volcanoes Depth 5 GR'
	dtree.runOnFolds(depth_5_GR_builder, volcanoes_partitions, volcanoes_attributes)


	print 'PART E'
	depth_2_builder = ID3(2,0)
	print 'Spam CV Depth 1'
	dtree.runOnFolds(depth_1_builder, spam_partitions, spam_attributes)
	print ''
	
	print 'Spam CV Depth 2'
	dtree.runOnFolds(depth_2_builder, spam_partitions, spam_attributes)
	print ''
	
	print 'Spam Full Depth 1'
	single_tree = depth_1_builder.buildTree(spam_examples, spam_attributes, {}, 1)
	print dtree.process_tree(single_tree, spam_examples)
	print ''
	
	print 'Spam Full Depth 2'
	single_tree = depth_2_builder.buildTree(spam_examples, spam_attributes, {}, 1)
	print dtree.process_tree(single_tree, spam_examples)
	print ''
	

	print 'Voting CV Depth 1'
	dtree.runOnFolds(depth_1_builder, voting_partitions, voting_attributes)
	print ''
	
	print 'Voting CV Depth 2'
	dtree.runOnFolds(depth_2_builder, voting_partitions, voting_attributes)
	print ''
	

	print 'Voting Full Depth 1'
	single_tree = depth_1_builder.buildTree(voting_examples, voting_attributes, {}, 1)
	print dtree.process_tree(single_tree, voting_examples)
	print ''
	print 'Voting Full Depth 2'
	single_tree = depth_2_builder.buildTree(voting_examples, voting_attributes, {}, 1)
	print dtree.process_tree(single_tree, voting_examples)
	print ''
	

	print 'Volcanoes CV Depth 1'
	dtree.runOnFolds(depth_1_builder, volcanoes_partitions, volcanoes_attributes)
	print ''
	
	print 'Volcanoes CV Depth 2'
	dtree.runOnFolds(depth_2_builder, volcanoes_partitions, volcanoes_attributes)
	print ''


	print 'Volcanoes Full Depth 1'
	single_tree = depth_1_builder.buildTree(volcanoes_examples, volcanoes_attributes, {}, 1)
	print dtree.process_tree(single_tree, volcanoes_examples)
	print ''
	print 'Volcanoes Full Depth 2'
	single_tree = depth_2_builder.buildTree(volcanoes_examples, volcanoes_attributes, {}, 1)
	print dtree.process_tree(single_tree, volcanoes_examples)




if __name__ == '__main__':
	main()
