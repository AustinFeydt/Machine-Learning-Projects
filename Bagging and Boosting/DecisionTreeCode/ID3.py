import numpy
import math
from copy import deepcopy
from decisionNode import DecisionNode
from mldata import *
from AttributeTest import AttributeTest
# This class holds all functions used to build the ID3 tree
class ID3:

	class SPLIT_CRITERION:
		INFORMATION_GAIN = 'INFORMATION_GAIN'
		GAIN_RATIO = 'GAIN_RATIO'

	def __init__(self, depth_limit, split_criterion):
		self.depth_limit = depth_limit
		if split_criterion == 0:
			self.split_criterion = ID3.SPLIT_CRITERION.INFORMATION_GAIN
		else:
			self.split_criterion = ID3.SPLIT_CRITERION.GAIN_RATIO
		self.max_depth = 0


	# This is the main recursive call!
	# It handles the base cases discussed in the pseudocode and splits
	# accordingly.
	def buildTree(self, examples, attributes, used_attributes, depth):

		if depth > self.depth_limit:
			return createLeaf(majority(examples), depth + 1)

		if depth > self.max_depth:
			self.max_depth = depth

		# empty examples, so just default to a positive
		if numpy.size(examples, 0) == 0:
			return createLeaf(0, depth + 1)

		if isPureNode(examples):
			return createLeaf(examples[0][-1], depth + 1)

		best_attribute = testAttributes(self, examples, attributes, used_attributes)

		if best_attribute == -1:
			return createLeaf(majority(examples), depth + 1)

		node = DecisionNode(best_attribute, depth)
		node.subtree_size = 1

		# We'll handle nominal and continuous differently
		if best_attribute.type == Feature.Type.BINARY or best_attribute.type == Feature.Type.NOMINAL:

			val_index = 0
			# Generate a child node for each discrete value
			while val_index < len(best_attribute.values):
				# the used_attributes dictionary is mutable, so we need to copy!
				new_used_attributes = deepcopy(used_attributes)
				new_used_attributes[best_attribute.name] = best_attribute
				node.children[val_index] = self.buildTree(examples[examples[:,best_attribute.index] == val_index], attributes, new_used_attributes, depth + 1)
				node.subtree_size += node.children[val_index].subtree_size
				
				val_index += 1
		else:

			# the used_attributes dictionary is mutable, so we need to copy!
			leq_used_attributes = deepcopy(used_attributes)
			greater_used_attributes = deepcopy(used_attributes)
			
			node.children[0] = self.buildTree(examples[examples[:, best_attribute.index] <= best_attribute.split], attributes, leq_used_attributes, depth + 1)
			node.subtree_size += node.children[0].subtree_size
			
			node.children[1] = self.buildTree(examples[examples[:, best_attribute.index] > best_attribute.split], attributes, greater_used_attributes, depth + 1)
			node.subtree_size += node.children[1].subtree_size

		return node

	# This tests each potential attribute and finds the best one
	global testAttributes
	def testAttributes(self, examples, attributes, used_attributes):
		max_metric = -1
		best_attribute = -1
		ex_entropy = discreteEntropy(examples, attributes['CLASS'])['hY_X']


		for attribute in attributes.values():
			if attribute.type == Feature.Type.BINARY or attribute.type == Feature.Type.NOMINAL:
				# Make sure attribute hasn't been used on this path
				if attribute.name not in used_attributes.keys():
					this_entropy = discreteEntropy(examples, attribute)
				else:
					continue
			elif attribute.type == Feature.Type.CONTINUOUS:
				this_entropy = continuousEntropy(examples,attribute, used_attributes)
			else:
				continue
			if self.split_criterion == ID3.SPLIT_CRITERION.INFORMATION_GAIN:
				this_metric = calcInformationGain(ex_entropy, this_entropy)
			else:
				this_metric = calcGainRatio(ex_entropy, this_entropy)

			# keep track of maximum metric and it's attribute
			if this_metric > max_metric:
				max_metric = this_metric
				best_attribute = attribute
				if attribute.type == Feature.Type.CONTINUOUS:
					best_attribute.split = this_entropy['best_split']

		return best_attribute


	# This method determines if we now have a pure node
	global isPureNode
	def isPureNode(examples):
		pos_count = len(list(ex for ex in examples if ex[-1] == 0))
		tot = numpy.size(examples, 0)

		if pos_count == tot or pos_count == 0:
			return True
		else:
			return False

	# Returns the majority of the example set
	global majority
	def majority(examples):
		pos_count = len(list(ex for ex in examples if ex[-1] == 0))
		neg_count = len(list(ex for ex in examples if ex[-1] == 1))
		if pos_count >= neg_count:
			return 0
		else:
			return 1

	# Returns a leaf node that is used for testing later
	global createLeaf
	def createLeaf(class_label, depth):
		attr = AttributeTest('LEAF', 'LEAF', class_label, -1)
		return DecisionNode(attr, depth)

	# Basic information gain calculation
	global calcInformationGain
	def calcInformationGain(ex_entropy, this_entropy):
		return ex_entropy - this_entropy['hY_X']

	# Basic gain ratio calculation
	global calcGainRatio
	def calcGainRatio(ex_entropy, this_entropy):
		# if this attribute has an entropy of 0, then this means that 
		# splitting on this attribute doesn't actually partition the example
		# set at all. Thus, we should ignore this attribute
		if this_entropy['h_X'] == 0:
			return 0
		else:
			return float(calcInformationGain(ex_entropy, this_entropy)) / float(this_entropy['h_X'])

	# Calculates the discrete entropy of an attribute
	global discreteEntropy
	def discreteEntropy(examples,attribute):
		# A dictionary of lists to keep track of positives/ negatives in each 
		# bucket
		frequencies = {}
		index = attribute.index
		hY_X = 0
		h_X = 0
		ex_size = float(numpy.size(examples, 0))
		for ex in examples:
			if ex[index] not in frequencies:
				frequencies[ex[index]] = [0,0]
			if ex[-1] == 0:
				frequencies[ex[index]][0] += 1
			else:
				frequencies[ex[index]][1] += 1

		for pair in frequencies.values():
			pos = float(pair[0])
			neg = float(pair[1])
			
			if attribute.type == Feature.Type.CLASS: 
				tot = ex_size
			else:
				tot = pos + neg
			if pos != 0:
				hY_X += -(tot / ex_size) * (pos / tot) * math.log(pos / tot,2)

			if neg != 0:
				hY_X += -(tot / ex_size) * (neg / tot) * math.log(neg / tot,2)

			if tot != 0:
				h_X += -(tot / ex_size) * math.log(tot / ex_size, 2)

		return {'hY_X':hY_X, 'h_X':h_X}

	# This function does all processing for a continuous attribute.
	#
	# It loops through the entire set of examples, which are ordered by the attr.
	# Any time it finds a split, it calculate the entropy on the spot, thus it
	# keeps a running total of number of positive and negative class labels on 
	# either side of the split. This was more optimal runtime-wise than finding all splits ahead of time and calculating all entropies after.
	#
	# This function returns the minimum entropy H(Y|X), as well as the split that 
	# contributed to this entropy and H(X)
	global continuousEntropy
	def continuousEntropy(examples, attribute, used_attributes):
		if attribute.name in used_attributes.keys():
			used_splits = used_attributes[attribute.name]
		else:
			used_splits = []
		hY_X = float("inf")
		h_X = 0
		best_split = 0

		index = attribute.index
		
		ex_size = numpy.size(examples, 0)
		tot_pos = len(list(ex for ex in examples if ex[-1] == 0))
		tot_neg = ex_size - tot_pos

		leq_buckets = [0,0]
		greater_buckets = [tot_pos,tot_neg]
		
		# sort based on attribute value
		examples = examples[numpy.argsort(examples[:, index])]

		prev_start = 0
		prev_end = 1
		curr_start = 0

		while curr_start < ex_size and prev_end < ex_size:
			prev_end_Class = int(examples[prev_end][-1])
			prev_end_Val = float(examples[prev_end][index])

			prev_start_Class = int(examples[prev_start][-1])
			prev_start_Val = float(examples[prev_start][index])
			
			# if there is a split to be found this iteration, it will definitely
			# be greater than prev_start, so adjust buckets accordingly
			leq_buckets[prev_start_Class] += 1
			greater_buckets[prev_start_Class] -= 1

			# increment prev_end til it's value differs from prev_start. Also
			# adjust buckets, because prev_end_val is the same as prev_start_val
			while prev_start_Val == prev_end_Val:
				leq_buckets[prev_end_Class] += 1
				greater_buckets[prev_end_Class] -= 1
				prev_end += 1
				if prev_end >= ex_size:
					break
				prev_end_Class = int(examples[prev_end][-1])
				prev_end_Val = float(examples[prev_end][index])

			if prev_end >= ex_size:
				break

			curr_start = prev_end
			prev_end -= 1

			curr_start_Class = int(examples[curr_start][-1])
			curr_start_Val = float(examples[curr_start][index])
			
			curr_end = curr_start
			curr_end_Val = float(examples[curr_end][index])

			# increment curr_end til it's value differs from curr_start
			while curr_start_Val == curr_end_Val:
				curr_end += 1
				if curr_end >= ex_size:
					break
				curr_end_Val = float(examples[curr_end][index])

			nxt_start = curr_end
			curr_end -= 1

			# check every combination of vectors in prev interval and curr interval
			split_found = False
			while prev_start <= prev_end and not split_found:
				prev_class = examples[prev_start][-1]
				
				while curr_start <= curr_end and not split_found:
					curr_class = examples[curr_start][-1]
					if prev_class != curr_class:
						# we found a split!
						split_found = True
						split_val = (prev_start_Val + curr_start_Val) / 2.0
						if split_val not in used_splits:
							entropy_pair = calculate_Split_Entropy(split_val, leq_buckets, greater_buckets)
							if entropy_pair['hY_X'] < hY_X:
								hY_X = entropy_pair['hY_X']
								h_X = entropy_pair['h_X']
								best_split = split_val
					curr_start += 1
				prev_start += 1

			# adjust indices and iterate
			prev_start = curr_start
			prev_end = prev_start + 1
			curr_start = nxt_start
			curr_end = curr_start + 1

		return {'hY_X':hY_X, 'best_split':best_split ,'h_X':h_X }

	# Calculates the entropy of the candidate split above
	# similar to discrete calculation above
	global calculate_Split_Entropy
	def calculate_Split_Entropy(split_val, leq_buckets, greater_buckets):
		leq_size = float(sum(leq_buckets))
		greater_size = float(sum(greater_buckets))
		tot = leq_size + greater_size
		hY_X = 0
		h_X = 0 

		pos = float(leq_buckets[0])
		if pos != 0:
			hY_X += -(leq_size / tot) * (pos / leq_size) * math.log(pos / leq_size,2)
		
		neg = float(leq_buckets[1])
		if neg != 0:
			hY_X += -(leq_size / tot) * (neg / leq_size) * math.log(neg / leq_size,2)

		pos = float(greater_buckets[0])
		if pos != 0:
			hY_X += -(greater_size / tot) * (pos / greater_size) * math.log(pos / greater_size,2)
		
		neg = float(greater_buckets[1])
		if neg != 0:
			hY_X += -(greater_size / tot) * (neg / greater_size) * math.log(neg / greater_size,2)

		h_X += -(leq_size / tot) * math.log(leq_size / tot,2)
		h_X += -(greater_size / tot) * math.log(greater_size / tot,2)

		return {'hY_X':hY_X, 'h_X':h_X}