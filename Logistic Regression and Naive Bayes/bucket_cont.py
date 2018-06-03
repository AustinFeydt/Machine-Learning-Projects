import numpy as np
from Attribute import Attribute
np.set_printoptions(threshold='nan')

#responsible for bucketing continuous features
def bucket_cont(buckets, examples, attributes):
	for attr in attributes.values():
		if attr.type == 'CONTINUOUS':
			examples = convert_vals(buckets, examples, attr.index)
	return examples

def convert_vals(buckets, examples, index):
	examples = examples[np.argsort(examples[:, index])]
	# find total gap in values of this column 
	min_val = examples[0, index]
	max_val = examples[np.size(examples,0)-1 , index]
	
	bins = np.linspace(min_val, max_val, buckets)
	digitized_col = np.digitize(examples[:, index], bins)
	examples[:, index] = digitized_col
	return examples