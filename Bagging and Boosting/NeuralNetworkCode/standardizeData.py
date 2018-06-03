import numpy as np
from Attribute import Attribute

def standardizeData(examples, attributes):
	for attribute in attributes.values():
		if attribute.type == 'NOMINAL':
		 	handleNominal(examples, attribute)
		if attribute.type == 'CONTINUOUS':
			handleContinuous(examples, attribute)

	return examples
# Converts nominal attributes to fixed integers ( 1 of N approach)
# mlData already converts nominal values to a 0 of N-1 approach, so
# we just add 1 to each value
def handleNominal(examples, attribute):
	nom_Mapping = {}
	index = attribute.index	
	for ex in examples:
		ex[index] += 1

# standardizes continuous attribute by centering it around the mean
# and normalizing by the variance
def handleContinuous(examples, attribute):
	index = attribute.index
	col = examples[:,index]
	mean = np.mean(col, dtype=np.float64)
	std_dev = (np.var(examples[:,index], dtype=np.float64))**(0.5)
	
	for ex in examples:
		ex[index] = (ex[index] - mean) / std_dev
