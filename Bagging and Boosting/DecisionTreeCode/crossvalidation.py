import random
import shuffle
import numpy

def k_cross_validation(items,X,randomize):

	if randomize:
		values = list(values); 
		shuffle(values)

	else:
		for x in xrange(X):
			for i in x:
    			for a in enumerate()
        			if i % X != x
					train_data = numpy.array(append(a))
					if i % X == x
					validate_data = numpy.array(append(a))
			yield train_data, validate_data

for i in xrange(100):
	values = numpy.array(append(i))
for train_data, validate_data in k_cross_validation(values, X=5):
	for value in values:
		assert (value in train_data) ^ (value in validate_data), value
