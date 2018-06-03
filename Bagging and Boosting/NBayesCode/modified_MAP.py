import numpy as np
def MAP(examples, m_est, w):
	# build a dictionary, mapping index to probabilities of each possible
	# value
	pos_probs, neg_probs = generate_probabilities(examples, m_est, w)
	return pos_probs, neg_probs

# This is modified for adaboost. rather than just counting frequencies,
# we sum up the weights
def generate_probabilities(examples, m_est, w):
	neg_probs = {}
	pos_probs = {}
	ex_size = float(np.size(examples,1))
	num_exs = float(np.size(examples,0))
	m = m_est
	p = 1 / ex_size
	if m < 0:
		m = float(float(1)/float(p))
	# loop through each attribute
	# the last value of i just counts number of positive and negative labels
	for i in xrange(int(ex_size)):
		neg_probs[i] = {}
		pos_probs[i] = {}
		
		# loop through each example, and increment according probability
		for j in xrange(int(num_exs)):
			ex = examples[j]
			val = ex[i]
			klass = ex[-1]
			if klass == 0:
				if val not in neg_probs[i].keys():
					neg_probs[i][val] = w[j] + m*p
				else:
					neg_probs[i][val] += w[j]
			else:
				if val not in pos_probs[i].keys():
					pos_probs[i][val] = w[j] + m*p
				else:
					pos_probs[i][val] += w[j]

	# fix the counts for our m-est formula (we want +m, not +mp for
	# denominator)
	neg_probs[ex_size -1][0] += -m*p + m
	pos_probs[ex_size -1][1] += -m*p + m

	return pos_probs, neg_probs