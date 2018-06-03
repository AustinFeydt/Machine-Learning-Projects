from Attribute import Attribute
#adds 1 to all nominal values to fit requirements
def fix_nominals(examples, attributes):
	for attribute in attributes.values():
		if attribute.type == 'NOMINAL':
			index = attribute.index
			for ex in examples:
				ex[index] += 1