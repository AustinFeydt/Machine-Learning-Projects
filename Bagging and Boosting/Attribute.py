# This class represents a test on an attribute
class Attribute:
	def __init__(self, name, attType, values, index):
		self.name = name
		self.type = attType
		self.values = values
		self.index = index
		self.split = 0