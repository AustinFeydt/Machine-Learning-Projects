from AttributeTest import AttributeTest

#This class represents a node in the decision tree
class DecisionNode:
	def __init__(self, attribute, level):
		# defines what attribute the node tests on
		self.attribute = attribute
		# defines all children nodes/leaves
		self.children = {}
		# defines the level of this node in the tree
		self.level = level
		self.subtree_size = 0