from neural_network.neural_network import *
from wikipedia_parser.wikipedia_parser import WikipediaParser

# define possible classifiers
classifiers = ["air","forests","polar","oceans","none"]

# instantiate the wikipedia parser
# which loads 'training_pages' from /data
wikipedia_parser = WikipediaParser()

# one_hot array to represent classifiers
# air | forests | polar | oceans | none
# [ 0 , 0 , 0 , 0 , 0 ]
# converts a one_hot array to the correct classifier
def classify(array):
	for i in range(len(array)):
		if array[i] == 1:
			return classifiers[i]
	return classifiers[-1]

wikipedia_parser.fill_all_data_files()