import nltk, random, pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import Counter

class Preprocessor:

	def __init__(self):
		self.lemmatizer = WordNetLemmatizer()
		self.number_of_lines = 1000000
		self.data_files = ["data/polar_data.txt",
					  	   "data/forests_data.txt",
					       "data/oceans_data.txt",
					       "data/air_data.txt"]
		self.classifications = [{"polar":[1,0,0,0,0],
								 "forests":[0,1,0,0,0],
								 "oceans":[0,0,1,0,0],
								 "air":[0,0,0,1,0],
								 "none":[0,0,0,0,1]}
								]
		self.lexicon = []

	def create_lexicon(self):
		lexicon = []
		for file in self.data_files:
			with open(file,'r') as f:
				contents = f.readlines()
				for line in contents[:self.number_of_lines]:
					all_words = word_tokenize(line.lower())
					lexicon += list(all_words)
		lexicon = [self.lemmatizer.lemmatize(i) for i in lexicon]
		word_count = Counter(lexicon)
		result = []
		for w in word_count:
			if 1000 > word_count[w] > 5:
				result.append(w)
		self.lexicon = result
		return self.lexicon

	def sample_handling(sample,lexicon,classification):
		featureset = []
		with open(sample,'r') as f:
			contents = f.readlines()
			for line in contents[:self.number_of_lines]:
				current_words = word_tokenize(line.lower())
				current_words = [self.lemmatizer.lemmatize(i) for i in current_words]
				features = np.zeros(len(lexicon))
				for word in current_words:
					if word.lower() in lexicon:
						index_value = lexicon.index(word.lower())
						features[index_value] += 1
				features = list(features)
				featureset.append([features, classification])
		return featureset

	def create_feature_sets_and_labels():
		pass
