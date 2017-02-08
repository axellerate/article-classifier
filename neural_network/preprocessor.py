import nltk, random, pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import Counter

class Preprocessor:

	def __init__(self):
		self.lemmatizer = WordNetLemmatizer()
		self.number_of_lines = 1000000
		self.data_files = {"polar":"data/polar_data.txt",
					  	   "forests":"data/forests_data.txt",
					       "oceans":"data/oceans_data.txt",
					       "air":"data/air_data.txt",
					       "none":"data/random_text_data.txt"}
		self.classifications = {"polar":[1,0,0,0,0],
								 "forests":[0,1,0,0,0],
								 "oceans":[0,0,1,0,0],
								 "air":[0,0,0,1,0],
								 "none":[0,0,0,0,1]}
		self.lexicon = []

	def create_lexicon(self):
		lexicon = []
		for f_type,file in self.data_files.items():
			with open(file,'r') as f:
				contents = f.readlines()
				for line in contents[:self.number_of_lines]:
					all_words = word_tokenize(line.lower())
					lexicon += list(all_words)
		lexicon = [self.lemmatizer.lemmatize(i) for i in lexicon]
		stop_words = set(stopwords.words('english'))
		lexicon = [word for word in lexicon if word not in stop_words]
		lexicon = [word for word in lexicon if word.isalpha()]
		word_count = Counter(lexicon)
		result = []
		for w in word_count:
			if 750 > word_count[w] > 10:
				result.append(w)
		self.lexicon = result
		print(result)
		print("Lexicon of size {} successfully created".format(len(self.lexicon)))
		return self.lexicon

	def sample_handling(self,sample,lexicon,classification):
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
		print("Featureset generated: {}".format(classification))
		return featureset

	def create_feature_sets_and_labels(self,test_size=0.1):
		lexicon = self.create_lexicon()
		features = []
		features += self.sample_handling(self.data_files['polar'],
									lexicon,
									self.classifications['polar'])
		features += self.sample_handling(self.data_files['forests'],
									lexicon,
									self.classifications['forests'])
		features += self.sample_handling(self.data_files['oceans'],
									lexicon,
									self.classifications['oceans'])
		features += self.sample_handling(self.data_files['air'],
									lexicon,
									self.classifications['air'])
		features += self.sample_handling(self.data_files['none'],
									lexicon,
									self.classifications['none'])
		random.shuffle(features)

		features = np.array(features)
		testing_size = int(test_size*len(features))

		print("Testing size: {}".format(testing_size))

		training_word_vectors = list(features[:,0][:-testing_size])
		train_labels = list(features[:,1][:-testing_size])

		testing_word_vectors = list(features[:,0][-testing_size:])
		testing_labels = list(features[:,1][-testing_size:])

		print("Training data length: ",len(training_word_vectors))
		print("Testing data length: ",len(testing_word_vectors))
		return training_word_vectors,train_labels,testing_word_vectors,testing_labels






