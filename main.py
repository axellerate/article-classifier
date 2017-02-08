from neural_network.preprocessor import Preprocessor
from wikipedia_parser.wikipedia_parser import WikipediaParser
import numpy as np
import tensorflow as tf

'''
input > weight > hidden layer 1 (activation function)
> weights > hidden layer 2 (activation function)
> weights > output layer

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer, gradient descent, etc...)

backpropogation

feed forward + backpropogation = 1 epoch (one cycle of the network) 
'''

preprocessor = Preprocessor()
training_word_vectors,training_labels,testing_word_vectors,testing_labels = preprocessor.create_feature_sets_and_labels()

batch_size = 100

n_nodes_hl1 = 100

n_classes = 5 # Example: [1,0,0,0,0] == 'polar'

test = training_word_vectors[50]
answer = training_labels[50]

x = tf.placeholder('float', [None,len(training_word_vectors[0])])
y = tf.placeholder('float')

def neural_network_model(data):

	# (input_data * weights) + biases

	# biases make sure that inputs of zero still
	# produce a non-zero output

	hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([len(training_word_vectors[0]), n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
					'biases':tf.Variable(tf.random_normal([n_classes]))}

	# (input_data * weights) + biases

	layer_1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
	layer_1 = tf.nn.relu(layer_1)

	output = tf.matmul(layer_1, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):

	prediction = neural_network_model(x)

	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y) )

	optimizer = tf.train.AdamOptimizer().minimize(cost)

	num_of_epochs = 40

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_of_epochs):
			epoch_loss = 0

			i = 0
			while  i < len(training_word_vectors):
				start = i
				end = i + batch_size

				batch_x = np.array(training_word_vectors[start:end])
				batch_y = np.array(training_labels[start:end])

				_,c = sess.run([optimizer,cost], feed_dict={x:batch_x, y:batch_y})
				epoch_loss += c
				i += batch_size
			print('Epoch', epoch+1, 'completed out of', num_of_epochs, 'loss', epoch_loss)
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))

		print('Accuracy',accuracy.eval({x:testing_word_vectors, y:testing_labels}))

train_neural_network(x)