from __future__ import print_function
import cv2
import numpy as np
import os
import pickle
import sys
import tensorflow as tf

def load_data(path, num_classes):
	data = []
	labels = []

	for dirs in sorted(os.listdir(path)):
		for file_name in sorted(os.listdir(os.path.join(path, dirs))):
			img_path = os.path.join(path, dirs, file_name)
			data.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
			# one_hot_label = np.zeros(num_classes)
			# one_hot_label[int(dirs)] = 1
			labels.append(int(dirs))

	data = np.array(data, dtype=np.float)
	data /= 255.0
	labels = np.array(labels)
	return data, labels

def shuffle(data, labels):
	data_size = len(data)
	permutation_index = np.random.permutation(data_size)
	shuffled_data = data[permutation_index]
	shuffled_labels = labels[permutation_index]
	return shuffled_data, shuffled_labels

def split_dataset(data, labels, train_percentage):
	data_size = len(data)

	train_size = int(data_size * train_percentage / 100)
	
	train_data = data[0:train_size]
	train_labels = labels[0:train_size]

	validation_data = data[train_size:data_size]
	validation_labels = labels[train_size:data_size]

	return train_data, train_labels, validation_data, validation_labels

def reshape_data(data):
	return data.reshape(np.shape(data)[0], -1)

class Model():
	def __init__(self, num_pixels, num_channels, num_classes, num_nodes):
		self.x = tf.placeholder(tf.float32, (None, num_pixels*num_channels))
		self.y = tf.placeholder(tf.int32, (None,))
		self.learning_rate = tf.placeholder(tf.float32)
		self.dropout_rate = tf.placeholder(tf.float32)
		self.is_train = tf.placeholder(tf.bool)

		fc = tf.layers.dense(self.x, num_nodes, activation=tf.nn.relu)

		dropout = tf.layers.dropout(fc, rate=self.dropout_rate, training=self.is_train);
		
		self.y_ = tf.layers.dense(dropout, num_classes, activation=None)

		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_, labels=self.y))

		self.train_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

		self.prediction = tf.cast(tf.argmax(self.y_, 1), tf.float32)
		self.label = tf.cast(self.y, tf.float32)

		self.ac_batch = tf.reduce_sum(tf.cast(tf.equal(self.prediction, self.label), tf.float32))

def train(train_data, train_labels, validation_data, validation_labels, model, num_epochs = 50):
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	batch_size = 8
	learning_rate = 5e-4
	dropout_rate = 0.0
	num_steps = len(train_data) / batch_size

	saver = tf.train.Saver(save_relative_paths=True)
	
	infile = open('./mlp_results/best_ac', 'r')
	best = pickle.load(infile)
	infile.close()


	for ep in range(num_epochs):
		best_now = 0
		ac_epoch = 0
		loss_epoch = 0
		train_data, train_labels = shuffle(train_data, train_labels)
		print('Epoca', ep)
		best_now = 0 
		for i in range(0, len(train_data), batch_size):
			batch_input = np.array(train_data[i : i + batch_size])
			batch_labels = np.array(train_labels[i : i + batch_size])

			feed_dict_train = {model.x: batch_input, model.y: batch_labels, model.learning_rate: learning_rate, model.dropout_rate: dropout_rate, model.is_train: True}
			loss_batch, ac_batch, _ = sess.run([model.loss, model.ac_batch, model.train_opt], feed_dict=feed_dict_train)
			loss_epoch += loss_batch
			ac_epoch += ac_batch

			feed_dict_validation = {model.x: validation_data, model.y: validation_labels, model.dropout_rate: 0.0, model.is_train: False}
			loss_validation, ac_validation = sess.run([model.loss, model.ac_batch], feed_dict=feed_dict_validation)
			ac_validation /= len(validation_data)

			best_now = max(best_now, ac_validation)
			if ac_validation > best:
				best = ac_validation
				saver.save(sess, './mlp_results/model_mlp')
				# print('best =', best)
		
		print('ac_validation =', best_now)
		print('ac_treino =', ac_epoch / len(train_data), 'loss_treino =', loss_epoch / num_steps)
	print('best =', best)
	outfile = open('./mlp_results/best_ac', 'w')
	pickle.dump(best, outfile)
	outfile.close()

def main():
	need_shuffle = True
	need_split = True

	num_classes = 10

	data, labels = load_data('../data_part1/train', num_classes)
	
	if need_shuffle:
		data, labels = shuffle(data, labels)

	if need_split:
		train_percentage = 80
		train_data, train_labels, \
			validation_data, validation_labels = split_dataset(data, labels, train_percentage)

	else:
		train_data = data
		train_labels = labels
		validation_data = np.array()
		validation_labels = np.array()

	train_data = reshape_data(train_data)
	validation_data = reshape_data(validation_data)

	num_pixels = train_data[0].shape[0]
	num_channels = 1

	num_nodes = 1024

	# print(num_pixels, num_channels)

	model = Model(num_pixels, num_channels, num_classes, num_nodes)

	train(train_data=train_data, train_labels=train_labels, validation_data=validation_data, validation_labels=validation_labels, model=model)

if __name__ == "__main__":
	np.random.seed(1)
	main()
