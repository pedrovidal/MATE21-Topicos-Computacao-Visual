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
		for file_name in os.listdir(os.path.join(path, dirs)):
			img_path = os.path.join(path, dirs, file_name)
			data.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
			one_hot_label = np.zeros(num_classes)
			one_hot_label[int(dirs)] = 1
			labels.append(one_hot_label)

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

def createGraph(num_pixels, num_channels, num_classes):
	graph = tf.Graph()
	with graph.as_default():
		x = tf.placeholder(tf.float32, (None, num_pixels*num_channels))
		y = tf.placeholder(tf.float32, (None, 10))
		# learning_rate = tf.placeholder(tf.float32, (1,))
		learning_rate = 5e-4

		y_ = tf.layers.dense(x, num_classes, activation=tf.nn.sigmoid)

		loss = tf.reduce_mean((y - y_) ** 2)

		train_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

		prediction = tf.cast(tf.argmax(y_, 1), tf.float32)

		ac_batch = tf.reduce_sum(tf.cast(tf.equal(prediction, y), tf.float32))
	return graph

def train(graph, train_data, train_labels, num_epochs = 100):
	sess = tf.Session(graph = graph)
	sess.run(tf.global_variables_initializer())

	batch_size = 50
	learning_rate = 5e-4
	num_steps = len(train_data) / batch_size

	for ep in range(num_epochs):
		best_now = 0
		ac_epoch = 0
		loss_epoch = 0
		for i in range(0, len(train_data), batch_size):
			batch_input = np.array(train_data[i : i + batch_size])
			batch_labels = np.array(train_labels[i : i + batch_size])
			loss_batch, ac_batch = sess.run([loss, ac_batch], feed_dict = {x: batch_input, y: batch_labels})
			loss_epoch += loss_batch
			ac_epoch += ac_batch
		print('Epoca', x, 'ac =', ac_epoch, 'loss =', loss_epoch)

def main():
	need_shuffle = True
	need_split = True

	num_classes = 10
	num_nodes = 1000 # numero de nos da camada hidden

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

	print(num_pixels, num_channels)

	graph = createGraph(num_pixels, num_channels, num_classes)

	train(graph = graph, train_data = train_data, train_labels = train_labels)

if __name__ == "__main__":
	np.random.seed(1)
	main()
