from __future__ import print_function
import cv2
import numpy as np
import os
import pickle
import sys
import tensorflow as tf

def load_data(path, num_classes):
	data = []
	names = []

	for file_name in sorted(os.listdir(path)):
		img_path = os.path.join(path, file_name)
		data.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
		names.append(file_name)

	data = np.array(data, dtype=np.float)
	data /= 255.0
	return data, names

def reshape_data(data):
	return data.reshape(np.shape(data)[0], -1)

class Model():
	def __init__(self, num_pixels, num_channels, num_classes):
		self.x = tf.placeholder(tf.float32, (None, num_pixels*num_channels))
		self.y = tf.placeholder(tf.float32, (None, 10))
		# learning_rate = tf.placeholder(tf.float32, (1,))
		self.learning_rate = 5e-3

		self.y_ = tf.layers.dense(self.x, num_classes, activation=tf.nn.sigmoid)

		self.loss = tf.reduce_mean((self.y - self.y_) ** 2)

		self.train_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

		self.prediction = tf.cast(tf.argmax(self.y_, 1), tf.float32)
		self.label = tf.cast(tf.argmax(self.y, 1), tf.float32)

		self.ac_batch = tf.reduce_sum(tf.cast(tf.equal(self.prediction, self.label), tf.float32))

def main():
	num_classes = 10

	test_data, names = load_data('../data_part1/test', num_classes)
	test_data = reshape_data(test_data)

	num_pixels = test_data[0].shape[0]
	num_channels = 1
	
	model = Model(num_pixels, num_channels, num_classes)
	
	sess = tf.Session()

	saver = tf.train.Saver()
	saver.restore(sess, tf.train.latest_checkpoint('./'))

	labels = sess.run(model.prediction, feed_dict={model.x: test_data})


	# print(labels)

	for i in range(len(names)):
		print(names[i], int(labels[i]))

if __name__ == '__main__':
	np.random.seed(1)
	main()
