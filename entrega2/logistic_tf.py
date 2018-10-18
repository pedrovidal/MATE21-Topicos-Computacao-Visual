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

def train(train_data, train_labels, validation_data, validation_labels, model, num_epochs = 50):
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver(save_relative_paths=True)

	infile = open('./logistic_results/best_ac', 'r')
	best = pickle.load(infile)
	infile.close()

	batch_size = 8
	num_steps = len(train_data) / batch_size

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
			loss_batch, ac_batch, _ = sess.run([model.loss, model.ac_batch, model.train_opt], feed_dict={model.x: batch_input, model.y: batch_labels})
			loss_epoch += loss_batch
			ac_epoch += ac_batch
			loss_validation, ac_validation = sess.run([model.loss, model.ac_batch], feed_dict={model.x: validation_data, model.y:validation_labels})
		
			ac_validation /= len(validation_data)

			best_now = max(best_now, ac_validation)

			if ac_validation > best:
				best = ac_validation
				saver.save(sess, './logistic_results/model_logistic')
				# print('best =', best)
		
		print('ac_validation =', best_now)
		print('ac_treino =', ac_epoch / len(train_data), 'loss_treino =', loss_epoch / num_steps)

	print('best =', best)
	outfile = open('./logistic_results/best_ac', 'w')
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

	# print(num_pixels, num_channels)

	model = Model(num_pixels, num_channels, num_classes)

	train(train_data=train_data, train_labels=train_labels, validation_data=validation_data, validation_labels=validation_labels, model=model)

if __name__ == '__main__':
	np.random.seed(1)
	main()
