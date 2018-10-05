from __future__ import print_function
import cv2
import numpy as np
import os
import pickle
import sys
import tensorflow as tf

def debug(data, labels):
	print(len(data), len(labels))
	if len(data):
		for i in range(10):
			# data *= 255
			cv2.imshow("debug", data[i])
			print(labels[i])
			cv2.waitKey(0)

def load_data(path, num_classes):
	data = []
	labels = []

	for dirs in sorted(os.listdir(path)):
		for file_name in os.listdir(os.path.join(path, dirs)):
			img_path = os.path.join(path, dirs, file_name)
			data.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
			one_hot = np.zeros(num_classes)
			one_hot[int(dirs)] = 1
			labels.append(one_hot)

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

def train(train_data, train_labels, validation_data, validation_labels, num_classes, num_nodes):
	# transforma imagem em vetor de pixels
	train_data = train_data.reshape(np.shape(train_data)[0], -1)
	validation_data = validation_data.reshape(np.shape(validation_data)[0], -1)

	train_size = len(train_data)
	validation_size = len(validation_data)
	# print(np.shape(train_data))

	num_pixels = train_data[0].shape[0]

	W0 = np.empty((num_nodes, num_pixels))
	b0 = np.empty(num_nodes)

	for i in range(num_nodes):
		W0[i] = initW(num_pixels)
		b0[i] = initB()

	W1 = np.empty((num_classes, num_nodes))
	b1 = np.empty(num_classes)

	for i in range(num_classes):
		W1[i] = initW(num_nodes)
		b1[i] = initB()

	batch_size = 50
	num_steps = train_size / batch_size
	learning_rate = 0.1

	infile = open('./mlp_matmul/ac', 'r')
	best = pickle.load(infile)
	infile.close()

	print("best =", best)

	for x in range(500):
		# if x % 100 == 0 and x > 0:
		# 	learning_rate = learning_rate - 1e-1
		print("Epoca", x)
		ini = 0
		fim = batch_size - 1
		best_now = 0

		# train_data, train_labels = shuffle(train_data, train_labels)

		for i in range(num_steps):
			# print("Step", i)
			batch_inputs = np.array(train_data[ini:fim])
			batch_labels = np.array(train_labels[ini:fim])
			
			W0, b0, W1, b1, loss = gradient_descent(W0, b0, W1, b1, batch_inputs, batch_labels, learning_rate)

			ini += batch_size
			fim += batch_size

			ac = validation(W0, b0, W1, b1, validation_data, validation_labels)
			# if i % 50 == 0:
			# print(i, "ac = ", ac)
			if ac > best:
				print(i, "ac = ", ac)
				best = ac
				outfile = open('./mlp_matmul/ac', 'w')
				pickle.dump(ac, outfile)
				outfile.close()

				best_W0 = W0
				outfile = open('./mlp_matmul/weights0', 'w')
				pickle.dump(W0, outfile)
				outfile.close()

				best_b0 = b0
				outfile = open('./mlp_matmul/bias0', 'w')
				pickle.dump(b0, outfile)
				outfile.close()

				best_W1 = W1
				outfile = open('./mlp_matmul/weights1', 'w')
				pickle.dump(W1, outfile)
				outfile.close()

				best_b1 = b1
				outfile = open('./mlp_matmul/bias1', 'w')
				pickle.dump(b1, outfile)
				outfile.close()

				outfile = open('./mlp_matmul/learning_rate', 'w')
				pickle.dump(learning_rate, outfile)
				outfile.close()

				outfile = open('./mlp_matmul/batch_size', 'w')
				pickle.dump(batch_size, outfile)
				outfile.close()

				outfile = open('./mlp_matmul/num_nodes', 'w')
				pickle.dump(num_nodes, outfile)
				outfile.close()

			best_now = max(ac, best_now)
			
		# print("best = ", best)
		print("best now = ", best_now)
		# print("lr = ", learning_rate)

def createGraph(input_size, num_classes):
	x = tf.placeholder(tf.float32, (None, input_size))
	y = tf.placeholder(tf.float32, (None,))
	learning_rate = tf.placeholder(tf.float32, (1,))

	y_ = tf.layers.dense(x, num_classes, activation=tf.nn.sigmoid)

	loss = tf.reduce_mean((y - y_) ** 2)

	ac = 


def main():
	need_shuffle = True
	need_split = True

	num_classes = 10
	num_nodes = 1000 # numero de nos da camada hidden

	data, labels = load_data('./data_part1/train', num_classes)
	
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

	graph = crateGraph()

	train(train_data, train_labels, validation_data, validation_labels, num_classes, num_nodes)

if __name__ == "__main__":
	np.random.seed(1)
	main()
