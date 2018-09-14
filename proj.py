from __future__ import print_function
import os
import sys
import numpy as np
import cv2

def debug(data, labels):
	print(len(data), len(labels))
	if len(data):
		for i in range(10):
			cv2.imshow("debug", data[i])
			print(labels[i])
			cv2.waitKey(0)

def initW(data_size):
	return np.zeros(data_size)

def initB():
	return 0
	
def load_data(path):
	train = []
	labels = []

	for dirs in sorted(os.listdir(path)):
		for file_name in os.listdir(os.path.join(path, dirs)):
			img_path = os.path.join(path, dirs, file_name)
			train.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
			labels.append(dirs)
	
	train = np.array(train)
	labels = np.array(labels)
	return train, labels

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

def gradient_descent(w0, b0, batch_inputs, batch_labels, learning_rate):
	# batch_size = numero de imagens
	batch_size = batch_inputs.shape[0]

	# num_pixels = numero de pixels de cada imagem
	num_pixels = batch_inputs[0].shape[0]

	print(batch_size, num_pixels, len(w0), len(b0))

	grad_b = 0
	grad_w = np.zeros(batch_size)


	for i in range(batch_size):
		y_i = batch_labels[i]
		# derivada = (2.0 / batch_size) * (w0[0] * x_i0 + ... + w0[M-1] * x_iM-1 + b0 - y_i)
		# derivada = (2.0 / batch_size) * soma
	# 	soma = b0 - y_i
	# 	for j in range(num_pixels):
	# 		x_ij = batch_inputs[i, j]
	# 		pre_calc += w0[j] * x_ij 
	# 	grad_b += (2.0 / batch_size) * parte
	# 	for j in range(num_pixels):
	# 			grad_w[j] += (2.0 / batch_size) * x_ij * parte

	# b1 = b0 - (learning_rate * grad_b)
	# w1 = w0 - (learning_rate * grad_w)

	# return b1, w1

def train(train_data, train_labels, validation_data, validation_labels, num_steps):
	# transforma imagem em vetor de pixels
	train_data = train_data.reshape(np.shape(train_data)[0], -1)
	
	train_size = len(train_data)
	print(np.shape(train_data))
	
	num_pixels = train_data[0].shape[0]

	w0 = initW(num_pixels)
	b0 = initB()

	batch_size = 5



	batch_inputs = np.array(train_data[0:batch_size])
	batch_labels = np.array(train_labels[0:batch_size])
	
	learning_rate = 0.5
	gradient_descent(w0, b0, batch_inputs, batch_labels, learning_rate)

def main():
	need_shuffle = True
	need_split = True
	data, labels = load_data('./data_part1/train')
	if need_shuffle:
		data, labels = shuffle(data, labels)
	if need_split:
		train_percentage = 80
		train_data, train_labels, \
			validation_data, validation_labels = split_dataset(data, labels, train_percentage)

		# debug(train_data, train_labels)
		# debug(validation_data, validation_labels)
	
	else:
		train_data = data
		train_labels = labels
		validation_data = []
		validation_labels = []

	train(train_data, train_labels, validation_data, validation_labels)

if __name__ == "__main__":
	np.random.seed(1)
	main()