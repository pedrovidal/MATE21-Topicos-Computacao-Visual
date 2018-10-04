from __future__ import print_function
import os
import sys
import numpy as np
import cv2
import pickle

def debug(data, labels):
	print(len(data), len(labels))
	if len(data):
		for i in range(10):
			data *= 255
			cv2.imshow("debug", data[i])
			print(labels[i])
			cv2.waitKey(0)

def initW(data_size):
	return np.random.uniform(-0.01, 0.01, data_size)

def initB():
	return np.random.random()
	
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

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))
	if x >= 0:
		return 1.0 / (1.0 + np.exp(-x))
	else:
		exp = np.exp(x)
		return exp / (1.0 + exp)

def gradient_descent(W, b, batch_inputs, batch_labels, learning_rate):
	# batch_size = numero de imagens
	batch_size = batch_inputs.shape[0]

	# num_pixels = numero de pixels de cada imagem
	num_pixels = batch_inputs[0].shape[0]

	num_classes = W.shape[0]

	# print(batch_size, num_pixels, num_classes, len(W), len(b))

	grad_W = np.zeros((num_classes, num_pixels))
	grad_b = np.zeros(num_classes)

	loss = 0
	y_ = np.empty(num_classes)
	for x, y in zip(batch_inputs, batch_labels):
		for i in range(num_classes):
			z = np.dot(W[i], x) + b[i]		
			y_[i] = sigmoid(z)

			# print(y_[i])

			# print(- learning_rate * (y_[i] - y[i]) * y_[i] * (1.0 - y_[i]), x)

			grad_W[i] += (y_[i] - y[i]) * y_[i] * (1.0 - y_[i]) * x
			grad_b[i] += (y_[i] - y[i]) * y_[i] * (1.0 - y_[i])
		
		loss += 1.0 / 2.0 * (np.sum((y_ - y)**2))
		
		# print(y_)

		prediction = np.argmax(y_)
		label = np.argmax(y)

		# print(prediction, label)


	# print(y_)

	loss /= batch_size
	# print("loss =", loss)

	for i in range(num_classes):
		grad_b[i] /= batch_size
		grad_W[i] /= batch_size


		b[i] = b[i] - (learning_rate * grad_b[i])
		W[i] = W[i] - (learning_rate * grad_W[i])

	# print(W)

	return b, W, loss

def validation(W, b, validation_data, validation_labels):
	ac = 0.0
	validation_size = len(validation_data)
	for i in range(validation_size):
		prediction = np.empty(10)
		for j in range(10):
			# print(W[j], b[j])
			prediction[j] = sigmoid(np.dot(W[j], validation_data[i]) + b[j])
		label_prediction = np.argmax(prediction)
		
		# print("Validation ", label_prediction, np.argmax(validation_labels[i]))

		if label_prediction == np.argmax(validation_labels[i]):
			ac += 1
	return ac / validation_size

def train(train_data, train_labels, validation_data, validation_labels, num_classes):
	# transforma imagem em vetor de pixels
	train_data = train_data.reshape(np.shape(train_data)[0], -1)
	validation_data = validation_data.reshape(np.shape(validation_data)[0], -1)
	
	train_size = len(train_data)
	validation_size = len(validation_data)
	# print(np.shape(train_data))
	
	num_pixels = train_data[0].shape[0]

	W = np.empty((num_classes, num_pixels))
	b = np.empty(num_classes)

	for i in range(num_classes):
		W[i] = initW(num_pixels)
		b[i] = initB()

	batch_size = 40
	num_steps = train_size / batch_size
	learning_rate = 0.5

	infile = open('ac', 'r')
	best = pickle.load(infile)
	infile.close()

	print("best =", best)
	
	infile = open('weights', 'r')
	best_W = pickle.load(infile)
	infile.close()

	infile = open('bias', 'r')
	best_b = pickle.load(infile)
	infile.close()

	# print("Validation =", validation(best_W, best_b, validation_data, validation_labels))

	for x in range(100):
		# if x % 100 == 0 and x > 0:
		# 	learning_rate = learning_rate - 1e-1
		print("Epoca", x)
		ini = 0
		fim = batch_size - 1
		best_now = 0
		for i in range(num_steps):
			# print("Step", i)
			batch_inputs = np.array(train_data[ini:fim])
			batch_labels = np.array(train_labels[ini:fim])
			
			batch_inputs, batch_labels = shuffle(batch_inputs, batch_labels)

			b, W, loss = gradient_descent(W, b, batch_inputs, batch_labels, learning_rate)

			ini += batch_size
			fim += batch_size

			ac = validation(W, b, validation_data, validation_labels)

			if ac > best:
				print("new best = ", ac)
				best = ac
				outfile = open('ac', 'w')
				pickle.dump(ac, outfile)
				outfile.close()

				best_W = W
				outfile = open('weights', 'w')
				pickle.dump(W, outfile)
				outfile.close()

				best_b = b
				outfile = open('bias', 'w')
				pickle.dump(b, outfile)
				outfile.close()

				outfile = open('learning_rate', 'w')
				pickle.dump(learning_rate, outfile)
				outfile.close()

				outfile = open('batch_size', 'w')
				pickle.dump(batch_size, outfile)
				outfile.close()

			best_now = max(ac, best_now)
			
		# print("best = ", best)
		print("best now = ", best_now)
		# print("lr = ", learning_rate)

def main():
	need_shuffle = True
	need_split = True

	num_classes = 10

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

	train(train_data, train_labels, validation_data, validation_labels, num_classes)

if __name__ == "__main__":
	np.random.seed(1)
	main()
