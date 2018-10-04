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

def gradient_descent(W0, b0, W1, b1, batch_inputs, batch_labels, learning_rate):
	# batch_size = numero de imagens
	batch_size = batch_inputs.shape[0]

	# num_pixels = numero de pixels de cada imagem
	num_pixels = batch_inputs[0].shape[0]

	num_nodes = b0.shape[0]

	num_classes = b1.shape[0]

	# print(batch_size, num_pixels, num_classes, len(W), len(b))

	grad_W0 = np.zeros((num_nodes, num_pixels))
	grad_b0 = np.zeros(num_nodes)
	
	grad_W1 = np.zeros((num_classes, num_nodes))
	grad_b1 = np.zeros(num_classes)

	# lossH = 0
	lossO = 0
	
	hidden = np.empty(num_nodes)
	deltaJ = np.empty(num_nodes)
	deltaK = np.empty(num_classes)

	y_ = np.empty(num_classes)
	
	for x, y in zip(batch_inputs, batch_labels):
		for j in range(num_nodes):
			z = np.dot(W0[j], x) + b0[j]
			hidden[j] = sigmoid(z) 

		for k in range(num_classes):
			z = np.dot(W1[k], hidden) + b1[k]		
			y_[k] = sigmoid(z)

			deltaK[k] = (y_[k] - y[k]) * y_[k] * (1.0 - y_[k]) 

			grad_W1[k] += deltaK[k] * hidden
			grad_b1[k] += deltaK[k]
		
		lossO += 1.0 / 2.0 * (np.sum((y_ - y)**2))
		
		for j in range(num_nodes):
			# soma = 0
			# for k in range(num_classes):
			# 	soma += W0[j][k] * deltaK[k]
			deltaJ[j] = hidden[j] * (1 - hidden[j]) * np.sum(W0[j] * deltaK[k])

			grad_W0[j] += deltaJ[j] * x
			grad_b0[j] += deltaJ[j]

	lossO /= batch_size

	for i in range(num_classes):
		grad_b0[i] /= batch_size
		grad_W0[i] /= batch_size
		
		grad_b1[i] /= batch_size
		grad_W1[i] /= batch_size


		b0[i] = b0[i] - (learning_rate * grad_b0[i])
		W0[i] = W0[i] - (learning_rate * grad_W0[i])

		b1[i] = b1[i] - (learning_rate * grad_b1[i])
		W1[i] = W1[i] - (learning_rate * grad_W1[i])

	# print(W)

	return W0, b0, W1, b1, lossO

def validation(W0, b0, W1, b1, validation_data, validation_labels):
	ac = 0.0
	validation_size = len(validation_data)

	num_nodes = b0.shape[0]

	num_classes = b1.shape[0]

	for i in range(validation_size):
		best = 0
		label_prediction = 0
		hidden = np.empty(num_nodes)
		prediction = np.empty(10)

		for j in range(num_nodes):
			z = np.dot(W0[j], validation_data[i]) + b0[j]
			hidden[j] = sigmoid(z)

		for k in range(num_classes):
			z = np.dot(W1[k], hidden) + b1[k]		
			prediction[k] = sigmoid(z)

		label_prediction = np.argmax(prediction)

		# print("Validation ", label_prediction, np.argmax(validation_labels[i]))

		if label_prediction == np.argmax(validation_labels[i]):
			ac += 1
	return ac / validation_size

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

	batch_size = 40
	num_steps = train_size / batch_size
	learning_rate = 0.1

	infile = open('./mlp/ac', 'r')
	best = pickle.load(infile)
	infile.close()

	print("best =", best)
	
	# infile = open('./mlp/weights', 'r')
	# best_W = pickle.load(infile)
	# infile.close()

	# infile = open('./mlp/bias', 'r')
	# best_b = pickle.load(infile)
	# infile.close()

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

			W0, b0, W1, b1, loss = gradient_descent(W0, b0, W1, b1, batch_inputs, batch_labels, learning_rate)

			ini += batch_size
			fim += batch_size

			ac = validation(W0, b0, W1, b1, validation_data, validation_labels)

			if ac >= best:
				print("ac = ", ac)
				best = ac
				outfile = open('./mlp/ac', 'w')
				pickle.dump(ac, outfile)
				outfile.close()

				best_W0 = W0
				outfile = open('./mlp/weights0', 'w')
				pickle.dump(W0, outfile)
				outfile.close()

				best_b0 = b0
				outfile = open('./mlp/bias0', 'w')
				pickle.dump(b0, outfile)
				outfile.close()

				best_W1 = W1
				outfile = open('./mlp/weights1', 'w')
				pickle.dump(W1, outfile)
				outfile.close()

				best_b1 = b1
				outfile = open('./mlp/bias1', 'w')
				pickle.dump(b1, outfile)
				outfile.close()

				outfile = open('./mlp/learning_rate', 'w')
				pickle.dump(learning_rate, outfile)
				outfile.close()

				outfile = open('./mlp/batch_size', 'w')
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
	num_nodes = 100 # numero de nos da camada hidden

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

	train(train_data, train_labels, validation_data, validation_labels, num_classes, num_nodes)

if __name__ == "__main__":
	np.random.seed(1)
	main()
