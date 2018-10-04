from __future__ import print_function
import os
import sys
import numpy as np
import cv2
import pickle
	
def load_data(path, num_classes):
	data = []
	name = []

	for file_name in os.listdir(path):
		img_path = os.path.join(path, file_name)
		data.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
		name.append(file_name)

	data = np.array(data, dtype=np.float)
	data /= 255.0
	data = data.reshape(np.shape(data)[0], -1)
	return data, name

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))
	if x >= 0:
		return 1.0 / (1.0 + np.exp(-x))
	else:
		exp = np.exp(x)
		return exp / (1.0 + exp)

def validation(data, name, num_classes):
	infile = open('weights', 'r')
	W = pickle.load(infile)
	infile.close()

	infile = open('bias', 'r')
	b = pickle.load(infile)
	infile.close()

	data_size = len(data)
	for i in range(data_size):
		prediction = np.empty(10)
		for j in range(10):
			prediction[j] = sigmoid(np.dot(W[j], data[i]) + b[j])
		label_prediction = np.argmax(prediction)
		
		print(name[i], label_prediction)

def main():

	num_classes = 10

	data, name = load_data('./data_part1/test', num_classes)	
	
	validation(data, name, num_classes)

if __name__ == "__main__":
	np.random.seed(1)
	main()
