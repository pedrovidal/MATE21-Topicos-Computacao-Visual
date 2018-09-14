from __future__ import print_function
import os
import sys
import numpy as np
import cv2

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

def debug(data, labels):
	print(len(data), len(labels))
	if len(data):
		for i in range(10):
			cv2.imshow("debug", data[i])
			print(labels[i])
			cv2.waitKey(0)

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

		debug(train_data, train_labels)
		debug(validation_data, validation_labels)
	else:
		train_data = data
		train_labels = labels

if __name__ == "__main__":
	np.random.seed(1)
	main()