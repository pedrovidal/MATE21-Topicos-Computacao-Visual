from __future__ import print_function
import os
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

def teste(data, labels):
	print(len(data), len(labels))
	for i in range(10):
		cv2.imshow("teste", data[i])
		print(labels[i])
		cv2.waitKey(0)
	
def split_dataset(data, labels, train_per_val):
	data_size = len(data)
	permutation_index = np.random.permutation(data_size)
	# print(permutation)
	train_size = int(data_size * train_per_val / 100)
	validation_size = data_size - train_size
	print(train_size, validation_size)
	
	train_data = data[permutation_index[0:train_size]]
	train_labels = labels[permutation_index[0:train_size]]
	validation_data = data[permutation_index[train_size:data_size]]
	validation_labels = labels[permutation_index[train_size:data_size]]
	return train_data, train_labels, validation_data, validation_labels

def main():
	data, labels = load_data('./data_part1/train')
	# teste(data, labels)
	# print(len(data))
	train_data, train_labels, \
		validation_data, validation_labels = split_dataset(data, labels, 80)
	print("teste validation")
	teste(validation_data, validation_labels)
	print("teste train")
	teste(train_data, train_labels)

if __name__ == "__main__":
	np.random.seed(1)
	main()