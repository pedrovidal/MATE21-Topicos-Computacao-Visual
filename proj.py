import os
import numpy as np
import cv2

def load_train_data(path):
	train = []
	labels = []
	for dirs in sorted(os.listdir(path)):
		for file_name in os.listdir(os.path.join(path, dirs)):
			img_path = os.path.join(path, dirs, file_name)
			train.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
			labels.append(dirs)
			# print dirs
			# print file_name
	return train, labels


def main():
	train_data, train_labels = load_train_data('./data_part1/train')
	# print train_data[0].shape
	for i in range(10):
		cv2.imshow("teste", train_data[i*500])
		print train_labels[i*500]
		cv2.waitKey(0)

if __name__ == "__main__":
	np.random.seed(1)
	main()