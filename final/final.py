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
    for file_name in sorted(os.listdir(os.path.join(path, dirs))):
      img_path = os.path.join(path, dirs, file_name)
      img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
      img = cv2.resize(img, (64, 64))
      data.append(img)
      # one_hot_label = np.zeros(num_classes)
      # one_hot_label[int(dirs)] = 1
      labels.append(int(dirs))

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
  return data.reshape(np.shape(data)[0], np.shape(data)[1], np.shape(data)[2], 1)

def augmentate(batch_input):
	new_batch = []
	for i in range(len(batch_input)):
		possible_values = [0, 1, 2, 3, 4, 5]
		# probabilidade de sortear cada numero
		# p = distribuicao de probabilidade
		p = [1.0 / (len(possible_values) + 1)] * len(possible_values)
		p[0] *= 2
		tx = np.random.choice(possible_values, p=p)
		ty = np.random.choice(possible_values)
		M_translacao = np.float32([[1, 0, tx], [0, 1, ty]])
		rows, cols, _ = batch_input[i].shape
		img = cv2.warpAffine(batch_input[i], M_translacao, (cols, rows))
		new_batch.append(img)

	new_batch = np.array(new_batch, dtype=np.float)
	return reshape_data(new_batch)


class Model():
  def __init__(self, image_h, image_w, num_channels, num_classes):
    self.x = tf.placeholder(tf.float32, (None, image_h, image_w, num_channels))
    self.y = tf.placeholder(tf.int32, (None,))
    self.learning_rate = tf.placeholder(tf.float32)
    self.is_train = tf.placeholder(tf.bool)

    # tf.layers.conv2d(input, filters, kernel_size, strides, padding)
    conv1 = tf.layers.conv2d(inputs=self.x, filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=tf.nn.relu)
    max_pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    
    conv2 = tf.layers.conv2d(inputs=max_pool1, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=tf.nn.relu)
    max_pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(2, 2), padding='valid')

    conv3 = tf.layers.conv2d(inputs=max_pool2, filters=128, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=tf.nn.relu)
    max_pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 2), strides=(2, 2), padding='valid')

    out1 = tf.reshape(max_pool3, [-1, max_pool3.shape[1]*max_pool3.shape[2]*max_pool3.shape[3]])

    fc1 = tf.layers.dense(out1, 128, activation=tf.nn.relu)

    fc2 = tf.layers.dense(fc1, 64, activation=tf.nn.relu)

    self.y_ = tf.layers.dense(fc2, num_classes, activation=None)

    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_, labels=self.y))

    self.train_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    self.prediction = tf.cast(tf.argmax(self.y_, 1), tf.float32)
    self.label = tf.cast(self.y, tf.float32)

    self.ac_batch = tf.reduce_sum(tf.cast(tf.equal(self.prediction, self.label), tf.float32))

def train(train_data, train_labels, validation_data, validation_labels, model, num_epochs=25, augmentation=True):
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  batch_size = 8
  learning_rate = 5e-4
  num_steps = len(train_data) / batch_size

  saver = tf.train.Saver(save_relative_paths=True)
  
  # infile = open('./result/best_ac', 'r')
  # best = pickle.load(infile)
  # infile.close()

  best = 0

  for ep in range(num_epochs):
    best_now = 0
    ac_epoch = 0
    loss_epoch = 0
    train_data, train_labels = shuffle(train_data, train_labels)
    print('Epoca', ep)
    best_now = 0
    cont = 0
    for i in range(0, len(train_data), batch_size):
      cont += 1
      batch_input = np.array(train_data[i : i + batch_size])
      batch_labels = np.array(train_labels[i : i + batch_size])

      if augmentation:
      	batch_input = augmentate(batch_input)

      feed_dict_train = {model.x: batch_input, model.y: batch_labels, model.learning_rate: learning_rate}
      loss_batch, ac_batch, _ = sess.run([model.loss, model.ac_batch, model.train_opt], feed_dict=feed_dict_train)
      loss_epoch += loss_batch
      ac_epoch += ac_batch

      feed_dict_validation = {model.x: validation_data, model.y: validation_labels}
      loss_validation, ac_validation = sess.run([model.loss, model.ac_batch], feed_dict=feed_dict_validation)
      ac_validation /= len(validation_data)

      if cont % 10 == 0:
        print('step[', cont, '/', num_steps, ']', ac_validation)

      best_now = max(best_now, ac_validation)
      if ac_validation > best:
        best = ac_validation
        saver.save(sess, './result/model_cnn')
        # print('best =', best)
    
    print('ac_validation =', best_now)
    print('ac_treino =', ac_epoch / len(train_data), 'loss_treino =', loss_epoch / num_steps)
  print('best =', best)

def main():
  need_shuffle = True
  need_split = True

  num_classes = 10

  data, labels = load_data('../data_part1/train', num_classes)
  
  data = reshape_data(data)

  image_h, image_w, num_channels = data[0].shape

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

  # print(num_pixels, num_channels)

  model = Model(image_h, image_w, num_channels, num_classes)

  train(train_data=train_data, train_labels=train_labels, validation_data=validation_data, validation_labels=validation_labels, model=model)

if __name__ == "__main__":
  np.random.seed(1)
  main()
