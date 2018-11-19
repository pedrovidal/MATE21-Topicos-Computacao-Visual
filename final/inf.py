from __future__ import print_function
import cv2
import numpy as np
import os
import pickle
import sys
import tensorflow as tf

def load_data(path, num_classes):
  data = []
  names = []

  for file_name in sorted(os.listdir(path)):
    img_path = os.path.join(path, file_name)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (64, 64))
    data.append(img)
    names.append(file_name)

  data = np.array(data, dtype=np.float)
  data /= 255.0
  return data, names

def reshape_data(data):
  return data.reshape(np.shape(data)[0], np.shape(data)[1], np.shape(data)[2], 1)

class Model():
  def __init__(self, image_h, image_w, num_channels, num_classes):
    self.x = tf.placeholder(tf.float32, (None, image_h, image_w, num_channels))
    self.y = tf.placeholder(tf.int32, (None,))
    self.learning_rate = tf.placeholder(tf.float32)
    self.is_train = tf.placeholder(tf.bool)

    # tf.layers.conv2d(input, filters, kernel_size, strides, padding)
    conv1 = tf.layers.conv2d(inputs=self.x, filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=tf.nn.relu)
    max_pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    
    conv2 = tf.layers.conv2d(inputs=max_pool1, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=tf.nn.relu)
    max_pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(2, 2), padding='valid')

    conv3 = tf.layers.conv2d(inputs=max_pool2, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=tf.nn.relu)
    max_pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 2), strides=(2, 2), padding='valid')

    out1 = tf.reshape(max_pool3, [-1, max_pool3.shape[1]*max_pool3.shape[2]*max_pool3.shape[3]])

    fc1 = tf.layers.dense(out1, 128, activation=tf.nn.relu)

    fc2 = tf.layers.dense(fc1, 256, activation=tf.nn.relu)

    self.y_ = tf.layers.dense(fc2, num_classes, activation=None)

    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_, labels=self.y))

    self.train_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    self.prediction = tf.cast(tf.argmax(self.y_, 1), tf.float32)
    self.label = tf.cast(self.y, tf.float32)

    self.ac_batch = tf.reduce_sum(tf.cast(tf.equal(self.prediction, self.label), tf.float32))

def main():

  num_classes = 10

  data, names = load_data('../data_part1/test', num_classes)
  
  data = reshape_data(data)

  image_h, image_w, num_channels = data[0].shape

  model = Model(image_h, image_w, num_channels, num_classes)

  sess = tf.Session()

  path = 'cnn_results_16_32_128'

  saver = tf.train.Saver()
  saver.restore(sess, tf.train.latest_checkpoint(path))

  labels = sess.run(model.prediction, feed_dict={model.x: data, model.is_train: False})

  for i in range(len(names)):
    print(names[i], int(labels[i]))

if __name__ == "__main__":
  np.random.seed(1)
  main()