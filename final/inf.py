from __future__ import print_function
import cv2
import numpy as np
import os
import pickle
import sys
import argparse
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

def compute_ensemble():
  # print('Computing ensemble...')
  votes = np.zeros((5490, 10))
  names = []
  for i, file_name in enumerate(sorted(os.listdir('result_files/'))):
    file = open('result_files/' + file_name)
    predictions = file.read().splitlines()
    for j, vote in enumerate(sorted(predictions)):
      img_name, label = vote.split()
      if i == 0:
        names.append(img_name)
      votes[j][int(label)] += 1
      # print(i, j, label, votes[j])
  for i in range(len(names)):
    print(names[i], np.argmax(votes[i]))

def compute_ac_validation():
  val_file = open('validation.txt')
  labels = val_file.read().splitlines()
  labels = sorted(labels)
  for i, file_name in enumerate(sorted(os.listdir('result_files/'))):
    ac = 0
    file = open('result_files/' + file_name)
    predictions = file.read().splitlines()
    predictions = sorted(predictions)
    for i in range(len(predictions)):
      if predictions[i] == labels[i]:
        ac += 1
    print(file_name, ac)

def main():

  if FLAGS.compute_ac:
    compute_ac_validation()
    return

  if FLAGS.ensemble:
    compute_ensemble()
    return

  num_classes = 10

  model = Model(64, 64, 1, num_classes)
  sess = tf.Session()

  if FLAGS.debug:
    print('Restoring model...')

  saver = tf.train.Saver()
  saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_path))

  if FLAGS.debug:
    print('Model restored.')

  for file_name in sorted(os.listdir(FLAGS.data_path)):
    data = []

    if FLAGS.debug:
      print('Loading image', file_name)
    
    img_path = os.path.join(FLAGS.data_path, file_name)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (64, 64))

    data.append(img)

    data = np.array(data, dtype=np.float)
    data /= 255.0

    data = reshape_data(data)

    if FLAGS.debug:
      print('Image loaded.')

    image_h, image_w, num_channels = data[0].shape

    if FLAGS.debug:
      print('Making prediction...')

    label = sess.run(model.prediction, feed_dict={model.x: data, model.is_train: False})

    if FLAGS.debug:
      print('Prediction made.')
    
    print(file_name, int(label))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--data_path',
    required=False,
    type=str,
    help='path to test dataset')
  parser.add_argument(
    '--model_path',
    required=False,
    type=str,
    help='path to model')
  parser.add_argument(
    '--ensemble',
    action='store_true',
    help='whether or not to perform ensemble')
  parser.add_argument(
    '--debug',
    action='store_true',
    help='whether or not to print debugs')
  parser.add_argument(
    '--compute_ac',
    action='store_true',
    help='whether or not to compute ac in validation, validation predictions must be in result_files/')
  FLAGS = parser.parse_args()
  np.random.seed(1)
  main()