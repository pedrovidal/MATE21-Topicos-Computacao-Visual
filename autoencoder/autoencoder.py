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

  data = np.array(data, dtype=np.float)
  data /= 255.0
  return data

def shuffle(data):
  data_size = len(data)
  permutation_index = np.random.permutation(data_size)
  shuffled_data = data[permutation_index]
  return shuffled_data

def split_dataset(data, train_percentage):
  data_size = len(data)

  train_size = int(data_size * train_percentage / 100)
  
  train_data = data[0:train_size]

  validation_data = data[train_size:data_size]

  return train_data, validation_data

def reshape_data(data):
  return data.reshape(np.shape(data)[0], np.shape(data)[1], np.shape(data)[2], 1)

class Model():
  def __init__(self, image_h, image_w, num_channels, num_classes):
    self.x = tf.placeholder(tf.float32, (None, image_h, image_w, num_channels))
    self.learning_rate = tf.placeholder(tf.float32)
    self.is_train = tf.placeholder(tf.bool)

    # tf.layers.conv2d(input, filters, kernel_size, strides, padding)
    conv1 = tf.layers.conv2d(inputs=self.x, filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation=tf.nn.relu)
  
    print('shape conv1', conv1.shape)

    max_pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
    
    print('shape max_pool1', max_pool1.shape)

    conv2 = tf.layers.conv2d(inputs=max_pool1, filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation=tf.nn.relu)
    
    print('shape conv2', conv2.shape)    
  
    max_pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(2, 2), padding='same')

    print('shape max_pool2', max_pool2.shape)

    convt1 = tf.layers.conv2d_transpose(inputs=max_pool2, filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', activation=tf.nn.relu)
    
    print('shape convt1', convt1.shape)

    convt2 = tf.layers.conv2d_transpose(inputs=convt1, filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation=tf.nn.relu)
   
    print('shape convt2', convt2.shape)

    self.y_ = tf.layers.conv2d_transpose(inputs=convt2, filters=1, kernel_size=(5, 5), strides=(1, 1), padding='same', activation=tf.nn.relu)
   
    print('shape output', self.y_.shape)
  
    # self.loss = tf.reduce_sum(abs(self.y_ - self.x)) # dist L1

    self.loss = tf.reduce_sum((self.y_ - self.x) ** 2) # dist L2

    self.train_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


def train(train_data, validation_data, model, num_epochs=200):
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  batch_size = 8
  learning_rate = 5e-4
  num_steps = len(train_data) / batch_size

  saver = tf.train.Saver(save_relative_paths=True)
  
  # infile = open('./result/best_ac', 'r')
  # best = pickle.load(infile)
  # infile.close()

  best = 1123456789

  for ep in range(num_epochs):
    best_now = 0
    ac_epoch = 0
    loss_epoch = 0
    train_data = shuffle(train_data)
    print('Epoca', ep)
    best_now = 0
    cont = 0
    for i in range(0, len(train_data), batch_size):
      cont += 1
      batch_input = np.array(train_data[i : i + batch_size])

      feed_dict_train = {model.x: batch_input, model.learning_rate: learning_rate}
      loss_batch, _ = sess.run([model.loss, model.train_opt], feed_dict=feed_dict_train)
      loss_epoch += loss_batch

      if cont % 100 == 0:
        feed_dict_validation = {model.x: validation_data}
        loss_validation, imagens = sess.run([model.loss, model.y_], feed_dict=feed_dict_validation)

        print('step[', cont, '/', num_steps, ']', 'loss validation =', loss_validation)
        
        ind = np.random.randint(len(validation_data))

        orig = validation_data[ind] * 255
        img = imagens[ind] * 255


        cv2.imwrite('teste/orig.png', orig)
        cv2.imwrite('teste/decodificada.png', img)

        best_now = max(best_now, loss_validation)
        if loss_validation < best:
          best = loss_validation
          saver.save(sess, './result/model_cnn')
          # print('best =', best)
    
    print('loss_validation =', best_now)
    print('loss_treino =', loss_epoch / num_steps)
  print('best =', best)

def main():
  need_shuffle = True
  need_split = True

  num_classes = 10

  data = reshape_data(load_data('../data_part1/train', num_classes))

  image_h, image_w, num_channels = data[0].shape

  if need_shuffle:
    data = shuffle(data)

  if need_split:
    train_percentage = 80
    train_data, validation_data = split_dataset(data, train_percentage)

  else:
    train_data = data
    validation_data = np.array()

  # print(num_pixels, num_channels)

  model = Model(image_h, image_w, num_channels, num_classes)

  train(train_data=train_data, validation_data=validation_data, model=model)

if __name__ == "__main__":
  np.random.seed(1)
  main()
