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
      img = cv2.resize(img, (16, 16))
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

def gen_noise(data_size):
  vet = []

  for i in range(data_size):
    aux = np.random.normal(size=64)
    aux = np.reshape(aux, (8, 8, 1))
    vet.append(aux)
  
  vet = np.array(vet)
  return vet

def generator(inputs, reuse=False):
  with tf.variable_scope('generator', reuse=reuse):
    fc = tf.layers.dense(inputs, 64, activation=tf.nn.relu)
    c1 = tf.layers.conv2d_transpose(inputs=fc, filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
    print(c1.shape)
    images = tf.layers.conv2d(inputs=c1, filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
    print(images.shape)
    
  return images

def discriminator(inputs, reuse=False):
  with tf.variable_scope('discriminator', reuse=reuse):
    c1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation=tf.nn.relu)
    print('shape conv1', c1.shape)
    mp1 = tf.layers.max_pooling2d(inputs=c1, pool_size=(2, 2), strides=(3, 3), padding='same')
    print('shape max_pool1', mp1.shape)
    
    c2 = tf.layers.conv2d(inputs=mp1, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation=tf.nn.relu)
    print('shape conv2', c2.shape)  
    mp2 = tf.layers.max_pooling2d(inputs=c2, pool_size=(2, 2), strides=(3, 3), padding='same')
    print('shape max_pool2', mp2.shape)
    
    c3 = tf.layers.conv2d(inputs=mp2, filters=1, kernel_size=(5, 5), strides=(1, 1), padding='same', activation=tf.nn.relu)
    print('shape conv3', c3.shape)  
    mp3 = tf.layers.max_pooling2d(inputs=c3, pool_size=(2, 2), strides=(3, 3), padding='same')
    print('shape max_pool3', mp3.shape)

  return mp3

class Model():
  def __init__(self):
    self.learning_rate_gen = tf.placeholder(tf.float32)
    self.learning_rate_disc = tf.placeholder(tf.float32)

    self.gen_input = tf.placeholder(tf.float32, (None, 8, 8, 1))
    self.gen_output = generator(self.gen_input)

    self.gen_output = tf.maximum(self.gen_output, 0)
    self.gen_output = tf.minimum(self.gen_output, 1)

    self.disc_input = tf.placeholder(tf.float32, (None, 16, 16, 1))

    self.r_logits = discriminator(self.disc_input)
    self.f_logits = discriminator(self.gen_output, reuse=True)

    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

    self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.f_logits, labels=tf.ones_like(self.f_logits)))
    self.disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.r_logits, labels=tf.ones_like(self.r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.f_logits, labels=tf.zeros_like(self.f_logits)))

    self.gen_train_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_gen).minimize(self.gen_loss, var_list=gen_vars)
    # self.gen_train_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_gen).minimize(self.gen_loss, var_list=gen_vars)
    self.disc_train_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_disc).minimize(self.disc_loss, var_list=disc_vars)
    # self.disc_train_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_disc).minimize(self.disc_loss, var_list=disc_vars)

def train(train_data, validation_data, model, num_epochs=200):
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  batch_size = 16
  learning_rate_gen = 2e-4
  learning_rate_disc = 5e-5
  num_steps = len(train_data) / batch_size

  saver = tf.train.Saver(save_relative_paths=True)

  best = 1123456789

  for ep in range(num_epochs):
    print('Epoca', ep)
    
    best_now = 0
    loss_epoch = 0

    train_data = shuffle(train_data)
    
    for cont, i in enumerate(range(0, len(train_data), batch_size // 2)):

      gen_input = gen_noise(batch_size // 2)
      disc_input = np.array(train_data[i : i + batch_size // 2])

      feed_dict = {model.gen_input: gen_input, model.disc_input: disc_input, model.learning_rate_gen: learning_rate_gen, model.learning_rate_disc: learning_rate_disc}

      disc_loss_batch, _ = sess.run([model.disc_loss, model.disc_train_opt], feed_dict=feed_dict)
      gen_loss_batch, gen_images, _ = sess.run([model.gen_loss, model.gen_output, model.gen_train_opt], feed_dict=feed_dict)

      if cont % 100 == 0:
        print('step[', cont, '/', num_steps * 2, ']', 'loss gen =', gen_loss_batch, 'loss disc =', disc_loss_batch)
        
        ind = np.random.randint(len(gen_images))
        for j, output in enumerate(gen_images):
          output *= 255

          cv2.imwrite('teste/output' + str(ep) + '_' + str(j) + '.png', output)


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

  model = Model()

  train(train_data=train_data, validation_data=validation_data, model=model)

if __name__ == "__main__":
  np.random.seed(1)
  main()
