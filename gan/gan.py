from __future__ import print_function
import cv2
import numpy as np
import os
import pickle
import sys
import tensorflow as tf

def load_data(path, num_classes, image_w, image_h):
  data = []
  labels = []

  for dirs in sorted(os.listdir(path)):
    # if dirs != '1':
    #   continue
    for file_name in sorted(os.listdir(os.path.join(path, dirs))):
      img_path = os.path.join(path, dirs, file_name)
      img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
      img = cv2.resize(img, (image_w, image_h))
      data.append(img)
      # one_hot_label = np.zeros(num_classes)
      # one_hot_label[int(dirs)] = 1
    # if int(dirs) == 3:
    #   break

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
  vet = np.random.randn(data_size, 64)
  return vet
  
def generator(net, image_h, image_w, reuse=False):
  with tf.variable_scope('generator', reuse=reuse):
    print(net.shape)

    # net = tf.layers.dense(net, 32, activation=tf.nn.relu)

    # net = tf.layers.dense(net, 64, activation=tf.nn.relu)

    net = tf.layers.dense(net, 128, activation=tf.nn.relu)
    
    # net = tf.layers.dense(net, 256, activation=tf.nn.relu)
    
    # net = tf.layers.dense(net, 512, activation=tf.nn.relu)

    net = tf.layers.dense(net, 1024, activation=tf.nn.sigmoid)
    print(net.shape)

    net = tf.reshape(net, (-1, 32, 32, 1))
    print(net.shape)

    # net = tf.layers.conv2d(inputs=net, filters = 16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)

    # net = tf.layers.conv2d(inputs=net, filters = 1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.sigmoid)

    # net = tf.layers.conv2d_transpose(inputs=net, filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
    print(net.shape)

    # net = tf.layers.conv2d_transpose(inputs=net, filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=None)
    # print(net.shape)

    # net = tf.image.resize_images(net, (image_h, image_w))
    
  return net

def discriminator(net, reuse=False):
  with tf.variable_scope('discriminator', reuse=reuse):
    # print('disc:')

    net = tf.reshape(net, (-1, 1024))

    net = tf.layers.dense(net, 64, activation=tf.nn.relu)
    
    # net = tf.layers.dense(net, 32, activation=tf.nn.relu)
    
    net = tf.layers.dense(net, 1, activation=None)

    # net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
    # print('shape conv1', net.shape)
    # net = tf.layers.max_pooling2d(inputs=net, pool_size=(3, 3), strides=(2, 2), padding='same')
    # print('shape max_pool1', net.shape)
    
    # net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
    # print('shape conv2', net.shape)  
    # net = tf.layers.max_pooling2d(inputs=net, pool_size=(3, 3), strides=(2, 2), padding='same')
    # print('shape max_pool2', net.shape)

    # NAO MUDAR ULTIMA CAMADA
    # net = tf.layers.conv2d(inputs=net, filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=None)
    # print('shape final conv', net.shape)  
    # net = tf.layers.max_pooling2d(inputs=net, pool_size=(3, 3), strides=(2, 2), padding='same')
    # print('shape final max_pool', net.shape)
    # NAO MUDAR ULTIMA CAMADA

  return net

class Model():
  def __init__(self, image_h, image_w):
    self.learning_rate_gen = tf.placeholder(tf.float32)
    self.learning_rate_disc = tf.placeholder(tf.float32)

    self.gen_input = tf.placeholder(tf.float32, (None, 64))
    self.gen_output = generator(self.gen_input, image_h, image_w)

    # self.gen_output = tf.maximum(self.gen_output, 0)
    # self.gen_output = tf.minimum(self.gen_output, 1)

    self.disc_input = tf.placeholder(tf.float32, (None, image_h, image_w, 1))

    self.r_logits = discriminator(self.disc_input)
    self.f_logits = discriminator(self.gen_output, reuse=True)

    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

    self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.f_logits, labels=tf.ones_like(self.f_logits)))
    self.disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.r_logits, labels=tf.ones_like(self.r_logits))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.f_logits, labels=tf.zeros_like(self.f_logits)))

    self.gen_train_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_gen).minimize(self.gen_loss, var_list=gen_vars)
    # self.gen_train_opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_gen).minimize(self.gen_loss, var_list=gen_vars)
    self.disc_train_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_disc).minimize(self.disc_loss, var_list=disc_vars)
    # self.disc_train_opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_disc).minimize(self.disc_loss, var_list=disc_vars)

def train(train_data, validation_data, model, num_epochs=10000):
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  batch_size = 128
  # learning_rate_gen = 4e-5
  learning_rate_gen = 5e-4
  # learning_rate_disc = 5e-5
  learning_rate_disc = 5e-5
  num_steps = len(train_data) / batch_size

  gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
  saver = tf.train.Saver(var_list=gen_vars, save_relative_paths=True)

  best = 1123456789

  validation_noise = gen_noise(11)

  for ep in range(num_epochs + 1):
    # print('Epoca', ep)
    
    best_now = 0
    loss_epoch = 0

    train_data = shuffle(train_data)
    
    for cont, i in enumerate(range(0, len(train_data), batch_size // 2)):

      gen_input = gen_noise(batch_size // 2)
      disc_input = np.array(train_data[i : i + batch_size // 2])

      feed_dict = {model.gen_input: gen_input, model.disc_input: disc_input, model.learning_rate_gen: learning_rate_gen, model.learning_rate_disc: learning_rate_disc}

      disc_loss_batch, _ = sess.run([model.disc_loss, model.disc_train_opt], feed_dict=feed_dict)

      gen_input = gen_noise(batch_size)
      feed_dict = {model.gen_input: gen_input, model.learning_rate_gen: learning_rate_gen}

      gen_loss_batch, gen_images, _ = sess.run([model.gen_loss, model.gen_output, model.gen_train_opt], feed_dict=feed_dict)

      if cont == ((num_steps * 2) - 1):
       print('epoca', ep, 'loss gen =', gen_loss_batch, 'loss disc =', disc_loss_batch)
       
      
      if ep % 25 == 0 and cont == ((num_steps * 2) - 1):
        for i in range(10):
          output = gen_images[i] * 255
          output = cv2.resize(output, (64, 64), interpolation=cv2.INTER_CUBIC)
          cv2.imwrite('teste/output' + str(ep) + '_' + str(i) + '.png', output)

        feed_dict = {model.gen_input: validation_noise, model.learning_rate_gen: learning_rate_gen}
        gen_loss_validation, gen_images_validation, _ = sess.run([model.gen_loss, model.gen_output, model.gen_train_opt], feed_dict=feed_dict)

        for i in range(len(gen_images_validation)):
          output = gen_images_validation[i] * 255
          output = cv2.resize(output, (64, 64), interpolation=cv2.INTER_CUBIC)
          cv2.imwrite('teste/validation/output' + str(ep) + '_' + str(i) + '.png', output)

      if ep % 100 == 0:
        # gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
        saver.save(sess, './model/gan_generator')


def main():
  need_shuffle = True
  need_split = True

  image_w, image_h = (32, 32)

  num_classes = 10

  data = reshape_data(load_data('../data_part1/train', num_classes, image_w, image_h))

  num_channels = data[0].shape[2]

  if need_shuffle:
    data = shuffle(data)

  if need_split:
    train_percentage = 80
    train_data, validation_data = split_dataset(data, train_percentage)

  else:
    train_data = data
    validation_data = np.array()

  model = Model(image_h, image_w)

  train(train_data=train_data, validation_data=validation_data, model=model)

if __name__ == "__main__":
  np.random.seed(1)
  main()
