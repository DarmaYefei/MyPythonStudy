#%%
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("D:\MyPythonDoc\MyPyResource\MNIST_data/", False, one_hot=True)

labelN = 300
for i in range(100):
    plt.subplot(10, 10, i + 1)
    image1 = mnist.train.images[labelN + i]
    label1 = mnist.train.labels[labelN + i]
    image1 = image1.reshape(28, 28)
    plt.imshow(image1, cmap='Greys', interpolation='nearest')
    plt.text(0, 1, str(list(label1).index(1)))
plt.show()
# print(mnist.train.images.shape, mnist.train.labels.shape)
# print(mnist.test.images.shape, mnist.test.labels.shape)
# print(mnist.validation.images.shape, mnist.validation.labels.shape)

# import tensorflow as tf
# sess = tf.InteractiveSession()
# x = tf.placeholder(tf.float32, [None, 784])
#
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
#
# y = tf.nn.softmax(tf.matmul(x, W) + b)
#
# y_ = tf.placeholder(tf.float32, [None, 10])
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# train_writer = tf.summary.FileWriter("mnist_logs/", sess.graph)
# train_writer.close()
#
# tf.global_variables_initializer().run()
#
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     train_step.run({x: batch_xs, y_: batch_ys})
#
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
