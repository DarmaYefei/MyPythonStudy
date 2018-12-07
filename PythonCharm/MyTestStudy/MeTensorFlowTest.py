import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# constant
# a = tf.constant(5.0, name='input_a')
# b = tf.constant(3.0, name='input_b')
# c = tf.multiply(a, b, name='mul_c')
# d = tf.add(a, b, name='add_d')
# e = tf.add(c, d, name='add_e')
#
# sess = tf.Session()
# output = sess.run(e)
# print(output)
#
# write = tf.summary.FileWriter("Graph/Shown", sess.graph)
# write.close()
# sess.close()

# variable
# a = tf.Variable(5.0, name='input_a')
# b = tf.Variable(3.0, name='input_b')
# c = tf.multiply(a, b, name='mul_c')
# d = tf.add(a, b, name='add_d')
# e = tf.add(c, d, name='add_e')
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# output = sess.run(e)
# print(output)
# sess.close()

# placeholder
# a = tf.placeholder(tf.int8, shape=[None], name='input_a')
# c = tf.reduce_prod(a, name='mul_c')
# d = tf.reduce_sum(a, name='add_d')
# e = tf.reduce_sum([c, d], name='add_e')
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# output = sess.run(e, feed_dict={a:[3, 5]})
# print(output)
# sess.close()

# graph = tf.Graph()
# with graph.as_default():
#     in_1 = tf.placeholder(tf.float32, shape=[], name='input_a')
#     in_2 = tf.placeholder(tf.float32, shape=[], name='input_b')
#     const = tf.constant(3, dtype=tf.float32, name='static_value')
#     with tf.name_scope('Tansformation'):
#         with tf.name_scope('A'):
#             A_mul = tf.multiply(in_1, const)
#             A_out = tf.subtract(A_mul, in_1)
#         with tf.name_scope('B'):
#             B_mul = tf.multiply(in_2, const)
#             B_out = tf.subtract(B_mul, in_2)
#         with tf.name_scope('C'):
#             C_div = tf.div(A_out, B_out)
#             C_out = tf.add(C_div, const)
#         with tf.name_scope('D'):
#             D_div = tf.div(B_out, A_out)
#             D_out = tf.add(D_div, const)
#             out = tf.maximum(C_out, D_out)
#
# write = tf.summary.FileWriter("Graph/Shown", graph=graph)
# write.close()

LR = 0.1
REAL_PARAMS = [1.2, 2.5]
INIT_PARAMS = [2, 4.5]

x = np.linspace(-1, 1, 200, dtype=np.float32)  # x data

y_fun = lambda a, b: np.sin(b * np.cos(a * x))
tf_y_fun = lambda a, b: tf.sin(b * tf.cos(a * x))

noise = np.random.randn(200) / 10
y = y_fun(*REAL_PARAMS) + noise  # 实际值

# tensorflow graph
# 定义a、b变量，并初始化
a = tf.Variable(2.0, dtype=tf.float32)
b = tf.Variable(4.5, dtype=tf.float32)
pred = tf_y_fun(a, b)  # 预测值
mse = tf.reduce_mean(tf.square(y - pred))
train_op = tf.train.GradientDescentOptimizer(LR).minimize(mse)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(400):
        a_, b_, mse_ = sess.run([a, b, mse])
        result, _ = sess.run([pred, train_op])  # 训练模型
# 可视化结果:
print('a=', a_, 'b=', b_)
plt.figure(1)
plt.scatter(x, y, c='b')  # plot data
plt.plot(x, result, 'r-', lw=2)  # plot line fitting
plt.show()
