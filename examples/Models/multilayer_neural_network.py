#coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

# 开启Eager Execution
tfe.enable_eager_execution()

# 使用TensorFlow自带的MNIST数据集，第一次会自动下载，会花费一定时间
mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

# 展示信息的间隔
verbose_interval = 500

dim_hidden = 150
num_class = 10

layer_fc0 = tf.layers.Dense(dim_hidden, activation = tf.nn.relu)
layer_dropout = tf.layers.Dropout(0.75)
layer_fc1 = tf.layers.Dense(num_class, activation = None)


# 构建多层神经网络
def mlp(step, x, y, is_train = True):
    inputs = tf.constant(x, name = "inputs")
    fc0 = layer_fc0(inputs)
    if is_train:
        fc0 = layer_dropout(fc0)
    logits = layer_fc1(fc0)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(loss)

    if step % verbose_interval == 0:
        # 计算准确率
        predict = tf.argmax(logits, 1).numpy()
        target = np.argmax(y, 1)
        accuracy = np.sum(predict == target)/len(target)

        print("step {}:\tloss = {}\taccuracy = {}".format(step, loss.numpy(), accuracy))

    return loss

batch_size = 128
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)

# 执行3000步
for step in range(3000):
    # 生成128个数据，batch_data是图像像素数据，batch_label是图像label信息
    batch_data, batch_label = mnist.train.next_batch(128)
    # 梯度下降优化网络参数
    optimizer.minimize(lambda: mlp(step, batch_data, batch_label))
