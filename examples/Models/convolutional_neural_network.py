#coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
tfe.enable_eager_execution()

# 使用TensorFlow自带的MNIST数据集，第一次会自动下载，会花费一定时间
mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

flat_size = 3136
num_class = 10
dim_hidden = 1024

# 展示信息的间隔
verbose_interval = 40

# 定义各种层
layer_cnn0 = tf.layers.Conv2D(32, 5, activation = tf.nn.relu) # 卷积层0
layer_pool0 = tf.layers.MaxPooling2D(2, 2) # pooling层0
layer_cnn1 = tf.layers.Conv2D(64, 5, activation = tf.nn.relu) # 卷积层1
layer_pool1 = tf.layers.MaxPooling2D(2, 2) # pooling层1
layer_flatten = tf.layers.Flatten() # 将pooling层1的结果flatten
layer_fc0 = tf.layers.Dense(dim_hidden, activation = tf.nn.relu) # 全连接层0
layer_dropout = tf.layers.Dropout(0.75) # DropOut层
layer_fc1 = tf.layers.Dense(num_class, activation = None) # 全连接层1


def loss(step, x, y):
    inputs = tf.constant(x, name = "inputs")
    # 调用各种层进行前向传播
    cnn0 = layer_cnn0(inputs)
    pool0 = layer_pool0(cnn0)
    cnn1 = layer_cnn1(pool0)
    pool1 = layer_pool1(cnn1)
    flatten = layer_flatten(pool1)
    fc0 = layer_fc0(flatten)
    dropout = layer_dropout(fc0)
    logits = layer_fc1(dropout)

    # 进行softmax，并使用cross entropy计算损失
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(loss)

    # 每隔verbose_interval步显示一下损失和准确率
    if step % verbose_interval == 0:
        # 计算准确率
        predict = tf.argmax(logits, 1).numpy()
        target = np.argmax(y, 1)
        accuracy = np.sum(predict == target)/len(target)

        print("step {}:\tloss = {}\taccuracy = {}".format(step, loss.numpy(), accuracy))

    return loss

optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
batch_size = 128
# 训练1000步
for step in range(1000):
    batch_data, batch_label = mnist.train.next_batch(batch_size)
    # 原始batch_data的shape为[batch_size, 784]，需要将其转换为[batch_size, height, weight, channel]
    batch_data = batch_data.reshape([-1,28,28,1])
    optimizer.minimize(lambda: loss(step, batch_data, batch_label))
