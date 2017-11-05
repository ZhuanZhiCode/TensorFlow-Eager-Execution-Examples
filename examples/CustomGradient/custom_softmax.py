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

# 加了注解之后，可以自定义梯度，如果不加注解，tf会自动计算梯度
# 加了注解之后，需要返回两个值，第一个值为loss，第二个值为梯度计算函数
# 本函数的参数中，step表示当前所在步骤，x表示Softmax层的输入，y是one-hot格式的label信息
@tfe.custom_gradient
def softmax_loss(step, x, y):
    # 将x限定在-20和20之间，防止产生过大的值
    x = tf.clip_by_value(x, -20, 20);
    exp = tf.exp(x)
    sum = tf.reduce_sum(exp, 1)
    sum = tf.reshape(sum, [-1, 1])
    # Softmax中的归一化
    sm = tf.divide(exp, sum)
    # 用Cross-Entropy计算Softmax的损失函数
    loss = -tf.log(tf.clip_by_value(sm, 1e-10, 1.0)) * y
    loss = tf.reduce_mean(loss)

    if step % verbose_interval == 0:
        # 计算准确率
        predict = tf.argmax(sm, 1).numpy()
        target = np.argmax(y, 1)
        accuracy = (predict == target).sum() / len(target)
        print("\nstep: {}".format(step))
        print("accuracy = {}".format(accuracy))

    # 定义梯度函数
    def grad(_):
        # Softmax在Cross-Entropy下梯度非常简单
        # 即 object - target
        d = sm - y
        # 需要返回损失函数相对于softmax_loss每个参数的梯度
        # 第一和第三个参数不需要训练，因此将梯度设置为None
        return None, d, None

    #返回损失函数和梯度函数
    return loss, grad

with tf.device("/gpu:0"):
    # 第一层网络的参数，输入为28*28=784维，隐藏层150维
    W0 = tf.get_variable("W0", shape=[784, 150])
    b0 = tf.get_variable("b0", shape=[150])
    # 第二层网络的参数，一共有10类数字，因此输出为10维
    W1 = tf.get_variable("W1", shape=[150, 10])
    b1 = tf.get_variable("b1", shape=[10])

    # 构建多层神经网络
    def mlp(step, x, y, is_train = True):
        hidden = tf.matmul(x, W0) + b0
        hidden = tf.nn.relu(hidden)
        # 如果在训练，使用dropout层防止过拟合
        # Eager Execution使得我们可以利用Python的if语句动态调整网络结构
        if is_train:
            hidden = tf.nn.dropout(hidden, keep_prob = 0.75)
        logits = tf.matmul(hidden, W1) + b1
        # 调用我们自定义的Softmax层
        loss = softmax_loss(step, logits, y)
        if step % verbose_interval == 0:
            print("loss = {}".format(loss.numpy()))
        return loss

    optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)

    # 执行3000步
    for step in range(3000):
        # 生成128个数据，batch_data是图像像素数据，batch_label是图像label信息
        batch_data, batch_label = mnist.train.next_batch(128)
        # 梯度下降优化网络参数
        optimizer.minimize(lambda: mlp(step, batch_data, batch_label))
