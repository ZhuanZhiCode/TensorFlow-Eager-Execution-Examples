#coding=utf-8

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
tfe.enable_eager_execution()

# 随机生成2个图像(batch_size=2)，每个图像的大小为 28 * 28 * 3
images = np.random.randn(2, 28, 28, 3).astype(np.float32)
# 卷积核参数
filter = tf.get_variable("conv_w0", shape = [5,5,3,20], initializer = tf.truncated_normal_initializer)
# 对生成的batch_size=2的数据进行卷积操作，立即获得结果
conv = tf.nn.conv2d(input = images, filter = filter, strides = [1,2,2,1], padding = 'VALID')
# 用结果的numpy()方法可获得结果的numpy表示，显示其shape
print(conv.numpy().shape)
