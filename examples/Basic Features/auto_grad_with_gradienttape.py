# coding=utf-8

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

tfe.enable_eager_execution()

data = np.random.randn(3, 5).astype(np.float32)

# explicit variable
W = tf.get_variable('W', [5, 4], initializer=tf.truncated_normal_initializer)

# Dense Layers have implicit variables
dense_layer_func = tf.layers.Dense(2)

with tf.GradientTape() as tape:
    data_variable = tf.get_variable("data", initializer=data, trainable=False)
    h0 = tf.matmul(data_variable, W)
    h1 = dense_layer_func(h0)

# get all watched variables
vars = tape.watched_variables()
print("watched variables:")
print(vars)


grads = tape.gradient(h1, vars)
print("gradients for watched variables:")
print(grads)


optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
# apply gradients
optimizer.apply_gradients(zip(grads, vars))