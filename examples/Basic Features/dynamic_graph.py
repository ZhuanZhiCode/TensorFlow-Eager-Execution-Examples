#coding=utf-8

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
tfe.enable_eager_execution()

def leaky_relu(x):
    if x < 0:
        return x * 0.1
    else:
        return x
grad = tfe.gradients_function(leaky_relu)
print(grad(4.0))
print(grad(-3.0))
