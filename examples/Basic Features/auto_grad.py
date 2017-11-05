#coding=utf-8

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
tfe.enable_eager_execution()

grad = tfe.gradients_function(lambda x: x * x + 4.0)
print(grad(10))
print(grad(5))
