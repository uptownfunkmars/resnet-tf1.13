import os
from dataProcess import *
from focal_loss import focal_loss

import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '9'

a = tf.Variable(tf.random_normal([5, 2]), name='a')
b = tf.constant([1, 1, 1, 0, 1], dtype=tf.int32, name='b')

ans = focal_loss(b, a)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(ans))

'''
negativePath = "/home/sdf/xujiping/tmb_bladder/data/HL_data/train/0"
positivePath = "/home/sdf/xujiping/tmb_bladder/data/HL_data/train/1"

dataGen = dataGen(positivePath, negativePath, 4)

inputs, labels = dataGen.__next__()
print(inputs.shape)
print(labels.shape)
'''
