import numpy as np
import os
import tensorflow as tf


# os.environ["CUDA_VISIBLE_DEVICES"] = "9"

class VGG(object):
    def __init__(self):
    # def __init__(self, isTraining):
        # self._is_training = isTraining
        pass
    # def model(self, x):
    def model(self, x, is_training):
        x = self._conv_layer(0, x, 64, 3, 1, padding="SAME")
        x = self._bachNorm(x, is_training, "BN0")
        x = self._conv_layer(1, x, 64, 3, 1, padding="SAME")
        x = self._bachNorm(x, is_training, "BN1")
        # x = self._max_pool(x, 3, 3, "maxPool1")
        x = self._max_pool(x, 2, 2, "maxPool1")

        print(x.get_shape().as_list())

        x = self._conv_layer(2, x, 128, 3, 1, padding="SAME")
        x = self._bachNorm(x, is_training, "BN2")
        x = self._conv_layer(3, x, 128, 3, 1, padding="SAME")
        x = self._bachNorm(x, is_training, "BN3")
        x = self._max_pool(x, 2, 2, "maxPool2")
 
        print(x.get_shape().as_list())

        x = self._conv_layer(4, x, 256, 3, 1, padding="SAME")
        x = self._bachNorm(x, is_training, "BN4")
        x = self._conv_layer(5, x, 256, 3, 1, padding="SAME")
        x = self._bachNorm(x, is_training, "BN5")
        x = self._conv_layer(6, x, 256, 1, 1, padding="SAME")
        x = self._max_pool(x, 2, 2, "maxPool3")

        print(x.get_shape().as_list())
        
        x = self._conv_layer(7, x, 512, 3, 1, padding="SAME")
        x = self._bachNorm(x, is_training, "BN6")
        x = self._conv_layer(8, x, 512, 3, 1, padding="SAME")
        x = self._bachNorm(x, is_training, "BN7")
        x = self._conv_layer(9, x, 512, 1, 1, padding="SAME")
        x = self._max_pool(x, 2, 2, "maxPool4")

        print(x.get_shape().as_list())
        
        x = self._conv_layer(10, x, 512, 3, 1, padding="SAME")
        x = self._bachNorm(x, is_training, "BN8")
        x = self._conv_layer(11, x, 512, 3, 1, padding="SAME")
        x = self._bachNorm(x, is_training, "BN9")
        x = self._conv_layer(12, x, 512, 1, 1, padding="SAME")
        x = self._max_pool(x, 2, 2, "maxPool5")

        print(x.get_shape().as_list())

        x = self._fc_layer(13, x, 4096, tf.nn.relu)
        x = self._bachNorm_1D(x, is_training, "BN10")
        x = self._fc_layer(14, x, 4096, tf.nn.relu)
        x = self._bachNorm_1D(x, is_training, "BN11")
        x = self._fc_layer(15, x, 2)

        print(x.get_shape().as_list())

        return x

    def _conv_layer(self, id, x, num_filters, filter_size, stride, padding):
        with tf.variable_scope("conv_%d" % id):
            in_channels = x.get_shape().as_list()[-1]
            weights = tf.get_variable("weight", [filter_size, filter_size, in_channels, num_filters], tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())
            feature_map = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)
            output = tf.nn.relu(feature_map)

        return output

    def _fc_layer(self, id, x, num_out, activation=None):
        with tf.variable_scope("fc_%d" % id):
            shapes = x.get_shape().as_list()

            dims = 1
            for val in shapes[1:]:
                dims *= val

            x = tf.reshape(x, [-1, dims])

            in_channels = x.get_shape().as_list()[-1]

            weights = tf.get_variable("weight", [in_channels, num_out], tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())
            # biases = tf.get_variable("biases", [num_out], tf.float32,
            #                          initializer=tf.contrib.layers.xavier_initializer())

            # outputs = tf.nn.xw_plus_b(x, weights, biases)
            
            outputs = tf.matmul(x, weights) 

            if (activation):
                outputs = activation(outputs)

        return outputs

    def _max_pool(self, x, filter_size, stride, scope, padding="VALID"):
        with tf.variable_scope(scope):
            outputs = tf.nn.max_pool(x, [1, filter_size, filter_size, 1], [1, stride, stride, 1], padding=padding)

        return outputs

    def _avg_pool(self, id, x, filter_size, stride, padding="VALID"):
        with tf.variable_scope("avgPool_%d" % id):
            outputs = tf.nn.avg_pool(x, [1, filter_size, filter_size, 1], [1, stride, stride, 1], padding=padding)

        return outputs

    def _batch_norm_template(self, inputs, is_training, scope, moments_dims, bn_decay):
        with tf.variable_scope(scope) as sc:
            num_channels = inputs.get_shape()[-1].value

            beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                                name='gamma', trainable=True)

            batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')

            decay = bn_decay if bn_decay is not None else 0.9
            ema = tf.train.ExponentialMovingAverage(decay=decay)
            # Operator that maintains moving averages of variables.
             
            ema_apply_op = tf.cond(is_training, true_fn=lambda: ema.apply([batch_mean, batch_var]), false_fn=lambda: tf.no_op())

            # Update moving average and return current batch's avg and var.
            def mean_var_with_update():
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            # ema.average returns the Variable holding the average of var.
            mean, var = tf.cond(is_training,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)

        return normed

    def _bachNorm(self, x, is_training, scope, bn_decay=None):
        return self._batch_norm_template(x, is_training, scope, [0, 1, 2], bn_decay)

    def _bachNorm_1D(self, x, is_training, scope, bn_decay=None):
        return self._batch_norm_template(x, is_training, scope, [0, 1], bn_decay)

    def _softmax(self, x):
        return tf.nn.softmax(x, name="softmax")

'''
is_train = tf.placeholder(tf.bool, name='is_training')
vgg = VGG()
x = tf.Variable(tf.random_normal([4, 224, 224, 3]))
x = vgg.model(x, is_train)
ans = tf.argmax(x, 1)
print(x.get_shape().as_list())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ans = sess.run(x, {is_train: True})
    print(ans)
'''
