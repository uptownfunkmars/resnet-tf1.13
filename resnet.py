import tensorflow as tf
import os 


os.environ["CUDA_VISIBLE_DEVICES"] = "9"

class ResNet(object):
    def __init__(self):
        pass

    def model(self, x, is_training):
        with tf.variable_scope("ResNet"):
            # conv1
            x = self._conv(0, x, 7, 64, 2, 3, is_training, tf.nn.relu)
            print(x.get_shape().as_list())

            # max pool 1
            x = self._max_pool(x, 3, 2, 1)
            print(x.get_shape().as_list())

            # block 1
            identity = x
            x = self._conv(1, x, 3, 64, 1, 1, is_training, tf.nn.relu)
            x = self._conv(2, x, 3, 64, 1, 1, is_training)
            x += identity
            x = self._relu(x)
            print(x.get_shape().as_list())

            # block 2
            identity = x
            x = self._conv(3, x, 3, 64, 1, 1, is_training, tf.nn.relu)
            x = self._conv(4, x, 3, 64, 1, 1, is_training)
            x += identity
            x = self._relu(x)
            print(x.get_shape().as_list())

            # block 3
            # identity = self._conv(5, x, 1, 128, 2, 0, is_training, tf.nn.relu)
            identity = self._conv(5, x, 1, 128, 2, 0, is_training)
            x = self._conv(6, x, 3, 128, 2, 1, is_training, tf.nn.relu)
            x = self._conv(7, x, 3, 128, 1, 1, is_training)
            x += identity
            x = self._relu(x)
            print(x.get_shape().as_list())

            # block 4
            identity = x
            x = self._conv(8, x, 3, 128, 1, 1, is_training, tf.nn.relu)
            x = self._conv(9, x, 3, 128, 1, 1, is_training)
            x += identity
            x = self._relu(x)
            print(x.get_shape().as_list())

            # block 5
            # identity = self._conv(10, x, 1, 256, 2, 0, is_training, tf.nn.relu)
            identity = self._conv(10, x, 1, 256, 2, 0, is_training)
            x = self._conv(11, x, 3, 256, 2, 1, is_training, tf.nn.relu)
            x = self._conv(12, x, 3, 256, 1, 1, is_training)
            x += identity
            x = self._relu(x)
            print(x.get_shape().as_list())

            # block 6
            identity = x
            x = self._conv(13, x, 3, 256, 1, 1, is_training, tf.nn.relu)
            x = self._conv(14, x, 3, 256, 1, 1, is_training)
            x += identity
            x = self._relu(x)
            print(x.get_shape().as_list())

            # block 7
            # identity = self._conv(15, x, 1, 512, 2, 0, is_training, tf.nn.relu)
            identity = self._conv(15, x, 1, 512, 2, 0, is_training)
            x = self._conv(16, x, 3, 512, 2, 1, is_training, tf.nn.relu)
            x = self._conv(17, x, 3, 512, 1, 1, is_training)
            x += identity
            x = self._relu(x)
            print(x.get_shape().as_list())

            # block 8
            identity = x
            x = self._conv(18, x, 3, 512, 1, 1, is_training, tf.nn.relu)
            x = self._conv(19, x, 3, 512, 1, 1, is_training)
            x += identity
            x = self._relu(x)
            print(x.get_shape().as_list())
            
            last_feature = x
            
            # avg pool
            x = self._avg_pool(x)
            print(x.get_shape().as_list())

            return x, last_feature
            # return x

    def _conv(self, idx, x, kernel_size, number_filters, stride, pad, is_training, activation=None):
        with tf.variable_scope("conv_%d" % idx):
            in_dims = x.get_shape().as_list()[-1]
            weights = tf.get_variable("weights", [kernel_size, kernel_size, in_dims, number_filters], tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())

            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], "CONSTANT", name="padding", constant_values=0)
            x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], "VALID")

            x = self._batchNorm_2D(x, is_training)

            if activation is not None:
                x = activation(x)

            return x

    def fc_layer(self, x, out_dims, is_training, activation=None):
        with tf.variable_scope("fc"):
            shapes = x.get_shape().as_list()[-1]

            # dims = 1
            # if len(shapes) > 1:
            #     for val in shapes[1:]:
            #         dims *= val
            # else:
            dims = shapes

            x = tf.reshape(x, [-1, dims])

            in_dims = x.get_shape().as_list()[-1]

            weights = tf.get_variable("weights", [in_dims, out_dims], tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())
            outputs = tf.matmul(x, weights)

            # outputs = self._batchNorm_1D(outputs, is_training)

            return outputs

    def _relu(self, x):
        return tf.nn.relu(x)

    def _max_pool(self, x, kernel_size, stride, pad):
        x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'CONSTANT', name='padding', constant_values=0)
        return tf.nn.max_pool(x, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1], padding="VALID")
    
    def _avg_pool(self, x):
        shape = x.get_shape().as_list()
        kernel_size = shape[1]
        return tf.nn.avg_pool(x, [1, kernel_size, kernel_size, 1], [1, 1, 1, 1], padding="VALID")

    def _batch_norm_template(self, x, is_training, moments_dims, bn_decay):
        num_channels = x.get_shape().as_list()[-1]

        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]), name='gamma', trainable=True)

        batch_mean, batch_var = tf.nn.moments(x, moments_dims, name='moments')

        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        ema_apply_op = tf.cond(is_training, true_fn=lambda: ema.apply([batch_mean, batch_var]), false_fn=lambda: tf.no_op())

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

        return normed

    def _batchNorm_2D(self, x, is_training, bn_decay=None):
        return self._batch_norm_template(x, is_training, [0, 1, 2], bn_decay)

    def _batchNorm_1D(self, x, is_training, bn_decay=None):
        return self._batch_norm_template(x, is_training, [0, 1], bn_decay)

    def _softmax(self, x):
        return tf.nn.softmax(x, name='softmax')

'''
resnet = ResNet()
#
x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='inputs')
is_training = tf.placeholder(tf.bool, name='is_training')
#
last_feature = resnet.model(x, is_training)
before_softmax = resnet.fc_layer(last_feature, 1000, is_training)
print(before_softmax.get_shape().as_list())
#
import numpy as np
#
img = np.random.random([1, 224, 224, 3])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    before_prob = sess.run(before_softmax, {x: img, is_training: True})
    print(before_prob.shape)

'''
