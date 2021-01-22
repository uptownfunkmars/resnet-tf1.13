import tensorflow as tf
import numpy as np


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2, epsilon=1e-6):
    positive = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    negative = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    return -alpha * tf.pow(1.0 - positive, gamma) * tf.log(positive + epsilon) - (1.0 - alpha) * tf.pow(negative, gamma) * tf.log(1.0 - negative + epsilon)






