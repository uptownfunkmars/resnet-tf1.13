import tensorflow as tf
import numpy as np


def focal_loss(labels, logits, alpha=0.25, gamma=2):
    print(labels.get_shape().as_list())
    print(logits.get_shape().as_list())

    pred = tf.nn.softmax(logits, axis=1)
    elements_number, depth = pred.get_shape().as_list()
    
    # print("elements_number : ", elements_number, " depth : ", depth)
    
    one_hot_labels = tf.one_hot(labels, depth)
    # print(one_hot_labels.get_shape().as_list())

    zeros = tf.zeros_like(pred, dtype=pred.dtype)
    ones = tf.ones_like(pred, dtype=pred.dtype)
    # print(zeros.get_shape().as_list())
    # print(ones.get_shape().as_list())

    alpha_matrix = tf.where(one_hot_labels > zeros, ones * alpha, ones * (1 - alpha))
    # print(alpha_matrix.get_shape().as_list())
    before_gamma_coefficients = tf.where(one_hot_labels > zeros, one_hot_labels - pred, pred)
    # print(before_gamma_coefficients.get_shape().as_list())
    coefficients = tf.pow(before_gamma_coefficients, gamma)
    # print(coefficients.get_shape().as_list())

    before_log_logits = tf.where(one_hot_labels > zeros, pred, one_hot_labels - pred)
    # print(before_log_logits.get_shape().as_list())
    log_logits = tf.log(tf.clip_by_value(before_log_logits, 1e-8, 1.0))
    # print(log_logits.get_shape().as_list())

    ans = tf.reduce_sum(- alpha_matrix * coefficients * log_logits, axis=1)

    return ans









