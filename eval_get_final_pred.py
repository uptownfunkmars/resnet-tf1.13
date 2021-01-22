import tensorflow as tf
import argparse
import os

from dataProcess import dataGen
from model import VGG
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
parser = argparse.ArgumentParser()

parser.add_argument("--is_training", type=bool, default=False, help="is training or not")
parser.add_argument("--height", type=int, default=256, help="input image's height")
parser.add_argument("--width", type=int, default=256, help="input image's width")
parser.add_argument("--channel", type=int, default=3, help="input image channels")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--test_data_pth", default='', help="test data saved path")
parser.add_argument("--model_saved_pth", default='', help="model saved path")

FLAGS = parser.parse_args()

is_train = FLAGS.is_training
height = FLAGS.height
width = FLAGS.width
channel = FLAGS.channel
batch_size = FLAGS.batch_size
test_data_path = FLAGS.test_data_pth
model_saved_path = FLAGS.model_saved_pth


def evaluate():
    data_generator = dataGen(test_data_path + "/0", test_data_path + "/1", batch_size)
    
    x = tf.placeholder(tf.float32, [None, height, width, channel], name="inputs")
    
    y_hat = tf.placeholder(tf.float32, [None], name="predicts")
    y_true = tf.placeholder(tf.float32, [None], name="labels")  
    
    is_training = tf.placeholder(tf.bool, name="is_train")

    vgg = VGG()
    out = vgg.model(x, is_training)
    prob = tf.nn.softmax(out)

    y_pred = tf.argmax(prob, axis=1)
        
    auc_value, auc_op = tf.metrics.auc(y_true, y_hat)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_saved_path)
        # inputs, labels = data_generator.__next__()
        
        y_hat_ = []
        y_true_ = []
        for inputs, labels in data_generator:
        #predict = sess.run(y_pred, {x: inputs, is_training: is_train})
            predict = sess.run(y_pred, {x: inputs, is_training: is_train})
            y_hat_ += list(predict)
            y_true_ += list(labels)
        #predict, _, val = sess.run([y_pred, auc_op, auc_value], {x: inputs, y_true: labels, is_training: is_train})
       
        sess.run(tf.local_variables_initializer())
        sess.run(auc_op, {y_true: y_true_, y_hat: y_hat_})
        val = sess.run(auc_value) 
        
        print(y_hat_)
        print(y_true_)
        print("AUC : ", val)
        print(metrics.accuracy_score(y_true_, y_hat_))

if __name__ == "__main__":
    evaluate()

