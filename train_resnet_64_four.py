import tensorflow as tf
import numpy as np
import argparse
import os

from dataProcess_4class import dataShuffle_4class, dataGen, valGen
from resnet import ResNet
from focal_loss import focal_loss

parser = argparse.ArgumentParser()

parser.add_argument('--is_training', type=bool, default=True, help='training or testing')
parser.add_argument('--height', type=int, default=224, help='input images height')
parser.add_argument('--width', type=int, default=224, help='input images width')
parser.add_argument('--channels', type=int, default=3, help='input images channels')
parser.add_argument('--batch_size', type=int, default=128, help='input images channels')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate base')
parser.add_argument('--epochs', type=int, default=10, help='how many epoch to train')
parser.add_argument('--moving_average_decay', type=float, default=0.99, help='moving average decay rate')
parser.add_argument('--regular_rate', type=float, default=5e-4, help='l2 regularization decay rate')
# parser.add_argument('--train_positive_pth', default='', help='saved negative samples path [default: NONE]')
# parser.add_argument('--train_negative_pth', default='', help='saved positive samples path [default: None]')
parser.add_argument('--data_pth', default='', help='saved negative samples path [default: NONE]')
parser.add_argument('--model_save_pth', default='', help='model save path [default: None]')
parser.add_argument('--log_dir', default='', help='saved log file path')
parser.add_argument('--gpu_number', default='', help='can used gpu idx')

FLAGS = parser.parse_args()


is_train = FLAGS.is_training
height = FLAGS.height
width = FLAGS.width
channels = FLAGS.channels
batch_size = FLAGS.batch_size
lr = FLAGS.lr
epochs = FLAGS.epochs
moving_average_decay = FLAGS.moving_average_decay
regular_rate = FLAGS.regular_rate
# positive_path = FLAGS.train_positive_pth
# negative_path = FLAGS.train_negative_pth
# val_positive_path = FLAGS.val_positive_pth
# val_negative_path = FLAGS.val_negative_pth
data_path = FLAGS.data_pth
model_saved_path = FLAGS.model_save_pth
log_saved_path = FLAGS.log_dir
gpu_number = FLAGS.gpu_number


os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number


# 数据预处理

# 搭建模型
# 迁移学习
'''
: 将部分Variable加入到指定集合中
    tf.add_to_collection("set name", tensor) 

: 获取集合中的全部元素
    tf.get_collection("set name")

: 将n个张量相加
    tf.add_n(tensor)  
'''
'''
:生成器定义数据（数据生成器的编写）
    yield关键字
'''
# 模型保存注意联系后续部署的问题
# 使用tf.train.Saver() 保存
# restore到session()中，使用tf.saved_model等函数保存为.pb后缀方便后续部署；
'''
: 正则化
    regularize = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    tf.add_to_collection("set name", regularize(tensor))
    l2_loss = tf.add_n(tf.get_collection("set name"))
'''
# 梯度裁剪
'''
: 参数使用argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=int, default=default_value, help)
    FLAGS = parser.parse_args()

    name = FLAGS.name
'''
# 评价指标的计算包括auc，以及折线图的绘制
# mAP等指标
'''
: tensorflow计算acc
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=1), y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''

# tensorboard 监控
# 多卡并行训练

# 剪枝，压缩，蒸馏，量化
# 部署
# C++部署


def train():
    # pf = os.listdir(positive_path)
    # nf = os.listdir(negative_path)

    # num_examples = int((len(pf) + len(nf)) * 0.7)

    num_examples = 0
    
    for dir_name in os.listdir(data_path):
        num_examples += len(os.listdir(data_path + '/' + dir_name))


    x = tf.placeholder(tf.float32, [None, height, width, channels], name="inputs")
    y_true = tf.placeholder(tf.int64, [None], name="labels")
    # y_true = tf.placeholder(tf.float32, [None], name="labels")
    is_training = tf.placeholder(tf.bool, name="is_train")

    # forward
    resnet = ResNet()
    last_feature = resnet.model(x, is_training)
    before_prob = resnet.fc_layer(last_feature, 4, is_training)
    prob = tf.nn.softmax(before_prob, axis=1, name="prob")
    # prob = tf.nn.sigmoid(before_prob, name="prob")

    print("prob shape : ", prob.get_shape().as_list())
    print("y_true shape : ", y_true.get_shape().as_list())
    # compute acc
    # y_pred = tf.where(prob > 0.5, True, False)
    # y_pred = tf.cast(tf.greater(prob, 0.5), tf.float32)
    y_pred = tf.argmax(prob, axis=1)
    correct_prediction = tf.equal(y_pred, y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('acc', accuracy)

    # loss function
    # labels = tf.expand_dims(y_true, axis=1)
    # cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels, logit, 9)
    
    # before_prob = tf.squeeze(before_prob)
    # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=before_prob)
    # prob = tf.squeeze(prob, axis=1) 
    # print("prob shape : ", prob.get_shape().as_list())

    # cross_entropy = focal_loss(y_true, prob) 
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=before_prob)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
   
    # print([v for v in tf.trainable_variables()])
 
    l2_loss = regular_rate * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])

    loss = cross_entropy_mean + l2_loss

    tf.summary.scalar('loss', loss)

    # global step
    global_step = tf.Variable(0, trainable=False)

    # exponential moving average
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)

    # update weight using moving average
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # learning rate exponential decay
    learning_rate = tf.train.exponential_decay(lr,
                                               global_step,
                                               num_examples // batch_size
                                               , 1, staircase=True)
    
    tf.summary.scalar('learning_rate', learning_rate)
    
    # Passing global_step to minimize() will increment it at each step.
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss, global_step=global_step)
    # train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    merged = tf.summary.merge_all()
    # merge train step and variables averages op
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # model save
    
    saver = tf.train.Saver(max_to_keep=20)

    with tf.Session() as sess:
        saver.restore(sess, "/home/sdc/xujiping_sde/saved_model_resnet_four/epoch_s1_9.ckpt")
        # sess.run(tf.global_variables_initializer())
        summary_writer_t = tf.summary.FileWriter(log_saved_path, sess.graph)

        inpu, outp, idx = dataShuffle_4class(data_path)
        # data_generator = dataGen(inpu, outp, idx, batch_size)
        val_generator = valGen(inpu, outp, idx, batch_size)

        for epoch in range(epochs):
            iteration = 0

            # inpu, outp, idx = dataShuffle(positive_path, negative_path)
            # data_generator = dataGen(inpu, outp, idx, batch_size)
            # val_generator = valGen(inpu, outp, idx, batch_size)
            data_generator = dataGen(inpu, outp, idx, batch_size)
            val_generator = valGen(inpu, outp, idx, batch_size)

            for inputs, labels in data_generator:

                probability, loss_value, acc_value, summary, step, clr, _ = sess.run(
                    [before_prob, loss, accuracy, merged, global_step, learning_rate, train_op],
                    {x: inputs, y_true: labels, is_training: is_train}
                )


                # print(probability)
                print("[epoch : %2d / iter : %5d] loss: %.5f acc: %.5f lr: %.5f" % (epoch, iteration, loss_value, acc_value, clr))           

                summary_writer_t.add_summary(summary, step)
                
                iteration += 1                

            # validation
            average_acc = []
            for vi, vl in val_generator:
                # val_inputs, val_labels = val_data_generator.__next__()
                va = sess.run(accuracy, {x: vi, y_true: vl, is_training: is_train})
                average_acc.append(va)  
            print("validation acc : ", np.mean(np.array(average_acc)))
            print("Saving model.....")
            if ((epoch + 1) % 10 == 0) : saver.save(sess, model_saved_path + "/epoch_s1_%d.ckpt" % epoch)

    summary_writer_t.close()


if __name__ == "__main__":
    train()
