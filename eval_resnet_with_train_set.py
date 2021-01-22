import numpy as np
import tensorflow as tf
import os
import pandas as pd
import collections
import argparse
import cv2 as cv
from time import time

from dataProcess_v import *
from resnet import ResNet

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

df = pd.DataFrame(pd.read_csv('../tsv_data/TCGA-BLCA.muse_snv.tsv', sep='	'))

print(df.head())

# 筛选出有害突变
samples = []
sampleDic = dict()
for i in range(len(df['Sample_ID'])):
    if df['filter'][i] == 'PASS' and ('coding_sequence_variant' in df['effect'][i]
    or 'frameshift_variant' in df['effect'][i]
    or 'inframe_' in df['effect'][i]
    or 'missense_variant' in df['effect'][i]
    or 'splice_' in df['effect'][i]
    or 'start_' in df['effect'][i]
    or 'stop_' in df['effect'][i]):
        samples.append(df['Sample_ID'][i])
        if not sampleDic.__contains__(df['Sample_ID'][i]):
            sampleDic[df['Sample_ID'][i]] = len(sampleDic)

# 对突变数目计数
c = dict(collections.Counter(samples))
for k in c.keys():
    c[k] /= 36
arr = list(zip(c.keys(), c.values()))
arr.sort(key = lambda x: x[1], reverse = True)

# 得到前41个病例
highTMB = set(e[0] for e in arr[:41])
print(highTMB)


# val_base_dir = "/home/sdd/bladder_clinical_shila_14_patch"
pos_pth = "/home/sdc/xujiping_sdf/data/train_shila/1"
neg_pth = "/home/sdc/xujiping_sdf/data/train_shila/0"
# val_base_dir = "/home/sdc/xujiping/train"
# sample_pth = "/home/sdc/xujiping/test"
# val_base_dir = "/home/sdf/xujiping/tmb_bladder/data/HL_data/test/TCGA-XF-AAN0-01A-01-TSA"

parser = argparse.ArgumentParser()

parser.add_argument("--is_training", type=bool, default=False, help="is training or not")
parser.add_argument("--height", type=int, default=224, help="input image's height")
parser.add_argument("--width", type=int, default=224, help="input image's width")
parser.add_argument("--channel", type=int, default=3, help="input image channels")
parser.add_argument("--model_saved_pth", default='', help="model saved path")

FLAGS = parser.parse_args()

is_train = FLAGS.is_training
height = FLAGS.height
width = FLAGS.width
channel = FLAGS.channel
model_saved_path = FLAGS.model_saved_pth


MEAN_B = 175.44276692465525
MEAN_G = 125.3624421292259
MEAN_R = 182.86357662647927

def get_result():
    # sample_name = os.listdir(val_base_dir)

    x, y, idx = dataShuffle(pos_pth, neg_pth)
    vg = valGen(x, y, idx, 64)
    
    x = tf.placeholder(tf.float32, [None, height, width, channel], name="inputs")
    is_training = tf.placeholder(tf.bool, name="is_train")

    resnet = ResNet()
    last_feature = resnet.model(x, is_training)
    # before_prob = resnet.fc_layer(last_feature, 2, is_training)
    before_prob = resnet.fc_layer(last_feature, 1, is_training)
    # prob = tf.nn.softmax(before_prob, axis=1, name="prob")
    prob = tf.nn.sigmoid(before_prob)

    # y_pred = tf.argmax(prob, axis=1)
    y_pred = prob > 0.5

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_saved_path)

        inp, lab = vg.__next__()
        pred = sess.run(y_pred, {x: inp, is_training: False})
        
        pred = np.squeeze(pred)      
        print(lab)
        print(pred.astype(int))
        
if __name__ == "__main__":
    get_result()



