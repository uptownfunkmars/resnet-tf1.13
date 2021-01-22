import numpy as np
import tensorflow as tf
import os
import pandas as pd
import collections
import argparse
import cv2 as cv

from dataProcess import dataGen
from model import VGG

os.environ["CUDA_VISIBLE_DEVICES"] = "8"

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

BASE_DIR = "/home/sdf/xujiping/tmb_bladder/data/HL_data/test_patch"

parser = argparse.ArgumentParser()

parser.add_argument("--is_training", type=bool, default=False, help="is training or not")
parser.add_argument("--height", type=int, default=256, help="input image's height")
parser.add_argument("--width", type=int, default=256, help="input image's width")
parser.add_argument("--channel", type=int, default=3, help="input image channels")
parser.add_argument("--model_saved_pth", default='', help="model saved path")

FLAGS = parser.parse_args()

is_train = FLAGS.is_training
height = FLAGS.height
width = FLAGS.width
channel = FLAGS.channel
model_saved_path = FLAGS.model_saved_pth


def get_result():
    sample_name = os.listdir(BASE_DIR)

    x = tf.placeholder(tf.float32, [None, height, width, channel], name="inputs")
    is_training = tf.placeholder(tf.bool, name="is_train")

    vgg = VGG()
    out = vgg.model(x, is_training)
    prob = tf.nn.softmax(out)

    y_pred = tf.argmax(prob, axis=1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_saved_path)

        ratio = []
        ans = []
        lab = []

        for name in sample_name:
            pth_name = BASE_DIR + '/' + name
            
            print(pth_name)

            pos_patch_num = 0
            patch_num = len(os.listdir(pth_name))

            for fname in os.listdir(pth_name):
                i# print(pth_name + '/' + fname)

                img = cv.imread(pth_name + '/' + fname)
                img = np.expand_dims(img, axis=0)

                i# print(img.shape)
 
                predict = sess.run(y_pred, {x: img, is_training: is_train})
                
                # print("predict : ", predict)

                pos_patch_num += (predict == 1)

            ratio.append(pos_patch_num / patch_num)
            ans.append((pos_patch_num / patch_num) >= 0.5)
            lab.append((1 if name[:16] in highTMB else 0))
            
            print("postive num : %d, patch num : %d, ratio : %.5f" % (pos_patch_num, patch_num, pos_patch_num / patch_num))
            print("predict : %d, labels : %d" % (int((pos_patch_num / patch_num) >= 0.5), int(1 if name[:16] in highTMB else 0)))
            print("Done")
        
        acc = np.mean(np.array(ans) == np.array(lab))

        print(acc)


if __name__ == "__main__":
    get_result()




