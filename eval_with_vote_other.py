import numpy as np
import tensorflow as tf
import os
import pandas as pd
import collections
import argparse
import cv2 as cv
from time import time

from dataProcess import dataGen, dataAugmentation
from model_one import VGG

os.environ["CUDA_VISIBLE_DEVICES"] = "9"

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
lowTMB = set(e[0] for e in arr[41:])

print(arr[100][1])
print(arr[200][1])
print(arr[300][1])

print(highTMB)
print(lowTMB)

highTMB = [e + " " for e in list(highTMB)]
lowTMB = [e + " " for e in list(lowTMB)]

highTMB[-1] = highTMB[-1] + "\n"

# with open("./high_low_tmb_file_name.txt", 'w') as f :
#      f.writelines(highTMB)
#      f.writelines(lowTMB)

base_dir = "/home/sdc/xujiping_sdf/data/test_patch"
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


MEAN_B = 172.49397218794172
MEAN_G = 134.6373524862326 
MEAN_R = 167.8962343929455


def get_result():

    x = tf.placeholder(tf.float32, [None, height, width, channel], name="inputs")
    is_training = tf.placeholder(tf.bool, name="is_train")

    vgg = VGG()
    out = vgg.model(x, is_training)
    prob = tf.nn.sigmoid(out)

    y_pred = tf.cast(tf.greater(prob, 0.5), tf.float32)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_saved_path)
        
        for dir_name in os.listdir(base_dir): 
            
            test_dir = base_dir + '/' + dir_name
            file_name = os.listdir(test_dir)
        
            cur_positive_sample_count = 0
            cur_sample_count = len(file_name)

            for name in file_name:
                img = cv.imread(test_dir + '/' + name)
    
                img = dataAugmentation(img)
                img = (img - [MEAN_B, MEAN_G, MEAN_R]) / 255.0

                if img.shape[0] != 224 :
                    img = cv.resize(img, (224, 224)) 
           
                img = np.expand_dims(img, axis=0)
 
                probability, predict = sess.run([prob, y_pred], {x: img, is_training: False})
            
                print("predict is : ", predict, "labels is : ", [1 if name[:16] in highTMB else 0])

                cur_positive_sample_count += predict
         
            print("current positive sample count from predict : ", cur_positive_sample_count)
            print("current sample count : ", cur_sample_count)

# if __name__ == '__main__':
#    get_result()
