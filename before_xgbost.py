import numpy as np
import tensorflow as tf
import os
import pandas as pd
import collections
import argparse
import cv2 as cv
from time import time
import json

from dataProcess import dataGen, dataAugmentation
from resnet import ResNet



os.environ["CUDA_VISIBLE_DEVICES"] = "9"
'''
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
'''

# val_base_dir = "/home/sdg/pangguangshila-FZ-patch"
val_base_dir = "/home/sdd/TCGA-SHILA-patch"
json_sav_dir = "/home/sdd"

# val_base_dir = "/home/sdc/xujiping_sdf/data/test_patch"
# val_base_dir = "/home/sdd/bladder_clinical_shila_14_patch"
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


MEAN_B = 175.43140134634166
MEAN_G = 125.49476437224308
MEAN_R = 183.10653607685623

VAR_B = 41.6515466570118
VAR_G = 59.2026254092725
VAR_R = 48.2724524058145


def get_result(val_base_dir, json_sav_dir):
    '''
       val_base_dir : patch saved dir
       json_sav_dir : json file saved dir
    '''
    val_base_pth = val_base_dir
    sample_name = os.listdir(val_base_dir)

    x = tf.placeholder(tf.float32, [None, height, width, channel], name="inputs")
    is_training = tf.placeholder(tf.bool, name="is_train")

    resnet = ResNet()
    last_feature = resnet.model(x, is_training)
    before_prob = resnet.fc_layer(last_feature, 1, is_training)
    prob = tf.nn.sigmoid(before_prob)

    
    my_dict = dict()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_saved_path)

        cnt = 0
        for name in sample_name:
            pth_name = val_base_dir + "/" + name
            prob_list = []
            t1 = time()
            
            for fname in os.listdir(pth_name):
                img = cv.imread(pth_name + '/' + fname)
            
                img = dataAugmentation(img)  
                img = (img - [MEAN_B, MEAN_G, MEAN_R]) / [VAR_B, VAR_G, VAR_R]
                if img.shape[0] != 224 : img = cv.resize(img, (224, 224))

                img = np.expand_dims(img, axis=0)

                p = sess.run(prob, {x: img, is_training: is_train})                
                p = np.squeeze(p) 

                prob_list.append(float(p))                

            t2 = time()
    
            print("%d : predict all patch spend %.5f second" % (cnt, (t2 - t1)))
            print(len(prob_list))
            print(type(prob_list)) 
            my_dict[name] = prob_list
            cnt += 1

        with open(json_sav_dir + '/prob_TCGA.json', "w") as f:
            json.dump(my_dict, f)
    
    return

if __name__ == "__main__":
    get_result(val_base_dir, json_sav_dir)




