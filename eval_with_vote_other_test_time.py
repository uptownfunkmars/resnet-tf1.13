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


# val_base_dir = "/home/sdc/xujiping/test/TCGA-ZF-AA54-01A-01-TS1"
sample_pth = "/home/sdc/xujiping/test_patch"
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

B_MEAN = 172.49397218794172
G_MEAN = 134.6373524862326
R_MEAN = 167.8962343929455 

def get_result():
    # sample_name = os.listdir(val_base_dir)

    x = tf.placeholder(tf.float32, [None, height, width, channel], name="inputs")
    is_training = tf.placeholder(tf.bool, name="is_train")

    vgg = VGG()
    out = vgg.model(x, is_training)
    prob = tf.nn.sigmoid(out)

    y_pred = tf.cast(tf.greater(prob, 0.5), tf.float32)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_saved_path)

        ava = []
        wc = 0       # whole count
        wp = 0       # whole positive count
        wn = 0       # whole negative count
        wpp = 0      # whole predict positive
        wpn = 0      # whole negative predict

        pssc = 0
        for sample_name in os.listdir(sample_pth): 
            t1 = time()
            pc = 0                                                         # 每个病例下的被预测为TMB高的癌灶区patch个数
            pos_patch_num = 0
            patch_num = len(os.listdir(sample_pth + '/' + sample_name))
            for name in os.listdir(sample_pth + '/' + sample_name):
                img_name = sample_pth + '/' + sample_name + "/" + name
            
                # print(pth_name)


                    # print(pth_name + '/' + fname)

                img = cv.imread(img_name) 
                img = dataAugmentation(img)
                img = (img - [B_MEAN, G_MEAN, R_MEAN]) / 255.0
                if img.shape[0] != 224 : 
                    img = cv.resize(img, (224, 224))

                img = np.expand_dims(img, axis=0)

                    # print(img.shape)
 
                predict, p = sess.run([y_pred, prob], {x: img, is_training: is_train})
                
                    # print("predict : ", predict, "probability : ", p)
                    # print("labels : ", (1 if fname[:16] in highTMB else 0))
                label = (1 if name[:16] in highTMB else 0)

                wc += 1
                wp += label
                wn += (1 - label)
                wpp += int((predict == 1) and (label == 1))
                wpn += int((predict == 0) and (label == 0)) 
                pos_patch_num += predict
                
                # a = input()
                
            
                # print("postive num : %d, patch num : %d, ratio : %.5f" % (pos_patch_num, patch_num, pos_patch_num / patch_num))
                # print("predict : %d, labels : %d" % (int((pos_patch_num / patch_num) >= 0.5), int(1 if name[:16] in highTMB else 0)))
                # print("Done")

            pc = int((pos_patch_num / patch_num) >= 0.5)
            
            t2 = time()
            case_predict_result = pc
            print("case predict : ", case_predict_result, "label is : ", int(1 if sample_name[:16] in highTMB else 0)) 
            ava.append(t2 - t1)

        t = np.mean(np.array(ava))
        print("average time is : ", t)
        print("recall : ", wpp / wp)
        print("acc for patch : ", (wpp + wpn) / wc)


if __name__ == "__main__":
    get_result()




