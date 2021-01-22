import os
import random
import numpy as np
import cv2


MEAN_B = 173.6302674305024
MEAN_G = 122.40541996506789
MEAN_R = 181.29120726077727

def dataAugmentation(img):
    # 旋转flip
    # 平移
    # 缩放
    # 滤波 考虑
    num = np.random.random()
    if num < 0.2 :
        img = imgRotate(img, 90)
    elif num < 0.4 :
        img = imgRotate(img, 180)
    elif num < 0.6 :
        img = imgRotate(img, 270)
    elif num < 0.8 :
        img = imgFlip(img)
    else :
        img = img
    
    
    # prob = np.random.random()
    # if prob < 0.2:
    #     img = inverseColor(img)
    
    # prob_1 = np.random.random()
    # if prob_1 < 0.2:
    #     img = gaussianBlur(img)
    # else:
    #    img = bilateralBlur(img)    

    return img


def inverseColor(img):
    img = 255 - img
    return img


def gaussianBlur(img):
    return cv2.GaussianBlur(img, (5, 5), 1.5)


def bilateralBlur(img):
    return cv2.bilateralFilter(img, 9, 75, 75)


def imgFlip(img):
    if np.random.random() < 0.25:
        img = cv2.flip(img, 0)
    elif np.random.random() > 0.85:
        img = cv2.flip(img, 1)

    return img


def imgRotate(img, angle):
    shape = img.shape

    h = shape[0]
    w = shape[1]
    m_rotate = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)

    img = cv2.warpAffine(img, m_rotate, (w, h))

    return img


def colorJitter(img):
    pass


def dataShuffle(positive_path, negative_path):
    x = []
    y = []

    positiveFileName = os.listdir(positive_path)
    negativeFileName = os.listdir(negative_path)

    for pfn in positiveFileName:
        absPth = positive_path + "/" + pfn

        # img = cv2.imread(absPth)
        # img = dataAugmentation(img)

        x.append(absPth)
        y.append(1)

    for nfn in negativeFileName:
        absPth = negative_path + "/" + nfn

        # img = cv2.imread(absPth)
        # img = dataAugmentation(img)

        x.append(absPth)
        y.append(0)

    assert len(x) == len(y), "length a and length b not equal!"

    idx = [i for i in range(len(x))]
    
    random.shuffle(idx)
    random.shuffle(idx)
    random.shuffle(idx)
    
    return x, y, idx


def dataGen(x, y, idx, batch_size, reshuffle=True):
    lent = int(len(x) * 0.7)
    idx = idx[:lent]   
    
    if reshuffle : 
        random.shuffle(idx)
        print("train set reshuffle done!")
 
    base = 0
    for i in range((len(idx) // batch_size)):
        inputs = []
        inputs_name = []
        labels = []        

        inputs_idx = idx[base:base + batch_size]
        
        for j in inputs_idx:
            inputs_name.append(x[j])
            labels.append(y[j])
        
 
        for j in inputs_name:
            img = cv2.imread(j)
           
            img = dataAugmentation(img)
            img = (img - [MEAN_B, MEAN_G, MEAN_R]) / 255.0           
 
            if img.shape[0] != 224 :
                img = cv2.resize(img, (224, 224))
            
            inputs.append(img)           
                        
            # img = dataAugmentation(img)
            # inputs.append(img * 1.0 / 255.0)
            # inputs.append(img)
        
        base += batch_size
        
        yield np.array(inputs), np.array(labels)


def valGen(x, y, idx, batch_size, reshuffle=False):
    lent = int(len(x) * 0.7)
    idx = idx[lent:]

    if reshuffle : 
        random.shuffle(idx)
        print("validation set reshuffle done !")

    base = 0
    for i in range((len(idx) // batch_size)):
        inputs = []
        inputs_name = []
        labels = []
 
        inputs_idx = idx[base:base + batch_size]

        for j in inputs_idx:
            inputs_name.append(x[j])
            labels.append(y[j])


        for j in inputs_name:
            img = cv2.imread(j)

            img = dataAugmentation(img)
            img = (img - [MEAN_B, MEAN_G, MEAN_R]) / 255.0

            if img.shape[0] != 224 :
                img = cv2.resize(img, (224, 224))
                  
            inputs.append(img)

            # img = dataAugmentation(img)
            # inputs.append(img * 1.0 / 255.0)
            # inputs.append(img)
 
        base += batch_size

        yield np.array(inputs), np.array(labels)


def dataShuffle_4class(data_pth):
    x = []
    y = []

    for dir_name in os.listdir(data_pth):
        for name in os.listdir(data_pth + '/' + dir_name):
            x.append(data_pth + '/' + dir_name + '/' + name)
            y.append(int(dir_name))
  
    idx = [e for e in range(len(x))]

    random.shuffle(idx)
    random.shuffle(idx)
    random.shuffle(idx)

    return x, y, idx


'''
x, y, idx = dataShuffle_4class("/home/sdg/4class_shila_patch")
dg = valGen(x, y, idx, 32)
for inputs, labels in dg :
    print(inputs)
    print(labels)
    break
print(len(x))
'''
