import os
import cv2 as cv
import numpy as np

def computeMean(data_pth):
    first_dir_name = os.listdir(data_pth)
    
    r_sum = 0.0
    g_sum = 0.0
    b_sum = 0.0
    
    sample_counts = 0
    
    for fdn in first_dir_name:
        sample_counts += len(os.listdir(data_pth + '/' + fdn))
        for img_name in os.listdir(data_pth + '/' + fdn):
            abs_img_name = data_pth + '/' + fdn + '/' + img_name
        
            img = cv.imread(abs_img_name)

            b_sum += img[:, :, 0]
            g_sum += img[:, :, 1]
            r_sum += img[:, :, 2]

    return np.mean(b_sum) / sample_counts, np.mean(g_sum) / sample_counts, np.mean(r_sum) / sample_counts


def computeVar(data_pth, mean):
    b_mean, g_mean, r_mean = mean
    
    first_dir_name = os.listdir(data_pth)

    sample_counts = 0

    b_var = 0.0
    g_var = 0.0
    r_var = 0.0
    
    for fdn in first_dir_name:
        sample_counts += len(os.listdir(data_pth + '/' + fdn))
        for img_name in os.listdir(data_pth + '/' + fdn):
            abs_img_name = data_pth + '/' + fdn + '/' + img_name

            img = cv.imread(abs_img_name)

            b_var += (img[:, :, 0] - b_mean) * (img[:, :, 0] - b_mean)
            g_var += (img[:, :, 1] - g_mean) * (img[:, : ,1] - g_mean)
            r_var += (img[:, :, 2] - r_mean) * (img[:, :, 2] - r_mean)
    
    return np.sqrt(np.mean(b_var) / sample_counts), np.sqrt(np.mean(g_var) / sample_counts), np.sqrt(np.mean(r_var) / sample_counts)





if __name__ == '__main__':
    # mean_b, mean_g, mean_r = computeMean("/home/sdc/xujiping_sdf/data/train_shila_other")
    mean_b, mean_g, mean_r = computeVar("/home/sdc/xujiping_sdf/data/train_shila_other", [175.43140134634166, 125.49476437224308, 183.10653607685623])
    # mean_b, mean_g, mean_r = computeMean("/home/sdg/4class_shila_patch")
    print(mean_b)
    print(mean_g)
    print(mean_r)    
