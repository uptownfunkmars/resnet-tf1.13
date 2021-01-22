import os
import json
import pandas as pd
import joblib
import random
import collections

from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

import numpy as np


def gen_histogram(json_pth, norm_hist=True):
    idx = []
    feature = []
    
    # p_feature = []
    # n_feature = []

    with open(json_pth, 'r') as f:
        load_dict = json.load(f)

    for nde in load_dict:
        if len(load_dict[nde]) == 0 : continue

        hist, bins = np.histogram(load_dict[nde], bins=200, range=(0, 1))
        idx.append(nde)
        feature.append(list(hist) if not norm_hist else list(hist / np.sum(hist)))

    return np.array(idx), np.array(feature)


def gen_jiezhichang_label_for_randomforest(idx, feature):
    jiechang = pd.DataFrame(pd.read_csv('/home/sda/jiezhichang_origin/csv/COAD111_tmb_jiechang.csv'))
    zhichang = pd.DataFrame(pd.read_csv('/home/sda/jiezhichang_origin/csv/READ111_tmb_zhichang.csv'))

    jiechang_tmb_low = list(jiechang['sample'][jiechang['tmb'] < 20])
    jiechang_tmb_high = list(jiechang['sample'][jiechang['tmb'] >= 20])

    zhichang_tmb_low = list(zhichang['sample'][zhichang['tmb'] < 20])
    zhichang_tmb_high = list(zhichang['sample'][zhichang['tmb'] >= 20])

    highTMB = zhichang_tmb_high + jiechang_tmb_high
    lowTMB = zhichang_tmb_low + jiechang_tmb_low 
   
    print(len(highTMB))
    print(len(lowTMB))
 
    label = []

    pos_cnt = 0
    neg_cnt = 0

    for name, fea in zip(idx, feature) :
        # print(fea.shape)
        if name[:16] in highTMB :
            label.append(1)
            pos_cnt += 1
        elif name[:16] in lowTMB :
            label.append(0)
            neg_cnt += 1

    print("positive sample number : ", pos_cnt)
    print("negative sample number : ", neg_cnt)

    return np.array(label)



def gen_label_for_randomforest(idx, feature):
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

   # 得到前#个病例
    highTMB = set(e[0][:12] for e in arr[:41]) # the number of high TMB cases are calculated from R script
    lowTMB = set(e[0][:12] for e in arr[41:])#low tmb的个数，[41:],41及之后的都是低级别

    print("highTMB: ", highTMB)
    print("lowTMB: ", lowTMB)#{'TCGA-YF-AA3M-01A', 'TCGA-BL-A13I-01A',,,,,,,,,,}

    label = []
 
    pos_cnt = 0
    neg_cnt = 0

    for name, fea in zip(idx, feature) :
        # print(fea.shape)
        if name[:12] in highTMB :
            label.append(1)
            pos_cnt += 1
        elif name[:12] in lowTMB : 
            label.append(0)
            neg_cnt += 1

    print("positive sample number : ", pos_cnt)
    print("negative sample number : ", neg_cnt)

    return np.array(label) 
 

def gen_jiezhichang_label_for_xgboost(idx, feature):
    jiechang = pd.DataFrame(pd.read_csv('/home/sda/jiezhichang_origin/csv/COAD111_tmb_jiechang.csv'))
    zhichang = pd.DataFrame(pd.read_csv('/home/sda/jiezhichang_origin/csv/READ111_tmb_zhichang.csv'))

    jiechang_tmb_low = list(jiechang['sample'][jiechang['tmb'] < 20])
    jiechang_tmb_high = list(jiechang['sample'][jiechang['tmb'] >= 20])

    zhichang_tmb_low = list(zhichang['sample'][zhichang['tmb'] < 20])
    zhichang_tmb_high = list(zhichang['sample'][zhichang['tmb'] >= 20])
    
    highTMB = zhichang_tmb_high + jiechang_tmb_high
    lowTMB = zhichang_tmb_low + jiechang_tmb_low

    print(len(highTMB))
    print(len(lowTMB))

    p_label = []
    n_label = []

    p_feature = []
    n_feature = []

    pos_cnt = 0
    neg_cnt = 0

    for name, fea in zip(idx, feature) :
        # print(fea.shape)
        if name[:16] in highTMB :
            p_feature.append(fea)
            p_label.append(1)
            pos_cnt += 1
        elif name[:16] in lowTMB :
            n_feature.append(fea)
            n_label.append(0)
            neg_cnt += 1

    print("positive sample number : ", pos_cnt)
    print("negative sample number : ", neg_cnt)

    return p_feature, n_feature, p_label, n_label

 

def gen_label_for_xgboost(idx, feature):
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

   # 得到前#个病例
    highTMB = set(e[0][:12] for e in arr[:41]) # the number of high TMB cases are calculated from R script
    lowTMB = set(e[0][:12] for e in arr[41:])#low tmb的个数，[41:],41及之后的都是低级别

    print("highTMB: ", highTMB)
    print("lowTMB: ", lowTMB)#{'TCGA-YF-AA3M-01A', 'TCGA-BL-A13I-01A',,,,,,,,,,}

    p_label = []
    n_label = []

    p_feature = []
    n_feature = []

    pos_cnt = 0
    neg_cnt = 0

    for name, fea in zip(idx, feature) :
        # print(fea.shape)
        if name[:12] in highTMB :
            p_feature.append(fea)
            p_label.append(1)
            pos_cnt += 1
        elif name[:12] in lowTMB :
            n_feature.append(fea)
            n_label.append(0)
            neg_cnt += 1

    print("positive sample number : ", pos_cnt)
    print("negative sample number : ", neg_cnt)

    return p_feature, n_feature, p_label, n_label


def gen_test(idx, feature):
    # p_feature, n_feature, p_label, n_label = gen_label_for_xgboost(idx, feature)
    p_feature, n_feature, p_label, n_label = gen_jiezhichang_label_for_xgboost(idx, feature)

    x_test = p_feature[-6:] + n_feature[-6:]
    y_test = p_label[-6:] + n_label[-6:]
    test = list(zip(x_test, y_test))
    random.shuffle(test)
    x_test, y_test = zip(*test)

    return np.array(x_test), np.array(y_test) 
 

def gen_train(idx, feature):
    # p_feature, n_feature, p_label, n_label = gen_label_for_xgboost(idx, feature)
    p_feature, n_feature, p_label, n_label = gen_jiezhichang_label_for_xgboost(idx, feature)

    pos = p_feature[:-6]
    batch_size = len(pos)
    start = 0
    for i in range((len(n_feature) - 6) // batch_size) :
        x_train = pos + n_feature[start : start + batch_size]
        y_train = p_label + n_label[start : start + batch_size]
 
        start += batch_size

        train = list(zip(x_train, y_train))
        random.shuffle(train)
        x_train, y_train = zip(*train)         

        yield np.array(x_train), np.array(y_train) 


def train_and_sav_randomforest(idx, feature, label=None):
    x_train, x_test, y_train, y_test = train_test_split(feature, label, random_state=1)
    
    # rf0 = RandomForestClassifier(oob_score=True, random_state=10)
    
    # rf0.fit(x_train, y_train)
    # print(rf0.oob_score_)
    
    # param_test1 = {'n_estimators':range(10,100, 10)}
    # gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
    #                         min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10), 
    #                         param_grid = param_test1, scoring='roc_auc',cv=5)
    # gsearch1.fit(x_train, y_train)
    # print(gsearch1.best_params_)

    # param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
    # gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 10, 
    #                         min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10),
    # param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
    # gsearch2.fit(x_train, y_train)
    # print(gsearch2.best_params_)

    # param_test3 = {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}
    # gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 10, max_depth=3,
    #                         max_features='sqrt' ,oob_score=True, random_state=10),
    # param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
    # gsearch3.fit(x_train, y_train)
    # print(gsearch3.best_params_)

    # param_test4 = {'max_features':range(3,11,2)}
    # gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 10, max_depth=3, min_samples_split=120,
    #                         min_samples_leaf=40 ,oob_score=True, random_state=10),
    # param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
    # gsearch4.fit(x_train,y_train)
    # print(gsearch4.best_params_)

    rf2 = RandomForestClassifier(n_estimators= 10, max_depth=3, min_samples_split=120,
                                 min_samples_leaf=40,max_features=9 ,oob_score=True, random_state=10)
    rf2.fit(x_train, y_train)
    print(rf2.oob_score_)


    # rf0.fit(x_train, y_train)
    # print(rf0.oob_score_)
    
    y_test_prob = rf2.predict_proba(x_test)[:, 1]
    # y_pred = rf2.predict(x_test)
    print(y_test)
    print(y_test_prob)
    print("Auc score(test) : %f" % metrics.roc_auc_score(y_test, y_test_prob))
    
    # joblib.dump(rf0, 'randomforest4TMB.pkl')
    
    return


def train_and_sav_xgboost(idx, feature, label=None):
    # x_train, x_test, y_train, y_test = train_test_split(feature, label, random_state=1)
    
    x_test, y_test = gen_test(idx, feature)
    gentrain = gen_train(idx, feature)
   
    bs = -1
    cnt = 0 
    for x_train, y_train in gentrain:
    
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        print(y_test)
        # model = XGBClassifier(max_depth=5, learning_rate=0.5, verbosity=1, objective='binary:logistic', random_state=1)

        # gsCv = GridSearchCV(model, {'max_depth': [2, 3, 4], 'n_estimators': [5,6,7,8]})

        # gsCv.fit(x_train, y_train, early_stopping_rounds=10, eval_metric="error", eval_set=[(x_test, y_test)])
        # model.fit(x_train, y_train)

        # print(gsCv.best_score_)
        # print(gsCv.best_params_)

        # model_1 = XGBClassifier(max_depth=2, n_estimators=5, learning_rate=0.5, verbosity=1, objective='binary:logistic', random_state=1) 
        
        # gsCv_1 = GridSearchCV(model_1, {'learning_rate' : [0.1, 0.2, 0.3]})
        
        # gsCv_1.fit(x_train, y_train)
        # gsCv_1.fit(x_train, y_train, early_stopping_rounds=10, eval_metric="error", eval_set=[(x_test, y_test)])
        
        # print(gsCv_1.best_score_)
        # print(gsCv_1.best_params_) 

        model_2 = XGBClassifier(max_depth=2, n_estimators=5, learning_rate=0.1, verbosity=1, objective='binary:logistic', random_state=1) 
        model_2.fit(x_train, y_train, early_stopping_rounds=10, eval_metric="error", eval_set=[(x_test, y_test)])
    
        # model_2 = RandomForestClassifier() 
        # model_2.fit(x_train, y_train, eval_set=[(x_test, y_test)])

        # if model_2.best_score_ > bs :
        #     joblib.dump(model_2, 'xgboost4TMB_%d.pkl' % cnt)
        #     bs = model_2.best_score_
        #     cnt += 1

        y_predprob = model_2.predict_proba(x_test)[:,1]
        print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))
        
        # joblib.dump(model_2, 'xgboost4TMB_%d.pkl' % cnt)
        # cnt += 1

    return


def eval_xgboost(json_pth, model_sav_pth):
    idx, hist = gen_histogram(json_pth)

    model = XGBClassifier()
    model = joblib.load(model_sav_pth)

    pred = model.predict(hist)
    pred_prob = model.predict_proba(hist)[:, 1]
    
    print(pred)
    print(pred_prob)

    return


def eval_randomforest(json_pth, model_sav_pth):
    idx, hist = gen_histogram(json_pth) 

    model = joblib.load(model_sav_pth)

    pred = model.predict(hist)
    pred_prob = model.predict_proba(hist)[:, 1]

    print(pred)
    print(pred_prob)


if __name__ == '__main__':
    idx, feature = gen_histogram("/home/sdc/xujiping_sdf/zzc/xyhomework/jiezhichang_patch_prob.txt")
    print(idx.shape)
    print(feature.shape)
    
    # train random forest
    # label = gen_label_for_randomforest(idx, feature)

    # idx = os.listdir('/home/sda/jiezhichang_origin/svs/jiezhichang')
    # feature = [i for i in range(len(idx))]

    # label = gen_jiezhichang_label_for_randomforest(idx, feature)
    
    # print(label) 
    # print(label.shape)
    # train_and_sav_randomforest([], feature, label)
    
    # train xgboost
    train_and_sav_xgboost(idx, feature)
    
    # eval_xgboost("/home/sdd/prob_TMBH.json", "xgboost4TMB.pkl")
    # eval_randomforest("/home/sdd/prob_TCGA.json", "randomforest4TMB.pkl")

