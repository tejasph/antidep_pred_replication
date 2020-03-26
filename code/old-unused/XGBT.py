# -*- coding: utf-8 -*-
"""
CPSC 532M project
Yihan, John-Hose, Teyden
"""

from utility import subsample
from utility import featureSelectionChi, featureSelectionELAS, drawROC, featureSelectionAgglo
from sklearn.model_selection import train_test_split
from scipy import stats
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix



def XGBTEnsemble():
    """ Train an ensemble of trees and report the accuracy as they did in the paper
    """
    param = {'nthread':5, 'booster': 'gbtree', 'max_depth':3, 'eta':0.1, 'silent':1, 'objective':'binary:logistic', 'eval_metric': 'error', 'colsample_bytree':0.8, 'lambda':0.5, 'lambda_bias': 0.5, 'subsample' : 1}
    num_round = 5
    
    pathData = 'C:/Users/jjnun/Documents/Sync/Research/1_CANBIND Replication/teyden-git/data/final_datasets/to_run_20201016/2_ExternalValidation/X_train_stard_extval.csv'
    pathLabel = 'C:/Users/jjnun/Documents/Sync/Research/1_CANBIND Replication/teyden-git/data/final_datasets/to_run_20201016/2_ExternalValidation/y_train_stard_extval.csv'
    testData = 'C:/Users/jjnun/Documents/Sync/Research/1_CANBIND Replication/teyden-git/data/final_datasets/to_run_20201016/2_ExternalValidation/X_test_cb_extval.csv'
    testLabel = 'C:/Users/jjnun/Documents/Sync/Research/1_CANBIND Replication/teyden-git/data/final_datasets/to_run_20201016/2_ExternalValidation/y_test_cb_extval.csv'
    # read data and chop the header
    X_train = np.genfromtxt(pathData, delimiter=',')[1:,1:]
    y_train = np.genfromtxt(pathLabel, delimiter=',')[1:,1]
    
    X_test = np.genfromtxt(testData, delimiter=',')[1:,1:]
    y_test = np.genfromtxt(testLabel, delimiter=',')[1:,1]
    
    _,m = X_train.shape
        
    # Feature selection
    features = range(m)
    #features = featureSelectionChi(X_train,y_train,30,100)
    #features = featureSelectionELAS(X_train,y_train,31)
    #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),50)

    # Subsampling data
    X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
    training, label = subsample(X_combined, t=30)
 
    # Train an ensemble of 30 decision trees
    clf = [None]*30
    for i in range(30):
        dtrain = xgb.DMatrix(training[i][:,features], label=label[i])      
        bst = xgb.train(param, dtrain, num_round)
        clf[i] = bst

    # Prediction for training
    m = X_train.shape[0]
    dtrain = xgb.DMatrix(X_train[:,features], label=y_train)
    # calculting the average probabilty of each class
    pred_prob = np.zeros((m,))
    for i in range(30):
        pred_prob += clf[0].predict(dtrain)
            
    pred_prob = pred_prob/30
    # Pick the class with the greastest probability to be the prediction
    pred = (pred_prob>=0.5)
    y_score = pred_prob
    
    # Report accuracy
    score = sum(pred==y_train)/m
    print("Accuracy is:", score)

    # Prediction
    n = X_test.shape[0]
    dtest = xgb.DMatrix(X_test[:,features], label=y_test)
    # calculting the average probabilty of each class
    pred_prob = np.zeros((n,))
    for i in range(30):
        pred_prob += clf[0].predict(dtest)
            
    pred_prob = pred_prob/30
    # Pick the class with the greastest probability to be the prediction
    pred = (pred_prob>=0.5)
    y_score = pred_prob
    
    # Report accuracy and draw ROC curve
    drawROC(y_test, y_score)
    #plt.savefig("../figs/XGBT-Can.eps",format="eps")
    #np.savetxt("../figs/XGBT-Can",y_score)
    score = sum(pred==y_test)/n
    print("Accuracy is:", score)
    
    tn, fp, fn, tp = confusion_matrix(y_test,pred).ravel()
    print("TN is: " + str(tn/n))
    print("FP is: " + str(fp/n))
    print("FN is: " + str(fn/n))
    print("TP IS: " + str(tp/n))
    

XGBTEnsemble()
