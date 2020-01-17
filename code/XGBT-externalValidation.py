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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import numpy as np


def XGBTEnsemble():
    """ Train an ensemble of trees and report the accuracy as they did in the paper
    """
    param = {'nthread':5, 'booster': 'gbtree', 'max_depth':3, 'eta':0.1, 'silent':1, 'objective':'binary:logistic', 'eval_metric': 'error', 'colsample_bytree':0.8, 'lambda':0.5, 'lambda_bias': 0.5, 'subsample' : 1}
    num_round = 5
    
    trainData = r'C:\Users\y374zhou\Documents\GitHub\antidep-project\code\data\X_train_stard_extval.csv'
    trainLabel = r'C:\Users\y374zhou\Documents\GitHub\antidep-project\code\data\y_train_stard_extval.csv'
    testData = r'C:\Users\y374zhou\Documents\GitHub\antidep-project\code\data\X_test_cb_extval.csv'
    testLabel = r'C:\Users\y374zhou\Documents\GitHub\antidep-project\code\data\y_test_cb_extval.csv'
    # read data and chop the header
    X = np.genfromtxt(trainData, delimiter=',')
    y = np.genfromtxt(trainLabel, delimiter=',')[1:,1]
    X = X[1:,1:]
    X_test = np.genfromtxt(testData, delimiter=',')
    X_test = X_test[1:,1:]
    y_test = np.genfromtxt(testLabel, delimiter=',')[1:,1]
    
    n,m = X.shape
    kf = KFold(n_splits=10, shuffle=True)

    j=1
    accu = np.empty([10,], dtype=float)
    auc = np.empty([10,], dtype=float)
    bscore = np.empty([10,], dtype=float)
    for train_index, test_index in kf.split(X):
        print("Fold:", j)
        X_train, _ = X[train_index], X[test_index]
        y_train, _ = y[train_index], y[test_index]
        
        # Feature selection
        #features = featureSelectionChi(X_train,y_train,30,50)
        features = featureSelectionELAS(X_train,y_train,50)
        #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),10)
        #features = np.arange(m)

        # Subsampling data
        X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
        training, label = subsample(X_combined, t=30)
 
        # Train an ensemble of 30 decision trees
        clf = [None]*30
        for i in range(30):
            dtrain = xgb.DMatrix(training[i][:,features], label=label[i])
            #dtrain = xgb.DMatrix(training[i], label=label[i])
            bst = xgb.train(param, dtrain, num_round)
            clf[i] = bst

        # Prediction
        n = X_test.shape[0]
        dtest = xgb.DMatrix(X_test[:,features], label=y_test)
        #dtest = xgb.DMatrix(X_test, label=y_test)
        # calculting the average probabilty of each class
        pred_prob = np.zeros((n,))
        for i in range(30):
            pred_prob += clf[0].predict(dtest)
            
        pred_prob = pred_prob/30
        # Pick the class with the greastest probability to be the prediction
        pred = (pred_prob>=0.5)
        y_score = pred_prob
    
        # Report accuracy and draw ROC curve
        auc[j-1] = drawROC(y_test, y_score)
        bscore[j-1] = balanced_accuracy_score(y_test, pred)
        score = sum(pred==y_test)/n
        print("Accuracy is:", score)
        print("Balanced Accuracy is:", bscore[j-1])
        accu[j-1] = score
        j = j+1
    print("Average accuracy is:",sum(accu)/10)
    print("Average AUC is:",sum(auc)/10)
    print("Average balanced accuracy is:",sum(bscore)/10)

XGBTEnsemble()
